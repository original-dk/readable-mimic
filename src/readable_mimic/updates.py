"""Eager FP32 world-model, actor and critic update equations adapted from BlinkMimic."""

from . import data_transforms as transforms
import torch
from torch import nn


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class TrainingUpdates:
    def __init__(
        self,
        *,
        config,
        actor,
        critic,
        world_model,
        buffer,
        task,
        motion,
        robot,
        actor_optimizer,
        critic_optimizer,
        world_model_optimizer,
        actor_average,
        mirror,
        state_error_floor,
        xi_center,
        xi_scale,
        command_frame_offsets,
    ):
        self.config = config
        self.actor = actor
        self.critic = critic
        self.world_model = world_model
        self.buffer = buffer
        self.task = task
        self.motion = motion
        self.robot = robot
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.world_model_optimizer = world_model_optimizer
        self.actor_average = actor_average
        self.mirror = mirror
        self.state_error_floor = state_error_floor
        self.device = config.device
        self.horizon = config.horizon
        self.history_length = config.hist
        self.command_frames = config.goal_frames
        self.default_joint_pos = torch.tensor(robot.default_joint_pos_list(), device=self.device)
        self.observation_noise_limits = transforms.compute_observation_noise_limits(
            robot.observation_noise, robot.num_joints
        ).to(self.device)
        self.xi_center = xi_center
        self.xi_scale = xi_scale
        self.command_frame_offsets = command_frame_offsets
        self.discount_factors = (
            config.gamma ** torch.arange(config.horizon, device=self.device, dtype=torch.float32)[:, None]
        )
        self.bootstrap_discount = config.gamma**config.horizon

    def world_model_rollout(self, start, action, last_action, xi):
        state = transforms.unflatten_state(start[:, 0])
        normalized_xi = transforms.compute_normalized_xi(xi, self.xi_center, self.xi_scale)
        predicted_states = []
        for step_index in range(self.config.wm_window):
            state = self.world_model.step(
                state,
                action[:, step_index],
                last_action[:, step_index],
                normalized_xi[:, step_index],
            )
            predicted_states.append(state)
        return tuple(
            (
                torch.stack(
                    [predicted_step_state[component_index] for predicted_step_state in predicted_states], 1
                )
                for component_index in range(6)
            )
        )

    def world_model_loss(self, predicted_state, next_states, mask):
        error_scale = torch.maximum(self.world_model.increment_normalizer.std(), self.state_error_floor)
        next_state = transforms.unflatten_state(next_states)
        squared_state_error = (self.world_model.state_error(predicted_state, next_state) / error_scale) ** 2
        squared_state_error = squared_state_error.mean(-1)
        return ((squared_state_error * mask).sum() / next_states.shape[0], squared_state_error[:, -1])

    def imagine_step(
        self,
        state,
        observation_history,
        last_action,
        motion_frame_index,
        xi,
        normalized_xi,
        command,
    ):
        action = self.actor(observation_history, command)
        next_state = self.world_model.step(state, action, last_action, normalized_xi)
        next_motion_frame_index = motion_frame_index + 1
        next_observation = transforms.apply_observation_noise(
            transforms.compute_observation(
                next_state,
                action,
                xi,
                next_motion_frame_index,
                self.motion,
                self.default_joint_pos,
            ),
            self.observation_noise_limits,
        )
        observation_history = transforms.update_observation_history(observation_history, next_observation)
        return (
            next_state,
            action,
            observation_history,
            next_motion_frame_index,
            transforms.compute_command(next_motion_frame_index, self.command_frame_offsets, self.motion),
        )

    def reward_and_termination(self, state, motion_frame_index, action, last_action):
        return (
            self.task.reward(
                state,
                motion_frame_index,
                action,
                last_action,
                foot_contact_reward_weight=self.config.contact_reward,
            ),
            self.task.soft_termination(state, motion_frame_index),
        )

    def critic_loss(self, critic_features, value_targets):
        return ((self.critic(critic_features) - value_targets) ** 2).mean()

    def mirror_windows(self, windows: dict, num_samples: int) -> dict:
        """Replace the first n windows by their mirror images (states, actions, sim context)."""
        out = dict(windows)
        mirror_flat_state = lambda x: transforms.flatten_state(
            self.mirror.state(transforms.unflatten_state(x))
        )
        for field_name, mirror_function in (
            ("state", mirror_flat_state),
            ("next_state", mirror_flat_state),
            ("action", self.mirror.joints),
            ("last_action", self.mirror.joints),
            ("xi", self.mirror.xi),
        ):
            out[field_name] = torch.cat(
                (mirror_function(windows[field_name][:num_samples]), windows[field_name][num_samples:])
            )
        return out

    def update_world_model(self):
        windows = self.buffer.sample_windows(self.config.wm_batch)
        if self.mirror is not None:
            windows = self.mirror_windows(windows, int(self.config.wm_batch * self.config.wm_mirror))
        predicted_state = self.world_model_rollout(
            windows["state"], windows["action"], windows["last_action"], windows["xi"]
        )
        loss, last_step_state_error = self.world_model_loss(
            predicted_state, windows["next_state"], windows["mask"]
        )
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()
        last_step_mask = windows["mask"][:, -1]
        return (
            loss.detach() / self.config.wm_window,
            (last_step_state_error.detach() * last_step_mask).sum() / last_step_mask.sum().clamp(min=1),
        )

    def sample_valid_starts(self, batch_size: int):
        starts, histories, num_samples = ([], [], 0)
        for attempt in range(100):
            if num_samples >= batch_size:
                break
            start, history = self.buffer.sample_starts(2 * batch_size)
            motion_frame_index = start["motion_frame_index"]
            valid_starts = (
                motion_frame_index + self.horizon
                <= self.motion.clip_end[self.motion.clip_id[motion_frame_index]]
            )
            selected_indices = valid_starts.nonzero(as_tuple=False).squeeze(-1)[: batch_size - num_samples]
            (
                starts.append({field_name: value[selected_indices] for field_name, value in start.items()}),
                histories.append(
                    {field_name: value[selected_indices] for field_name, value in history.items()}
                ),
            )
            num_samples += len(selected_indices)
        if num_samples < batch_size:
            raise RuntimeError("No valid rollout starts: collect more data or reduce horizon.")
        concatenate_batches = lambda batches: {
            field_name: torch.cat(
                [termination_probabilities[field_name] for termination_probabilities in batches]
            )
            for field_name in batches[0]
        }
        return (concatenate_batches(starts), concatenate_batches(histories))

    def update_actor(self):
        start, history = self.sample_valid_starts(self.config.pi_batch)
        state = transforms.unflatten_state(start["state"])
        xi, motion_frame_index, last_action = (
            start["xi"],
            start["motion_frame_index"],
            start["last_action"],
        )
        normalized_xi = transforms.compute_normalized_xi(xi, self.xi_center, self.xi_scale)
        with torch.no_grad():
            observation_history = transforms.compute_replay_observation_history(
                history, self.motion, self.default_joint_pos, self.observation_noise_limits
            )
        set_requires_grad(self.world_model, False)
        set_requires_grad(self.critic, False)
        batch_size = self.config.pi_batch
        initial_command = transforms.compute_command(
            motion_frame_index, self.command_frame_offsets, self.motion
        )
        command = initial_command
        initial_state = state
        rollout_states, rollout_actions, rollout_last_actions, rollout_frames, rollout_commands = (
            [],
            [],
            [],
            [],
            [],
        )
        for step_index in range(self.horizon):
            rollout_last_actions.append(last_action)
            state, last_action, observation_history, motion_frame_index, command = self.imagine_step(
                state,
                observation_history,
                last_action,
                motion_frame_index,
                xi,
                normalized_xi,
                command,
            )
            (
                rollout_states.append(state),
                rollout_actions.append(last_action),
                rollout_frames.append(motion_frame_index),
                rollout_commands.append(command),
            )
        flat = lambda xs: torch.stack(xs).flatten(0, 1)
        all_states = tuple(
            (
                flat([step_state[component_index] for step_state in rollout_states])
                for component_index in range(6)
            )
        )
        all_frames = flat(rollout_frames)
        rewards, termination_probabilities = self.reward_and_termination(
            all_states, all_frames, flat(rollout_actions), flat(rollout_last_actions)
        )
        rewards, termination_probabilities = (
            rewards.view(self.horizon, batch_size),
            termination_probabilities.view(self.horizon, batch_size),
        )
        bootstrap_value = self.critic(
            transforms.compute_critic_input(
                state,
                motion_frame_index,
                normalized_xi,
                command,
                self.motion,
                self.task.kinematics,
            )
        )
        actor_return, survival_weights, final_survival = transforms.compute_actor_return(
            rewards,
            termination_probabilities,
            bootstrap_value,
            self.discount_factors,
            self.bootstrap_discount,
        )
        reward_sum = (survival_weights * rewards).detach().sum(0)
        detached_rewards, detached_terminations = (rewards.detach(), termination_probabilities.detach())
        loss = -actor_return.mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        if self.config.pi_clip_value > 0:
            nn.utils.clip_grad_value_(self.actor.parameters(), self.config.pi_clip_value)
        if torch.isfinite(gradient_norm):
            self.actor_optimizer.step()
        if self.actor_average is not None:
            self.actor_average.update_parameters(self.actor)
        set_requires_grad(self.world_model, True)
        set_requires_grad(self.critic, True)
        critic_loss = self.update_critic(
            initial_state,
            rollout_states,
            start,
            rollout_frames,
            initial_command,
            rollout_commands,
            normalized_xi,
            detached_rewards,
            detached_terminations,
            batch_size,
        )
        return (
            loss.detach(),
            critic_loss.detach(),
            (reward_sum / self.horizon / 0.1).mean(),
            gradient_norm.detach(),
            final_survival.mean().detach(),
        )

    def update_critic(
        self,
        initial_state,
        rollout_states,
        start,
        rollout_frames,
        initial_command,
        rollout_commands,
        normalized_xi,
        detached_rewards,
        detached_terminations,
        batch_size,
    ):
        value_loss = torch.zeros(())
        with torch.no_grad():
            critic_states = tuple(
                (
                    torch.cat(
                        (
                            initial_state[component_index].unsqueeze(0),
                            torch.stack(
                                [step_state[component_index].detach() for step_state in rollout_states]
                            ),
                        ),
                        0,
                    ).flatten(0, 1)
                    for component_index in range(6)
                )
            )
            critic_frames = torch.cat(
                (start["motion_frame_index"].unsqueeze(0), torch.stack(rollout_frames)), 0
            ).flatten()
            critic_commands = torch.cat(
                (initial_command.unsqueeze(0), torch.stack(rollout_commands)), 0
            ).flatten(0, 1)
            critic_xi = normalized_xi.unsqueeze(0).expand(self.horizon + 1, -1, -1).flatten(0, 1)
            critic_features = transforms.compute_critic_input(
                critic_states,
                critic_frames,
                critic_xi,
                critic_commands,
                self.motion,
                self.task.kinematics,
            ).view(self.horizon + 1, batch_size, -1)
        self.critic.norm.update(critic_features)
        with torch.no_grad():
            targets = transforms.compute_value_targets(
                self.critic(critic_features),
                detached_rewards,
                detached_terminations,
                self.config.gamma,
                self.config.lam,
            )
        flat_critic_features, flat_value_targets = (
            critic_features[:-1].reshape(-1, critic_features.shape[-1]),
            targets.reshape(-1),
        )
        num_samples = flat_critic_features.shape[0]
        minibatch_size = num_samples // self.config.critic_minibatches
        for _ in range(self.config.critic_epochs):
            sample_permutation = torch.randperm(num_samples, device=self.device)
            for component_index in range(self.config.critic_minibatches):
                selected_indices = sample_permutation[
                    component_index * minibatch_size : (component_index + 1) * minibatch_size
                ]
                value_loss = self.critic_loss(
                    flat_critic_features[selected_indices], flat_value_targets[selected_indices]
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_optimizer.step()
        return value_loss
