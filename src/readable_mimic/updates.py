"""Eager FP32 world-model, actor and critic update equations adapted from BlinkMimic."""

import torch
from torch import nn
from .task import flatten_state, unflatten_state
from .world_model import WorldModel
from .objective import actor_objective


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class Updates:
    def world_model_rollout(self, start, action, last_action, simulation_parameters):
        state = unflatten_state(start[:, 0])
        predicted_states = []
        for step_index in range(self.config.wm_window):
            state = self.world_model.step(
                state,
                action[:, step_index],
                last_action[:, step_index],
                self.normalize_parameters(simulation_parameters[:, step_index]),
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
        next_state = unflatten_state(next_states)
        squared_state_error = (self.world_model.state_error(predicted_state, next_state) / error_scale) ** 2
        squared_state_error = squared_state_error.mean(-1)
        return ((squared_state_error * mask).sum() / next_states.shape[0], squared_state_error[:, -1])

    def imagine_step(
        self,
        state,
        observation_history,
        last_action,
        motion_frame_index,
        simulation_parameters,
        normalized_simulation_parameters,
        command,
    ):
        action = self.actor(observation_history, command)
        next_state = self.world_model.step(state, action, last_action, normalized_simulation_parameters)
        next_motion_frame_index = motion_frame_index + 1
        next_observation = self.task.add_noise(
            self.task.observation(next_state, action, simulation_parameters, next_motion_frame_index)
        )
        observation_history = torch.cat((observation_history[:, 1:], next_observation.unsqueeze(1)), 1)
        return (
            next_state,
            action,
            observation_history,
            next_motion_frame_index,
            self.motion.command(next_motion_frame_index, self.command_frames),
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

    def value_targets(self, state_values, detached_rewards, detached_terminations):
        """GAE(lambda) returns over the imagined transitions; V is (H+1, B), bootstrapped from V[-1]."""
        return_target = state_values[-1]
        targets = []
        for step_index in range(self.horizon - 1, -1, -1):
            return_target = detached_rewards[step_index] + self.config.gamma * (
                1 - detached_terminations[step_index]
            ) * ((1 - self.config.lam) * state_values[step_index + 1] + self.config.lam * return_target)
            targets.append(return_target)
        return torch.stack(targets[::-1])

    def critic_loss(self, critic_features, value_targets):
        return ((self.critic(critic_features) - value_targets) ** 2).mean()

    def world_model_normalization_data(
        self, initial_state, next_state, action, last_action, simulation_parameters
    ):
        return (
            self.world_model.inputs(
                initial_state, action, last_action, self.normalize_parameters(simulation_parameters)
            ),
            self.world_model.targets(initial_state, next_state),
            WorldModel.increments(initial_state, next_state),
        )

    def critic_input(self, state, motion_frame_index, normalized_simulation_parameters, command):
        return torch.cat(
            (self.task.critic_features(state, motion_frame_index, normalized_simulation_parameters), command),
            -1,
        )

    def mirror_windows(self, windows: dict, num_samples: int) -> dict:
        """Replace the first n windows by their mirror images (states, actions, sim context)."""
        out = dict(windows)
        mirror_flat_state = lambda x: flatten_state(self.mirror.state(unflatten_state(x)))
        for field_name, mirror_function in (
            ("state", mirror_flat_state),
            ("next_state", mirror_flat_state),
            ("action", self.mirror.joints),
            ("last_action", self.mirror.joints),
            ("simulation_parameters", self.mirror.simulation_parameters),
        ):
            out[field_name] = torch.cat(
                (mirror_function(windows[field_name][:num_samples]), windows[field_name][num_samples:])
            )
        return out

    def update_world_model(self):
        windows = self.buffer.sample_windows(self.config.wm_batch, self.config.wm_window)
        if self.mirror is not None:
            windows = self.mirror_windows(windows, int(self.config.wm_batch * self.config.wm_mirror))
        predicted_state = self.world_model_rollout(
            windows["state"], windows["action"], windows["last_action"], windows["simulation_parameters"]
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
            start, history = self.buffer.sample_starts(2 * batch_size, self.history_length)
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
        state = unflatten_state(start["state"])
        simulation_parameters, motion_frame_index, last_action = (
            start["simulation_parameters"],
            start["motion_frame_index"],
            start["last_action"],
        )
        normalized_simulation_parameters = self.normalize_parameters(simulation_parameters)
        with torch.no_grad():
            observation_history = self.task.observation(
                unflatten_state(history["state"]),
                history["last_action"],
                history["simulation_parameters"],
                history["motion_frame_index"],
            )
        observation_history = self.task.add_noise(observation_history)
        set_requires_grad(self.world_model, False)
        set_requires_grad(self.critic, False)
        batch_size = self.config.pi_batch
        initial_command = self.motion.command(motion_frame_index, self.command_frames)
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
                simulation_parameters,
                normalized_simulation_parameters,
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
            self.critic_input(state, motion_frame_index, normalized_simulation_parameters, command)
        )
        actor_return, survival_weights, final_survival = actor_objective(
            rewards, termination_probabilities, bootstrap_value, self.config.gamma
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
            normalized_simulation_parameters,
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
        normalized_simulation_parameters,
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
            critic_parameters = (
                normalized_simulation_parameters.unsqueeze(0).expand(self.horizon + 1, -1, -1).flatten(0, 1)
            )
            critic_features = self.critic_input(
                critic_states, critic_frames, critic_parameters, critic_commands
            ).view(self.horizon + 1, batch_size, -1)
        self.critic.norm.update(critic_features)
        with torch.no_grad():
            targets = self.value_targets(
                self.critic(critic_features), detached_rewards, detached_terminations
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
