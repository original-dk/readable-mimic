"""Update equations with optional compilation of WM and imagined actor computations."""

from . import data_transforms as transforms
from .timing import TrainingTimer
import torch
from .task import MAX_STEP_REWARD
from torch import nn
from .grad_optimization import GradientOptimizationConfig, GradientStartSampler, capped_action_gradients


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
        self.grad_config = GradientOptimizationConfig(**getattr(config, "grad_optimization", {}))
        self.start_sampler = GradientStartSampler(self.grad_config, config.device)
        self.grad_metrics = {}
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
        self.timer = TrainingTimer(config.profile_speed, self.device)
        self.horizon = config.horizon
        self.history_length = config.hist
        self.command_frames = config.goal_frames
        self.action_scale = torch.tensor(robot.action_scale_list(), device=self.device)
        self.default_joint_pos = torch.tensor(robot.default_joint_pos_list(), device=self.device)
        self.observation_noise_limits = transforms.compute_observation_noise_limits(
            robot.observation_noise, robot.num_joints
        ).to(self.device)
        self.xi_center = xi_center
        self.xi_scale = xi_scale
        self.command_frame_offsets = command_frame_offsets
        self.horizon_offsets = torch.arange(config.horizon + 1, device=self.device)
        self.policy_gradient_chunk = 32768
        self.discount_factors = (
            config.gamma ** torch.arange(config.horizon, device=self.device, dtype=torch.float32)[:, None]
        )
        self.bootstrap_discount = config.gamma**config.horizon
        if config.compile_world_model and torch.device(self.device).type == "cuda":
            self.world_model_rollout = torch.compile(
                self.world_model_rollout,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            self.world_model_loss = torch.compile(
                self.world_model_loss,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
        if config.compile_actor and torch.device(self.device).type == "cuda":
            self.policy_gradient_loss = torch.compile(
                self.policy_gradient_loss,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            self.critic_features = torch.compile(
                self.critic_features,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            self.imagine_step = torch.compile(
                self.imagine_step,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            self.imagine_step_precomputed = torch.compile(
                self.imagine_step_precomputed,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            self.compute_commands = torch.compile(
                self.compute_commands,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            self.encode_commands = torch.compile(
                self.encode_commands,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            self.reward_and_termination = torch.compile(
                self.reward_and_termination,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            self.reward_termination_and_bodies = torch.compile(
                self.reward_termination_and_bodies,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            self.compute_critic_targets = torch.compile(
                self.compute_critic_targets,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            self.critic_loss = torch.compile(
                self.critic_loss,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )

    def critic_features(
        self,
        state,
        motion_frame_index,
        normalized_xi,
        command,
        init_root,
        init_f,
        previous_target=None,
        body_state=None,
        ref_body_state=None,
    ):
        if self.robot.has_action_delay and previous_target is None:
            raise ValueError("Delayed critic requires previous_target")
        return transforms.compute_critic_input(
            state,
            motion_frame_index,
            normalized_xi,
            command,
            self.task,
            init_root,
            init_f,
            previous_target,
            body_state=body_state,
            ref_body_state=ref_body_state,
        )

    def compute_commands(self, motion_frame_index):
        return transforms.compute_command(motion_frame_index, self.command_frame_offsets, self.motion)

    def encode_commands(self, command):
        return self.actor.encode(command)

    def policy_gradient_loss(self, history, command, action_gradient):
        return -(self.actor(history, command) * action_gradient).sum()

    @torch.no_grad()
    def sample_actor_starts(self):
        batch_size = self.config.pi_batch
        count = batch_size * self.grad_config.candidates if self.grad_config.start_draw else batch_size
        start, history = self.sample_valid_starts(count)
        importance = torch.ones(batch_size, device=self.device)
        quantiles = None
        if self.grad_config.start_draw:
            state = transforms.unflatten_state(start["state"])
            command = transforms.compute_command(start["motion_frame_index"], self.command_frame_offsets, self.motion)
            normalized_xi = transforms.compute_normalized_xi(start["xi"], self.xi_center, self.xi_scale)
            values = self.critic(self.critic_features(
                state, start["motion_frame_index"], normalized_xi, command, start["init_root"], start["init_f"],
                start.get("previous_target"),
            ))
            selected, importance, quantiles = self.start_sampler.draw(values, batch_size)
            start = {key: value[selected] for key, value in start.items()}
            history = {key: value[selected] for key, value in history.items()}
        return start, history, importance, quantiles

    @torch.no_grad()
    def rank_reference(self, frames, horizon):
        state = self.motion.ref_state(frames)
        init_root = transforms.flatten_state(state)[:, :7]
        init_f = frames
        xi = transforms.default_xi(self.robot, self.device).expand(len(frames), -1).clone()
        xi[:, 0] = 1.0
        normalized_xi = transforms.compute_normalized_xi(xi, self.xi_center, self.xi_scale)
        previous_target = state[4] if self.robot.has_action_delay else None
        action = torch.zeros(len(frames), self.robot.num_joints, device=self.device)
        observation = transforms.compute_observation(
            state, action, xi, frames, self.motion, self.default_joint_pos,
        )
        history = transforms.compute_observation_history(observation, self.history_length)
        command = transforms.compute_command(frames, self.command_frame_offsets, self.motion)
        total = torch.zeros(len(frames), device=self.device)
        survival = torch.ones_like(total)
        for step in range(horizon):
            previous_action = action
            state, action, history, frames, command = self.imagine_step(
                state, history, action, frames, xi, normalized_xi, command, noisy=False,
                previous_target=previous_target,
            )
            if self.robot.has_action_delay:
                previous_target = self.default_joint_pos + xi[:, 1:1 + self.robot.num_joints] + self.action_scale * action
            # Rank the same contact-free tracking objective reported in simulation.
            reward = self.task.reward(state, frames, action, previous_action, init_root=init_root, init_f=init_f)
            termination = self.task.soft_termination(state, frames)
            total += self.config.gamma**step * survival * reward
            survival *= 1 - termination
        bootstrap = self.critic(self.critic_features(state, frames, normalized_xi, command, init_root, init_f, previous_target))
        return total + self.config.gamma**horizon * survival * bootstrap

    def world_model_rollout(self, start, action, last_action, xi, previous_targets=None):
        state = transforms.unflatten_state(start[:, 0])
        normalized_xi = transforms.compute_normalized_xi(xi, self.xi_center, self.xi_scale)
        predicted_states = []
        for step_index in range(self.config.wm_window):
            state = self.world_model.step(
                state,
                action[:, step_index],
                last_action[:, step_index],
                normalized_xi[:, step_index],
                None if previous_targets is None else previous_targets[:, step_index],
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

    def _imagine_step_core(
        self,
        state,
        observation_history,
        last_action,
        next_motion_frame_index,
        xi,
        normalized_xi,
        command,
        action_perturbation=None,
        noisy=True,
        previous_target=None,
        motion_latent=None,
    ):
        action = self.actor(observation_history, command, motion_latent=motion_latent)
        if action_perturbation is not None:
            action = action + action_perturbation
        next_state = self.world_model.step(state, action, last_action, normalized_xi, previous_target)
        next_observation = transforms.compute_observation(
            next_state, action, xi, next_motion_frame_index, self.motion, self.default_joint_pos,
        )
        if noisy:
            next_observation = transforms.apply_observation_noise(next_observation, self.observation_noise_limits)
        observation_history = transforms.update_observation_history(observation_history, next_observation)
        return next_state, action, observation_history

    def imagine_step(
        self,
        state,
        observation_history,
        last_action,
        motion_frame_index,
        xi,
        normalized_xi,
        command,
        action_perturbation=None,
        noisy=True,
        previous_target=None,
    ):
        next_motion_frame_index = motion_frame_index + 1
        next_state, action, observation_history = self._imagine_step_core(
            state,
            observation_history,
            last_action,
            next_motion_frame_index,
            xi,
            normalized_xi,
            command,
            action_perturbation=action_perturbation,
            noisy=noisy,
            previous_target=previous_target,
        )
        return (
            next_state,
            action,
            observation_history,
            next_motion_frame_index,
            transforms.compute_command(next_motion_frame_index, self.command_frame_offsets, self.motion),
        )

    def imagine_step_precomputed(
        self,
        state,
        observation_history,
        last_action,
        next_motion_frame_index,
        xi,
        normalized_xi,
        command,
        motion_latent=None,
        action_perturbation=None,
        noisy=True,
        previous_target=None,
    ):
        return self._imagine_step_core(
            state,
            observation_history,
            last_action,
            next_motion_frame_index,
            xi,
            normalized_xi,
            command,
            action_perturbation=action_perturbation,
            noisy=noisy,
            previous_target=previous_target,
            motion_latent=motion_latent,
        )

    def reward_termination_and_bodies(self, state, motion_frame_index, action, last_action, init_root, init_f):
        body_state = self.task.kinematics.forward(*state)
        ref_body_state = self.motion.ref_body_state(motion_frame_index)
        reward = self.task.reward(
            state,
            motion_frame_index,
            action,
            last_action,
            init_root=init_root,
            init_f=init_f,
            foot_contact_reward_weight=self.config.contact_reward,
            body_state=body_state,
            ref_body_state=ref_body_state,
        )
        termination = self.task.soft_termination(state, motion_frame_index)
        return reward, termination, body_state, ref_body_state

    def reward_and_termination(self, state, motion_frame_index, action, last_action, init_root, init_f):
        return (
            self.task.reward(
                state,
                motion_frame_index,
                action,
                last_action,
                init_root=init_root,
                init_f=init_f,
                foot_contact_reward_weight=self.config.contact_reward,
            ),
            self.task.soft_termination(state, motion_frame_index),
        )

    def compute_critic_targets(self, critic_features, detached_rewards, detached_terminations):
        return transforms.compute_value_targets(
            self.critic(critic_features),
            detached_rewards,
            detached_terminations,
            self.config.gamma,
            self.config.lam,
        )

    def _backward_policy_gradients(self, histories, commands, weighted_gradients, batch_size):
        flat_histories = histories.flatten(0, 1)
        flat_commands = commands.flatten(0, 1)
        flat_gradients = (weighted_gradients / batch_size).flatten(0, 1)
        total_samples = flat_histories.shape[0]
        chunk_size = self.policy_gradient_chunk
        if total_samples > chunk_size and total_samples % chunk_size == 0:
            for start_index in range(0, total_samples, chunk_size):
                end_index = start_index + chunk_size
                surrogate = self.policy_gradient_loss(
                    flat_histories[start_index:end_index],
                    flat_commands[start_index:end_index],
                    flat_gradients[start_index:end_index],
                )
                surrogate.backward()
        else:
            self.policy_gradient_loss(flat_histories, flat_commands, flat_gradients).backward()

    def critic_loss(self, critic_features, value_targets, importance=None):
        errors = (self.critic(critic_features) - value_targets) ** 2
        if importance is not None:
            errors = errors * importance
        return errors.mean()

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
            ("previous_target", self.mirror.joints),
            ("xi", self.mirror.xi),
        ):
            if field_name not in windows:
                continue
            out[field_name] = torch.cat(
                (mirror_function(windows[field_name][:num_samples]), windows[field_name][num_samples:])
            )
        return out

    def update_world_model(self):
        windows = self.buffer.sample_windows(self.config.wm_batch)
        if self.mirror is not None:
            windows = self.mirror_windows(windows, int(self.config.wm_batch * self.config.wm_mirror))
        self.timer.record("world_model_prepare")
        predicted_state = self.world_model_rollout(
            windows["state"], windows["action"], windows["last_action"], windows["xi"],
            windows.get("previous_target"),
        )
        loss, last_step_state_error = self.world_model_loss(
            predicted_state, windows["next_state"], windows["mask"]
        )
        self.timer.record("world_model_forward")
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.timer.record("world_model_backward")
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.timer.record("world_model_clip")
        self.world_model_optimizer.step()
        self.timer.record("world_model_optimizer")
        last_step_mask = windows["mask"][:, -1]
        result = (
            loss.detach() / self.config.wm_window,
            (last_step_state_error.detach() * last_step_mask).sum() / last_step_mask.sum().clamp(min=1),
        )
        self.timer.record("world_model_other")
        return result

    def sample_valid_starts(self, batch_size: int):
        starts, histories, num_samples = ([], [], 0)
        for attempt in range(100):
            if num_samples >= batch_size:
                break
            start, history = self.buffer.sample_starts(2 * batch_size)
            motion_frame_index = start["motion_frame_index"]
            valid_starts = (
                motion_frame_index + self.horizon
                <= self.motion.end_frame(motion_frame_index)
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
        self.grad_metrics = {}
        start, history, importance, quantiles = self.sample_actor_starts()
        state = transforms.unflatten_state(start["state"])
        xi, motion_frame_index, last_action = (
            start["xi"],
            start["motion_frame_index"],
            start["last_action"],
        )
        previous_target = start.get("previous_target")
        normalized_xi = transforms.compute_normalized_xi(xi, self.xi_center, self.xi_scale)
        with torch.no_grad():
            observation_history = transforms.compute_replay_observation_history(
                history, self.motion, self.default_joint_pos, self.observation_noise_limits
            )
        set_requires_grad(self.world_model, False)
        set_requires_grad(self.critic, False)
        capped = self.grad_config.cap > 0
        if capped:
            set_requires_grad(self.actor, False)
        action_perturbations, policy_histories = [], []
        batch_size = self.config.pi_batch
        all_frames = motion_frame_index[None, :] + self.horizon_offsets[:, None]
        with torch.no_grad():
            all_commands = self.compute_commands(all_frames.flatten()).view(
                self.horizon + 1, batch_size, -1
            )
            motion_latents = (
                self.encode_commands(all_commands[:-1].flatten(0, 1)).view(self.horizon, batch_size, -1)
                if capped
                else None
            )
        initial_state = state
        rollout_states, rollout_actions, rollout_last_actions = [], [], []
        self.timer.record("actor_prepare")
        for step_index in range(self.horizon):
            perturbation = None
            if capped:
                perturbation = torch.zeros_like(last_action, requires_grad=True)
                action_perturbations.append(perturbation)
                policy_histories.append(observation_history.detach())
            rollout_last_actions.append(last_action)
            state, last_action, observation_history = self.imagine_step_precomputed(
                state,
                observation_history,
                last_action,
                all_frames[step_index + 1],
                xi,
                normalized_xi,
                all_commands[step_index],
                motion_latent=None if motion_latents is None else motion_latents[step_index],
                action_perturbation=perturbation,
                previous_target=previous_target,
            )
            if self.robot.has_action_delay:
                previous_target = (
                    self.default_joint_pos
                    + xi[:, 1 : 1 + self.robot.num_joints]
                    + self.action_scale * last_action
                )
            rollout_states.append(state)
            rollout_actions.append(last_action)
        self.timer.record("actor_rollout")
        flat = lambda xs: torch.stack(xs).flatten(0, 1)
        all_states = tuple(
            (
                flat([step_state[component_index] for step_state in rollout_states])
                for component_index in range(6)
            )
        )
        rewards, termination_probabilities, rollout_body_state, rollout_ref_body_state = (
            self.reward_termination_and_bodies(
                all_states,
                all_frames[1:].flatten(),
                flat(rollout_actions),
                flat(rollout_last_actions),
                start["init_root"].repeat(self.horizon, 1),
                start["init_f"].repeat(self.horizon),
            )
        )
        rewards, termination_probabilities = (
            rewards.view(self.horizon, batch_size),
            termination_probabilities.view(self.horizon, batch_size),
        )
        self.timer.record("reward")
        bootstrap_value = self.critic(
            self.critic_features(
                state,
                all_frames[-1],
                normalized_xi,
                all_commands[-1],
                start["init_root"],
                start["init_f"],
                previous_target,
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
        self.timer.record("actor_return")
        self.actor_optimizer.zero_grad(set_to_none=True)
        if capped:
            gradients = torch.stack(torch.autograd.grad(actor_return.sum(), action_perturbations))
            weighted_gradients, raw_norms, self.grad_metrics = capped_action_gradients(
                gradients, importance, self.grad_config.cap,
            )
            if quantiles is not None:
                self.start_sampler.observe(quantiles, raw_norms)
            self.grad_metrics["start_importance_min"] = importance.min()
            self.grad_metrics["start_importance_max"] = importance.max()
            set_requires_grad(self.actor, True)
            self._backward_policy_gradients(
                torch.stack(policy_histories), all_commands[:-1], weighted_gradients, batch_size
            )
        else:
            loss.backward()
        self.timer.record("actor_backward_gradients")
        gradient_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        if self.config.pi_clip_value > 0:
            nn.utils.clip_grad_value_(self.actor.parameters(), self.config.pi_clip_value)
        self.timer.record("actor_backward_clip")
        if torch.isfinite(gradient_norm):
            self.actor_optimizer.step()
        self.timer.record("actor_backward_optimizer")
        if self.actor_average is not None:
            self.actor_average.update_parameters(self.actor)
        self.timer.record("actor_backward_ema")
        set_requires_grad(self.world_model, True)
        set_requires_grad(self.critic, True)
        self.timer.record("actor_backward_other")
        critic_loss = self.update_critic(
            initial_state,
            rollout_states,
            start,
            all_frames,
            all_commands,
            normalized_xi,
            detached_rewards,
            detached_terminations,
            batch_size,
            importance,
            rollout_actions,
            rollout_body_state=rollout_body_state,
            rollout_ref_body_state=rollout_ref_body_state,
        )
        self.timer.record("critic")
        return (
            loss.detach(),
            critic_loss.detach(),
            (reward_sum / self.horizon / MAX_STEP_REWARD).mean(),
            gradient_norm.detach(),
            final_survival.mean().detach(),
        )

    def update_critic(
        self,
        initial_state,
        rollout_states,
        start,
        all_frames,
        all_commands,
        normalized_xi,
        detached_rewards,
        detached_terminations,
        batch_size,
        importance=None,
        rollout_actions=None,
        rollout_body_state=None,
        rollout_ref_body_state=None,
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
            critic_frames = all_frames.flatten()
            critic_commands = all_commands.flatten(0, 1)
            critic_xi = normalized_xi.unsqueeze(0).expand(self.horizon + 1, -1, -1).flatten(0, 1)
            critic_previous_target = None
            if self.robot.has_action_delay:
                issued_actions = torch.stack([action.detach() for action in rollout_actions])
                issued_targets = (
                    self.default_joint_pos + start["xi"][None, :, 1:1 + self.robot.num_joints]
                    + self.action_scale * issued_actions
                )
                critic_previous_target = torch.cat((start["previous_target"][None], issued_targets), 0).flatten(0, 1)
            critic_body_state = None
            critic_ref_body_state = None
            if rollout_body_state is not None and rollout_ref_body_state is not None:
                initial_body_state = self.task.kinematics.forward(*initial_state)
                initial_ref_body_state = self.motion.ref_body_state(all_frames[0])
                critic_body_state = tuple(
                    torch.cat((init_value, rollout_value.detach()), 0)
                    for init_value, rollout_value in zip(initial_body_state, rollout_body_state)
                )
                critic_ref_body_state = tuple(
                    torch.cat((init_value, rollout_value.detach()), 0)
                    for init_value, rollout_value in zip(initial_ref_body_state, rollout_ref_body_state)
                )
            critic_features = self.critic_features(
                critic_states,
                critic_frames,
                critic_xi,
                critic_commands,
                start["init_root"].repeat(self.horizon + 1, 1),
                start["init_f"].repeat(self.horizon + 1),
                critic_previous_target,
                body_state=critic_body_state,
                ref_body_state=critic_ref_body_state,
            ).view(self.horizon + 1, batch_size, -1)
        self.critic.norm.update(critic_features)
        with torch.no_grad():
            targets = self.compute_critic_targets(critic_features, detached_rewards, detached_terminations)
        flat_critic_features, flat_value_targets = (
            critic_features[:-1].reshape(-1, critic_features.shape[-1]),
            targets.reshape(-1),
        )
        if importance is None:
            importance = torch.ones(batch_size, device=self.device)
        critic_importance = importance.repeat(self.horizon)
        num_samples = flat_critic_features.shape[0]
        minibatch_size = num_samples // self.config.critic_minibatches
        for _ in range(self.config.critic_epochs):
            sample_permutation = torch.randperm(num_samples, device=self.device)
            for component_index in range(self.config.critic_minibatches):
                selected_indices = sample_permutation[
                    component_index * minibatch_size : (component_index + 1) * minibatch_size
                ]
                value_loss = self.critic_loss(
                    flat_critic_features[selected_indices], flat_value_targets[selected_indices],
                    critic_importance[selected_indices],
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_optimizer.step()
        return value_loss
