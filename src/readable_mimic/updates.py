"""Eager FP32 world-model, actor and critic update equations adapted from BlinkMimic."""

import torch
from torch import nn
from .mdp import flatten_state, unflatten_state
from .networks import WorldModel
from .objective import actor_objective


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class Updates:
    def wm_rollout(self, start, action, last_action, simulation_parameters):
        state = unflatten_state(start[:, 0])
        preds = []
        for j in range(self.config.wm_window):
            state = self.world_model.step(
                state, action[:, j], last_action[:, j], self.normalize_parameters(simulation_parameters[:, j])
            )
            preds.append(state)
        return tuple((torch.stack([P[i] for P in preds], 1) for i in range(6)))

    def wm_err(self, predicted_state, next_states, mask):
        std = torch.maximum(self.world_model.inc_norm.std(), self.error_floor)
        next_state = unflatten_state(next_states)
        e = (self.world_model.state_error(predicted_state, next_state) / std) ** 2
        e = e.mean(-1)
        return ((e * mask).sum() / next_states.shape[0], e[:, -1])

    def imag_step(
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
        f1 = motion_frame_index + 1
        next_observation = self.mdp.add_noise(self.mdp.obs(next_state, action, simulation_parameters, f1))
        observation_history = torch.cat((observation_history[:, 1:], next_observation.unsqueeze(1)), 1)
        return (next_state, action, observation_history, f1, self.motion.command(f1, self.goal_frames))

    def reward_term(self, state, motion_frame_index, action, last_action):
        return (
            self.mdp.reward(
                state, motion_frame_index, action, last_action, w_contact=self.config.contact_reward
            ),
            self.mdp.soft_termination(state, motion_frame_index),
        )

    def gae_targets(self, state_values, detached_rewards, detached_terminations):
        """GAE(lambda) returns over the imagined transitions; V is (H+1, B), bootstrapped from V[-1]."""
        return_target = state_values[-1]
        targets = []
        for t in range(self.horizon - 1, -1, -1):
            return_target = detached_rewards[t] + self.config.gamma * (1 - detached_terminations[t]) * (
                (1 - self.config.lam) * state_values[t + 1] + self.config.lam * return_target
            )
            targets.append(return_target)
        return torch.stack(targets[::-1])

    def critic_loss(self, critic_features, value_targets):
        return ((self.critic(critic_features) - value_targets) ** 2).mean()

    def wm_norm_data(self, initial_state, next_state, action, last_action, simulation_parameters):
        return (
            self.world_model.inputs(
                initial_state, action, last_action, self.normalize_parameters(simulation_parameters)
            ),
            self.world_model.targets(initial_state, next_state),
            WorldModel.increments(initial_state, next_state),
        )

    def critic_input(self, state, motion_frame_index, normalized_simulation_parameters, command):
        return torch.cat(
            (self.mdp.critic_features(state, motion_frame_index, normalized_simulation_parameters), command),
            -1,
        )

    def mirror_windows(self, windows: dict, n: int) -> dict:
        """Replace the first n windows by their mirror images (states, actions, sim context)."""
        out = dict(windows)
        ms = lambda x: flatten_state(self.mirror.state(unflatten_state(x)))
        for k, fn in (
            ("state", ms),
            ("next_state", ms),
            ("action", self.mirror.joints),
            ("last_action", self.mirror.joints),
            ("simulation_parameters", self.mirror.simulation_parameters),
        ):
            out[k] = torch.cat((fn(windows[k][:n]), windows[k][n:]))
        return out

    def wm_update(self):
        windows = self.buffer.sample_windows(self.config.wm_batch, self.config.wm_window)
        if self.mirror is not None:
            windows = self.mirror_windows(windows, int(self.config.wm_batch * self.config.wm_mirror))
        predicted_state = self.wm_rollout(
            windows["state"], windows["action"], windows["last_action"], windows["simulation_parameters"]
        )
        loss, e_last = self.wm_err(predicted_state, windows["next_state"], windows["mask"])
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()
        m = windows["mask"][:, -1]
        return (loss.detach() / self.config.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1))

    def sample_valid_starts(self, batch_size: int):
        starts, hists, n = ([], [], 0)
        for attempt in range(100):
            if n >= batch_size:
                break
            start, history = self.buffer.sample_starts(2 * batch_size, self.history_length)
            motion_frame_index = start["motion_frame_index"]
            ok = (
                motion_frame_index + self.horizon
                <= self.motion.clip_end[self.motion.clip_id[motion_frame_index]]
            )
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[: batch_size - n]
            (
                starts.append({k: value[idx] for k, value in start.items()}),
                hists.append({k: value[idx] for k, value in history.items()}),
            )
            n += len(idx)
        if n < batch_size:
            raise RuntimeError("No valid rollout starts: collect more data or reduce horizon.")
        cat = lambda ds: {
            k: torch.cat([termination_probabilities[k] for termination_probabilities in ds]) for k in ds[0]
        }
        return (cat(starts), cat(hists))

    def policy_update(self):
        start, history = self.sample_valid_starts(self.config.pi_batch)
        state = unflatten_state(start["state"])
        simulation_parameters, motion_frame_index, last_action = (
            start["simulation_parameters"],
            start["motion_frame_index"],
            start["last_action"],
        )
        normalized_simulation_parameters = self.normalize_parameters(simulation_parameters)
        with torch.no_grad():
            observation_history = self.mdp.obs(
                unflatten_state(history["state"]),
                history["last_action"],
                history["simulation_parameters"],
                history["motion_frame_index"],
            )
        observation_history = self.mdp.add_noise(observation_history)
        set_requires_grad(self.world_model, False)
        set_requires_grad(self.critic, False)
        batch_size = self.config.pi_batch
        initial_command = self.motion.command(motion_frame_index, self.goal_frames)
        command = initial_command
        initial_state = state
        rollout_states, rollout_actions, rollout_last_actions, rollout_frames, rollout_commands = (
            [],
            [],
            [],
            [],
            [],
        )
        for t in range(self.horizon):
            rollout_last_actions.append(last_action)
            state, last_action, observation_history, motion_frame_index, command = self.imag_step(
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
        all_states = tuple((flat([step_state[i] for step_state in rollout_states]) for i in range(6)))
        all_frames = flat(rollout_frames)
        rewards, termination_probabilities = self.reward_term(
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
            rewards, termination_probabilities, bootstrap_value, self.config.gamma, loop_cumprod=False
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
                            initial_state[i].unsqueeze(0),
                            torch.stack([step_state[i].detach() for step_state in rollout_states]),
                        ),
                        0,
                    ).flatten(0, 1)
                    for i in range(6)
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
            targets = self.gae_targets(self.critic(critic_features), detached_rewards, detached_terminations)
        flat_critic_features, flat_value_targets = (
            critic_features[:-1].reshape(-1, critic_features.shape[-1]),
            targets.reshape(-1),
        )
        n = flat_critic_features.shape[0]
        mb = n // self.config.critic_minibatches
        for _ in range(self.config.critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(self.config.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                value_loss = self.critic_loss(flat_critic_features[idx], flat_value_targets[idx])
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_optimizer.step()
        return value_loss
