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

    def wm_rollout(self, s, action, last_action, simulation_parameters):
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(self.config.wm_window):
            S = self.world_model.step(S, action[:, j], last_action[:, j], self.normalize_parameters(simulation_parameters[:, j]))
            preds.append(S)
        return tuple((torch.stack([P[i] for P in preds], 1) for i in range(6)))

    def wm_err(self, S_hat, s1, mask):
        std = torch.maximum(self.world_model.inc_norm.std(), self.error_floor)
        S1 = unflatten_state(s1)
        e = (self.world_model.state_error(S_hat, S1) / std) ** 2
        e = e.mean(-1)
        return ((e * mask).sum() / s1.shape[0], e[:, -1])

    def imag_step(self, S, o_hist, last_action, motion_frame_index, simulation_parameters, normalized_simulation_parameters, g):
        action = self.actor(o_hist, g)
        S1 = self.world_model.step(S, action, last_action, normalized_simulation_parameters)
        f1 = motion_frame_index + 1
        o1 = self.mdp.add_noise(self.mdp.obs(S1, action, simulation_parameters, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return (S1, action, o_hist, f1, self.motion.command(f1, self.goal_frames))

    def reward_term(self, S, motion_frame_index, action, last_action):
        return (self.mdp.reward(S, motion_frame_index, action, last_action, w_contact=self.config.contact_reward), self.mdp.soft_termination(S, motion_frame_index))

    def gae_targets(self, V, rews, dones):
        """GAE(lambda) returns over the imagined transitions; V is (H+1, B), bootstrapped from V[-1]."""
        return_target = V[-1]
        targets = []
        for t in range(self.horizon - 1, -1, -1):
            return_target = rews[t] + self.config.gamma * (1 - dones[t]) * ((1 - self.config.lam) * V[t + 1] + self.config.lam * return_target)
            targets.append(return_target)
        return torch.stack(targets[::-1])

    def critic_loss(self, X, Y):
        return ((self.critic(X) - Y) ** 2).mean()

    def wm_norm_data(self, S0, S1, action, last_action, simulation_parameters):
        return (self.world_model.inputs(S0, action, last_action, self.normalize_parameters(simulation_parameters)), self.world_model.targets(S0, S1), WorldModel.increments(S0, S1))

    def critic_input(self, S, motion_frame_index, normalized_simulation_parameters, g):
        return torch.cat((self.mdp.critic_features(S, motion_frame_index, normalized_simulation_parameters), g), -1)

    def mirror_windows(self, windows: dict, n: int) -> dict:
        """Replace the first n windows by their mirror images (states, actions, sim context)."""
        out = dict(windows)
        ms = lambda x: flatten_state(self.mirror.state(unflatten_state(x)))
        for k, fn in (('state', ms), ('next_state', ms), ('action', self.mirror.joints), ('last_action', self.mirror.joints), ('simulation_parameters', self.mirror.simulation_parameters)):
            out[k] = torch.cat((fn(windows[k][:n]), windows[k][n:]))
        return out

    def wm_update(self):
        windows = self.buffer.sample_windows(self.config.wm_batch, self.config.wm_window)
        if self.mirror is not None:
            windows = self.mirror_windows(windows, int(self.config.wm_batch * self.config.wm_mirror))
        S_hat = self.wm_rollout(windows['state'], windows['action'], windows['last_action'], windows['simulation_parameters'])
        loss, e_last = self.wm_err(S_hat, windows['next_state'], windows['mask'])
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()
        m = windows['mask'][:, -1]
        return (loss.detach() / self.config.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1))

    def sample_valid_starts(self, B: int):
        starts, hists, n = ([], [], 0)
        for attempt in range(100):
            if n >= B:
                break
            s, h = self.buffer.sample_starts(2 * B, self.history_length)
            motion_frame_index = s['motion_frame_index']
            ok = motion_frame_index + self.horizon <= self.motion.clip_end[self.motion.clip_id[motion_frame_index]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[:B - n]
            (starts.append({k: value[idx] for k, value in s.items()}), hists.append({k: value[idx] for k, value in h.items()}))
            n += len(idx)
        if n < B:
            raise RuntimeError('No valid rollout starts: collect more data or reduce horizon.')
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}
        return (cat(starts), cat(hists))

    def policy_update(self):
        s, h = self.sample_valid_starts(self.config.pi_batch)
        S = unflatten_state(s['state'])
        simulation_parameters, motion_frame_index, last_action = (s['simulation_parameters'], s['motion_frame_index'], s['last_action'])
        normalized_simulation_parameters = self.normalize_parameters(simulation_parameters)
        with torch.no_grad():
            o_hist = self.mdp.obs(unflatten_state(h['state']), h['last_action'], h['simulation_parameters'], h['motion_frame_index'])
        o_hist = self.mdp.add_noise(o_hist)
        set_requires_grad(self.world_model, False)
        set_requires_grad(self.critic, False)
        B = self.config.pi_batch
        g0 = self.motion.command(motion_frame_index, self.goal_frames)
        g = g0
        S0 = S
        Ss, As, Aps, Fs, Gs = ([], [], [], [], [])
        for t in range(self.horizon):
            Aps.append(last_action)
            S, last_action, o_hist, motion_frame_index, g = self.imag_step(S, o_hist, last_action, motion_frame_index, simulation_parameters, normalized_simulation_parameters, g)
            (Ss.append(S), As.append(last_action), Fs.append(motion_frame_index), Gs.append(g))
        flat = lambda xs: torch.stack(xs).flatten(0, 1)
        S_all = tuple((flat([St[i] for St in Ss]) for i in range(6)))
        f_all = flat(Fs)
        r, d = self.reward_term(S_all, f_all, flat(As), flat(Aps))
        r, d = (r.view(self.horizon, B), d.view(self.horizon, B))
        v_boot = self.critic(self.critic_input(S, motion_frame_index, normalized_simulation_parameters, g))
        J, c_t, c = actor_objective(r, d, v_boot, self.config.gamma, loop_cumprod=False)
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = (r.detach(), d.detach())
        loss = -J.mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        if self.config.pi_clip_value > 0:
            nn.utils.clip_grad_value_(self.actor.parameters(), self.config.pi_clip_value)
        if torch.isfinite(gn):
            self.actor_optimizer.step()
        if self.actor_average is not None:
            self.actor_average.update_parameters(self.actor)
        set_requires_grad(self.world_model, True)
        set_requires_grad(self.critic, True)
        critic_loss = self.update_critic(S0, Ss, s, Fs, g0, Gs, normalized_simulation_parameters, rews, dones, B)
        return (loss.detach(), critic_loss.detach(), (r_sum / self.horizon / 0.1).mean(), gn.detach(), c.mean().detach())

    def update_critic(self, S0, Ss, s, Fs, g0, Gs, normalized_simulation_parameters, rews, dones, B):
        v_loss = torch.zeros(())
        with torch.no_grad():
            S_c = tuple((torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6)))
            f_c = torch.cat((s['motion_frame_index'].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = normalized_simulation_parameters.unsqueeze(0).expand(self.horizon + 1, -1, -1).flatten(0, 1)
            X = self.critic_input(S_c, f_c, xi_c, g_c).view(self.horizon + 1, B, -1)
        self.critic.norm.update(X)
        with torch.no_grad():
            targets = self.gae_targets(self.critic(X), rews, dones)
        Xf, Yf = (X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1))
        n = Xf.shape[0]
        mb = n // self.config.critic_minibatches
        for _ in range(self.config.critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(self.config.critic_minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                v_loss = self.critic_loss(Xf[idx], Yf[idx])
                self.critic_optimizer.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_optimizer.step()
        return v_loss
