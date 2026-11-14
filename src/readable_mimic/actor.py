from __future__ import annotations
import torch
import torch.nn as nn
from .mdp import inv_dim, invariant_features
from .rotations import mm3, mv3, so3_exp, yaw_mat
from .common import mlp, RunningNorm, run_mlp


class Actor(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh.

    ``ref_skip`` = (offset, q0, alpha) (--pi_ref_skip): reference-action skip connection,
    a = (q_ref(f+1) - q0) / alpha + dec(...), with q_ref(f+1) read from the raw (unnormalized) goal vector at
    ``offset``; the network only learns the correction to tracking the reference with the PD controller."""

    def __init__(self, obs_dim: int, command_dim: int, act_dim: int, hist: int=10, z_dim: int=32, enc_hidden=(2048, 1024, 512, 256), dec_hidden=(2048, 1024, 512, 256), ref_skip=None):
        super().__init__()
        self.hist = hist
        self.ref_off = None
        if ref_skip is not None:
            self.ref_off = ref_skip[0]
            self.register_buffer('ref_q0', torch.tensor(ref_skip[1], dtype=torch.float32))
            self.register_buffer('ref_alpha', torch.tensor(ref_skip[2], dtype=torch.float32))
        self.enc = mlp(command_dim, list(enc_hidden), z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, list(dec_hidden), act_dim)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(command_dim)

    def forward(self, observation_history: torch.Tensor, command: torch.Tensor, z: torch.Tensor | None=None) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last. ``z`` (optional) is
        ``encode(goal)`` computed beforehand."""
        o = self.obs_norm(observation_history).flatten(-2)
        if z is None:
            z = self.encode(command)
        action = run_mlp(self.dec, torch.cat((z, o), -1))
        if self.ref_off is not None:
            q_ref = command[..., self.ref_off:self.ref_off + action.shape[-1]]
            action = action + (q_ref - self.ref_q0) / self.ref_alpha
        return action

    def encode(self, command: torch.Tensor) -> torch.Tensor:
        """Motion latent z from the goal features (reference only, independent of the robot state)."""
        return torch.tanh(run_mlp(self.enc, self.goal_norm(command)))

def check_pruned(a_: dict, nets_only: bool=False, what: str='this run') -> None:
    """Raise if the run arguments ``a_`` use a removed switch. ``nets_only``: only the switches that change the
    networks (enough to rebuild and evaluate a policy / world model); otherwise also every removed training switch
    (for tools that recompute the training objective)."""
    keys = PRUNED_NET if nets_only else PRUNED_OFF
    bad = {k: a_[k] for k in keys if k in a_ and a_[k] != PRUNED_OFF[k] and (not (PRUNED_OFF[k] is None and (not a_[k])))}
    if bad:
        raise ValueError(f'{what} uses switches that this code does not have: {bad}')

def policy_kwargs(a_: dict, robot) -> dict:
    """Policy keyword arguments of a run with the training arguments ``a_``."""
    check_pruned(a_, nets_only=True)
    return {'ref_skip': policy_ref_skip(a_, robot)}

def policy_ref_skip(a_: dict, robot):
    """Policy(ref_skip=...) argument of a run with the training arguments ``a_`` (None without --pi_ref_skip)."""
    if not a_.get('pi_ref_skip', False):
        return None
    from .motion import GOAL_Q_OFFSET
    return (GOAL_Q_OFFSET, robot.default_joint_pos_list(), robot.action_scale_list())

PRUNED_OFF = {'actor_lam': 1.0, 'grad_decay': 1.0, 'anchored': False, 'demo_replay': False, 'imag_action_noise': 0.0, 'wm_prio': 0.0, 'imag_prio': 0.0, 'critic_alive': False, 'critic_prior': False, 'critic_grad': 0.0, 'critic_ema': 0.0, 'lookahead_k': 0, 'wm_batches_early': 0, 'wm_foot_weight': 0.0, 'bc_sharp_weight': 0.0, 'pi_grad_batch': 0, 'no_critic': False, 'wm_imag_ema': 0.0, 'anchor_coef': 0.0, 'boot_coef': 1.0, 'grad_clip_step': 0.0, 'det_frac': 0.0, 'wm_load': None, 'wm_freeze': False, 'wm_ready': 0.0, 'wm_pd': False, 'pi_arch': 'mlp', 'pi_fourier': None, 'pi_goal_feet': False, 'wm_contact': 0.0}

PRUNED_NET = ('wm_pd', 'pi_arch', 'pi_fourier', 'pi_goal_feet', 'wm_contact')
