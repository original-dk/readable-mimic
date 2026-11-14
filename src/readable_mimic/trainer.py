"""Collect -> world model -> imagined actor -> critic; no compilation or CUDA graphs."""
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from .buffer import ReplayBuffer
from .mdp import MDP, RewardCfg, inv_dim, unflatten_state
from .motion import CONTROL_DT, MotionLib
from .kinematics import Kinematics
from .dr import DomainRandCfg, XiLayout
from .sensing import SensingCfg
from .mirror import Mirror
from .networks import Actor, Critic, WorldModel, policy_kwargs
from .optimizers import make_optimizer
from .robot import get_robot
from .updates import Updates


class Trainer(Updates):
    def __init__(self, config):
        self.config = config
        self.device = config.device
        torch.set_num_threads(config.num_threads)
        torch.manual_seed(config.seed)
        self.horizon = config.horizon
        self.history_length = config.hist
        self.goal_frames = config.goal_frames
        self.robot = get_robot(config.robot)
        self.robot = replace(self.robot, dr=DomainRandCfg(**config.domain_randomization),
                             sensing=SensingCfg(**config.sensing))
        self.kinematics = Kinematics(self.robot, self.device)
        self.motion = MotionLib.from_dir(config.motion_dir, self.robot, self.kinematics, self.device)
        self.parameter_layout = XiLayout(self.robot, self.device)
        self.normalize_parameters = self.parameter_layout.normalize
        self.mdp = MDP(self.kinematics, self.motion,
                       torch.tensor(self.robot.default_joint_pos_list(), device=self.device),
                       self.parameter_layout.dim, self.robot.sensing, RewardCfg(**config.reward))
        assert not self.robot.sensing.stateful, 'First version uses the paper observation noise only.'
        self.mdp.feet = [self.robot.tracked_bodies.index(name)
                         for name in ('left_ankle_roll_link', 'right_ankle_roll_link')]
        if config.ref_contact_fix:
            self.motion.set_ref_contact(self.mdp.feet)
        joint_count = self.robot.num_joints
        command_dim = self.motion.command_dim(config.goal_frames)
        self.world_model = WorldModel(joint_count, self.parameter_layout.dim,
            use_prev_action=self.robot.has_action_delay, hidden=self.widths(config.wm_hidden),
            structured=config.wm_structured, dt=CONTROL_DT).to(self.device)
        self.actor = Actor(self.mdp.obs_dim, command_dim, joint_count, hist=config.hist,
            enc_hidden=self.widths(config.enc_hidden), dec_hidden=self.widths(config.dec_hidden),
            **policy_kwargs(vars(config), self.robot)).to(self.device)
        self.critic = Critic(inv_dim(joint_count) + self.mdp.critic_extra + command_dim).to(self.device)
        self.world_model_optimizer = make_optimizer(config.wm_opt, self.world_model,
            [self.world_model.net], config.lr, (0.9, 0.999))
        self.actor_optimizer = make_optimizer(config.pi_opt, self.actor,
            [self.actor.enc, self.actor.dec], config.lr, (config.pi_beta1, 0.95))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        self.actor_average = torch.optim.swa_utils.AveragedModel(self.actor,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config.pi_ema), use_buffers=True)
        self.mirror = Mirror(self.robot, self.parameter_layout.slices, self.device) if config.wm_mirror else None
        self.error_floor = torch.tensor([0.003]*3 + [0.005]*3 + [0.02]*3 + [0.05]*3
                                       + [0.003]*joint_count + [0.05]*joint_count, device=self.device)
        self.buffer = ReplayBuffer(config.buffer_steps, config.num_envs, self.device,
                                   joint_count, self.parameter_layout.dim)
        self.observation_history = torch.zeros(config.num_envs, config.hist, self.mdp.obs_dim, device=self.device)
        self.iteration = 0
        self.collection_remainder = 0.0
        self.decay_start = None
        self.environment = None

    @staticmethod
    def widths(text):
        return tuple(int(value) for value in text.split(','))

    def create_environment(self):
        from .environment import TrackingSim, make_sim
        self.environment = TrackingSim(make_sim(self.robot), self.config.num_envs, self.robot,
            self.motion, self.mdp, self.device, gamma=self.config.gamma,
            max_episode_steps=self.config.max_episode_steps)

    @torch.no_grad()
    def collect(self, steps):
        episode_returns = []
        for _ in range(steps):
            observation = self.environment.observe()
            history = torch.cat((self.observation_history[:, 1:], observation.unsqueeze(1)), 1)
            fresh = self.environment.ep_step == 0
            self.observation_history = torch.where(fresh[:, None, None],
                observation[:, None].expand_as(history), history)
            command = self.motion.command(self.environment.motion_frame_index, self.goal_frames)
            self.actor.obs_norm.update(observation)
            self.actor.goal_norm.update(command)
            action = self.actor(self.observation_history, command)
            action = (action + self.config.action_noise * torch.randn_like(action)).clamp(-20, 20)
            transition = self.environment.step(action)
            self.buffer.add(transition)
            inputs, targets, increments = self.wm_norm_data(
                unflatten_state(transition['state']), unflatten_state(transition['next_state']),
                transition['action'], transition['last_action'], transition['simulation_parameters'])
            self.world_model.in_norm.update(inputs)
            self.world_model.out_norm.update(targets)
            self.world_model.inc_norm.update(increments)
            if 'ep_ret_norm' in transition:
                episode_returns.extend(transition['ep_ret_norm'].tolist())
        return episode_returns

    def set_learning_rate(self, iteration):
        config = self.config
        progress = min(iteration / config.lr_decay_iters, 1)
        rate = config.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * progress)))
        multiplier = config.pi_lr_mult
        if self.decay_start is not None:
            fraction = min((iteration-self.decay_start) / max(config.lr_decay_len, 1), 1)
            multiplier += (1 - multiplier) * fraction
        for optimizer, scale in ((self.world_model_optimizer, 1),
                                  (self.actor_optimizer, multiplier), (self.critic_optimizer, 1)):
            for group in optimizer.param_groups:
                group['lr'] = rate * scale

    def run(self):
        config = self.config
        output = Path(config.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'config.json').write_text(json.dumps(vars(config), indent=2), encoding='utf-8')
        self.create_environment()
        self.collect(config.warmup_steps)
        for iteration in range(self.iteration, config.iterations):
            self.set_learning_rate(iteration)
            self.collection_remainder += 1 / config.utd
            steps = int(self.collection_remainder)
            self.collection_remainder -= steps
            returns = self.collect(steps)
            for _ in range(config.wm_batches):
                world_model_loss, _ = self.wm_update()
            metrics = {'iteration': iteration+1, 'world_model_loss': float(world_model_loss)}
            if iteration >= config.wm_only_iters:
                actor_loss, critic_loss, imagined_reward, gradient_norm, survival = self.policy_update()
                metrics.update(actor_loss=float(actor_loss), critic_loss=float(critic_loss),
                    imagined_reward=float(imagined_reward), gradient_norm=float(gradient_norm), survival=float(survival))
            average_return = sum(returns) / max(len(returns), 1)
            deadline = config.lr_decay_by > 0 and iteration+1 >= config.lr_decay_by
            if self.decay_start is None and config.lr_decay_ret > 0 and (average_return >= config.lr_decay_ret or deadline):
                self.decay_start = iteration+1
            self.iteration = iteration+1
            if self.iteration % config.log_every == 0:
                print(json.dumps(metrics), flush=True)
                with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(metrics) + '\n')
            if self.iteration % config.ckpt_every == 0:
                self.save(output / 'last.pt')
        self.save(output / 'last.pt')

    def save(self, path):
        state = {name: getattr(self, name).state_dict() for name in
                 ('actor', 'actor_average', 'critic', 'world_model', 'actor_optimizer',
                  'critic_optimizer', 'world_model_optimizer')}
        state.update(config=vars(self.config), iteration=self.iteration,
                     collection_remainder=self.collection_remainder, decay_start=self.decay_start,
                     torch_rng_state=torch.get_rng_state())
        temporary = Path(str(path) + '.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        for name in ('actor', 'actor_average', 'critic', 'world_model', 'actor_optimizer',
                     'critic_optimizer', 'world_model_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        self.iteration = state['iteration']
        self.collection_remainder = state['collection_remainder']
        self.decay_start = state['decay_start']
        torch.set_rng_state(state['torch_rng_state'].cpu())
