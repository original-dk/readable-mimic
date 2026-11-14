"""Collect -> world model -> imagined actor -> critic; no compilation or CUDA graphs."""

from . import data_transforms as transforms
from .updates import TrainingUpdates
from dataclasses import replace
import json
import math
import time
from pathlib import Path
import torch
from .buffer import ReplayBuffer
from .task import TrackingTask, RewardConfig
from .motion import CONTROL_DT, ReferenceMotion
from .kinematics import Kinematics
from .simulation_parameters import DomainRandomizationConfig, simulation_parameter_count
from .mirror import Mirror
from .actor import Actor, actor_kwargs
from .critic import Critic
from .world_model import WorldModel
from .optimizers import make_optimizer
from .robot import get_robot


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        torch.set_num_threads(config.num_threads)
        torch.manual_seed(config.seed)
        self.horizon = config.horizon
        self.history_length = config.hist
        self.command_frames = config.goal_frames
        self.robot = get_robot(config.robot)
        self.robot = replace(
            self.robot,
            domain_randomization=DomainRandomizationConfig(**config.domain_randomization),
            observation_noise=ObservationNoiseConfig(**config.observation_noise),
        )
        self.kinematics = Kinematics(self.robot, self.device)
        self.motion = ReferenceMotion.from_dir(config.motion_dir, self.robot, self.kinematics, self.device)
        self.num_simulation_parameters = simulation_parameter_count(self.robot)
        self.default_joint_pos = torch.tensor(self.robot.default_joint_pos_list(), device=self.device)
        self.observation_dim = transforms.compute_observation_dim(self.robot.num_joints)
        self.observation_noise_limits = transforms.compute_observation_noise_limits(
            self.robot.observation_noise, self.robot.num_joints
        ).to(self.device)
        self.task = TrackingTask(self.kinematics, self.motion, RewardConfig(**config.reward))
        self.task.feet = [
            self.robot.tracked_bodies.index(name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        if config.ref_contact_fix:
            self.motion.set_ref_foot_contact(self.task.feet)
        joint_count = self.robot.num_joints
        command_dim = transforms.compute_command_dim(self.robot.num_joints, config.goal_frames)
        self.world_model = WorldModel(
            joint_count,
            self.num_simulation_parameters,
            use_last_action=self.robot.has_action_delay,
            hidden_dims=self.widths(config.wm_hidden),
            structured=config.wm_structured,
            control_time_step=CONTROL_DT,
        ).to(self.device)
        self.actor = Actor(
            self.observation_dim,
            command_dim,
            joint_count,
            history_length=config.hist,
            encoder_hidden_dims=self.widths(config.enc_hidden),
            decoder_hidden_dims=self.widths(config.dec_hidden),
            **actor_kwargs(vars(config), self.robot),
        ).to(self.device)
        self.critic = Critic(
            transforms.compute_critic_input_dim(
                joint_count, len(self.robot.tracked_bodies), self.num_simulation_parameters, command_dim
            )
        ).to(self.device)
        self.world_model_optimizer = make_optimizer(
            config.wm_opt, self.world_model, [self.world_model.network], config.lr, (0.9, 0.999)
        )
        self.actor_optimizer = make_optimizer(
            config.pi_opt,
            self.actor,
            [self.actor.command_encoder, self.actor.action_decoder],
            config.lr,
            (config.pi_beta1, 0.95),
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        self.actor_average = torch.optim.swa_utils.AveragedModel(
            self.actor,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config.pi_ema),
            use_buffers=True,
        )
        self.mirror = Mirror(self.robot, self.device) if config.wm_mirror else None
        self.state_error_floor = torch.tensor(
            [0.003] * 3
            + [0.005] * 3
            + [0.02] * 3
            + [0.05] * 3
            + [0.003] * joint_count
            + [0.05] * joint_count,
            device=self.device,
        )
        self.buffer = ReplayBuffer(
            config.buffer_steps, config.num_envs, self.device, joint_count, self.num_simulation_parameters
        )
        self.observation_history = torch.zeros(
            config.num_envs, config.hist, self.observation_dim, device=self.device
        )
        self.iteration = 0
        self.collection_remainder = 0.0
        self.decay_start = None
        self.environment = None
        self.updates = TrainingUpdates(
            config=config,
            actor=self.actor,
            critic=self.critic,
            world_model=self.world_model,
            buffer=self.buffer,
            task=self.task,
            motion=self.motion,
            robot=self.robot,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            world_model_optimizer=self.world_model_optimizer,
            actor_average=self.actor_average,
            mirror=self.mirror,
            state_error_floor=self.state_error_floor,
        )

    @staticmethod
    def widths(text):
        return tuple((int(value) for value in text.split(",")))

    def create_environment(self):
        from .environment import TrackingEnvironment, make_sim

        if self.device.startswith("cuda") and (not torch.cuda.is_available()):
            raise RuntimeError(
                "G1 training requires a CUDA-enabled PyTorch and an NVIDIA GPU. CPU tests: uv run pytest -q"
            )
        self.environment = TrackingEnvironment(
            make_sim(self.robot, self.config.simulation),
            self.config.num_envs,
            self.robot,
            self.motion,
            self.task,
            self.device,
            max_episode_steps=self.config.max_episode_steps,
        )

    @torch.no_grad()
    def collect(self, steps):
        episode_returns = []
        for _ in range(steps):
            observation = self.environment.observe()
            self.observation_history = transforms.update_observation_history(
                self.observation_history, observation, self.environment.episode_step == 0
            )
            command = transforms.compute_command(
                self.environment.motion_frame_index, self.command_frames, self.motion
            )
            self.actor.observation_normalizer.update(observation)
            self.actor.command_normalizer.update(command)
            action = self.actor(self.observation_history, command)
            action = (action + self.config.action_noise * torch.randn_like(action)).clamp(-20, 20)
            transition = self.environment.step(action)
            self.buffer.add(transition)
            inputs, targets, increments = transforms.compute_world_model_training_data(
                self.world_model,
                transforms.unflatten_state(transition["state"]),
                transforms.unflatten_state(transition["next_state"]),
                transition["action"],
                transition["last_action"],
                transition["simulation_parameters"],
                self.robot,
            )
            self.world_model.input_normalizer.update(inputs)
            self.world_model.output_normalizer.update(targets)
            self.world_model.increment_normalizer.update(increments)
            if "normalized_episode_return" in transition:
                episode_returns.append(float(transition["normalized_episode_return"].mean()))
        return episode_returns

    def set_learning_rate(self, iteration):
        config = self.config
        progress = min(iteration / config.lr_decay_iters, 1)
        rate = config.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * progress)))
        multiplier = config.pi_lr_mult
        if self.decay_start is not None:
            fraction = min((iteration - self.decay_start) / max(config.lr_decay_len, 1), 1)
            multiplier += (1 - multiplier) * fraction
        for optimizer, scale in (
            (self.world_model_optimizer, 1),
            (self.actor_optimizer, multiplier),
            (self.critic_optimizer, 1),
        ):
            for group in optimizer.param_groups:
                group["lr"] = rate * scale

    def run(self):
        config = self.config
        output = Path(config.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(vars(config), indent=2), encoding="utf-8")
        self.create_environment()
        return_window = self.collect(config.warmup_steps)
        start_time = time.monotonic()
        next_checkpoint_time = start_time + config.ckpt_minutes * 60
        for iteration in range(self.iteration, config.iterations):
            self.set_learning_rate(iteration)
            self.collection_remainder += 1 / config.utd
            steps = int(self.collection_remainder)
            self.collection_remainder -= steps
            returns = self.collect(steps)
            return_window.extend(returns)
            for _ in range(config.wm_batches):
                world_model_loss, _ = self.updates.update_world_model()
            metrics = {"iteration": iteration + 1, "world_model_loss": float(world_model_loss)}
            if iteration >= config.wm_only_iters:
                actor_loss, critic_loss, imagined_reward, gradient_norm, survival = (
                    self.updates.update_actor()
                )
                metrics.update(
                    actor_loss=float(actor_loss),
                    critic_loss=float(critic_loss),
                    imagined_reward=float(imagined_reward),
                    gradient_norm=float(gradient_norm),
                    survival=float(survival),
                )
            self.iteration = iteration + 1
            if self.iteration % config.log_every == 0:
                average_return = sum(return_window) / max(len(return_window), 1)
                deadline = config.lr_decay_by > 0 and self.iteration >= config.lr_decay_by
                if (
                    self.decay_start is None
                    and config.lr_decay_ret > 0
                    and (average_return >= config.lr_decay_ret or deadline)
                ):
                    self.decay_start = self.iteration
                metrics["episode_return"] = average_return
                return_window.clear()
                print(json.dumps(metrics), flush=True)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics) + "\n")
            timed_checkpoint = config.ckpt_minutes > 0 and time.monotonic() >= next_checkpoint_time
            if self.iteration % config.ckpt_every == 0 or timed_checkpoint:
                self.save(output / "last.pt")
                next_checkpoint_time = time.monotonic() + config.ckpt_minutes * 60
            if config.max_hours is not None and time.monotonic() - start_time >= config.max_hours * 3600:
                break
        self.save(output / "last.pt")

    def save(self, path):
        state = {
            name: getattr(self, name).state_dict()
            for name in (
                "actor",
                "actor_average",
                "critic",
                "world_model",
                "actor_optimizer",
                "critic_optimizer",
                "world_model_optimizer",
            )
        }
        state.update(
            config=vars(self.config),
            iteration=self.iteration,
            collection_remainder=self.collection_remainder,
            decay_start=self.decay_start,
            torch_rng_state=torch.get_rng_state(),
        )
        temporary = Path(str(path) + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        for name in (
            "actor",
            "actor_average",
            "critic",
            "world_model",
            "actor_optimizer",
            "critic_optimizer",
            "world_model_optimizer",
        ):
            getattr(self, name).load_state_dict(state[name])
        self.iteration = state["iteration"]
        self.collection_remainder = state["collection_remainder"]
        self.decay_start = state["decay_start"]
        torch.set_rng_state(state["torch_rng_state"].cpu())


from .config import ObservationNoiseConfig
