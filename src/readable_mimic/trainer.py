"""Collect -> world model -> imagined actor -> critic, with optional compiled WM updates."""

from .sim import joint_pos_offset_limits, CONTROL_DT, DomainRandomizationConfig, compute_xi_dim
import random, shutil
import numpy as np
from . import data_transforms as transforms
from .updates import TrainingUpdates
from .config import ObservationNoiseConfig
from dataclasses import replace
import json
import math
import time
from pathlib import Path
import torch
from .buffer import ReplayBuffer
from .task import TrackingTask, RewardConfig, MAX_STEP_REWARD
from .motion import ReferenceMotion
from .kinematics import Kinematics
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
        if torch.device(self.device).type == "cuda":
            # PyTorch 2.9 Inductor still reads allow_tf32 (pad_mm.py).
            # Keep backend state on the legacy API until that dependency is upgraded.
            torch.set_float32_matmul_precision("high" if config.allow_tf32 else "highest")
            torch.backends.cudnn.allow_tf32 = config.allow_tf32
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
        self.xi_dim = compute_xi_dim(self.robot)
        self.joint_pos_offset_limits = joint_pos_offset_limits(self.robot, self.device)
        self.xi_center, self.xi_scale = transforms.compute_xi_normalization(
            self.robot, self.joint_pos_offset_limits
        )
        self.command_frame_offsets = torch.arange(1, config.goal_frames + 1, device=self.device)
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
            self.xi_dim,
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
                joint_count, len(self.robot.tracked_bodies), self.xi_dim, command_dim
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
            config.buffer_steps,
            config.num_envs,
            self.device,
            joint_count,
            self.xi_dim,
            window_length=config.wm_window,
            history_length=config.hist,
        )
        self.observation_history = torch.zeros(
            config.num_envs, config.hist, self.observation_dim, device=self.device
        )
        self.iteration = 0
        self.collection_remainder = 0.0
        self.decay_start = None
        self.env = None
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
            xi_center=self.xi_center,
            xi_scale=self.xi_scale,
            command_frame_offsets=self.command_frame_offsets,
        )
        self.pending_random_states = None
        self.sim_reward_sum = torch.zeros((), device=self.device)
        self.sim_reward_count = 0
        self.failure_counts = torch.zeros(self.motion.num_frames, device=self.device, dtype=torch.int32)

    @staticmethod
    def widths(text):
        return tuple((int(value) for value in text.split(",")))

    def create_env(self):
        from .env import TrackingEnv

        if self.device.startswith("cuda") and (not torch.cuda.is_available()):
            raise RuntimeError(
                "G1 training requires a CUDA-enabled PyTorch and an NVIDIA GPU. CPU tests: uv run pytest -q"
            )
        if self.config.profile_speed and self.device.startswith("cuda"):
            print(
                json.dumps(
                    {
                        "speed_settings": {
                            "simulator": self.config.simulator,
                            "gpu": torch.cuda.get_device_name(self.device),
                            "pytorch": torch.__version__,
                            "cuda": torch.version.cuda,
                            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "compile_world_model": self.config.compile_world_model,
                            "compile_actor": self.config.compile_actor,
                        }
                    }
                ),
                flush=True,
            )
        self.env = TrackingEnv(
            self.config.simulation,
            self.config.num_envs,
            self.robot,
            self.motion,
            self.task,
            self.device,
            max_episode_steps=self.config.max_episode_steps,
            timer=self.updates.timer,
            backend=self.config.simulator,
            genesis_settings=self.config.genesis,
        )

    @torch.no_grad()
    def collect(self, steps):
        episode_returns = []
        for _ in range(steps):
            observation = self.env.observe()
            self.observation_history = transforms.update_observation_history(
                self.observation_history, observation, self.env.episode_step == 0
            )
            command = transforms.compute_command(
                self.env.motion_frame_index, self.command_frame_offsets, self.motion
            )
            self.actor.observation_normalizer.update(observation)
            self.actor.command_normalizer.update(command)
            self.updates.timer.record("collect_prepare")
            action = self.actor(self.observation_history, command)
            action = (action + self.config.action_noise * torch.randn_like(action)).clamp(-20, 20)
            self.updates.timer.record("collect_actor")
            transition = self.env.step(action)
            self.updates.timer.record("collect_env_other")
            self.sim_reward_sum.add_(transition["reward"].sum())
            self.sim_reward_count += transition["reward"].numel()
            # Termination is evaluated against the next reference frame, before reset.
            self.failure_counts.index_add_(
                0, transition["motion_frame_index"] + 1, transition["termination"].to(torch.int32)
            )
            self.buffer.add(transition)
            inputs, targets, increments = transforms.compute_world_model_training_data(
                self.world_model,
                transforms.unflatten_state(transition["state"]),
                transforms.unflatten_state(transition["next_state"]),
                transition["action"],
                transition["last_action"],
                transition["xi"],
                self.xi_center,
                self.xi_scale,
            )
            self.world_model.input_normalizer.update(inputs)
            self.world_model.output_normalizer.update(targets)
            self.world_model.increment_normalizer.update(increments)
            if "normalized_episode_return" in transition:
                episode_returns.append(float(transition["normalized_episode_return"].mean()))
            self.updates.timer.record("collect_buffer")
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
        self.create_env()
        return_window = self.collect(config.warmup_steps)
        self.failure_counts.zero_()
        self.sim_reward_sum.zero_()
        self.sim_reward_count = 0
        if self.pending_random_states is not None:
            self.restore_random_states()
        last_saved_iteration = None
        start_time = time.monotonic()
        last_log_time = start_time
        last_log_iteration = self.iteration
        next_checkpoint_time = start_time + config.ckpt_minutes * 60
        for iteration in range(self.iteration, config.iterations):
            self.updates.timer.start_iteration()
            self.set_learning_rate(iteration)
            self.collection_remainder += 1 / config.utd
            steps = int(self.collection_remainder)
            self.collection_remainder -= steps
            self.updates.timer.record("prepare")
            returns = self.collect(steps)
            return_window.extend(returns)
            self.updates.timer.record("collect_other")
            for _ in range(config.wm_batches):
                world_model_loss, _ = self.updates.update_world_model()
            metrics = {"iteration": iteration + 1, "world_model_loss": world_model_loss}
            self.updates.timer.record("world_model_other")
            if iteration >= config.wm_only_iters:
                actor_loss, critic_loss, imagined_reward, gradient_norm, survival = (
                    self.updates.update_actor()
                )
                metrics.update(
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    imagined_reward=imagined_reward,
                    gradient_norm=gradient_norm,
                    survival=survival,
                )
            self.iteration = iteration + 1
            self.updates.timer.end_iteration()
            if self.iteration % config.log_every == 0:
                metrics = {
                    name: float(value) if isinstance(value, torch.Tensor) else value
                    for name, value in metrics.items()
                }
                metrics.update(self.updates.timer.report())
                average_return = sum(return_window) / max(len(return_window), 1)
                deadline = config.lr_decay_by > 0 and self.iteration >= config.lr_decay_by
                if (
                    self.decay_start is None
                    and config.lr_decay_ret > 0
                    and (average_return >= config.lr_decay_ret or deadline)
                ):
                    self.decay_start = self.iteration
                metrics["episode_return"] = average_return
                metrics["sim_r"] = (
                    float(self.sim_reward_sum) / max(self.sim_reward_count, 1) / MAX_STEP_REWARD
                )
                self.sim_reward_sum.zero_()
                self.sim_reward_count = 0
                metrics["top_failures"] = self.motion.top_failure_frames(self.failure_counts)
                self.failure_counts.zero_()
                return_window.clear()
                log_time = time.monotonic()
                metrics["iterations_per_second"] = (self.iteration - last_log_iteration) / max(
                    log_time - last_log_time, 1e-9
                )
                last_log_time = log_time
                last_log_iteration = self.iteration
                formatted_metrics = ", ".join(
                    f'"{name}": {value:.{4 if name.endswith("loss") else 3}f}'
                    if isinstance(value, float)
                    else f'"{name}": {json.dumps(value, ensure_ascii=False)}'
                    for name, value in metrics.items()
                )
                print(
                    f"{{{formatted_metrics}}} | {metrics['iterations_per_second']:.3f} it/s",
                    flush=True,
                )
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics) + "\n")
            timed_checkpoint = config.ckpt_minutes > 0 and time.monotonic() >= next_checkpoint_time
            if self.iteration % config.ckpt_every == 0 or timed_checkpoint:
                self.save_training_checkpoint(output)
                next_checkpoint_time = time.monotonic() + config.ckpt_minutes * 60
                last_saved_iteration = self.iteration
            if config.max_hours is not None and time.monotonic() - start_time >= config.max_hours * 3600:
                break
        if self.iteration != last_saved_iteration:
            self.save_training_checkpoint(output)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
            format_version=2,
            python_random_state=random.getstate(),
            numpy_random_state=np.random.get_state(),
            cuda_random_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        )
        temporary = Path(str(path) + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format_version", 1) not in (1, 2):
            raise ValueError("Unsupported checkpoint format.")
        if state["iteration"] >= self.config.iterations:
            raise ValueError(
                f"Checkpoint is at iteration {state['iteration']}; set config.iterations to a larger total target."
            )
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
        self.pending_random_states = {
            name: state[name]
            for name in ("torch_rng_state", "python_random_state", "numpy_random_state", "cuda_random_states")
            if name in state
        }
        print(f"Resuming iteration {self.iteration}; replay buffer will be refilled.", flush=True)

    def save_training_checkpoint(self, output_directory):
        checkpoint_path = Path(output_directory) / "checkpoints" / f"iteration_{self.iteration:06d}.pt"
        self.save(checkpoint_path)
        temporary_last = Path(output_directory) / "last.pt.tmp"
        shutil.copyfile(checkpoint_path, temporary_last)
        temporary_last.replace(Path(output_directory) / "last.pt")
        print(f"Saved checkpoint: {checkpoint_path}", flush=True)

    def restore_random_states(self):
        states = self.pending_random_states
        torch.set_rng_state(states["torch_rng_state"].cpu())
        if "python_random_state" in states:
            random.setstate(states["python_random_state"])
        if "numpy_random_state" in states:
            np.random.set_state(states["numpy_random_state"])
        if states.get("cuda_random_states") and torch.cuda.is_available():
            if len(states["cuda_random_states"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(states["cuda_random_states"])
        self.pending_random_states = None
