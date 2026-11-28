"""Gradient reconstruction and sampling regressions for the paper update."""
from types import SimpleNamespace

import pytest
import torch

from test_pipeline import tiny_trainer as tiny_trainer
from test_action_delay import delayed_trainer as delayed_trainer
from readable_mimic import data_transforms as transforms
from readable_mimic.grad_optimization import (
    GradientOptimizationConfig, GradientStartSampler, ModelRankedReset, capped_action_gradients,
)
from readable_mimic.updates import set_requires_grad


def test_cap_uses_entire_rollout_and_importance():
    gradients = torch.tensor([[[1.], [2.], [100.]], [[1.], [2.], [100.]]])
    importance = torch.tensor([1., 2., 1.])
    actual, norms, metrics = capped_action_gradients(gradients, importance, 2)
    median = (norms * importance).median()
    expected_weights = (2 * median / (norms * importance)).clamp(max=1)
    torch.testing.assert_close(actual, gradients * (importance * expected_weights)[None, :, None])
    assert metrics["gradient_cap_fraction"] == pytest.approx(1 / 3)
    bad = gradients.clone()
    bad[0, 2, 0] = float("nan")
    actual, _, _ = capped_action_gradients(bad, importance, 10)
    assert torch.isfinite(actual).all()
    assert actual[:, 2].eq(0).all()
    zeros, _, _ = capped_action_gradients(torch.zeros_like(gradients), importance, 10)
    assert zeros.eq(0).all()


def test_start_importance_matches_candidate_distribution():
    config = GradientOptimizationConfig(cap=10, start_draw=True, quantiles=4)
    sampler = GradientStartSampler(config, "cpu")
    sampler.norms.copy_(torch.tensor([1., 2., 3., 4.]))
    values = torch.arange(8).float()
    indices, importance, quantiles = sampler.draw(values, 100)
    probabilities = torch.tensor([1., 1., 2., 2., 3., 3., 4., 4.]) / 20
    torch.testing.assert_close(importance, 1 / (8 * probabilities[indices]))
    torch.testing.assert_close(quantiles, indices // 2)
    # Inverse weighting recovers the uniform expectation exactly.
    errors = values.square()
    torch.testing.assert_close((probabilities * errors / (8 * probabilities)).sum(), errors.mean())
    sampler.observe(torch.tensor([0, 0, 1]), torch.tensor([2., 4., 8.]))
    torch.testing.assert_close(sampler.norms[:2], torch.tensor([3., 8.]))
    assert sampler.seen[:2].all()


def reset_motion(ground_supported=False):
    motion = SimpleNamespace(
        bin_start=torch.arange(0, 400, 50), bin_clip=torch.arange(8) // 4,
        clip_end=torch.tensor([199, 399]), reset_starts=None,
    )
    if ground_supported:
        starts = motion.bin_start[:, None] + torch.tensor([3, 12, 48])[None]
        motion.reset_starts = (starts, torch.full((8,), 3))
    return motion


@pytest.mark.parametrize("ground_supported", [False, True])
def test_model_ranked_resets_preserve_legal_starts_and_state(ground_supported):
    config = GradientOptimizationConfig(model_resets=True, hard_fraction=0.25, rank_batch=2)
    sampler = ModelRankedReset(reset_motion(ground_supported), config)
    frames = sampler.sample(1000)
    assert ((frames % 200) <= 149).all()
    if ground_supported:
        assert torch.isin(frames % 50, torch.tensor([3, 12, 48])).all()
    sampler.refresh(lambda frame, horizon: -frame.float(), 2000)
    first_hard = sampler.hard.clone()
    sampler.refresh(lambda frame, horizon: frame.float(), 4000)
    assert sampler.past[first_hard].all()
    restored = ModelRankedReset(reset_motion(ground_supported), config)
    restored.load_state_dict(sampler.state_dict())
    torch.testing.assert_close(restored.hard, sampler.hard)
    torch.testing.assert_close(restored.past, sampler.past)
    torch.manual_seed(7)
    expected = sampler.sample(200)
    torch.manual_seed(7)
    torch.testing.assert_close(restored.sample(200), expected)


@pytest.mark.parametrize("action_delay", [False, True])
def test_two_pass_gradient_matches_closed_loop_backprop(tiny_trainer, delayed_trainer, action_delay):
    updates = (delayed_trainer if action_delay else tiny_trainer).updates
    start, history = updates.sample_valid_starts(4)
    normalized_xi = transforms.compute_normalized_xi(start["xi"], updates.xi_center, updates.xi_scale)
    initial_history = transforms.compute_replay_observation_history(
        history, updates.motion, updates.default_joint_pos, updates.observation_noise_limits,
    )
    with torch.no_grad():
        updates.world_model.network[-1].weight.normal_(0, 0.01)
    set_requires_grad(updates.world_model, False)
    set_requires_grad(updates.critic, False)
    compiled_step = torch.compile(updates.imagine_step, backend="aot_eager", fullgraph=True, dynamic=False)
    compiled_surrogate = torch.compile(
        updates.policy_gradient_loss, backend="aot_eager", fullgraph=True, dynamic=False,
    )

    def gradients(two_pass):
        set_requires_grad(updates.actor, not two_pass)
        updates.actor.zero_grad(set_to_none=True)
        state = transforms.unflatten_state(start["state"])
        frames, last_action = start["motion_frame_index"], start["last_action"]
        observations = initial_history
        previous_target = start.get("previous_target")
        command = transforms.compute_command(frames, updates.command_frame_offsets, updates.motion)
        objective = torch.zeros(4)
        perturbations, stored = [], []
        torch.manual_seed(42)
        for _ in range(2):
            perturbation = torch.zeros_like(last_action, requires_grad=True) if two_pass else None
            if two_pass:
                perturbations.append(perturbation)
                stored.append((observations.detach(), command.detach()))
            previous_action = last_action
            step = compiled_step if two_pass else updates.imagine_step
            state, last_action, observations, frames, command = step(
                state, observations, last_action, frames, start["xi"], normalized_xi, command,
                action_perturbation=perturbation,
                previous_target=previous_target,
            )
            if action_delay:
                previous_target = updates.default_joint_pos + start["xi"][:, 1:3] + updates.action_scale * last_action
            reward, termination = updates.reward_and_termination(
                state, frames, last_action, previous_action, start["init_root"], start["init_f"],
            )
            objective = objective + reward * (1 - termination)
        objective += 0.5 * updates.critic(updates.critic_features(
            state, frames, normalized_xi, command, start["init_root"], start["init_f"], previous_target,
        ))
        sample_weights = torch.tensor([0.2, 0.5, 1.0, 0.8])
        if two_pass:
            action_gradients = torch.autograd.grad(objective.sum(), perturbations)
            set_requires_grad(updates.actor, True)
            for (observations, command), action_gradient in zip(stored, action_gradients):
                loss = compiled_surrogate(observations, command, action_gradient.detach() * sample_weights[:, None] / 4)
                loss.backward()
        else:
            (-(objective * sample_weights).mean()).backward()
        return [p.grad.clone() for p in updates.actor.parameters()]

    expected = gradients(False)
    actual = gradients(True)
    for value, target in zip(actual, expected):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, target, atol=2e-5, rtol=2e-4)


def test_paper_updates_and_sampler_checkpoint(tiny_trainer, tmp_path):
    trainer = tiny_trainer
    config = GradientOptimizationConfig(
        cap=10, start_draw=True, candidates=2, model_resets=True,
        reset_horizon=3, reset_tail=5, rank_batch=8,
    )
    trainer.updates.grad_config = config
    trainer.updates.start_sampler = GradientStartSampler(config, trainer.device)
    trainer.model_reset = ModelRankedReset(trainer.motion, config)
    before = [p.clone() for p in trainer.actor.parameters()]
    for _ in range(2):
        result = trainer.updates.update_actor()
        assert all(torch.isfinite(value).all() for value in result)
    assert trainer.updates.start_sampler.seen.any()
    assert any(not torch.equal(p, old) for p, old in zip(trainer.actor.parameters(), before))
    metrics = trainer.model_reset.refresh(trainer.updates.rank_reference, 2000)
    assert metrics["model_reset_hard_bins"] > 0
    path = tmp_path / "paper.pt"
    trainer.save(path)
    norms = trainer.updates.start_sampler.norms.clone()
    hard = trainer.model_reset.hard.clone()
    trainer.updates.start_sampler.norms.zero_()
    trainer.model_reset.hard.zero_()
    trainer.load(path)
    torch.testing.assert_close(trainer.updates.start_sampler.norms, norms)
    torch.testing.assert_close(trainer.model_reset.hard, hard)


def test_model_ranked_reset_mixture_and_short_clip_guard():
    config = GradientOptimizationConfig(model_resets=True)
    sampler = ModelRankedReset(reset_motion(), config)
    sampler.hard[0] = True
    sampler.past[1] = True
    torch.manual_seed(17)
    draws = sampler.sample(100000)
    chosen_bins = draws // 50
    uniform = 0.5 / len(sampler.valid_bins)
    first = float((chosen_bins == sampler.valid_bins[0]).float().mean())
    second = float((chosen_bins == sampler.valid_bins[1]).float().mean())
    assert first == pytest.approx(uniform + 0.4, abs=0.01)
    assert second == pytest.approx(uniform + 0.1, abs=0.01)
    short = SimpleNamespace(bin_start=torch.tensor([0]), bin_clip=torch.tensor([0]),
                            clip_end=torch.tensor([10]), reset_starts=None)
    with pytest.raises(ValueError, match="No legal"):
        ModelRankedReset(short, config)


def test_legacy_resume_keeps_original_gradient_rule(tiny_trainer, tmp_path):
    import json
    from readable_mimic.config import load_config
    trainer = tiny_trainer
    trainer.config.warmup_steps = 4
    checkpoint = tmp_path / "legacy.pt"
    trainer.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    saved["config"].pop("grad_optimization", None)
    torch.save(saved, checkpoint)
    path = tmp_path / "config.json"
    current = dict(vars(trainer.config))
    current["grad_optimization"] = {"cap": 10, "start_draw": True}
    path.write_text(json.dumps(current))
    resumed = load_config(path, resume=str(checkpoint))
    assert resumed.grad_optimization == {}


@pytest.mark.parametrize("action_delay", [False, True])
def test_precomputed_rollout_batched_backward_and_reused_fk(tiny_trainer, delayed_trainer, action_delay):
    updates = (delayed_trainer if action_delay else tiny_trainer).updates
    start, history = updates.sample_valid_starts(4)
    normalized_xi = transforms.compute_normalized_xi(start["xi"], updates.xi_center, updates.xi_scale)
    initial_history = transforms.compute_replay_observation_history(
        history, updates.motion, updates.default_joint_pos, updates.observation_noise_limits,
    )
    with torch.no_grad():
        updates.world_model.network[-1].weight.normal_(0, 0.01)
    set_requires_grad(updates.world_model, False)
    set_requires_grad(updates.critic, False)

    compiled_commands = torch.compile(updates.compute_commands, backend="aot_eager", fullgraph=True, dynamic=False)
    compiled_encode = torch.compile(updates.encode_commands, backend="aot_eager", fullgraph=True, dynamic=False)
    compiled_step = torch.compile(
        updates.imagine_step_precomputed, backend="aot_eager", fullgraph=True, dynamic=False,
    )
    compiled_reward_bodies = torch.compile(
        updates.reward_termination_and_bodies, backend="aot_eager", fullgraph=True, dynamic=False,
    )
    compiled_critic_features = torch.compile(
        updates.critic_features, backend="aot_eager", fullgraph=True, dynamic=False,
    )
    compiled_targets = torch.compile(
        updates.compute_critic_targets, backend="aot_eager", fullgraph=True, dynamic=False,
    )
    compiled_surrogate = torch.compile(
        updates.policy_gradient_loss, backend="aot_eager", fullgraph=True, dynamic=False,
    )

    # Reference unbatched two-pass gradients.
    set_requires_grad(updates.actor, False)
    updates.actor.zero_grad(set_to_none=True)
    state = transforms.unflatten_state(start["state"])
    frames, last_action = start["motion_frame_index"], start["last_action"]
    observations = initial_history
    previous_target = start.get("previous_target")
    command = transforms.compute_command(frames, updates.command_frame_offsets, updates.motion)
    objective = torch.zeros(4)
    perturbations, stored = [], []
    torch.manual_seed(99)
    for _ in range(updates.horizon):
        perturbation = torch.zeros_like(last_action, requires_grad=True)
        perturbations.append(perturbation)
        stored.append((observations.detach(), command.detach()))
        previous_action = last_action
        state, last_action, observations, frames, command = updates.imagine_step(
            state, observations, last_action, frames, start["xi"], normalized_xi, command,
            action_perturbation=perturbation, previous_target=previous_target,
        )
        if action_delay:
            previous_target = (
                updates.default_joint_pos + start["xi"][:, 1:3] + updates.action_scale * last_action
            )
        reward, termination = updates.reward_and_termination(
            state, frames, last_action, previous_action, start["init_root"], start["init_f"],
        )
        objective = objective + reward * (1 - termination)
    objective += 0.5 * updates.critic(updates.critic_features(
        state, frames, normalized_xi, command, start["init_root"], start["init_f"], previous_target,
    ))
    ref_action_grads = torch.stack(torch.autograd.grad(objective.sum(), perturbations))
    set_requires_grad(updates.actor, True)
    for (obs_step, cmd_step), action_grad in zip(stored, ref_action_grads):
        updates.policy_gradient_loss(obs_step, cmd_step, action_grad.detach() / 4).backward()
    expected_grads = [p.grad.clone() for p in updates.actor.parameters()]

    # Optimized precomputed rollout + compiled reward/bodies + chunked batched backward.
    set_requires_grad(updates.actor, False)
    updates.actor.zero_grad(set_to_none=True)
    state = transforms.unflatten_state(start["state"])
    initial_state = state
    last_action = start["last_action"]
    observations = initial_history
    previous_target = start.get("previous_target")
    all_frames = start["motion_frame_index"][None, :] + updates.horizon_offsets[:, None]
    with torch.no_grad():
        all_commands = compiled_commands(all_frames.flatten()).view(updates.horizon + 1, 4, -1)
        motion_latents = compiled_encode(all_commands[:-1].flatten(0, 1)).view(updates.horizon, 4, -1)
    perturbations, policy_histories = [], []
    rollout_states, rollout_actions, rollout_last_actions = [], [], []
    torch.manual_seed(99)
    for step_index in range(updates.horizon):
        perturbation = torch.zeros_like(last_action, requires_grad=True)
        perturbations.append(perturbation)
        policy_histories.append(observations.detach())
        rollout_last_actions.append(last_action)
        state, last_action, observations = compiled_step(
            state, observations, last_action, all_frames[step_index + 1], start["xi"], normalized_xi,
            all_commands[step_index], motion_latent=motion_latents[step_index],
            action_perturbation=perturbation, previous_target=previous_target,
        )
        if action_delay:
            previous_target = (
                updates.default_joint_pos + start["xi"][:, 1:3] + updates.action_scale * last_action
            )
        rollout_states.append(state)
        rollout_actions.append(last_action)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)
    all_states = tuple(flat([s[i] for s in rollout_states]) for i in range(6))
    rewards, terminations, body_state, ref_body_state = compiled_reward_bodies(
        all_states, all_frames[1:].flatten(), flat(rollout_actions), flat(rollout_last_actions),
        start["init_root"].repeat(updates.horizon, 1), start["init_f"].repeat(updates.horizon),
    )
    rewards = rewards.view(updates.horizon, 4)
    terminations = terminations.view(updates.horizon, 4)
    opt_objective = (rewards * (1 - terminations)).sum(0) + 0.5 * updates.critic(compiled_critic_features(
        state, all_frames[-1], normalized_xi, all_commands[-1], start["init_root"], start["init_f"],
        previous_target,
    ))
    torch.testing.assert_close(opt_objective, objective)
    opt_action_grads = torch.stack(torch.autograd.grad(opt_objective.sum(), perturbations))
    torch.testing.assert_close(opt_action_grads, ref_action_grads)
    set_requires_grad(updates.actor, True)
    updates.policy_gradient_loss, orig_pg = compiled_surrogate, updates.policy_gradient_loss
    updates.policy_gradient_chunk, orig_chunk = 6, updates.policy_gradient_chunk
    try:
        updates._backward_policy_gradients(
            torch.stack(policy_histories), all_commands[:-1], opt_action_grads.detach(), 4,
        )
    finally:
        updates.policy_gradient_loss = orig_pg
        updates.policy_gradient_chunk = orig_chunk
    for actual, expected in zip([p.grad for p in updates.actor.parameters()], expected_grads):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

    # Reused FK critic features and compiled GAE targets match uncached FK.
    with torch.no_grad():
        init_body = updates.task.kinematics.forward(*initial_state)
        init_ref = updates.motion.ref_body_state(all_frames[0])
        critic_body = tuple(torch.cat((a, b.detach()), 0) for a, b in zip(init_body, body_state))
        critic_ref = tuple(torch.cat((a, b.detach()), 0) for a, b in zip(init_ref, ref_body_state))
        critic_states = tuple(
            torch.cat((initial_state[i].unsqueeze(0), torch.stack([s[i].detach() for s in rollout_states])), 0)
            .flatten(0, 1)
            for i in range(6)
        )
        critic_xi = normalized_xi.unsqueeze(0).expand(updates.horizon + 1, -1, -1).flatten(0, 1)
        critic_prev = None
        if action_delay:
            issued_targets = (
                updates.default_joint_pos + start["xi"][None, :, 1:3]
                + updates.action_scale * torch.stack([a.detach() for a in rollout_actions])
            )
            critic_prev = torch.cat((start["previous_target"][None], issued_targets), 0).flatten(0, 1)
        reused = compiled_critic_features(
            critic_states, all_frames.flatten(), critic_xi, all_commands.flatten(0, 1),
            start["init_root"].repeat(updates.horizon + 1, 1), start["init_f"].repeat(updates.horizon + 1),
            critic_prev, body_state=critic_body, ref_body_state=critic_ref,
        )
        uncached = updates.critic_features(
            critic_states, all_frames.flatten(), critic_xi, all_commands.flatten(0, 1),
            start["init_root"].repeat(updates.horizon + 1, 1), start["init_f"].repeat(updates.horizon + 1),
            critic_prev,
        )
        torch.testing.assert_close(reused, uncached)
        reused_grid = reused.view(updates.horizon + 1, 4, -1)
        actual_targets = compiled_targets(reused_grid, rewards.detach(), terminations.detach())
        expected_targets = transforms.compute_value_targets(
            updates.critic(reused_grid), rewards.detach(), terminations.detach(),
            updates.config.gamma, updates.config.lam,
        )
        torch.testing.assert_close(actual_targets, expected_targets)
        raw_ref_body = updates.motion.kinematics.forward(*updates.motion.ref_state(all_frames.flatten()))
        cached_ref_body = updates.motion.ref_body_state(all_frames.flatten())
        for cached_val, raw_val in zip(cached_ref_body, raw_ref_body):
            torch.testing.assert_close(cached_val, raw_val)
