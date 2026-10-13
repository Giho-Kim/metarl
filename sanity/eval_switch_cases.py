#!/usr/bin/env python3
"""Exhaustively evaluate three task switches after round 2."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import torch

import compose2 as envs


ROUND_STEPS = envs.ROUND_STEPS
PRE_ROUNDS = 2
POST_ROUNDS = 4
PRE_STEPS = PRE_ROUNDS * ROUND_STEPS
POST_STEPS = POST_ROUNDS * ROUND_STEPS
TOTAL_ROUNDS = PRE_ROUNDS + POST_ROUNDS
TOTAL_STEPS = TOTAL_ROUNDS * ROUND_STEPS


def transition_pairs(condition: str) -> tuple[np.ndarray, np.ndarray]:
    """Return every valid (before task, after task) transition exactly once."""
    rows = []
    for m, g in itertools.product(range(len(envs.PERMUTATIONS)), range(len(envs.GOALS))):
        next_ms = [x for x in range(len(envs.PERMUTATIONS)) if x != m]
        next_gs = [x for x in range(len(envs.GOALS)) if x != g]
        if condition == "switch_dynamics":
            rows.extend(((m, g), (m2, g)) for m2 in next_ms)
        elif condition == "switch_reward":
            rows.extend(((m, g), (m, g2)) for g2 in next_gs)
        elif condition == "switch_both":
            rows.extend(((m, g), (m2, g2)) for m2 in next_ms for g2 in next_gs)
        else:
            raise ValueError(condition)
    before = np.asarray([row[0] for row in rows], dtype=np.int64)
    after = np.asarray([row[1] for row in rows], dtype=np.int64)
    return before, after


@torch.no_grad()
def rollout(agent, before: np.ndarray, after: np.ndarray) -> torch.Tensor:
    batch = len(before)
    world = envs.Environment(batch, seed=0, condition=0, tasks=before)
    device = next(agent.parameters()).device
    memory = agent.initial_memory(batch, device)
    x = torch.zeros(batch, dtype=torch.long, device=device)
    rewards = []
    for t in range(TOTAL_STEPS):
        if t and t % ROUND_STEPS == 0:
            if t == PRE_STEPS:
                world.m = after[:, 0].copy()
                world.g = after[:, 1].copy()
            world.x.fill(0)
            x = torch.zeros(batch, dtype=torch.long, device=device)
        # Agent memory persists. The clock spans 0..1 over R1-R2, then resets
        # and spans 0..1 over the longer R3-R6 post-switch block.
        if t < PRE_STEPS:
            clock_fraction = t / (PRE_STEPS - 1)
        else:
            clock_fraction = (t - PRE_STEPS) / (POST_STEPS - 1)
        # heads() normalises by the checkpoint's training horizon internally.
        policy_timestep = clock_fraction * (envs.META_STEPS - 1)
        logits, _ = agent.heads(memory, x, policy_timestep)
        action = logits.argmax(-1)
        next_x, reward = world.step(action.cpu().numpy())
        next_x = (next_x + envs.LIMIT).to(device)
        reward = reward.to(device)
        memory = agent.observe(memory, action, next_x, reward, sample=False)
        rewards.append(reward.cpu())
        x = next_x
    return torch.stack(rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    conditions = ("switch_dynamics", "switch_reward", "switch_both")
    transitions = {condition: transition_pairs(condition) for condition in conditions}
    rows = []

    for checkpoint in sorted(args.results.glob("*.pt")):
        saved = torch.load(checkpoint, map_location=device)
        agent = envs.Agent(saved["kind"], saved["deter"], saved["latent_size"]).to(device)
        agent.load_state_dict(saved["state_dict"])
        agent.eval()
        for condition in conditions:
            before, after = transitions[condition]
            chunks = []
            for start in range(0, len(before), args.chunk_size):
                stop = min(start + args.chunk_size, len(before))
                chunks.append(rollout(agent, before[start:stop], after[start:stop]))
            reward = torch.cat(chunks, dim=1)
            rounds = [
                reward[i * ROUND_STEPS:(i + 1) * ROUND_STEPS].sum(0)
                for i in range(TOTAL_ROUNDS)
            ]
            post = torch.stack(rounds[PRE_ROUNDS:]).sum(0)
            rows.append({
                "model": saved["kind"], "condition": condition,
                "transitions": len(before),
                **{f"round{i + 1}": rounds[i].mean().item() for i in range(TOTAL_ROUNDS)},
                "pre_switch": (rounds[0] + rounds[1]).mean().item(),
                "post_switch": post.mean().item(),
                "post_switch_se": post.std().item() / np.sqrt(len(post)),
            })

    output = args.results / "switch_cases_r3_to_r6_evaluation.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"device={device}")
    print("model        condition          combos     R1     R2     R3     R4     R5     R6   post    SE")
    for row in rows:
        print(
            f"{row['model']:12s} {row['condition']:17s} {row['transitions']:6d} "
            f"{row['round1']:6.3f} {row['round2']:6.3f} "
            f"{row['round3']:6.3f} {row['round4']:6.3f} "
            f"{row['round5']:6.3f} {row['round6']:6.3f} "
            f"{row['post_switch']:6.3f} {row['post_switch_se']:6.4f}"
        )
    print(output.resolve())


if __name__ == "__main__":
    main()
