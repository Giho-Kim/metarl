#!/usr/bin/env python3
"""Evaluate saved Compose-2 agents across a dynamics switch after round 2.

Rounds 1-2 share one MDP. At the start of round 3, ``m`` is replaced by a
different dynamics permutation while the goal and recurrent memory are kept.
The policy clock alone is reset, so rounds 3-4 receive the same 0..27 clock
range that the checkpoint saw during training.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import compose2 as envs


N_ROUNDS = 4
BLOCK_STEPS = 2 * envs.ROUND_STEPS
TOTAL_STEPS = N_ROUNDS * envs.ROUND_STEPS


@torch.no_grad()
def rollout(agent, batch: int, seed: int, switch_m: bool) -> torch.Tensor:
    env = envs.Environment(batch, seed, condition=0)
    device = next(agent.parameters()).device
    memory = agent.initial_memory(batch, device)
    # Preserve the checkpoint's training-time convention at round starts.
    x = torch.zeros(batch, dtype=torch.long, device=device)
    rewards = []

    for t in range(TOTAL_STEPS):
        if t and t % envs.ROUND_STEPS == 0:
            if switch_m and t == BLOCK_STEPS:
                change = np.ones(batch, dtype=bool)
                env.m = env._resample(env.m, len(envs.PERMUTATIONS), change)
            env.x.fill(0)
            x = torch.zeros(batch, dtype=torch.long, device=device)

        # This is the only agent-side reset at the R2 -> R3 MDP switch.
        policy_timestep = t % BLOCK_STEPS
        logits, _ = agent.heads(memory, x, policy_timestep)
        action = logits.argmax(-1)
        next_x, reward = env.step(action.cpu().numpy())
        next_x = (next_x + envs.LIMIT).to(device)
        reward = reward.to(device)
        memory = agent.observe(memory, action, next_x, reward, sample=False)
        rewards.append(reward.cpu())
        x = next_x

    return torch.stack(rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "results" / "compose2_seed0",
    )
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    rows = []
    for checkpoint in sorted(args.results.glob("*.pt")):
        saved = torch.load(checkpoint, map_location=device)
        agent = envs.Agent(saved["kind"], saved["deter"], saved["latent_size"]).to(device)
        agent.load_state_dict(saved["state_dict"])
        agent.eval()
        for condition, switch_m in (("keep", False), ("switch_m_r3", True)):
            seed_rewards = []
            for i in range(args.seeds):
                seed_rewards.append(rollout(agent, args.batch, 930_000 + i, switch_m))
            reward = torch.cat(seed_rewards, dim=1)
            rounds = [
                reward[r * envs.ROUND_STEPS:(r + 1) * envs.ROUND_STEPS].sum(0)
                for r in range(N_ROUNDS)
            ]
            totals = torch.stack(rounds)
            row = {
                "model": saved["kind"],
                "condition": condition,
                **{f"round{r + 1}": rounds[r].mean().item() for r in range(N_ROUNDS)},
                "pre_switch": totals[:2].sum(0).mean().item(),
                "post_switch": totals[2:].sum(0).mean().item(),
                "total": totals.sum(0).mean().item(),
                "post_switch_se": totals[2:].sum(0).std().item() / np.sqrt(totals.shape[1]),
                "episodes": totals.shape[1],
            }
            rows.append(row)

    output = args.results / "mdp_switch_evaluation.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"device={device} episodes/condition={args.batch * args.seeds}")
    print("model        condition       R1     R2     R3     R4    pre   post  post SE")
    for row in rows:
        print(
            f"{row['model']:12s} {row['condition']:11s} "
            f"{row['round1']:6.3f} {row['round2']:6.3f} "
            f"{row['round3']:6.3f} {row['round4']:6.3f} "
            f"{row['pre_switch']:6.3f} {row['post_switch']:6.3f} "
            f"{row['post_switch_se']:7.4f}"
        )
    print(output.resolve())


if __name__ == "__main__":
    main()
