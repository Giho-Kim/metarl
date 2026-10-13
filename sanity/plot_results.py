#!/usr/bin/env python3
"""Plot the Compose-2 sanity-check training and evaluation curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {"joint": "#2563eb", "conditional": "#dc2626"}
LABELS = {"joint": "Joint", "conditional": "Conditional"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_model_rows(results_dir: Path, suffix: str) -> list[dict[str, str]]:
    """Prefer collision-safe per-model logs, with legacy combined fallback."""
    combined_path = results_dir / f"{suffix}.csv"
    combined = read_csv(combined_path) if combined_path.exists() else []
    rows = []
    for model in ("joint", "conditional"):
        model_path = results_dir / f"{model}_{suffix}.csv"
        if model_path.exists():
            rows.extend(read_csv(model_path))
        else:
            rows.extend(row for row in combined if row["model"] == model)
    return rows


def series(rows: list[dict[str, str]], key: str) -> tuple[list[float], list[float]]:
    return (
        [float(row["update"]) for row in rows],
        [float(row[key]) for row in rows],
    )


def add_model_lines(ax, rows_by_model, key, *, styles=None):
    for model, rows in rows_by_model.items():
        x, y = series(rows, key)
        kwargs = styles.get(model, {}) if styles else {}
        ax.plot(x, y, color=COLORS[model], label=LABELS[model], **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "results" / "compose2_seed0",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    output = args.output or results_dir / "result_curves.png"
    curves = {
        model: read_csv(results_dir / f"{model}_curve.csv")
        for model in ("joint", "conditional")
    }
    evaluation_rows = read_model_rows(results_dir, "evaluation")
    probe_rows = read_model_rows(results_dir, "probe")
    evaluations = {
        model: [row for row in evaluation_rows if row["model"] == model]
        for model in curves
    }
    probes = {
        model: [row for row in probe_rows if row["model"] == model]
        for model in curves
    }

    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)

    add_model_lines(axes[0, 0], curves, "train_total")
    axes[0, 0].set(title="Training return", ylabel="Return")

    for model, rows in curves.items():
        x, y1 = series(rows, "train_round1")
        _, y2 = series(rows, "train_round2")
        axes[0, 1].plot(x, y1, color=COLORS[model], label=f"{LABELS[model]} R1")
        axes[0, 1].plot(
            x, y2, color=COLORS[model], linestyle="--", label=f"{LABELS[model]} R2"
        )
    axes[0, 1].set(title="Training return by round", ylabel="Return")

    if evaluation_rows and "split" in evaluation_rows[0]:
        for model in curves:
            for split, linestyle in (("train", "--"), ("heldout", "-")):
                rows = [row for row in evaluations[model] if row["split"] == split]
                x, y = series(rows, "total")
                axes[0, 2].plot(
                    x, y, color=COLORS[model], marker="o", linestyle=linestyle,
                    label=f"{LABELS[model]} {split}",
                )
        axes[0, 2].set(title="Train vs held-out tasks", ylabel="Return")
    else:
        add_model_lines(axes[0, 2], evaluations, "total", styles={
            "joint": {"marker": "o"}, "conditional": {"marker": "o"}
        })
        axes[0, 2].set(title="Evaluation", ylabel="Return")

    for model, rows in curves.items():
        x, state = series(rows, "state_loss")
        _, reward = series(rows, "reward_loss")
        axes[1, 0].plot(x, state, color=COLORS[model], label=f"{LABELS[model]} state")
        axes[1, 0].plot(
            x, reward, color=COLORS[model], linestyle="--", label=f"{LABELS[model]} reward"
        )
    axes[1, 0].set(title="Prediction losses", ylabel="Loss")
    axes[1, 0].set_yscale("log")

    add_model_lines(axes[1, 1], curves, "raw_kl")
    axes[1, 1].set(title="Latent KL", ylabel="Raw KL")

    probe_styles = {
        "m|z_dyn": "-",
        "g|z_dyn": "--",
        "m|z_task": "-.",
        "g|z_task": ":",
    }
    for model, rows in probes.items():
        for metric, linestyle in probe_styles.items():
            x, y = series(rows, metric)
            axes[1, 2].plot(
                x,
                y,
                color=COLORS[model],
                linestyle=linestyle,
                label=f"{LABELS[model]} {metric}",
            )
    axes[1, 2].axhline(0.5, color="#64748b", linewidth=1, alpha=0.7)
    axes[1, 2].set(title="Linear-probe accuracy", ylabel="Accuracy", ylim=(0, 1))

    for ax in axes.flat:
        ax.set_xlabel("Update")
        ax.legend(fontsize=8, frameon=True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Compose-2 sanity check (seed 0)", fontsize=16, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(output)
    print(output.with_suffix(".pdf"))

    switch_path = results_dir / "mdp_switch_evaluation.csv"
    if switch_path.exists():
        switch_rows = read_csv(switch_path)
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
        style = {"keep": "--", "switch_m_r3": "-"}
        condition_label = {"keep": "Keep MDP", "switch_m_r3": "Switch MDP at R3"}
        for ax, model in zip(axes2, ("joint", "conditional")):
            for row in switch_rows:
                if row["model"] != model:
                    continue
                condition = row["condition"]
                ax.plot(
                    range(1, 5),
                    [float(row[f"round{i}"]) for i in range(1, 5)],
                    marker="o",
                    linestyle=style[condition],
                    linewidth=2,
                    label=condition_label[condition],
                )
            ax.axvline(2.5, color="#475569", linestyle=":", linewidth=1.5)
            ax.text(2.55, 2.55, "MDP switch\nclock = 0", fontsize=9, color="#334155")
            ax.set(title=LABELS[model], xlabel="Round", xticks=range(1, 5))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=9)
        axes2[0].set_ylabel("Mean return")
        fig2.suptitle("Dynamics switch after round 2 (seed-0 checkpoints)", fontweight="bold")
        switch_output = results_dir / "mdp_switch_curves.png"
        fig2.savefig(switch_output, dpi=180, bbox_inches="tight")
        fig2.savefig(switch_output.with_suffix(".pdf"), bbox_inches="tight")
        print(switch_output)
        print(switch_output.with_suffix(".pdf"))

    extended_cases_path = results_dir / "switch_cases_r3_to_r6_evaluation.csv"
    cases_path = (extended_cases_path if extended_cases_path.exists()
                  else results_dir / "switch_cases_evaluation.csv")
    if cases_path.exists():
        case_rows = read_csv(cases_path)
        fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
        case_style = {
            "switch_dynamics": ("Dynamics switch", "o"),
            "switch_reward": ("Reward switch", "s"),
            "switch_both": ("Both switch", "^"),
        }
        case_colors = {
            "switch_dynamics": "#2563eb",
            "switch_reward": "#16a34a",
            "switch_both": "#dc2626",
        }
        rounds = 6 if cases_path == extended_cases_path else 4
        switch_boundary = 2.5
        for ax, model in zip(axes3, ("joint", "conditional")):
            for row in case_rows:
                if row["model"] != model:
                    continue
                condition = row["condition"]
                label, marker = case_style[condition]
                ax.plot(
                    range(1, rounds + 1),
                    [float(row[f"round{i}"]) for i in range(1, rounds + 1)],
                    color=case_colors[condition], marker=marker, linewidth=2, label=label,
                )
            ax.axvline(switch_boundary, color="#475569", linestyle=":", linewidth=1.5)
            ax.set(title=LABELS[model], xlabel="Round", xticks=range(1, rounds + 1))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=9)
        axes3[0].set_ylabel("Mean return")
        fig3.suptitle("Exhaustive task switches at round 3", fontweight="bold")
        cases_output = results_dir / (
            "switch_cases_r3_to_r6_curves.png" if rounds == 6 else "switch_cases_curves.png")
        fig3.savefig(cases_output, dpi=180, bbox_inches="tight")
        fig3.savefig(cases_output.with_suffix(".pdf"), bbox_inches="tight")
        print(cases_output)
        print(cases_output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
