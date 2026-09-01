#!/usr/bin/env python3
"""Plot the four Copy-Paste optimizer/schedule ablation runs together."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS = (
    ("Original (wd=0, constant)", "wheat50-foreground-copypaste-bs16-e250", "#2563EB"),
    ("WD only (wd=0.05, constant)", "wheat50-foreground-copypaste-wd-only-bs16-e250", "#EA580C"),
    ("Cosine only (wd=0)", "wheat50-foreground-copypaste-cosine-only-bs16-e250", "#16A34A"),
    ("WD + cosine (wd=0.05)", "wheat50-foreground-copypaste-cosine-wd-bs16-e250", "#DC2626"),
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/copy_paste_optimizer_schedule_ablation.png"),
    )
    parser.add_argument("--eval-freq", type=int, default=5)
    return parser.parse_args()


def load_scalars(run_dir):
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    required = ("metric/mae", "metric/mse", "loss/loss")
    missing = set(required) - set(accumulator.Tags()["scalars"])
    if missing:
        raise RuntimeError(f"{run_dir} is missing scalar tags: {sorted(missing)}")
    return {
        tag: (
            np.asarray([event.step for event in accumulator.Scalars(tag)]),
            np.asarray([event.value for event in accumulator.Scalars(tag)]),
        )
        for tag in required
    }


def exponential_smooth(values, factor=0.6):
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = factor * smoothed[index - 1] + (1 - factor) * values[index]
    return smoothed


def main():
    args = parse_args()
    data = {}
    for label, directory, color in RUNS:
        data[label] = (load_scalars(args.runs_dir / directory), color)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=False)
    mae_ax, mse_ax, loss_ax, tail_ax = axes.flat

    legend_handles = []
    for label, (scalars, color) in data.items():
        mae_steps, mae = scalars["metric/mae"]
        mse_steps, mse = scalars["metric/mse"]
        loss_steps, loss = scalars["loss/loss"]
        mae_epochs = (mae_steps + 1) * args.eval_freq
        mse_epochs = (mse_steps + 1) * args.eval_freq

        # Skip only the epoch-5 initialization transient in the overview panels.
        handle = mae_ax.plot(
            mae_epochs[1:], mae[1:], color=color, linewidth=1.8, alpha=0.9, label=label
        )[0]
        legend_handles.append(handle)
        mse_ax.plot(mse_epochs[1:], mse[1:], color=color, linewidth=1.8, alpha=0.9)

        loss_ax.plot(loss_steps, loss, color=color, linewidth=0.7, alpha=0.14)
        loss_ax.plot(
            loss_steps,
            exponential_smooth(loss),
            color=color,
            linewidth=2.0,
        )

        tail_epochs = mae_epochs[-10:]
        tail = mae[-10:]
        mean = float(np.mean(tail))
        std = float(np.std(tail))
        tail_ax.plot(
            tail_epochs,
            tail,
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=3.5,
            alpha=0.85,
        )
        tail_ax.hlines(
            mean,
            tail_epochs[0],
            tail_epochs[-1],
            color=color,
            linewidth=2.2,
            linestyle="--",
            label=f"{label}: {mean:.3f} +/- {std:.3f}",
        )
        tail_ax.fill_between(
            tail_epochs,
            mean - std,
            mean + std,
            color=color,
            alpha=0.08,
        )

    mae_ax.set_title("Validation MAE (epoch 5 transient omitted)")
    mae_ax.set_ylabel("MAE (lower is better)")
    mae_ax.set_ylim(bottom=3)

    mse_ax.set_title("Validation MSE (epoch 5 transient omitted)")
    mse_ax.set_ylabel("MSE (lower is better)")
    mse_ax.set_ylim(bottom=4)

    loss_ax.set_title("Training total loss (raw + TensorBoard-style smoothing 0.6)")
    loss_ax.set_ylabel("Loss (log scale)")
    loss_ax.set_yscale("log")

    tail_ax.set_title("Last 10 validation MAE evaluations")
    tail_ax.set_ylabel("MAE (line/band = mean +/- std)")
    tail_ax.legend(loc="upper left", frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(True, color="#CBD5E1", linewidth=0.7, alpha=0.55)
        axis.set_axisbelow(True)

    fig.legend(
        handles=legend_handles,
        labels=[item[0] for item in RUNS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.944),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Copy-Paste Ablation: Weight Decay x Learning-rate Schedule",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.958,
        "Wheat50 | VGG16-BN | batch 16 | 250 epochs | validation every 5 epochs",
        ha="center",
        color="#475569",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.07, right=0.98, hspace=0.30, wspace=0.19)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
