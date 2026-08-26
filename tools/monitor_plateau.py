"""Check P2PNet training progress and decide if MAE has plateaued.

Parses the training log (one eval per epoch, eval_freq=1), finds the best MAE
and the epoch at which it was last improved, and reports whether the run has
gone `--patience` epochs without improvement, finished, or crashed.

Usage:
  python tools/monitor_plateau.py --log logs/wheat50_v2/train.log \
      --pid 3551675 --patience 400 --total_epochs 3500

Prints a one-line DECISION (CONTINUE | PLATEAU | FINISHED | CRASHED) plus stats.
Exit code: 0 = CONTINUE, 1 = should notify (PLATEAU/FINISHED/CRASHED).
"""
import argparse
import os
import re
import sys


def proc_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--patience", type=int, default=400,
                    help="Epochs without a new best AFTER MAE first improved -> PLATEAU.")
    ap.add_argument("--stuck_patience", type=int, default=600,
                    help="Epochs with MAE never improving from its initial value -> STUCK "
                         "(longer, to tolerate P2PNet's slow warm-up).")
    ap.add_argument("--total_epochs", type=int, default=3500)
    args = ap.parse_args()

    with open(args.log) as f:
        text = f.read()

    # epochs completed (training side)
    ep_nums = [int(m) for m in re.findall(r"\[ep (\d+)\]", text)]
    latest_epoch = max(ep_nums) if ep_nums else -1

    # eval MAEs, in order (one per epoch with eval_freq=1).
    # The eval line is "mae: X mse: Y ... best mae: Z" -- exclude the "best mae:"
    # occurrence so we count one value per eval, not two.
    maes = [float(m) for m in re.findall(r"(?<!best )mae:\s*([0-9.]+)", text)]

    alive = proc_alive(args.pid)

    if not maes:
        decision = "CRASHED" if not alive else "CONTINUE"
        print(f"DECISION={decision} no_eval_yet alive={alive} latest_epoch={latest_epoch}")
        sys.exit(0 if decision == "CONTINUE" else 1)

    # running-min: index where a new best was last achieved
    best = float("inf")
    best_idx = 0
    for i, v in enumerate(maes):
        if v < best - 1e-9:
            best = v
            best_idx = i
    n_evals = len(maes)
    last_eval_idx = n_evals - 1
    since_improve = last_eval_idx - best_idx  # in epochs (eval per epoch)

    improved_at_least_once = best_idx > 0
    if not alive:
        decision = "FINISHED" if latest_epoch >= args.total_epochs - 1 else "CRASHED"
    elif improved_at_least_once and since_improve >= args.patience:
        decision = "PLATEAU"
    elif (not improved_at_least_once) and since_improve >= args.stuck_patience:
        decision = "STUCK"
    else:
        decision = "CONTINUE"

    print(
        f"DECISION={decision} latest_epoch={latest_epoch} n_evals={n_evals} "
        f"best_mae={best:.4f} best_epoch={best_idx} since_improve={since_improve} "
        f"patience={args.patience} alive={alive} latest_mae={maes[-1]:.4f}"
    )
    sys.exit(0 if decision == "CONTINUE" else 1)


if __name__ == "__main__":
    main()
