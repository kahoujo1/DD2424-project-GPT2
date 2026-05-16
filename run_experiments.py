#!/usr/bin/env python3
"""
Experiment launcher for DD2424 / CS224n GPT-2 project.

Creates:
  experiments/results.csv
  experiments/logs/
  experiments/plots/

Supports:
  - full finetuning vs last-layer vs LoRA vs ReFT
  - LoRA rank ablation
  - limited-data PEFT experiments
  - sonnet decoding sweeps

Assumptions:
  - LoRA is enabled with: --enable_lora
  - ReFT is enabled with: --enable_reft
  - LoRA rank is set with: --lora_r
  - limited data is set with: --train_fraction
  - sonnet decoding may support: --top_k and --line_count_stopping
"""

import argparse
import csv
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


EXPERIMENT_DIR = Path("experiments")
LOG_DIR = EXPERIMENT_DIR / "logs"
PLOT_DIR = EXPERIMENT_DIR / "plots"
RESULTS_CSV = EXPERIMENT_DIR / "results.csv"


@dataclass
class Experiment:
    name: str
    group: str
    task: str
    method: str
    script: str
    args: List[str]
    notes: str = ""


def ensure_dirs() -> None:
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def get_supported_flags(script: str) -> set:
    """
    Best-effort extraction of supported argparse flags from script --help.

    This is optional. By default we do NOT filter flags, because some flags are
    placeholders that will be added later, e.g. --enable_reft.
    """
    try:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        help_text = result.stdout
    except Exception:
        return set()

    flags = set(re.findall(r"(--[A-Za-z0-9_-]+)", help_text))
    return flags


def flag_takes_value(flag: str) -> bool:
    """
    Flags that are boolean switches and do not consume the following argument.
    """
    boolean_flags = {
        "--use_gpu",
        "--enable_lora",
        "--enable_reft",
        "--line_count_stopping",
    }
    return flag not in boolean_flags


def filter_supported_args(script: str, args: List[str]) -> List[str]:
    """
    Optional helper for running scripts before all future flags are implemented.
    Use only with --filter-unsupported-flags.
    """
    supported = get_supported_flags(script)

    if not supported:
        print(f"[WARN] Could not read supported flags for {script}; keeping all flags.")
        return args

    filtered = []
    i = 0

    while i < len(args):
        item = args[i]

        if not item.startswith("--"):
            filtered.append(item)
            i += 1
            continue

        flag = item.split("=")[0]

        if flag in supported:
            filtered.append(item)

            if "=" not in item and flag_takes_value(flag) and i + 1 < len(args):
                filtered.append(args[i + 1])
                i += 2
            else:
                i += 1
        else:
            print(f"[WARN] {script} does not support {flag}; dropping it.")

            if "=" not in item and flag_takes_value(flag) and i + 1 < len(args):
                i += 2
            else:
                i += 1

    return filtered


def extract_metric(patterns: List[str], text: str) -> Optional[float]:
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                pass
    return None


def parse_metrics(output: str) -> Dict[str, Optional[float]]:
    """
    Best-effort metric extraction from stdout.
    Logs remain the source of truth.
    """
    return {
        "dev_acc": extract_metric(
            [
                r"dev acc\s*::\s*([0-9]+(?:\.[0-9]+)?)",
                r"dev paraphrase acc\s*::\s*([0-9]+(?:\.[0-9]+)?)",
                r"dev accuracy\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            ],
            output,
        ),
        "train_loss": extract_metric(
            [
                r"train loss\s*::\s*([0-9]+(?:\.[0-9]+)?)",
                r"Epoch\s+\d+:\s+train loss\s*::\s*([0-9]+(?:\.[0-9]+)?)",
            ],
            output,
        ),
        "trainable_percent": extract_metric(
            [
                r"trainable%\s*:\s*([0-9]+(?:\.[0-9]+)?)%",
            ],
            output,
        ),
    }


def append_result(row: Dict[str, object]) -> None:
    fieldnames = [
        "timestamp",
        "name",
        "group",
        "task",
        "method",
        "status",
        "returncode",
        "duration_sec",
        "dev_acc",
        "train_loss",
        "trainable_percent",
        "command",
        "log_file",
        "notes",
    ]

    file_exists = RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_experiment(exp: Experiment, dry_run: bool, filter_unsupported_flags: bool) -> None:
    exp_args = exp.args

    if filter_unsupported_flags:
        exp_args = filter_supported_args(exp.script, exp.args)

    command = [sys.executable, exp.script] + exp_args
    printable = " ".join(shlex.quote(x) for x in command)

    print("\n" + "=" * 100)
    print(f"[EXPERIMENT] {exp.name}")
    print(f"[GROUP]      {exp.group}")
    print(f"[TASK]       {exp.task}")
    print(f"[METHOD]     {exp.method}")
    print(f"[COMMAND]    {printable}")

    if dry_run:
        append_result(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "name": exp.name,
                "group": exp.group,
                "task": exp.task,
                "method": exp.method,
                "status": "dry-run",
                "returncode": "",
                "duration_sec": 0,
                "dev_acc": "",
                "train_loss": "",
                "trainable_percent": "",
                "command": printable,
                "log_file": "",
                "notes": exp.notes,
            }
        )
        return

    start = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + sanitize_name(exp.name)
    log_file = LOG_DIR / f"{run_id}.log"

    with open(log_file, "w") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    duration = time.time() - start

    with open(log_file, "r", errors="replace") as f:
        output = f.read()

    metrics = parse_metrics(output)
    status = "ok" if process.returncode == 0 else "failed"

    append_result(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "name": exp.name,
            "group": exp.group,
            "task": exp.task,
            "method": exp.method,
            "status": status,
            "returncode": process.returncode,
            "duration_sec": round(duration, 2),
            "dev_acc": metrics["dev_acc"] if metrics["dev_acc"] is not None else "",
            "train_loss": metrics["train_loss"] if metrics["train_loss"] is not None else "",
            "trainable_percent": metrics["trainable_percent"] if metrics["trainable_percent"] is not None else "",
            "command": printable,
            "log_file": str(log_file),
            "notes": exp.notes,
        }
    )

    print(f"[STATUS]     {status}")
    print(f"[DURATION]   {duration:.1f} sec")
    print(f"[LOG]        {log_file}")

    if metrics["dev_acc"] is not None:
        print(f"[DEV ACC]    {metrics['dev_acc']}")

    if metrics["train_loss"] is not None:
        print(f"[LOSS]       {metrics['train_loss']}")

    if metrics["trainable_percent"] is not None:
        print(f"[TRAINABLE]  {metrics['trainable_percent']}%")


def base_runtime_args(args: argparse.Namespace) -> List[str]:
    out = [
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
    ]

    if args.use_gpu:
        out.append("--use_gpu")

    return out


def lora_flags(rank: int, alpha: float, target_modules: List[str]) -> List[str]:
    flags = [
        "--enable_lora",
        "--lora_r",
        str(rank),
        "--lora_alpha",
        str(alpha),
    ]

    if target_modules:
        flags.append("--lora_target_modules")
        flags.extend(target_modules)

    return flags


def build_comparison_experiments(args: argparse.Namespace) -> List[Experiment]:
    common = base_runtime_args(args)
    exps = []

    # Sentiment comparison.
    exps.append(
        Experiment(
            name="sentiment_last_linear_layer",
            group="comparison",
            task="sentiment",
            method="last-linear-layer",
            script="classifier.py",
            args=common + ["--fine-tune-mode", "last-linear-layer"],
        )
    )

    exps.append(
        Experiment(
            name="sentiment_full_model",
            group="comparison",
            task="sentiment",
            method="full-model",
            script="classifier.py",
            args=common + ["--fine-tune-mode", "full-model"],
        )
    )

    # Your classifier asserts that LoRA should not be combined with full-model.
    # So LoRA uses last-linear-layer mode while LoRA params stay trainable.
    exps.append(
        Experiment(
            name="sentiment_lora_r4",
            group="comparison",
            task="sentiment",
            method="lora",
            script="classifier.py",
            args=common
            + ["--fine-tune-mode", "last-linear-layer"]
            + lora_flags(4, args.lora_alpha, args.lora_target_modules),
        )
    )

    exps.append(
        Experiment(
            name="sentiment_reft_placeholder",
            group="comparison",
            task="sentiment",
            method="reft",
            script="classifier.py",
            args=common + ["--fine-tune-mode", "full-model", "--enable_reft"],
            notes="ReFT placeholder. Assumes --enable_reft exists but does nothing until ReFT is implemented.",
        )
    )

    # Paraphrase comparison.
    exps.append(
        Experiment(
            name="paraphrase_full_model",
            group="comparison",
            task="paraphrase",
            method="full-model",
            script="paraphrase_detection.py",
            args=common,
        )
    )

    exps.append(
        Experiment(
            name="paraphrase_lora_r4",
            group="comparison",
            task="paraphrase",
            method="lora",
            script="paraphrase_detection.py",
            args=common + lora_flags(4, args.lora_alpha, args.lora_target_modules),
        )
    )

    exps.append(
        Experiment(
            name="paraphrase_reft_placeholder",
            group="comparison",
            task="paraphrase",
            method="reft",
            script="paraphrase_detection.py",
            args=common + ["--enable_reft"],
            notes="ReFT placeholder. Assumes --enable_reft exists but does nothing until ReFT is implemented.",
        )
    )

    # Sonnet comparison.
    exps.append(
        Experiment(
            name="sonnet_full_model",
            group="comparison",
            task="sonnet",
            method="full-model",
            script="sonnet_generation.py",
            args=common + ["--temperature", str(args.temperature), "--top_p", str(args.top_p)],
        )
    )

    exps.append(
        Experiment(
            name="sonnet_lora_r4",
            group="comparison",
            task="sonnet",
            method="lora",
            script="sonnet_generation.py",
            args=common
            + lora_flags(4, args.lora_alpha, args.lora_target_modules)
            + ["--temperature", str(args.temperature), "--top_p", str(args.top_p)],
        )
    )

    exps.append(
        Experiment(
            name="sonnet_reft_placeholder",
            group="comparison",
            task="sonnet",
            method="reft",
            script="sonnet_generation.py",
            args=common + ["--enable_reft", "--temperature", str(args.temperature), "--top_p", str(args.top_p)],
            notes="ReFT placeholder. Assumes --enable_reft exists but does nothing until ReFT is implemented.",
        )
    )

    return exps


def build_lora_rank_experiments(args: argparse.Namespace) -> List[Experiment]:
    common = base_runtime_args(args)
    exps = []

    scripts = {
        "sentiment": "classifier.py",
        "paraphrase": "paraphrase_detection.py",
        "sonnet": "sonnet_generation.py",
    }

    for task, script in scripts.items():
        for rank in args.lora_ranks:
            extra = lora_flags(rank, args.lora_alpha, args.lora_target_modules)

            if task == "sentiment":
                extra = ["--fine-tune-mode", "last-linear-layer"] + extra

            if task == "sonnet":
                extra += ["--temperature", str(args.temperature), "--top_p", str(args.top_p)]

            exps.append(
                Experiment(
                    name=f"{task}_lora_rank_{rank}",
                    group="lora_rank",
                    task=task,
                    method=f"lora_r{rank}",
                    script=script,
                    args=common + extra,
                )
            )

    return exps


def build_limited_data_experiments(args: argparse.Namespace) -> List[Experiment]:
    common = base_runtime_args(args)
    exps = []

    scripts = {
        "sentiment": "classifier.py",
        "paraphrase": "paraphrase_detection.py",
        "sonnet": "sonnet_generation.py",
    }

    for task, script in scripts.items():
        for fraction in args.train_fractions:
            # Full model.
            full_extra = ["--train_fraction", str(fraction)]
            if task == "sentiment":
                full_extra = ["--fine-tune-mode", "full-model"] + full_extra
            if task == "sonnet":
                full_extra += ["--temperature", str(args.temperature), "--top_p", str(args.top_p)]

            exps.append(
                Experiment(
                    name=f"{task}_full_model_data_{fraction}",
                    group="limited_data",
                    task=task,
                    method="full-model",
                    script=script,
                    args=common + full_extra,
                )
            )

            # LoRA.
            lora_extra = ["--train_fraction", str(fraction)] + lora_flags(
                4, args.lora_alpha, args.lora_target_modules
            )
            if task == "sentiment":
                lora_extra = ["--fine-tune-mode", "last-linear-layer"] + lora_extra
            if task == "sonnet":
                lora_extra += ["--temperature", str(args.temperature), "--top_p", str(args.top_p)]

            exps.append(
                Experiment(
                    name=f"{task}_lora_data_{fraction}",
                    group="limited_data",
                    task=task,
                    method="lora",
                    script=script,
                    args=common + lora_extra,
                )
            )

            # ReFT placeholder.
            reft_extra = ["--train_fraction", str(fraction), "--enable_reft"]
            if task == "sentiment":
                reft_extra = ["--fine-tune-mode", "full-model"] + reft_extra
            if task == "sonnet":
                reft_extra += ["--temperature", str(args.temperature), "--top_p", str(args.top_p)]

            exps.append(
                Experiment(
                    name=f"{task}_reft_data_{fraction}",
                    group="limited_data",
                    task=task,
                    method="reft",
                    script=script,
                    args=common + reft_extra,
                    notes="ReFT placeholder. Assumes --enable_reft exists but does nothing until ReFT is implemented.",
                )
            )

    return exps


def build_sonnet_decoding_experiments(args: argparse.Namespace) -> List[Experiment]:
    common = base_runtime_args(args)
    exps = []

    for temp in args.temperature_sweep:
        exps.append(
            Experiment(
                name=f"sonnet_decode_temperature_{temp}",
                group="sonnet_decoding",
                task="sonnet",
                method="temperature_sweep",
                script="sonnet_generation.py",
                args=common + ["--temperature", str(temp), "--top_p", str(args.top_p)],
            )
        )

    for top_p in args.top_p_sweep:
        exps.append(
            Experiment(
                name=f"sonnet_decode_top_p_{top_p}",
                group="sonnet_decoding",
                task="sonnet",
                method="top_p_sweep",
                script="sonnet_generation.py",
                args=common + ["--temperature", str(args.temperature), "--top_p", str(top_p)],
            )
        )

    for top_k in args.top_k_sweep:
        exps.append(
            Experiment(
                name=f"sonnet_decode_top_k_{top_k}",
                group="sonnet_decoding",
                task="sonnet",
                method="top_k_sweep",
                script="sonnet_generation.py",
                args=common
                + [
                    "--temperature",
                    str(args.temperature),
                    "--top_p",
                    str(args.top_p),
                    "--top_k",
                    str(top_k),
                ],
            )
        )

    exps.append(
        Experiment(
            name="sonnet_decode_line_count_stopping",
            group="sonnet_decoding",
            task="sonnet",
            method="line_count_stopping",
            script="sonnet_generation.py",
            args=common
            + [
                "--temperature",
                str(args.temperature),
                "--top_p",
                str(args.top_p),
                "--line_count_stopping",
            ],
        )
    )

    return exps


def build_experiments(args: argparse.Namespace) -> List[Experiment]:
    exps = []

    if args.experiment in ["comparison", "all"]:
        exps.extend(build_comparison_experiments(args))

    if args.experiment in ["lora_rank", "all"]:
        exps.extend(build_lora_rank_experiments(args))

    if args.experiment in ["limited_data", "all"]:
        exps.extend(build_limited_data_experiments(args))

    if args.experiment in ["sonnet_decoding", "all"]:
        exps.extend(build_sonnet_decoding_experiments(args))

    if args.task != "all":
        exps = [e for e in exps if e.task == args.task]

    if args.max_runs is not None:
        exps = exps[: args.max_runs]

    return exps


def plot_results() -> None:
    """
    Creates a simple dev accuracy bar chart from completed runs.
    Dry-runs and failed runs are ignored.
    """
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        print("[PLOT] No results.csv found.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib is not available.")
        return

    rows = []
    with open(RESULTS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("status") != "ok":
                continue

            try:
                dev_acc = float(row.get("dev_acc", ""))
            except (ValueError, TypeError):
                continue

            rows.append(
                (
                    row.get("task", ""),
                    row.get("method", ""),
                    row.get("group", ""),
                    dev_acc,
                )
            )

    if not rows:
        print("[PLOT] No numeric dev_acc rows to plot.")
        return

    labels = [f"{task}\n{method}" for task, method, _, _ in rows]
    values = [acc for _, _, _, acc in rows]

    plt.figure(figsize=(max(8, len(labels) * 0.7), 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Dev accuracy")
    plt.title("Experiment comparison")
    plt.tight_layout()

    out_path = PLOT_DIR / "dev_accuracy_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[PLOT] Saved {out_path}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["comparison", "lora_rank", "limited_data", "sonnet_decoding", "all"],
        default="comparison",
    )

    parser.add_argument(
        "--task",
        choices=["all", "sentiment", "paraphrase", "sonnet"],
        default="all",
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--filter-unsupported-flags", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        choices=["query", "key", "value", "dense"],
        default=["query", "value"],
    )
    parser.add_argument("--lora_ranks", type=int, nargs="+", default=[2, 4, 8, 16])

    parser.add_argument("--train_fractions", type=float, nargs="+", default=[0.10, 0.25, 0.50, 1.00])

    parser.add_argument("--temperature_sweep", type=float, nargs="+", default=[0.7, 1.0, 1.2])
    parser.add_argument("--top_p_sweep", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    parser.add_argument("--top_k_sweep", type=int, nargs="+", default=[20, 50, 100])

    return parser.parse_args()


def main() -> None:
    args = get_args()
    ensure_dirs()

    if args.plot_only:
        plot_results()
        return

    experiments = build_experiments(args)

    print(f"[INFO] Python executable: {sys.executable}")
    print(f"[INFO] Planned experiments: {len(experiments)}")

    for exp in experiments:
        run_experiment(
            exp,
            dry_run=args.dry_run,
            filter_unsupported_flags=args.filter_unsupported_flags,
        )

    plot_results()


if __name__ == "__main__":
    main()