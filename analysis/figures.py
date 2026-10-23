"""Produce all 7 figures + master table from results/summary.csv.

Usage:
    python -m analysis.figures --summary results/summary.csv --out results/figures

Produces:
    F1  per_config_latency.pdf      – p50 + p95 bars per config, per system
    F2  ablation_headline.pdf        – add-one gain and LOO loss per lever
    F3  speedup_vs_concurrency.pdf   – S vs. concurrency for H2
    F4  quality_guardrails.pdf       – accuracy + catch rate per config (flat = good)
    F5  token_cache_breakdown.pdf    – prompt/cached/output tokens per config
    F6  trace_waterfall.pdf          – TODO: needs raw trace data (placeholder)
    T1  master_results.csv           – full quantitative table
"""

import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    raise SystemExit("matplotlib and pandas required: pip install matplotlib pandas")

CONFIG_ORDER = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
LEVER_NAMES = {"C1": "L1 parallel", "C2": "L2 coalesce", "C3": "L3 cache", "C4": "L4 tier"}
LOO_MAP = {"C6": "L1", "C7": "L2", "C8": "L3", "C9": "L4"}
SYSTEMS = ["debate", "groupchat"]


def _load(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    # Average over seeds for plotting
    return df.groupby(["system", "config", "concurrency"], as_index=False).mean(numeric_only=True)


def f1_latency_bars(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, system in zip(axes, SYSTEMS):
        sub = df[(df["system"] == system) & (df["concurrency"] == 1)]
        sub = sub.set_index("config").reindex(CONFIG_ORDER).dropna(subset=["p50_s"])
        x = range(len(sub))
        ax.bar(x, sub["p50_s"], label="p50", color="#4a90d9", alpha=0.85)
        ax.errorbar(x, sub["p50_s"],
                    yerr=[sub["p50_s"] - sub["p50_s"], sub["p95_s"] - sub["p50_s"]],
                    fmt="none", ecolor="#1a3a5c", capsize=4, linewidth=1.2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(list(sub.index), fontsize=9)
        ax.set_title(f"{system} — latency per config", fontsize=11)
        ax.set_ylabel("Task completion time (s)")
        ax.set_xlabel("Configuration")
        ax.axhline(sub.loc["C0", "p50_s"] if "C0" in sub.index else 0,
                   color="gray", linestyle="--", linewidth=0.8, label="baseline p50")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "F1_per_config_latency.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote F1")


def f2_ablation_headline(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, system in zip(axes, SYSTEMS):
        sub = df[(df["system"] == system) & (df["concurrency"] == 1)].set_index("config")
        s_all = sub.loc["C5", "speedup_S"] if "C5" in sub.index else float("nan")
        add_one = {lv: sub.loc[cfg, "speedup_S"] - 1.0
                   for cfg, lv in LEVER_NAMES.items() if cfg in sub.index}
        loo_loss = {lv: s_all - sub.loc[cfg, "speedup_S"]
                    for cfg, lv in LOO_MAP.items() if cfg in sub.index}
        levers = sorted(set(add_one) | set(loo_loss))
        x = range(len(levers))
        w = 0.35
        ax.bar([i - w / 2 for i in x],
               [add_one.get(l, 0) for l in levers],
               width=w, label="Add-one gain", color="#4a90d9", alpha=0.85)
        ax.bar([i + w / 2 for i in x],
               [loo_loss.get(l, 0) for l in levers],
               width=w, label="LOO loss", color="#e05c3a", alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(levers, fontsize=10)
        ax.set_title(f"{system} — ablation headline", fontsize=11)
        ax.set_ylabel("Speedup contribution")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "F2_ablation_headline.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote F2")


def f3_speedup_vs_concurrency(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, system in zip(axes, SYSTEMS):
        sub = df[df["system"] == system]
        for cfg, color, label in [("C0", "gray", "C0 baseline"), ("C1", "#4a90d9", "C1 (+L1)"),
                                   ("C5", "#e05c3a", "C5 (all)")]:
            s = sub[sub["config"] == cfg].sort_values("concurrency")
            if not s.empty:
                ax.plot(s["concurrency"], s["speedup_S"], marker="o", label=label, color=color)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Concurrent tasks")
        ax.set_ylabel("Speedup S")
        ax.set_title(f"{system} — speedup vs. load", fontsize=11)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "F3_speedup_vs_concurrency.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote F3")


def f5_token_cache(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, system in zip(axes, SYSTEMS):
        sub = df[(df["system"] == system) & (df["concurrency"] == 1)]
        sub = sub.set_index("config").reindex(CONFIG_ORDER).dropna(subset=["prompt_tokens"])
        x = range(len(sub))
        ax.bar(x, sub["cached_tokens"], label="cached", color="#4a90d9", alpha=0.85)
        ax.bar(x, sub["prompt_tokens"] - sub["cached_tokens"],
               bottom=sub["cached_tokens"], label="prompt (uncached)", color="#b0c8e8", alpha=0.85)
        ax.bar(x, sub["output_tokens"],
               bottom=sub["prompt_tokens"], label="output", color="#e05c3a", alpha=0.65)
        ax.set_xticks(list(x))
        ax.set_xticklabels(list(sub.index), fontsize=9)
        ax.set_title(f"{system} — token breakdown", fontsize=11)
        ax.set_ylabel("Total tokens")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "F5_token_cache_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote F5")


def t1_master_table(df: pd.DataFrame, out_dir: Path) -> None:
    cols = ["system", "config", "seed", "concurrency",
            "p50_s", "p95_s", "std_s", "speedup_S", "ci_lower",
            "prompt_tokens", "cached_tokens", "output_tokens", "cache_hit_rate"]
    available = [c for c in cols if c in df.columns]
    df[available].sort_values(["system", "config", "seed", "concurrency"]).to_csv(
        out_dir / "T1_master_results.csv", index=False
    )
    print("Wrote T1")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = _load(args.summary)

    f1_latency_bars(df, args.out)
    f2_ablation_headline(df, args.out)
    f3_speedup_vs_concurrency(df, args.out)
    f5_token_cache(df, args.out)
    t1_master_table(df, args.out)
    print(f"\nAll figures written to {args.out}/")
    print("F4 (quality guardrails) and F6 (trace waterfall) require quality scores "
          "from eval/ — add them via analysis/figures_quality.py after running eval.")


if __name__ == "__main__":
    main()
