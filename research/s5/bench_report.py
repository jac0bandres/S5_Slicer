#!/usr/bin/env python
"""bench_report.py — aggregate bench_results/<tag>/ into a single report.

Reads every per-model benchmark directory written by ``bench_sweep.py``
(``timings.csv`` / ``correctness.csv`` / ``env.json`` / ``meta.json``) and emits:

    bench_results/REPORT.md          — env, per-model overview, S4-vs-S5 speedup
                                        table, S5-scaling table, correctness table
    bench_results/speedup.png        — geomean S4/S5 speedup per pipeline stage

Only directories that look like a sweep run are included; the stale Windows
``pi_3mm_subdiv_0`` directory is skipped by default (mixing platforms in one
timing table is misleading). Override with --include.

Usage:
    .venv/bin/python bench_report.py
    .venv/bin/python bench_report.py --include pi_3mm_subdiv_0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("bench_results")
SKIP_DIRS = {"pi_3mm_subdiv_0"}  # superseded Windows run

# Preferred display order for stages and models.
STAGE_ORDER = ["adjacency", "smoothing", "deformation",
               "vertex_rotation", "volume_scales", "barycentric"]
# tag -> pretty label; falls back to the tag if unknown.
MODEL_LABELS = {
    "big_pi": "Big pi", "letter_z": "Letter Z", "pi_3mm": "pi 3mm",
    "dino": "dino", "bridge": "bridge", "squirtle": "Squirtle",
    "benchy": "benchy", "tree": "tree", "catstretch": "catstretch",
}
MODEL_ORDER = ["big_pi", "letter_z", "pi_3mm", "dino",
               "bridge", "squirtle", "benchy", "tree", "catstretch"]


def _parse_context(ctx: str) -> dict:
    out = {}
    for pair in (ctx or "").split(";"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def load_run(d: Path) -> dict | None:
    timings = d / "timings.csv"
    if not timings.is_file():
        return None
    run = {"tag": d.name, "timings": [], "correctness": [],
           "env": {}, "meta": {}, "n_cells": None, "n_points": None}
    with timings.open() as f:
        for row in csv.DictReader(f):
            try:
                row["wall_s"] = float(row["wall_s"])
            except (TypeError, ValueError):
                row["wall_s"] = None
            run["timings"].append(row)
            ctx = _parse_context(row.get("context", ""))
            if run["n_cells"] is None and "n_cells" in ctx:
                run["n_cells"] = int(ctx["n_cells"])
                run["n_points"] = int(ctx.get("n_points", 0))
    corr = d / "correctness.csv"
    if corr.is_file():
        with corr.open() as f:
            run["correctness"] = list(csv.DictReader(f))
    for name in ("env", "meta"):
        p = d / f"{name}.json"
        if p.is_file():
            run[name] = json.loads(p.read_text())
    return run


def stage_means(run: dict) -> dict:
    """(stage) -> {'s5': mean_wall, 's4': mean_wall or None, 's4_ok': bool}."""
    by = defaultdict(lambda: defaultdict(list))
    ok = defaultdict(lambda: defaultdict(list))
    for row in run["timings"]:
        by[row["stage"]][row["impl"]].append(row["wall_s"])
        ok[row["stage"]][row["impl"]].append(row.get("status") == "ok")
    out = {}
    for stage, impls in by.items():
        s5 = _mean(impls.get("s5", []))
        s4 = _mean([w for w, o in zip(impls.get("s4", []), ok[stage].get("s4", []))
                    if o]) if "s4" in impls else None
        out[stage] = {"s5": s5, "s4": s4}
    return out


def ordered(keys, order):
    seen = list(dict.fromkeys(keys))
    head = [k for k in order if k in seen]
    tail = [k for k in seen if k not in order]
    return head + tail


def fmt_ms(s):
    return "—" if s is None else f"{s*1000:,.1f}"


def fmt_x(v):
    return "—" if v is None else f"{v:,.1f}×"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", default="", help="Comma-separated extra dirs to include.")
    args = ap.parse_args()
    skip = SKIP_DIRS - set(t.strip() for t in args.include.split(",") if t.strip())

    runs = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir() or d.name in skip:
            continue
        r = load_run(d)
        if r:
            runs[d.name] = r
    if not runs:
        raise SystemExit("No benchmark runs found in bench_results/.")

    tags = ordered(runs.keys(), MODEL_ORDER)
    means = {t: stage_means(runs[t]) for t in tags}

    # env (take the first run that has one)
    env = next((runs[t]["env"] for t in tags if runs[t]["env"]), {})
    s4_tags = [t for t in tags if any(m["s4"] is not None for m in means[t].values())]

    # ── speedup per stage, geomean across S4-capable models ──────────────────
    stage_speedups = defaultdict(list)  # stage -> [speedup per model]
    for t in s4_tags:
        for stage, m in means[t].items():
            if m["s4"] and m["s5"]:
                stage_speedups[stage].append(m["s4"] / m["s5"])
    geomean = {}
    for stage, xs in stage_speedups.items():
        if xs:
            geomean[stage] = math.exp(sum(math.log(x) for x in xs) / len(xs))

    lines = []
    W = lines.append
    W("# S5_Slicer benchmark report")
    W("")
    W("Per-stage timing and correctness of the optimized S5 pipeline (`S5.py`) "
      "against the naive S4 reference (`s4_reference.py`), generated by "
      "`bench_report.py` from the `bench_sweep.py` runs under `bench_results/`.")
    W("")

    # Environment
    pkgs = env.get("packages", {})
    W("## Environment")
    W("")
    W(f"- **Platform:** {env.get('platform', '?')}")
    W(f"- **Python:** {env.get('python', '?')}")
    if "cpu_count_logical" in env:
        W(f"- **CPU:** {env.get('cpu_count_physical', '?')} physical / "
          f"{env.get('cpu_count_logical', '?')} logical, "
          f"{env.get('ram_gb', '?')} GB RAM")
    key_pkgs = ", ".join(f"{k} {pkgs[k]}" for k in
                         ("numpy", "scipy", "pyvista", "tetgen", "igl")
                         if k in pkgs)
    if key_pkgs:
        W(f"- **Packages:** {key_pkgs}")
    W("")

    # Per-model overview
    W("## Models")
    W("")
    W("| Model | Cells | Points | S4 compared | Instrumented S5 total |")
    W("|---|--:|--:|:--:|--:|")
    for t in tags:
        r = runs[t]
        s5_total = sum(m["s5"] for m in means[t].values() if m["s5"])
        s4c = "yes" if t in s4_tags else "—"
        W(f"| {MODEL_LABELS.get(t, t)} | {r['n_cells'] or '—':,} | "
          f"{r['n_points'] or '—':,} | {s4c} | {s5_total*1000:,.0f} ms |")
    W("")
    W("*Instrumented S5 total = sum of the benchmarked stages only, not full "
      "wall-clock (tetgen, the CuraEngine slice, and G-code reprojection are not "
      "instrumented here).*")
    W("")

    # Speedup table
    if s4_tags:
        stages = ordered(
            [s for s in geomean], STAGE_ORDER)
        W("## S4 → S5 speedup")
        W("")
        W("Mean wall time per stage on the models where the S4 reference is "
          "tractable, and the S5 speedup (`S4 / S5`). Times in ms.")
        W("")
        header = "| Stage | " + " | ".join(
            f"{MODEL_LABELS.get(t, t)} S5 / S4 / ×" for t in s4_tags) + " |"
        W(header)
        W("|---|" + "---|" * len(s4_tags))
        for stage in stages:
            cells = []
            for t in s4_tags:
                m = means[t].get(stage, {"s5": None, "s4": None})
                sp = (m["s4"] / m["s5"]) if (m["s4"] and m["s5"]) else None
                cells.append(f"{fmt_ms(m['s5'])} / {fmt_ms(m['s4'])} / {fmt_x(sp)}")
            W(f"| `{stage}` | " + " | ".join(cells) + " |")
        W("")
        W("**Geomean speedup per stage** (across the S4-compared models):")
        W("")
        W("| Stage | " + " | ".join(f"`{s}`" for s in stages) + " |")
        W("|---|" + "---|" * len(stages))
        W("| ×faster | " + " | ".join(fmt_x(geomean.get(s)) for s in stages) + " |")
        W("")
        W("![Speedup per stage](speedup.png)")
        W("")

    # S5 scaling across all models
    all_stages = ordered(
        {s for t in tags for s in means[t]}, STAGE_ORDER)
    W("## S5 stage timing across all models")
    W("")
    W("S5 mean wall time per stage (ms) — includes the large models the S4 "
      "reference is too slow to compare against.")
    W("")
    W("| Stage | " + " | ".join(MODEL_LABELS.get(t, t) for t in tags) + " |")
    W("|---|" + "---:|" * len(tags))
    for stage in all_stages:
        cells = [fmt_ms(means[t].get(stage, {}).get("s5")) for t in tags]
        W(f"| `{stage}` | " + " | ".join(cells) + " |")
    W("")

    # Correctness
    W("## Correctness (S4 vs S5)")
    W("")
    W("Agreement between the S4 reference output and the S5 output per stage. "
      "`deformation` is **expected** to diverge: S5 replaced S4's non-converged "
      "`lsqr` with an exact direct sparse solve (see `OPTIMIZE.md` §3), so this is "
      "a change of result, not a regression.")
    W("")
    W("| Model | Stage | Metric | max_abs | cosine / pearson | exact | n |")
    W("|---|---|---|--:|--:|:--:|--:|")
    for t in s4_tags:
        seen = set()
        for row in runs[t]["correctness"]:
            stage = row.get("stage", "")
            key = (stage, row.get("metric", ""))
            if key in seen:
                continue
            seen.add(key)
            cos = row.get("cosine_mean", "") or row.get("pearson_r", "")
            try:
                cos = f"{float(cos):.4f}"
            except (TypeError, ValueError):
                cos = "—"
            ma = row.get("max_abs", "")
            try:
                ma = f"{float(ma):.3g}"
            except (TypeError, ValueError):
                ma = "—"
            exact = row.get("exact_match", "") or "—"
            n = row.get("n_compared", "") or row.get("n_s5", "") or "—"
            W(f"| {MODEL_LABELS.get(t, t)} | `{stage}` | {row.get('metric','')} "
              f"| {ma} | {cos} | {exact} | {n} |")
    W("")

    report_path = RESULTS_DIR / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"wrote {report_path}")

    # ── chart ────────────────────────────────────────────────────────────────
    if geomean:
        _render_speedup_chart(geomean)


def _render_speedup_chart(geomean: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # dataviz reference palette (light surface) — single series, so one hue.
    SURFACE = "#fcfcfb"
    BLUE = "#2a78d6"
    INK = "#0b0b0b"
    MUTED = "#898781"
    GRID = "#e1e0d9"

    stages = ordered(list(geomean), STAGE_ORDER)
    vals = [geomean[s] for s in stages]

    fig, ax = plt.subplots(figsize=(8, 0.62 * len(stages) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y = range(len(stages))
    ax.barh(list(y), vals, color=BLUE, height=0.62, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(stages, color=INK, fontsize=11)
    ax.invert_yaxis()

    ax.set_xscale("log")
    ax.set_xlabel("Speedup  (S4 wall time ÷ S5 wall time, geomean)", color=MUTED, fontsize=10)
    ax.axvline(1.0, color=MUTED, lw=1, ls="--", zorder=2)  # parity

    for i, v in zip(y, vals):
        ax.text(v * 1.05, i, f"{v:,.1f}×", va="center", ha="left",
                color=INK, fontsize=10, fontweight="bold")

    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_xlim(right=max(vals) * 1.6)
    ax.set_title("S5 per-stage speedup over the S4 reference",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=10)

    fig.tight_layout()
    out = RESULTS_DIR / "speedup.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
