#!/usr/bin/env python
"""bench_sweep.py — run the S4-vs-S5 per-stage benchmark across input_models/.

Drives ``S5.slice()`` with ``BenchConfig`` enabled, one ``run_tag`` per model,
writing ``timings.csv`` / ``correctness.csv`` / ``env.json`` / ``meta.json``
under ``bench_results/<tag>/`` (exactly what ``bench.finish()`` emits).

The naive S4 reference (``s4_reference.py``) is only tractable on small meshes —
e.g. the barycentric stage is ~80 s on pi's 25 k cells and scales with cell
count — so S4-vs-S5 *correctness* is gated by mesh size: small models run the
full comparison, large models record S5-only timings (which also serve as the
end-to-end CuraEngine compatibility check).

Usage:
    .venv/bin/python bench_sweep.py                 # default set
    .venv/bin/python bench_sweep.py --only pi_3mm,dino
    .venv/bin/python bench_sweep.py --include-catstretch
    .venv/bin/python bench_sweep.py --no-s4         # timing only, everywhere

Then aggregate with:
    .venv/bin/python bench_report.py
"""
from __future__ import annotations

import argparse
import os
import time
import traceback

# ── Model set ────────────────────────────────────────────────────────────────
# (tag, filename, run_s4).  run_s4=True only where the naive reference finishes
# in a sane time; large meshes are S5-only.
MODELS = [
    # small — full S4-vs-S5 correctness + timing
    ("big_pi",   "Big pi.stl",                 True),
    ("letter_z", "Letter_Z.stl",               True),
    ("pi_3mm",   "pi 3mm.stl",                  True),
    ("dino",     "dino.stl",                    True),
    # large — S5-only timing (also the CuraEngine end-to-end compat check)
    ("bridge",   "bridge.stl",                  False),
    ("squirtle", "Squirtle.stl",               False),
    ("benchy",   "benchy upsidedown tilted.stl", False),
    ("tree",     "tree.stl",                    False),
]
CATSTRETCH = ("catstretch", "catstretch.stl", False)  # 84 MB — opt-in only

# ── Fixed slice() defaults (mirror S5.py's CLI defaults) ─────────────────────
CURA_PATH = "/usr/bin/CuraEngine"
CONFIG_PATH = "config/core.def.json"
_CURA_DEFS = "/usr/share/cura/resources/definitions"


def _def_path(name: str) -> str:
    """System Cura def if present (engine-matched), else the bundled copy.

    Mirrors the ``_def_path`` helper in S5.py's __main__ block.
    """
    sys_path = os.path.join(_CURA_DEFS, name)
    return sys_path if os.path.isfile(sys_path) else f"config/{name}"


def run_one(tag: str, filename: str, run_s4: bool, out_root: str) -> dict:
    """Run one model end-to-end with bench enabled; write bench_results/<tag>/.

    Returns a small status dict for the sweep-level summary.
    """
    import S5
    from s5_bench import bench, BenchConfig

    model_path = os.path.join("input_models", filename)
    output_path = os.path.join(out_root, tag)

    BenchConfig.enabled = True
    BenchConfig.output_dir = "./bench_results"
    BenchConfig.record_correctness = True
    BenchConfig.track_memory = False
    BenchConfig.verbose = True
    BenchConfig.run_tag = tag
    BenchConfig.run_s4_comparison = run_s4

    bench.reset()
    t0 = time.time()
    status, err = "ok", ""
    try:
        S5.slice(
            model_path=model_path,
            config_path=CONFIG_PATH,
            cura_path=CURA_PATH,
            extruder_path=_def_path("fdmextruder.def.json"),
            printer_path=_def_path("fdmprinter.def.json"),
            output_path=output_path,
            offset=[0.0, 0.0, 0.0],
            scale=1.0,
            rotation_multiplier=2.0,
            neighbor_loss_weight=30.0,
            max_overhang=30.0,
            nozzle_offset=41.5,
            reorient=None,
            gcode_dialect="rtheta",
        )
        # slice() calls bench.finish() on the happy path.
    except Exception as exc:  # noqa: BLE001 — record any model that fails to slice
        status = "error"
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        # Salvage whatever timings/correctness accumulated before the failure.
        try:
            bench.finish()
        except Exception:  # noqa: BLE001
            pass
    wall = time.time() - t0
    return {"tag": tag, "model": filename, "run_s4": run_s4,
            "status": status, "error": err, "wall_s": round(wall, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None,
                    help="Comma-separated tags to run (default: the full set).")
    ap.add_argument("--include-catstretch", action="store_true",
                    help="Also run the 84 MB catstretch.stl (S5-only).")
    ap.add_argument("--no-s4", action="store_true",
                    help="Force run_s4=False for every model (timing only).")
    ap.add_argument("--out-root", default="bench_output",
                    help="Where per-model slice artifacts (STL/G-code/pkl) go.")
    args = ap.parse_args()

    models = list(MODELS)
    if args.include_catstretch:
        models.append(CATSTRETCH)
    if args.only:
        want = {t.strip() for t in args.only.split(",")}
        models = [m for m in models if m[0] in want]
        missing = want - {m[0] for m in models}
        if missing:
            ap.error(f"unknown tag(s): {sorted(missing)}")
    if args.no_s4:
        models = [(t, f, False) for (t, f, _) in models]

    os.makedirs(args.out_root, exist_ok=True)
    summary = []
    sweep_t0 = time.time()
    for i, (tag, filename, run_s4) in enumerate(models, 1):
        print(f"\n{'='*70}\n[sweep {i}/{len(models)}] {tag}  ({filename})  "
              f"s4={run_s4}\n{'='*70}", flush=True)
        summary.append(run_one(tag, filename, run_s4, args.out_root))

    total = time.time() - sweep_t0
    print(f"\n{'='*70}\nSWEEP SUMMARY  ({total/60:.1f} min total)\n{'='*70}")
    for s in summary:
        mark = "OK " if s["status"] == "ok" else "ERR"
        print(f"  {mark}  {s['tag']:10s} {s['wall_s']:7.1f}s  s4={s['run_s4']}"
              f"  {s['error']}")


if __name__ == "__main__":
    main()
