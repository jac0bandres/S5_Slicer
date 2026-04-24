"""s5_bench — per-stage benchmarking for S5_Slicer.

Mirrors the s5_viz pattern: a module-level singleton that you drop into your
pipeline at hot-path sites. When BenchConfig.enabled is False, every call is a
no-op. When True, timings + correctness go to CSV at bench.finish().

Basic usage (drop into S5.py top, just like viz):

    from s5_bench import bench, BenchConfig
    BenchConfig.enabled       = True
    BenchConfig.output_dir    = './bench_results'
    BenchConfig.run_tag       = 'pi_3mm_subdiv_0'
    BenchConfig.run_s4_comparison = True

    # ... then at hot paths in your code:

    with bench.stage('smoothing', context={'n_cells': n_c}):
        rotation_field = spsolve(A_rot, b_rot)

    bench.reference('smoothing',
        s4=lambda: s4_ref.smooth_rotation_field(n_c, cell_face_nb,
                                                initial_rotation_field, W),
        s5_result=rotation_field,
        correctness=bench.compare_scalar_fields,
    )

    # at the end:
    bench.finish()

Output files (under BenchConfig.output_dir / BenchConfig.run_tag /):
    timings.csv      — one row per (stage, impl) measurement
    correctness.csv  — one row per S4-vs-S5 comparison
    env.json         — machine + package snapshot
    meta.json        — run tag, timestamp, stages seen
"""
from __future__ import annotations

import csv
import gc
import json
import platform
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration (class-level attrs so users can override exactly like VizConfig)
# ---------------------------------------------------------------------------

class BenchConfig:
    enabled: bool = False
    """Master switch. When False, all bench calls are no-ops with zero overhead."""

    run_s4_comparison: bool = True
    """If False, bench.reference() skips the s4 callable entirely — useful on
    large meshes where S4 TRF takes minutes."""

    record_correctness: bool = True
    """If False, bench.reference() times both impls but skips correctness fn."""

    output_dir: str = "./bench_results"
    run_tag: Optional[str] = None   # None → timestamp

    verbose: bool = True
    """Print measurements to stderr as they're taken."""

    track_memory: bool = True
    """Use tracemalloc for peak Python memory. Adds ~5% overhead."""


# ---------------------------------------------------------------------------
# The singleton
# ---------------------------------------------------------------------------

@dataclass
class _TimingRow:
    stage: str
    impl: str
    wall_s: float
    peak_mb: float
    iterations: Optional[int] = None
    status: str = "ok"
    error_msg: str = ""
    context: str = ""


class _Bench:
    def __init__(self):
        self._timings: list[_TimingRow] = []
        self._correctness: list[dict] = []
        self._context_stack: list[dict] = []
        self._stages_seen: set = set()

    # ---- context helpers ------------------------------------------------

    def set_global_context(self, **kwargs):
        """Attach key/value pairs to every subsequent row.

        Typical use: bench.set_global_context(n_cells=tet.number_of_cells,
        model='pi_3mm', subdivide=0).
        """
        if not BenchConfig.enabled:
            return
        if not self._context_stack:
            self._context_stack.append({})
        self._context_stack[-1].update(kwargs)

    def _current_context(self, extra: Optional[dict] = None) -> str:
        """Flatten context dicts into a short string for the CSV column."""
        ctx = {}
        for d in self._context_stack:
            ctx.update(d)
        if extra:
            ctx.update(extra)
        if not ctx:
            return ""
        return ";".join(f"{k}={v}" for k, v in sorted(ctx.items()))

    # ---- core timing primitives ----------------------------------------

    @contextmanager
    def stage(self, name: str, impl: str = "s5", context: Optional[dict] = None):
        """Time a block of code.

        Nest calls freely — they don't interfere. The impl arg defaults to 's5'
        so most S5.py sites are one-word labels. Use impl='s5_subpart' or similar
        to distinguish sub-blocks within a stage if you want a breakdown.
        """
        if not BenchConfig.enabled:
            yield
            return

        self._stages_seen.add(name)
        gc.collect()
        if BenchConfig.track_memory:
            tracemalloc.start()
        t0 = time.perf_counter()
        try:
            yield
            status, err = "ok", ""
        except Exception as exc:
            status, err = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            t1 = time.perf_counter()
            if BenchConfig.track_memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_mb = peak / (1024 * 1024)
            else:
                peak_mb = float("nan")

            row = _TimingRow(
                stage=name, impl=impl,
                wall_s=t1 - t0, peak_mb=peak_mb,
                status=status, error_msg=err,
                context=self._current_context(context),
            )
            self._timings.append(row)
            if BenchConfig.verbose:
                print(f"[bench] {name:20s} {impl:6s} {row.wall_s*1000:8.2f} ms"
                      f"  peak={row.peak_mb:6.1f} MB  {row.context}",
                      file=sys.stderr, flush=True)

    def reference(
        self,
        name: str,
        s4: Callable[[], Any],
        s5_result: Any = None,
        correctness: Optional[Callable[[Any, Any], dict]] = None,
        context: Optional[dict] = None,
    ):
        """Run the s4 reference impl, time it, and compare its output to s5_result.

        When BenchConfig.enabled is False: no-op.
        When BenchConfig.run_s4_comparison is False: no-op (saves wall time on
            large meshes where S4 takes minutes).
        When correctness is None: times s4 but skips comparison.

        The s4 callable receives no arguments — capture the inputs you need in a
        closure. Example:
            bench.reference('smoothing',
                s4=lambda: s4_ref.smooth_rotation_field(n_c, cell_face_nb, init, W),
                s5_result=rotation_field,
                correctness=bench.compare_scalar_fields)
        """
        if not BenchConfig.enabled or not BenchConfig.run_s4_comparison:
            return None

        gc.collect()
        if BenchConfig.track_memory:
            tracemalloc.start()
        t0 = time.perf_counter()
        status = "ok"
        err = ""
        s4_result = None
        try:
            s4_result = s4()
        except Exception as exc:
            status = "error"
            err = f"{type(exc).__name__}: {exc}"
        t1 = time.perf_counter()
        if BenchConfig.track_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / (1024 * 1024)
        else:
            peak_mb = float("nan")

        row = _TimingRow(
            stage=name, impl="s4",
            wall_s=t1 - t0, peak_mb=peak_mb,
            status=status, error_msg=err,
            context=self._current_context(context),
        )
        self._timings.append(row)
        if BenchConfig.verbose:
            print(f"[bench] {name:20s} {'s4':6s} {row.wall_s*1000:8.2f} ms"
                  f"  peak={row.peak_mb:6.1f} MB  {row.context}  {row.status}",
                  file=sys.stderr, flush=True)

        # Correctness check
        if (BenchConfig.record_correctness and correctness is not None
                and s4_result is not None and s5_result is not None):
            try:
                metrics = correctness(s4_result, s5_result)
            except Exception as exc:
                metrics = {"metric": "error", "error": f"{type(exc).__name__}: {exc}"}
            metrics.update({
                "stage": name,
                "context": self._current_context(context),
            })
            self._correctness.append(metrics)
            if BenchConfig.verbose:
                # Pick out the most informative numeric field for the log line
                summary_bits = []
                for k in ("max_abs", "cosine_mean", "pearson_r", "exact_match"):
                    if k in metrics and metrics[k] not in (None, ""):
                        v = metrics[k]
                        summary_bits.append(f"{k}={v:.3g}" if isinstance(v, float)
                                            else f"{k}={v}")
                print(f"[bench] {name:20s} correctness  {' '.join(summary_bits)}",
                      file=sys.stderr, flush=True)

        return s4_result

    # ---- correctness helpers (methods so users type bench.compare_*) ---

    @staticmethod
    def compare_scalar_fields(a: np.ndarray, b: np.ndarray) -> dict:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        both_finite = ~nan_a & ~nan_b
        if both_finite.sum() == 0:
            return {"metric": "scalar_compare", "n_compared": 0}
        d = a[both_finite] - b[both_finite]
        r = (np.corrcoef(a[both_finite], b[both_finite])[0, 1]
             if a[both_finite].std() > 0 and b[both_finite].std() > 0 else float("nan"))
        return {
            "metric": "scalar_compare",
            "max_abs": float(np.max(np.abs(d))),
            "rms": float(np.sqrt(np.mean(d ** 2))),
            "pearson_r": float(r),
            "nan_disagree": int(np.sum(nan_a != nan_b)),
            "n_compared": int(both_finite.sum()),
        }

    @staticmethod
    def compare_vector_fields(a: np.ndarray, b: np.ndarray) -> dict:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        assert a.shape == b.shape
        ok = ~(np.any(np.isnan(a), axis=1) | np.any(np.isnan(b), axis=1))
        if ok.sum() == 0:
            return {"metric": "vector_compare", "n_compared": 0}
        diff = a[ok] - b[ok]
        max_abs = float(np.max(np.linalg.norm(diff, axis=1)))
        rms = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        na = np.linalg.norm(a[ok], axis=1)
        nb = np.linalg.norm(b[ok], axis=1)
        nz = (na > 1e-12) & (nb > 1e-12)
        cos = (float(np.mean(np.sum(a[ok][nz] * b[ok][nz], axis=1) / (na[nz] * nb[nz])))
               if nz.any() else float("nan"))
        return {
            "metric": "vector_compare",
            "max_abs": max_abs, "rms": rms, "cosine_mean": cos,
            "n_compared": int(ok.sum()),
        }

    @staticmethod
    def compare_edge_sets(a: np.ndarray, b: np.ndarray) -> dict:
        def canon(arr):
            arr = np.asarray(arr)
            if arr.size == 0:
                return set()
            lo = np.minimum(arr[:, 0], arr[:, 1])
            hi = np.maximum(arr[:, 0], arr[:, 1])
            return set(zip(lo.tolist(), hi.tolist()))
        sa, sb = canon(a), canon(b)
        return {
            "metric": "edge_set_compare",
            "n_s4": len(sa), "n_s5": len(sb),
            "only_in_s4": len(sa - sb), "only_in_s5": len(sb - sa),
            "exact_match": (sa == sb),
        }

    @staticmethod
    def compare_bool_arrays(a: np.ndarray, b: np.ndarray) -> dict:
        a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
        return {
            "metric": "bool_array_compare",
            "n_compared": int(a.size),
            "exact_match": bool(np.array_equal(a, b)),
            "n_disagree": int(np.sum(a != b)),
        }

    # ---- output --------------------------------------------------------

    def finish(self):
        """Write timings.csv, correctness.csv, env.json, meta.json to disk.

        Safe to call multiple times; it dumps the accumulated rows each time.
        """
        if not BenchConfig.enabled:
            return
        run_tag = BenchConfig.run_tag or time.strftime("%Y%m%d_%H%M%S")
        out = Path(BenchConfig.output_dir) / run_tag
        out.mkdir(parents=True, exist_ok=True)

        # timings.csv
        with (out / "timings.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "stage", "impl", "wall_s", "peak_mb", "iterations",
                "status", "error_msg", "context",
            ])
            w.writeheader()
            for row in self._timings:
                w.writerow({
                    "stage": row.stage, "impl": row.impl,
                    "wall_s": f"{row.wall_s:.6f}",
                    "peak_mb": f"{row.peak_mb:.3f}",
                    "iterations": row.iterations if row.iterations is not None else "",
                    "status": row.status, "error_msg": row.error_msg,
                    "context": row.context,
                })

        # correctness.csv (schema is sparse because different metrics emit different keys)
        if self._correctness:
            all_keys = set()
            for row in self._correctness:
                all_keys.update(row.keys())
            preferred_order = [
                "stage", "metric", "max_abs", "rms", "cosine_mean", "pearson_r",
                "exact_match", "n_compared", "only_in_s4", "only_in_s5",
                "nan_disagree", "n_disagree", "n_s4", "n_s5", "error", "context",
            ]
            ordered = [k for k in preferred_order if k in all_keys]
            ordered += sorted(k for k in all_keys if k not in preferred_order)

            with (out / "correctness.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
                w.writeheader()
                for row in self._correctness:
                    w.writerow(row)

        # env.json
        env: dict = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
        }
        try:
            import psutil
            env["cpu_count_physical"] = psutil.cpu_count(logical=False)
            env["cpu_count_logical"] = psutil.cpu_count(logical=True)
            env["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        except ImportError:
            pass
        pkgs = {}
        for name in ["numpy", "scipy", "pyvista", "tetgen", "open3d",
                     "networkx", "potpourri3d", "igl"]:
            try:
                mod = __import__(name)
                pkgs[name] = getattr(mod, "__version__", "unknown")
            except ImportError:
                pkgs[name] = "missing"
        env["packages"] = pkgs
        (out / "env.json").write_text(json.dumps(env, indent=2))

        # meta.json
        meta = {
            "run_tag": run_tag,
            "stages_seen": sorted(self._stages_seen),
            "timing_rows": len(self._timings),
            "correctness_rows": len(self._correctness),
            "config": {
                k: getattr(BenchConfig, k)
                for k in dir(BenchConfig)
                if not k.startswith("_") and not callable(getattr(BenchConfig, k))
            },
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

        if BenchConfig.verbose:
            print(f"[bench] wrote {len(self._timings)} timing rows, "
                  f"{len(self._correctness)} correctness rows to {out}",
                  file=sys.stderr)

    def reset(self):
        """Clear accumulated rows without writing. Useful if you want to run
        multiple independent benchmarks in one process."""
        self._timings.clear()
        self._correctness.clear()
        self._context_stack.clear()
        self._stages_seen.clear()


bench = _Bench()