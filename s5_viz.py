"""
s5_viz.py — PyVista visualization helpers for the S5_Slicer pipeline.

One function per pipeline stage.  Each one takes whatever data it needs
and writes a labeled screenshot to VizConfig.output_dir.  Flip
VizConfig.enabled = False to make every call a no-op (zero overhead).

Typical integration:

    from s5_viz import viz, VizConfig
    VizConfig.output_dir = './plots'
    VizConfig.interactive = False       # True for pop-up windows

    viz.input_mesh(input_tet)
    viz.overhang_analysis(input_tet)
    viz.distance_field(input_tet, cell_distance_to_bottom)
    ...
"""

import os
import numpy as np
import pyvista as pv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class VizConfig:
    """Global configuration for visualization.  Mutate class attrs directly."""

    enabled = True
    output_dir = "./plots"
    save_screenshots = True
    interactive = False  # True = pop-up, False = off-screen
    window_size = (1400, 1000)
    edge_width = 0.3
    default_cpos = [-1, -2, 0]  # PyVista camera preset


# Color maps used consistently across the pipeline
CMAP_SEQ = "viridis"  # monotonic scalars (distances, volumes)
CMAP_DIV = "RdBu_r"  # signed scalars (rotations, gradients)
CMAP_ANG = "plasma"  # angles / magnitudes


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _plotter(title=None, shape=(1, 1)):
    p = pv.Plotter(
        off_screen=not VizConfig.interactive,
        window_size=VizConfig.window_size,
        shape=shape,
    )
    if title and shape == (1, 1):
        p.add_text(title, font_size=12, position="upper_edge")
    return p


def _finish(p, filename):
    if VizConfig.save_screenshots and filename:
        os.makedirs(VizConfig.output_dir, exist_ok=True)
        path = os.path.join(VizConfig.output_dir, filename)
        p.screenshot(path)
        print(f"  [viz] saved {path}")
    if VizConfig.interactive:
        p.show()
    else:
        p.close()


def _guard(fn):
    """Decorator: skip all viz work when disabled."""

    def wrapped(*args, **kwargs):
        if not VizConfig.enabled:
            return
        return fn(*args, **kwargs)

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class _Viz:
    """Namespace that holds all pipeline plot functions."""

    # ---------- Stage 1: Input mesh ---------------------------------------
    @staticmethod
    @_guard
    def input_mesh(tet, filename="01_input_mesh.png"):
        """Raw tetrahedral mesh after TetGen + scaling + origin shift."""
        p = _plotter("Input tetrahedral mesh")
        p.add_mesh(
            tet,
            show_edges=True,
            color="lightsteelblue",
            opacity=0.85,
            line_width=VizConfig.edge_width,
        )
        p.show_axes()
        p.camera_position = VizConfig.default_cpos
        _finish(p, filename)

    # ---------- Stage 2: Overhang analysis + bottom cells ------------------
    @staticmethod
    @_guard
    def overhang_analysis(tet, filename="02_overhang_analysis.png"):
        """Cells colored by overhang angle (deg).  Bottom cells in red.

        NaN overhang = interior cell (no surface face) → rendered translucent.
        """
        p = _plotter()
        t = tet.copy()
        overhang_deg = np.rad2deg(t.cell_data["overhang_angle"])
        t.cell_data["overhang_deg"] = overhang_deg

        p.add_mesh(
            t,
            scalars="overhang_deg",
            cmap=CMAP_ANG,
            nan_color="lightgray",
            nan_opacity=0.15,
            show_edges=False,
            opacity=0.9,
            scalar_bar_args={"title": "Overhang (deg)", "n_labels": 5, "fmt": "%.0f"},
        )

        is_bottom = t.cell_data["is_bottom"]
        if is_bottom.any():
            bottom = t.extract_cells(np.where(is_bottom)[0])
            p.add_mesh(bottom, color="crimson", opacity=0.9)

        p.show_axes()
        p.camera_position = VizConfig.default_cpos
        _finish(p, filename)

    # ---------- Stage 3: Dijkstra distance field --------------------------
    @staticmethod
    @_guard
    def distance_field(tet, cell_distance_to_bottom, filename="03_distance_field.png"):
        """Graph-distance from each cell to the nearest bottom cell.

        Infinite / NaN entries (unreachable cells) are rendered translucent.
        """
        p = _plotter()
        t = tet.copy()
        d = np.array(cell_distance_to_bottom, dtype=float).copy()
        d[np.isinf(d)] = np.nan
        t.cell_data["distance_to_base"] = d

        p.add_mesh(
            t,
            scalars="distance_to_base",
            cmap=CMAP_SEQ,
            nan_color="lightgray",
            nan_opacity=0.15,
            show_edges=False,
            scalar_bar_args={"title": "d  (graph units)", "n_labels": 5},
        )
        p.show_axes()
        p.camera_position = VizConfig.default_cpos
        _finish(p, filename)

    # ---------- Stage 4: Path-length gradient (Section 4.3) ---------------
    @staticmethod
    @_guard
    def path_length_gradient(tet, gradient, filename="04_path_length_gradient.png"):
        """Per-cell radial gradient of the distance-to-base field.

        Signed scalar, diverging colormap centered at 0.
        """
        p = _plotter()
        t = tet.copy()
        g = np.array(gradient, dtype=float).copy()
        t.cell_data["path_gradient"] = g

        clim_mag = np.nanmax(np.abs(g[np.isfinite(g)])) if np.isfinite(g).any() else 1.0
        p.add_mesh(
            t,
            scalars="path_gradient",
            cmap=CMAP_DIV,
            clim=[-clim_mag, clim_mag],
            nan_color="lightgray",
            nan_opacity=0.15,
            scalar_bar_args={"title": "∇_r d", "n_labels": 5},
        )
        p.show_axes()
        p.camera_position = VizConfig.default_cpos
        _finish(p, filename)

    # ---------- Stage 5a: INITIAL rotation field --------------------------
    @staticmethod
    @_guard
    def initial_rotation_field(
        tet, initial_rotation, filename="05a_initial_rotation_field.png"
    ):
        """Pre-smoothing rotation field from overhang + gradient product."""
        p = _plotter()
        t = tet.copy()
        t.cell_data["r_init"] = np.rad2deg(np.array(initial_rotation))

        clim = np.nanmax(
            np.abs(t.cell_data["r_init"][np.isfinite(t.cell_data["r_init"])])
        )
        clim = float(clim) if np.isfinite(clim) else 30.0

        p.add_mesh(
            t,
            scalars="r_init",
            cmap=CMAP_DIV,
            clim=[-clim, clim],
            nan_color="lightgray",
            nan_opacity=0.15,
            scalar_bar_args={"title": "r_init  (deg)", "n_labels": 5},
        )
        p.show_axes()
        p.camera_position = VizConfig.default_cpos
        _finish(p, filename)

    # ---------- Stage 5b: SMOOTHED rotation field (Section 4.4) -----------
    @staticmethod
    @_guard
    def smoothed_rotation_field(
        tet, rotation_field, filename="05b_smoothed_rotation_field.png"
    ):
        """Post-smoothing rotation field (UMFPACK direct solve output)."""
        p = _plotter()
        t = tet.copy()
        t.cell_data["r"] = np.rad2deg(np.array(rotation_field))

        clim = float(np.nanmax(np.abs(t.cell_data["r"]))) or 30.0
        p.add_mesh(
            t,
            scalars="r",
            cmap=CMAP_DIV,
            clim=[-clim, clim],
            scalar_bar_args={"title": "r  (deg)", "n_labels": 5},
        )
        p.show_axes()
        p.camera_position = VizConfig.default_cpos
        _finish(p, filename)

    # ---------- Stage 5c: initial vs smoothed side-by-side ----------------
    @staticmethod
    @_guard
    def rotation_field_comparison(
        tet,
        initial_rotation,
        rotation_field,
        filename="05c_rotation_smoothing_pair.png",
    ):
        """Side-by-side: initial rotation vs smoothed rotation.

        Visually encodes the effect of the Laplacian QP direct solve.
        """
        p = _plotter(shape=(1, 2))

        r0 = np.rad2deg(np.array(initial_rotation))
        r1 = np.rad2deg(np.array(rotation_field))
        fin = np.isfinite(np.concatenate([r0, r1]))
        clim_mag = float(np.nanmax(np.abs(np.concatenate([r0, r1])[fin]))) or 30.0

        # Left: initial
        p.subplot(0, 0)
        p.add_text("Initial (pre-smoothing)", font_size=10)
        t0 = tet.copy()
        t0.cell_data["r"] = r0
        p.add_mesh(
            t0,
            scalars="r",
            cmap=CMAP_DIV,
            clim=[-clim_mag, clim_mag],
            nan_color="lightgray",
            nan_opacity=0.2,
            scalar_bar_args={"title": "r (deg)"},
        )
        p.camera_position = VizConfig.default_cpos

        # Right: smoothed
        p.subplot(0, 1)
        p.add_text("Smoothed (UMFPACK solve)", font_size=10)
        t1 = tet.copy()
        t1.cell_data["r"] = r1
        p.add_mesh(
            t1,
            scalars="r",
            cmap=CMAP_DIV,
            clim=[-clim_mag, clim_mag],
            scalar_bar_args={"title": "r (deg)"},
        )
        p.camera_position = VizConfig.default_cpos

        p.link_views()
        _finish(p, filename)

    # ---------- Stage 6: Deformation (Section 4.5) ------------------------
    @staticmethod
    @_guard
    def deformation_pair(
        undeformed_tet, deformed_tet, filename="06_deformation_pair.png"
    ):
        """Side-by-side: undeformed vs deformed mesh (parallel LSQR output)."""
        p = _plotter(shape=(1, 2))

        p.subplot(0, 0)
        p.add_text("Undeformed", font_size=10)
        p.add_mesh(
            undeformed_tet,
            show_edges=True,
            color="steelblue",
            opacity=0.85,
            line_width=VizConfig.edge_width,
        )
        p.show_axes()
        p.camera_position = VizConfig.default_cpos

        p.subplot(0, 1)
        p.add_text("Deformed (parallel LSQR, Section 4.5)", font_size=10)
        p.add_mesh(
            deformed_tet,
            show_edges=True,
            color="indianred",
            opacity=0.85,
            line_width=VizConfig.edge_width,
        )
        p.show_axes()
        p.camera_position = VizConfig.default_cpos

        p.link_views()
        _finish(p, filename)

    @staticmethod
    @_guard
    def deformation_displacement(
        undeformed_tet, deformed_tet, filename="06b_deformation_displacement.png"
    ):
        """Deformed mesh colored by per-vertex displacement magnitude."""
        p = _plotter()
        disp = np.linalg.norm(deformed_tet.points - undeformed_tet.points, axis=1)
        t = deformed_tet.copy()
        t.point_data["displacement"] = disp
        p.add_mesh(
            t,
            scalars="displacement",
            cmap=CMAP_SEQ,
            show_edges=False,
            scalar_bar_args={"title": "‖Δv‖  (mm)", "n_labels": 5},
        )
        p.show_axes()
        p.camera_position = VizConfig.default_cpos
        _finish(p, filename)

    # ---------- Stage 7: Volume scaling / z-squish (Section 4.8) ----------
    @staticmethod
    @_guard
    def volume_scaling(deformed_tet, z_squish_scales, filename="07_volume_scaling.png"):
        """Extrusion-compensation ratio ζ = vol_orig / vol_def per cell."""
        p = _plotter()
        t = deformed_tet.copy()
        z = np.array(z_squish_scales, dtype=float).copy()
        # Clip extremes so the colorbar isn't dominated by degenerate cells
        p1, p99 = np.nanpercentile(z[np.isfinite(z)], [1, 99])
        t.cell_data["z_squish"] = np.clip(z, p1, p99)

        p.add_mesh(
            t,
            scalars="z_squish",
            cmap=CMAP_DIV,
            clim=[p1, p99],
            scalar_bar_args={"title": "ζ_c", "n_labels": 5},
        )
        p.show_axes()
        p.camera_position = VizConfig.default_cpos
        _finish(p, filename)

    # ---------- Stage 8: Final G-code toolpaths ---------------------------
    @staticmethod
    @_guard
    def gcode_toolpaths(
        new_gcode_points,
        background_mesh=None,
        filename="08_gcode_toolpaths.png",
        extrusion_only=True,
    ):
        """Toolpaths as line segments colored by B-axis rotation (deg).

        Parameters
        ----------
        new_gcode_points : list of dicts
            Must contain 'position' (3,), 'rotation' (rad), 'command' (str).
        background_mesh : pv.UnstructuredGrid or None
            Optional mesh to show faintly behind the paths for context.
        extrusion_only : bool
            If True, show only G01 (printing) moves and drop G00 travel moves.
        """
        positions = np.array([g["position"] for g in new_gcode_points], dtype=float)
        rotations = np.array(
            [g.get("rotation", 0.0) or 0.0 for g in new_gcode_points], dtype=float
        )
        commands = np.array([g.get("command", "G01") for g in new_gcode_points])
        extrusions = np.array(
            [g.get("extrusion", 0.0) or 0.0 for g in new_gcode_points], dtype=float
        )

        n = len(positions)
        if n < 2:
            print("  [viz] fewer than 2 valid points — skipping")
            return

        # Build explicit 2-point segments between CONSECUTIVE toolpath points.
        # Segment k links point k → k+1; the extrusion recorded on a move
        # describes material laid down arriving at the destination, so segment k
        # is a printing move iff extrusions[k+1] > 0.  Masking per-segment (vs.
        # filtering points then drawing one polyline) is essential: a single
        # polyline over a filtered point set would bridge every gap left by a
        # removed travel move, drawing spurious straight lines across the part.
        finite = np.isfinite(positions).all(axis=1)
        both_finite = finite[:-1] & finite[1:]
        if extrusion_only:
            keep = both_finite & (extrusions[1:] > 0.0)
        else:
            keep = both_finite
        seg = np.flatnonzero(keep)  # indices k of kept segments (k → k+1)
        if len(seg) == 0:
            print("  [viz] no extrusion segments to draw — skipping")
            return

        rotations = np.rad2deg(rotations)
        lines = np.column_stack([np.full(len(seg), 2), seg, seg + 1]).ravel()
        poly = pv.PolyData(positions)
        poly.lines = lines

        # Per-point rotation scalars; only points touched by a kept segment are
        # tubed, so the colouring follows the printed path.
        poly.point_data["rotation_deg"] = rotations

        p = _plotter()
        if background_mesh is not None:
            p.add_mesh(
                background_mesh, color="whitesmoke", opacity=0.15, show_edges=False
            )

        tube = poly.tube(radius=0.15)
        clim = float(np.max(np.abs(rotations))) or 30.0
        p.add_mesh(
            tube,
            scalars="rotation_deg",
            cmap=CMAP_DIV,
            clim=[-clim, clim],
            scalar_bar_args={"title": "B (deg)", "n_labels": 5},
        )
        p.show_axes()
        p.camera_position = VizConfig.default_cpos
        _finish(p, filename)

    # ---------- Stage 8b: Animated layer-by-layer build -------------------
    @staticmethod
    @_guard
    def gcode_layer_build(
        new_gcode_points,
        background_mesh=None,
        filename="10_layer_build.gif",
        max_frames=100000,
        fps=15,
    ):
        """Animated GIF of the print building up one layer at a time.

        Layers are keyed off the *sliced* (deformed) Z — ``original_position[2]``
        — which is planar (Cura's layer heights), so binning is clean even
        though the displayed ``position`` is the nonplanar real-space toolpath.
        Only extrusion segments are drawn (travel moves omitted), coloured by
        B-axis rotation.  Layers are grouped into at most ``max_frames`` frames.
        """
        pos = np.array([g["position"] for g in new_gcode_points], dtype=float)
        # sliced-space Z gives the true planar layer; fall back to real Z
        layer_z = np.array(
            [
                (
                    g["original_position"]
                    if g.get("original_position") is not None
                    else g["position"]
                )[2]
                for g in new_gcode_points
            ],
            dtype=float,
        )
        rot = np.rad2deg(
            np.array(
                [g.get("rotation", 0.0) or 0.0 for g in new_gcode_points], dtype=float
            )
        )
        extr = np.array(
            [g.get("extrusion", 0.0) or 0.0 for g in new_gcode_points], dtype=float
        )

        # Segment k links point k → k+1; it prints iff the move into k+1 extrudes
        # (same masking as gcode_toolpaths — never bridge across removed travels).
        finite = np.isfinite(pos).all(axis=1)
        keep = finite[:-1] & finite[1:] & (extr[1:] > 0.0)
        seg = np.flatnonzero(keep)
        if len(seg) < 2:
            print("  [viz] no extrusion segments — skipping layer animation")
            return

        # Reveal in true print order.  new_gcode_points is already sequential
        # (built in gcode order), so `seg` is ascending = deposition order.
        # Equal segments per frame → smooth, monotonic growth.  (Binning by
        # sliced-Z instead piles most segments into a few frames when the
        # deformation warps layer thickness, so the build appears to stall.)
        n_frames = int(min(max_frames, len(seg)))
        frame_of_seg = (np.arange(len(seg)) * n_frames) // len(seg)

        # Layer label = how many distinct sliced-Z layers have been laid so far.
        sz = np.round(layer_z[seg + 1], 3)
        layers = np.unique(sz)
        n_layers = len(layers)
        layer_of_seg = np.searchsorted(layers, sz)

        clim = float(np.nanmax(np.abs(rot))) or 30.0
        os.makedirs(VizConfig.output_dir, exist_ok=True)
        path = os.path.join(VizConfig.output_dir, filename)

        p = _plotter()
        if background_mesh is not None:
            p.add_mesh(
                background_mesh, color="whitesmoke", opacity=0.1, show_edges=False
            )
        p.show_axes()
        p.camera_position = VizConfig.default_cpos
        p.open_gif(path, fps=fps)

        for f in range(n_frames):
            mask = frame_of_seg <= f
            sel = seg[mask]
            if len(sel) == 0:
                continue
            lines = np.column_stack([np.full(len(sel), 2), sel, sel + 1]).ravel()
            # NB: pass lines= to the constructor — pv.PolyData(pos) alone adds a
            # vertex cell per point, which would render the *entire* point cloud
            # every frame (the whole part appears built from frame 0).
            poly = pv.PolyData(pos, lines=lines)
            poly.point_data["rotation_deg"] = rot
            # name= replaces the previous actors in place (no stacking).
            p.add_mesh(
                poly,
                name="paths",
                scalars="rotation_deg",
                cmap=CMAP_DIV,
                clim=[-clim, clim],
                line_width=2,
                reset_camera=(f == 0),
                scalar_bar_args={"title": "B (deg)", "n_labels": 5},
            )
            cur_layer = int(layer_of_seg[mask].max()) + 1
            p.add_text(
                f"{cur_layer}/{n_layers} layers",
                name="progress",
                font_size=11,
                position="upper_left",
            )
            p.write_frame()

        p.close()
        print(f"  [viz] saved {path} ({n_frames} frames over {n_layers} layers)")

    # ---------- Bonus: vertex rotations onto deformed mesh ----------------
    @staticmethod
    @_guard
    def vertex_rotations(
        deformed_tet, vertex_rotations, filename="09_vertex_rotations.png"
    ):
        """Scalar plot of per-vertex rotations (Section 4.7 output)."""
        p = _plotter("Per-vertex rotations  (Section 4.7)")
        t = deformed_tet.copy()
        vr = np.rad2deg(np.array(vertex_rotations))
        t.point_data["vertex_rot"] = vr

        clim = float(np.max(np.abs(vr))) or 30.0
        p.add_mesh(
            t,
            scalars="vertex_rot",
            cmap=CMAP_DIV,
            clim=[-clim, clim],
            scalar_bar_args={"title": "φ_v (deg)", "n_labels": 5},
        )
        p.show_axes()
        p.camera_position = VizConfig.default_cpos
        _finish(p, filename)

    @staticmethod
    @_guard
    def radial_projection_field(tet, output_path="./plots/04c_radial_projection.png"):
        """Visualize raw = r̂ · ∇d on the surface. Red = positive (outward), blue = negative."""
        import pyvista as pv

        grad = tet.cell_data["surface_gradient"]
        cc = tet.cell_data["cell_center"]
        r_xy = cc[:, :2]
        r_n = np.linalg.norm(r_xy, axis=1, keepdims=True) + 1e-8
        r_hat = r_xy / r_n
        raw = np.sum(r_hat * grad[:, :2], axis=1)

        print(
            f"raw projection: min={np.nanmin(raw):.3f}, "
            f"max={np.nanmax(raw):.3f}, "
            f"abs-mean={np.nanmean(np.abs(raw)):.3f}"
        )
        print(f"  fraction positive: {(raw > 0).sum() / np.isfinite(raw).sum():.2%}")
        print(f"  fraction negative: {(raw < 0).sum() / np.isfinite(raw).sum():.2%}")

        surf = tet.extract_surface()
        # Map cell-data raw onto surface faces
        orig_ids = surf.cell_data["vtkOriginalCellIds"]
        surf.cell_data["raw_projection"] = raw[orig_ids]

        plotter = pv.Plotter(off_screen=True, window_size=(1400, 1000))
        # Symmetric range around 0 so red/blue map cleanly to sign
        absmax = max(abs(np.nanmin(raw)), abs(np.nanmax(raw)))
        plotter.add_mesh(
            surf,
            scalars="raw_projection",
            cmap="RdBu_r",
            clim=[-absmax, absmax],
            scalar_bar_args={"title": "r̂·∇d"},
        )
        plotter.add_title("Radial projection of ∇d  (red = outward, blue = inward)")
        plotter.add_axes()
        plotter.screenshot(output_path)
        plotter.close()
        print(f"  [viz] saved {output_path}")

    @staticmethod
    @_guard
    def surface_gradient_field(tet, output_path="./plots/04b_surface_gradient.png"):
        """Visualize the heat-method surface gradient as arrows on surface cells.
        Arrow color = magnitude, arrow direction = ∇d (geodesic-distance gradient).
        """
        import pyvista as pv

        grad = tet.cell_data["surface_gradient"]  # (n_cells, 3)
        cc = tet.cell_data["cell_center"]  # (n_cells, 3)

        # Surface cells only (interior cells have NaN gradients)
        valid = ~np.isnan(grad).any(axis=1)
        origins = cc[valid]
        vectors = grad[valid]
        magnitudes = np.linalg.norm(vectors, axis=1)

        # Normalize arrow lengths for display but color by true magnitude
        print(
            f"Gradient magnitude: "
            f"min={magnitudes.min():.3f}, "
            f"median={np.median(magnitudes):.3f}, "
            f"max={magnitudes.max():.3f}"
        )

        xmin, xmax, ymin, ymax, zmin, zmax = tet.bounds
        diag = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
        cell_scale = diag / 50.0
        vectors_display = vectors / (magnitudes[:, None] + 1e-12) * cell_scale

        # Subsample if too many arrows clutter the view
        n = len(origins)
        if n > 3000:
            stride = n // 3000
            origins = origins[::stride]
            vectors_display = vectors_display[::stride]
            magnitudes = magnitudes[::stride]

        arrows = pv.PolyData(origins)
        arrows["vectors"] = vectors_display
        arrows["magnitude"] = magnitudes
        glyphs = arrows.glyph(orient="vectors", scale=False, factor=1.0)

        plotter = pv.Plotter(off_screen=True, window_size=(1400, 1000))
        plotter.add_mesh(
            tet.extract_surface(), color="lightgray", opacity=0.3, show_edges=False
        )
        plotter.add_mesh(
            glyphs,
            scalars="magnitude",
            cmap="viridis",
            scalar_bar_args={"title": "‖∇d‖"},
        )
        plotter.add_title("Surface gradient ∇d (heat-method)")
        plotter.add_axes()
        plotter.screenshot(output_path)
        plotter.close()
        print(f"  [viz] saved {output_path}")

    @staticmethod
    @_guard
    def heat_field(tet, cell_distance_to_base, cell_gradient):
        import pyvista as pv

        surface = tet.extract_surface()
        orig_ids = surface.cell_data["vtkOriginalCellIds"]

        # --- 1. Heat distance field on surface ---
        surf_dist = pv.PolyData(surface.points, surface.faces)
        surf_dist.cell_data["distance_to_base"] = cell_distance_to_base[orig_ids]

        p = _plotter()

        p.add_mesh(surf_dist, scalars="distance_to_base", cmap="plasma")
        p.screenshot("./plots/heatmap.png")


# Singleton convenience namespace
viz = _Viz()

