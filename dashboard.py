"""
S5 Dashboard — Streamlit UI for slicer configuration, STL upload, and GCode visualization.

Run with:
    streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import configparser
import json
import os
import re
import io
import time
import tempfile
from pathlib import Path

# ──────────────────────────────────────────────────────────
# Theme — edit these to restyle the entire dashboard.
# CSS variables below mirror these so both Python (Plotly)
# and CSS (Streamlit widgets) stay in sync.
# ──────────────────────────────────────────────────────────
T = {
    # Core palette
    "accent":         "#0f766e",   # primary buttons, active tabs, highlights
    "accent_hover":   "#0d6a62",   # button hover
    "accent_light":   "#f0fdf9",   # banner backgrounds, light tints
    "accent_border":  "#5eead4",   # banner borders

    # Backgrounds — all white or near-white, no dark surfaces
    "bg":             "#f5f7fa",   # app background
    "surface":        "#ffffff",   # cards, sidebar, chart backgrounds
    "plot_bg":        "#ffffff",   # 3D scene / 2D plot area background

    # Borders & grid
    "border":         "#d1d5db",   # card borders, tab underlines
    "grid":           "#e5e7eb",   # plot gridlines
    "zeroline":       "#d1d5db",   # plot zero lines

    # Text — all dark, no light grays
    "text":           "#111827",   # primary text
    "text_muted":     "#374151",   # labels, captions  ← was #6b7280 (too light)
    "text_subtle":    "#374151",   # stat labels        ← was #9ca3af (too light)

    # Semantic
    "danger":         "#dc2626",
    "warning":        "#d97706",
    "info":           "#0369a1",
    "info_light":     "#f0f9ff",
    "info_border":    "#bae6fd",

    # GCode viewer — move type colourmap
    "cmap_type":      "Turbo",     # colorscale for print-order in type mode
    "travel_color":   "#9ca3af",   # travel move line color

    # GCode viewer — layer / speed colourscales
    "cmap_layer":     "Plasma",
    "cmap_speed":     "RdYlGn",
}



st.set_page_config(
    page_title="S5 Dashboard",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
    --accent:        {T["accent"]};
    --accent-hover:  {T["accent_hover"]};
    --accent-light:  {T["accent_light"]};
    --accent-border: {T["accent_border"]};
    --bg:            {T["bg"]};
    --surface:       {T["surface"]};
    --plot-bg:       {T["plot_bg"]};
    --border:        {T["border"]};
    --text:          {T["text"]};
    --text-muted:    {T["text_muted"]};
    --text-subtle:   {T["text_subtle"]};
    --info:          {T["info"]};
    --info-light:    {T["info_light"]};
    --info-border:   {T["info_border"]};
}}

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif;
    color: var(--text);
}}

/* ── App & sidebar — white / off-white, no dark surfaces ── */
.stApp {{ background-color: var(--bg); }}

section[data-testid="stSidebar"] {{
    background: var(--surface);
    border-right: 1px solid var(--border);
}}

/* ── Typography ── */
.dash-header {{
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.2rem;
}}
.dash-sub {{
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 1.2rem;
}}
.panel-title {{
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}}

/* ── Stat cards ── */
.stat-row {{ display: flex; gap: 0.6rem; margin-bottom: 1rem; flex-wrap: wrap; }}
.stat-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    flex: 1;
    min-width: 90px;
}}
.stat-label {{
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-subtle);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.stat-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 500;
    color: var(--text);
    margin-top: 0.1rem;
}}

/* ── Sidebar expanders ── */
.streamlit-expanderHeader {{
    background: var(--surface) !important;
    color: var(--text) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border-radius: 4px !important;
    border: 1px solid var(--border) !important;
}}
.streamlit-expanderContent {{
    border: 1px solid var(--border) !important;
    border-top: none !important;
}}

/* ── Buttons — no transition ── */
.stButton > button {{
    background: var(--accent);
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.45rem 1.2rem;
}}
.stButton > button:hover {{ background: var(--accent-hover); }}

/* ── Tabs — full opacity, dark text on unselected ── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 2px solid var(--border);
    gap: 0;
    padding: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text) !important;
    padding: 0.65rem 1.3rem;
    border-radius: 0;
    border-bottom: 3px solid transparent;
    background: transparent;
    opacity: 1;
}}
.stTabs [aria-selected="true"] {{
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
    background: transparent !important;
}}

/* ── All inputs — white bg, dark text ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {{
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}}

/* ── Selectbox & dropdown — white everywhere ── */
[data-baseweb="select"] > div:first-child {{
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}}
[data-baseweb="select"] span,
[data-baseweb="select"] input {{
    color: var(--text) !important;
    background: transparent !important;
}}
[data-baseweb="popover"],
[data-baseweb="menu"] {{
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}}
[role="option"] {{
    background-color: var(--surface) !important;
    color: var(--text) !important;
}}
[role="option"]:hover,
[aria-selected="true"][role="option"] {{
    background-color: var(--accent-light) !important;
    color: var(--accent) !important;
}}

/* ── Radio, checkbox, label text — always dark ── */
.stRadio label span,
label[data-baseweb="checkbox"] span,
.stMarkdown p, .stMarkdown li,
.stCaption, .stText {{
    color: var(--text) !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {{
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 8px !important;
}}

/* ── GCode preview box ── */
.gcode-display {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.9rem 1rem;
    color: var(--text);
    max-height: 320px;
    overflow-y: auto;
    line-height: 1.7;
    white-space: pre;
}}

/* ── Banners ── */
.banner-core {{
    background: var(--accent-light);
    border: 1px solid var(--accent-border);
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    font-size: 0.8rem;
    color: var(--text);
    margin-bottom: 0.75rem;
}}
.banner-xyz {{
    background: var(--info-light);
    border: 1px solid var(--info-border);
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    font-size: 0.8rem;
    color: var(--text);
    margin-bottom: 0.75rem;
}}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────
CONFIG_PATH = "config/easys4.ini"
CORE_DEF_PATH = "config/core.def.json"

KEY_SETTINGS = {
    # (label, type, min, max, step, description)
    "layer_height":            ("Layer Height (mm)",            float, 0.01, 0.5,   0.01,  "Height of each printed layer"),
    "layer_height_0":          ("First Layer Height (mm)",      float, 0.01, 0.5,   0.01,  "Height of the very first layer"),
    "material_print_temperature": ("Print Temp (°C)",           int,   150,  300,   1,     "Nozzle print temperature"),
    "material_bed_temperature":   ("Bed Temp (°C)",             int,   0,    120,   1,     "Heated bed temperature"),
    "material_diameter":       ("Filament Diameter (mm)",       float, 1.0,  3.5,   0.05,  "Diameter of filament"),
    "wall_line_count":         ("Wall Line Count",              int,   1,    8,     1,     "Number of perimeter walls"),
    "top_layers":              ("Top Layers",                   int,   0,    10,    1,     "Solid top layer count"),
    "bottom_layers":           ("Bottom Layers",                int,   0,    10,    1,     "Solid bottom layer count"),
    "infill_sparse_density":   ("Infill Density (%)",           int,   0,    100,   1,     "Percentage fill inside the part"),
    "infill_pattern":          ("Infill Pattern",               str,   None, None,  None,  "Pattern used for infill"),
    "speed_print":             ("Print Speed (mm/s)",           float, 5,    300,   5,     "Default extrusion speed"),
    "speed_travel":            ("Travel Speed (mm/s)",          float, 20,   500,   10,    "Non-printing move speed"),
    "speed_wall":              ("Wall Speed (mm/s)",            float, 5,    200,   5,     "Wall print speed"),
    "retraction_enable":       ("Enable Retraction",            bool,  None, None,  None,  "Pull filament back during travel"),
    "retraction_amount":       ("Retraction Amount (mm)",       float, 0.0,  10.0,  0.1,   "Distance to retract"),
    "retraction_speed":        ("Retraction Speed (mm/s)",      float, 5,    100,   1,     "Speed of retraction move"),
    "support_enable":          ("Enable Support",               bool,  None, None,  None,  "Generate support structures"),
    "support_angle":           ("Support Overhang Angle (°)",   float, 0,    90,    1,     "Min overhang angle to support"),
    "cool_fan_speed":          ("Fan Speed (%)",                float, 0,    100,   5,     "Part cooling fan percentage"),
    "cool_min_layer_time":     ("Min Layer Time (s)",           float, 1,    60,    1,     "Minimum time per layer"),
}

INFILL_PATTERNS = [
    "grid", "lines", "triangles", "trihexagon", "cubic", "cubicsubdiv",
    "tetrahedral", "quarter_cubic", "concentric", "zigzag", "cross",
    "cross_3d", "gyroid", "lightning"
]


def load_config():
    cfgp = configparser.ConfigParser()
    if os.path.exists(CONFIG_PATH):
        cfgp.read(CONFIG_PATH)
    return cfgp


def save_config(cfgp):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        cfgp.write(f)


def get_setting(cfgp, key, default=""):
    try:
        return cfgp["SLICER"][key]
    except (KeyError, TypeError):
        return default


def set_setting(cfgp, key, value):
    if "SLICER" not in cfgp:
        cfgp["SLICER"] = {}
    cfgp["SLICER"][key] = str(value)


# ──────────────────────────────────────────────────────────
# GCode parsing (lightweight, no pygcode dependency in UI)
# ──────────────────────────────────────────────────────────

# One regex per axis — order-independent, handles any GCode axis arrangement
_GCMD_RE = re.compile(r"^(G0|G00|G1|G01)\b", re.IGNORECASE)
_AXIS_RE  = {k: re.compile(rf"\b{k}([-\d.]+)", re.IGNORECASE)
             for k in ("X", "Y", "Z", "E", "B", "C", "F")}


@st.cache_data(show_spinner=False)
def parse_gcode(content: str):
    """
    Parse GCode into a DataFrame. Handles both XYZ and Core-R-Theta (B/C axis) files.
    Axes are parsed independently so order in the line doesn't matter.
    B and C are stored as None when absent from a line, so check_coord_type() can
    distinguish coordinate systems without false positives.
    """
    rows = []
    # Carry-forward state for axes that accumulate (absolute mode)
    carry = dict(X=0.0, Y=0.0, Z=0.0, E=None, B=None, C=None, F=None)

    for raw_line in content.splitlines():
        line = raw_line.split(";")[0].strip()
        if not line:
            continue
        cmd_m = _GCMD_RE.match(line)
        if not cmd_m:
            continue

        cmd = cmd_m.group(1).upper()
        is_travel = cmd in ("G0", "G00")

        # Parse each axis independently from the full line text
        seen = {}
        for key, rx in _AXIS_RE.items():
            m = rx.search(line)
            if m:
                seen[key] = float(m.group(1))

        # Update carry-forward for axes present in this line
        for key in ("X", "Y", "Z", "E", "F"):
            if key in seen:
                carry[key] = seen[key]

        # B and C: carry forward once set (absolute rotary/tilt axes)
        if "B" in seen:
            carry["B"] = seen["B"]
        if "C" in seen:
            carry["C"] = seen["C"]

        rows.append({
            "G":  "G00" if is_travel else "G01",
            "X":  carry["X"],
            "Y":  carry["Y"],
            "Z":  carry["Z"],
            "E":  carry["E"],
            "F":  carry["F"],
            # Store None when B/C have never been set — lets check_coord_type()
            # correctly identify pure-XYZ files even after B/C appear once in core files.
            "B":  carry["B"],
            "C":  carry["C"],
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────
# Coordinate system detection & kinematic transform
# ──────────────────────────────────────────────────────────

def check_coord_type(df: pd.DataFrame) -> str:
    """Return 'core' if B and C axes are present, else 'xyz'."""
    if df["C"].isnull().all() or df["B"].isnull().all():
        return "xyz"
    return "core"


@st.cache_data(show_spinner=False)
def core_to_xyz(content_hash: str, df: pd.DataFrame, b_len: float, angle_step: float = 1.0) -> pd.DataFrame:
    """
    Kinematic transform from Core-R-Theta machine coordinates to Cartesian XYZ.
    Mirrors gcode.py::to_xyz() exactly, including angle interpolation.

    content_hash is only used as a cache key — pass hash(gcode_content) so
    Streamlit re-runs the transform when the file changes or b_len changes.
    """
    # Fill B/C forward so interpolation works across all rows
    df = df.copy()
    df["B"] = df["B"].ffill().fillna(0.0)
    df["C"] = df["C"].ffill().fillna(0.0)
    df["E"] = df["E"].fillna(0.0)

    diff_b    = np.abs(df["B"].diff().shift(-1).fillna(0))
    diff_c    = np.abs(df["C"].diff().shift(-1).fillna(0))
    max_diff  = np.maximum(diff_b, diff_c)

    interpolated_rows = []

    for i in range(len(df) - 1):
        start_row = df.iloc[i]
        end_row   = df.iloc[i + 1]

        steps = max(int(np.ceil(max_diff.iloc[i] / angle_step)), 1)

        b_interp = np.linspace(start_row["B"], end_row["B"], steps, endpoint=False)
        c_interp = np.linspace(start_row["C"], end_row["C"], steps, endpoint=False)

        if start_row["E"] == end_row["E"] or start_row["G"] == "G00":
            e_interp = np.full(steps, start_row["E"])
        else:
            e_interp = np.linspace(start_row["E"], end_row["E"], steps, endpoint=False)

        e_interp = np.nan_to_num(e_interp, nan=0.0)

        for b, c, e in zip(b_interp, c_interp, e_interp):
            interpolated_rows.append({
                "G":      start_row["G"],
                "X_mach": start_row["X"],
                "Z_mach": start_row["Z"],
                "B":      b,
                "C":      c,
                "E":      e,
                "F":      start_row["F"],
            })

    last = df.iloc[-1]
    interpolated_rows.append({
        "G": last["G"], "X_mach": last["X"], "Z_mach": last["Z"],
        "B": last["B"], "C": last["C"], "E": last["E"], "F": last["F"],
    })

    idf = pd.DataFrame(interpolated_rows)

    # Kinematic transform — reverse machine offset to recover Cartesian coords
    z_hop = np.where(idf["G"] == "G00", 1, 0)
    L     = b_len + z_hop

    rad_b = np.radians(idf["B"])
    rad_c = np.radians(idf["C"])

    r_orig = idf["X_mach"] + (np.sin(rad_b) * L)
    z_orig = idf["Z_mach"] - ((np.cos(rad_b) - 1) * L) - z_hop

    idf["X"] = r_orig * np.cos(rad_c)
    idf["Y"] = r_orig * np.sin(rad_c)
    idf["Z"] = z_orig

    return idf


def build_toolpath_figure(
    df: pd.DataFrame,
    z_cutoff,
    color_mode: str = "layer",
    show_travel: bool = False,
):
    """
    Build an interactive 3D toolpath figure.

    color_mode:
      "layer" — markers colored by Z height (T["cmap_layer"]). Best for dense
                paths; shows layer progression like Cura/PrusaSlicer.
      "speed" — markers colored by feedrate F (T["cmap_speed"]).
      "type"  — markers colored by print order (T["cmap_type"]), travel in gray.
    """
    plot    = df[df["Z"] <= z_cutoff].copy() if z_cutoff is not None else df.copy()
    extrude = plot[plot["G"] == "G01"]
    travel  = plot[plot["G"] == "G00"]

    fig = go.Figure()

    # ── Shared colorbar style ────────────────────────────────
    cbar_base = dict(
        tickfont=dict(size=10, color=T["text_muted"]),
        thickness=12,
        len=0.6,
        x=1.01,
    )

    if color_mode == "layer":
        color_vals = extrude["Z"]
        cbar       = dict(**cbar_base, title=dict(text="Z (mm)",    font=dict(size=11, color=T["text_muted"])))
        cscale     = T["cmap_layer"]

    elif color_mode == "speed":
        color_vals = extrude["F"].fillna(0)
        cbar       = dict(**cbar_base, title=dict(text="Feedrate",  font=dict(size=11, color=T["text_muted"])))
        cscale     = T["cmap_speed"]

    else:  # "type" — print order colormap
        color_vals = np.linspace(0, 1, max(len(extrude), 1))
        cbar       = dict(**cbar_base, title=dict(text="Print order", font=dict(size=11, color=T["text_muted"])))
        cscale     = T["cmap_type"]

    fig.add_trace(go.Scatter3d(
        x=extrude["X"], y=extrude["Y"], z=extrude["Z"],
        mode="markers",
        marker=dict(
            size=1.8,
            color=color_vals,
            colorscale=cscale,
            showscale=True,
            colorbar=cbar,
            opacity=0.85,
        ),
        name="Extrusion",
        hovertemplate="X: %{x:.2f}  Y: %{y:.2f}  Z: %{z:.2f}<extra></extra>",
    ))

    if show_travel and not travel.empty:
        fig.add_trace(go.Scatter3d(
            x=travel["X"], y=travel["Y"], z=travel["Z"],
            mode="lines",
            line=dict(color=T["travel_color"], width=1),
            name="Travel",
            opacity=0.45,
            hoverinfo="skip",
        ))

    # ── Layout ───────────────────────────────────────────────
    axis_common = dict(
        backgroundcolor=T["plot_bg"],
        gridcolor=T["grid"],
        showbackground=True,
        zerolinecolor=T["zeroline"],
        color=T["text_muted"],
    )
    fig.update_layout(
        paper_bgcolor=T["surface"],
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            font=dict(family="Inter", size=12, color=T["text"]),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor=T["border"],
            borderwidth=1,
            x=0.01, y=0.99,
        ),
        scene=dict(
            bgcolor=T["plot_bg"],
            aspectmode="data",
            xaxis=dict(**axis_common, title=dict(text="X (mm)", font=dict(size=11))),
            yaxis=dict(**axis_common, title=dict(text="Y (mm)", font=dict(size=11))),
            zaxis=dict(**axis_common, title=dict(text="Z (mm)", font=dict(size=11))),
        ),
    )
    return fig


def build_extrusion_figure(df: pd.DataFrame):
    """Hexbin-style extrusion heatmap using Plotly density contour."""
    ext = df[df["E"].notna() & (df["G"] == "G01")]
    if ext.empty:
        return None

    fig = go.Figure(go.Histogram2dContour(
        x=ext["X"], y=ext["Y"],
        colorscale=T["cmap_layer"],
        contours=dict(coloring="heatmap", showlabels=False),
        line=dict(width=0),
        ncontours=40,
        name="Extrusion density",
        showscale=True,
        colorbar=dict(
            title=dict(text="Density", font=dict(size=11, color=T["text_muted"])),
            tickfont=dict(size=10, color=T["text_muted"]),
            thickness=12,
        ),
    ))

    fig.update_layout(
        paper_bgcolor=T["surface"],
        plot_bgcolor=T["plot_bg"],
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            gridcolor=T["grid"], zerolinecolor=T["zeroline"],
            color=T["text_muted"], title="X (mm)", scaleanchor="y",
        ),
        yaxis=dict(
            gridcolor=T["grid"], zerolinecolor=T["zeroline"],
            color=T["text_muted"], title="Y (mm)",
        ),
    )
    return fig


# ──────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────
st.markdown('<div class="dash-sub">S4 Slicer Utility</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-header">S5 Dashboard</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# Sidebar — Configuration Panel
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="panel-title">⚙ Slicer Config</div>', unsafe_allow_html=True)

    cfgp = load_config()
    changed_values = {}

    # Group settings into sections
    SECTIONS = {
        "Print Quality": ["layer_height", "layer_height_0"],
        "Material": ["material_diameter"],
        "Walls": ["wall_line_count", "top_layers", "bottom_layers"],
        "Infill": ["infill_sparse_density", "infill_pattern"],
        "Speed": ["speed_print", "speed_travel", "speed_wall"],
        "Retraction": ["retraction_enable", "retraction_amount", "retraction_speed"],
    }

    for section_name, keys in SECTIONS.items():
        with st.expander(section_name, expanded=(section_name in ("Print Quality", "Speed"))):
            for key in keys:
                if key not in KEY_SETTINGS:
                    continue
                label, typ, mn, mx, step, desc = KEY_SETTINGS[key]
                raw_val = get_setting(cfgp, key, "")

                if typ == bool:
                    try:
                        current_bool = raw_val.lower() not in ("false", "0", "no", "")
                    except Exception:
                        current_bool = False
                    new_val = st.checkbox(label, value=current_bool, help=desc, key=f"cfg_{key}")
                    changed_values[key] = str(new_val)

                elif typ == str and key == "infill_pattern":
                    try:
                        idx = INFILL_PATTERNS.index(raw_val) if raw_val in INFILL_PATTERNS else 0
                    except ValueError:
                        idx = 0
                    new_val = st.selectbox(label, INFILL_PATTERNS, index=idx, help=desc, key=f"cfg_{key}")
                    changed_values[key] = new_val

                elif typ == int:
                    try:
                        current_int = int(float(raw_val)) if raw_val else int((mn + mx) / 2)
                    except Exception:
                        current_int = int((mn + mx) / 2)
                    new_val = st.number_input(label, min_value=mn, max_value=mx, value=current_int,
                                              step=step, help=desc, key=f"cfg_{key}")
                    changed_values[key] = str(int(new_val))

                else:  # float
                    try:
                        current_float = float(raw_val) if raw_val else (mn + mx) / 2
                    except Exception:
                        current_float = (mn + mx) / 2
                    new_val = st.number_input(label, min_value=float(mn), max_value=float(mx),
                                              value=float(current_float), step=float(step),
                                              format="%.3f", help=desc, key=f"cfg_{key}")
                    changed_values[key] = str(new_val)

    st.markdown("<hr style='border-top:1px solid #1a2530; margin:0.8rem 0'/>", unsafe_allow_html=True)

    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button("Save", use_container_width=True):
            for k, v in changed_values.items():
                set_setting(cfgp, k, v)
            save_config(cfgp)
            st.success("Saved!")

    with col_reset:
        if st.button("↺ Reload", use_container_width=True):
            st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)

    # CuraEngine path
    st.markdown('<div class="panel-title" style="margin-top:0.5rem">🔧 Paths</div>', unsafe_allow_html=True)
    cura_path_val = ""
    try:
        cura_path_val = cfgp["PATHS"]["curaengine"]
    except Exception:
        pass
    new_cura_path = st.text_input("CuraEngine Path", value=cura_path_val, key="cura_path_input")
    if new_cura_path != cura_path_val:
        if "PATHS" not in cfgp:
            cfgp["PATHS"] = {}
        cfgp["PATHS"]["curaengine"] = new_cura_path

    cura_exists = os.path.exists(new_cura_path) if new_cura_path else False
    if new_cura_path:
        if cura_exists:
            st.success("✓ CuraEngine found")
        else:
            st.warning("⚠ CuraEngine not found at this path")

# ──────────────────────────────────────────────────────────
# Main area — Tabs
# ──────────────────────────────────────────────────────────
tab_slice, tab_viewer, tab_extrusion, tab_rawconfig = st.tabs([
    "  SLICE  ", "  GCODE VIEWER  ", "  EXTRUSION MAP  ", "  RAW CONFIG  "
])


# ──────────────────────────────────────────────────────────
# Tab 1 — Slice
# ──────────────────────────────────────────────────────────
with tab_slice:
    col_up, col_info = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown('<div class="panel-title">STL Upload</div>', unsafe_allow_html=True)
        stl_file = st.file_uploader(
            "Drop an STL file here",
            type=["stl"],
            label_visibility="collapsed",
        )

        if stl_file:
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-label">File</div>
                    <div class="stat-value" style="font-size:0.85rem">{stl_file.name}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Size</div>
                    <div class="stat-value">{stl_file.size / 1024:.1f} KB</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            b_len = st.number_input(
                "B Nozzle Length (mm) — hinge to tip",
                min_value=10.0, max_value=200.0, value=41.5, step=0.5,
                help="Length of the B-axis nozzle arm from hinge to tip."
            )

            slice_btn = st.button("⬡ Run S4 Slice", use_container_width=True)

            if slice_btn:
                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                    tmp.write(stl_file.read())
                    tmp_path = tmp.name

                progress = st.progress(0, text="Initialising slicer…")

                try:
                    # Import here so the dashboard can still load if deps are missing
                    from S5 import slice 

                    progress.progress(10, text="Loading mesh…")
                    time.sleep(0.1)

                    import threading, queue
                    result_q = queue.Queue()

                    def run_slice():
                        try:
                            slice(tmp_path)
                            result_q.put(("ok", None))
                        except Exception as e:
                            result_q.put(("err", str(e)))

                    t = threading.Thread(target=run_slice, daemon=True)
                    t.start()

                    steps = ["Tetrahedralizing…", "Building graph…", "Optimising rotations…",
                             "Deforming mesh…", "CuraEngine slice…", "Reforming GCode…"]
                    for i, step_label in enumerate(steps):
                        progress.progress(15 + i * 12, text=step_label)
                        t.join(timeout=30)
                        if not t.is_alive():
                            break

                    t.join()
                    progress.progress(100, text="Done!")
                    status, err = result_q.get()

                    if status == "ok":
                        st.success("✓ Slice complete. Output written to output_gcode/")
                    else:
                        st.error(f"Slice error: {err}")

                except ImportError:
                    progress.empty()
                    st.warning(
                        "⚠ Slicer dependencies (open3d, tetgen, pyvista, etc.) are not installed "
                        "in this environment. To slice, run `python easys4.py s4 <file.stl>` from "
                        "the command line."
                    )
                finally:
                    os.unlink(tmp_path)
        else:
            st.markdown("""
            <div style="color:#3a5060; font-family:'IBM Plex Mono',monospace; font-size:0.8rem;
                        text-align:center; padding:3rem 1rem; border:1px dashed #1a2a35; border-radius:4px;
                        margin-top:0.5rem">
                Upload an STL to begin.<br/>
                <span style="font-size:2rem; display:block; margin-top:0.8rem; opacity:0.4">⬡</span>
            </div>
            """, unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="panel-title">Active Settings</div>', unsafe_allow_html=True)
        cfgp2 = load_config()
        preview_keys = [
            ("Layer Height", "layer_height", "mm"),
            ("Wall Count", "wall_line_count", ""),
            ("Infill", "infill_sparse_density", "%"),
            ("Pattern", "infill_pattern", ""),
            ("Support", "support_enable", ""),
        ]
        cols = st.columns(2)
        for i, (display, key, unit) in enumerate(preview_keys):
            val = get_setting(cfgp2, key, "—")
            cols[i % 2].markdown(f"""
            <div class="stat-box" style="margin-bottom:0.5rem">
                <div class="stat-label">{display}</div>
                <div class="stat-value" style="font-size:0.95rem">{val} {unit}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border-top:1px solid #1a2530'/>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title" style="margin-top:0.8rem">Output Files</div>', unsafe_allow_html=True)

        output_dirs = ["output_gcode", "input_gcode/deformed", "output_models/deformed"]
        for d in output_dirs:
            if os.path.isdir(d):
                files = sorted(Path(d).glob("*.gcode"), key=os.path.getmtime, reverse=True)
                if files:
                    for f in files[:3]:
                        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f)))
                        size_kb = os.path.getsize(f) / 1024
                        st.markdown(f"""
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
                                    color:#6a8a7a; padding:0.3rem 0; border-bottom:1px solid #131820">
                            <span style="color:#2a9d8f">{f.name}</span><br/>
                            <span style="color:#3a5060">{mtime} · {size_kb:.0f} KB</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown(f'<span style="font-family:\'IBM Plex Mono\',monospace; font-size:0.72rem; color:#3a5060">{d}/ not found</span>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Tab 2 — GCode Viewer
# ──────────────────────────────────────────────────────────
with tab_viewer:
    st.markdown('<div class="panel-title">Gcode Viewer</div>', unsafe_allow_html=True)

    gcode_source = st.radio(
        "Source",
        ["Upload GCode", "Load from output_gcode/"],
        horizontal=True,
        label_visibility="collapsed",
    )

    gcode_content = None

    if gcode_source == "Upload GCode":
        gcode_up = st.file_uploader("Upload .gcode file", type=["gcode", "g"], label_visibility="collapsed")
        if gcode_up:
            gcode_content = gcode_up.read().decode("utf-8", errors="replace")
            st.caption(f"Loaded: {gcode_up.name} ({len(gcode_content)//1024} KB)")
    else:
        output_path = Path("output_gcode")
        if output_path.is_dir():
            gcode_files = sorted(output_path.glob("*.gcode"), key=os.path.getmtime, reverse=True)
            if gcode_files:
                chosen = st.selectbox(
                    "Select file",
                    [f.name for f in gcode_files],
                    label_visibility="collapsed",
                )
                with open(output_path / chosen, "r", errors="replace") as fh:
                    gcode_content = fh.read()
            else:
                st.info("No .gcode files found in output_gcode/")
        else:
            st.info("output_gcode/ directory not found. Run a slice first.")

    if gcode_content:
        with st.spinner("Parsing GCode…"):
            df_raw = parse_gcode(gcode_content)

        if df_raw.empty:
            st.error("No G0/G1 moves found in this file.")
        else:
            coord_type = check_coord_type(df_raw)

            # ── Coordinate system banner + b_len input ──────────────
            if coord_type == "core":
                banner_col, blen_col = st.columns([3, 1])
                with banner_col:
                    st.markdown(
                        '<div class="banner-core">⟳ <strong>Core-R-Theta detected</strong> '
                        '— applying kinematic transform (X<sub>mach</sub>, B, C) → Cartesian XYZ</div>',
                        unsafe_allow_html=True,
                    )
                with blen_col:
                    b_len_viewer = st.number_input(
                        "B nozzle len (mm)", min_value=1.0, max_value=300.0,
                        value=41.5, step=0.5, key="viewer_b_len",
                        help="Length from B-axis hinge to nozzle tip",
                    )

                content_hash = str(hash(gcode_content + str(b_len_viewer)))
                with st.spinner("Applying kinematic transform…"):
                    df = core_to_xyz(content_hash, df_raw, b_len_viewer)
            else:
                st.markdown(
                    '<div class="banner-xyz">✓ <strong>Cartesian XYZ</strong> — no coordinate transform needed</div>',
                    unsafe_allow_html=True,
                )
                df = df_raw

            # ── Stats row ───────────────────────────────────────────
            n_ext  = (df["G"] == "G01").sum()
            n_trav = (df["G"] == "G00").sum()
            layers = sorted(df["Z"].dropna().unique())
            n_layers = len(layers)
            x_range = f"{df['X'].min():.1f} → {df['X'].max():.1f}"
            y_range = f"{df['Y'].min():.1f} → {df['Y'].max():.1f}"
            z_range = f"{df['Z'].min():.1f} → {df['Z'].max():.1f}"
            coord_label = "CORE→XYZ" if coord_type == "core" else "XYZ"

            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-box"><div class="stat-label">Coord Space</div><div class="stat-value" style="font-size:0.85rem">{coord_label}</div></div>
                <div class="stat-box"><div class="stat-label">Extrusion Moves</div><div class="stat-value">{n_ext:,}</div></div>
                <div class="stat-box"><div class="stat-label">Travel Moves</div><div class="stat-value">{n_trav:,}</div></div>
                <div class="stat-box"><div class="stat-label">Layers</div><div class="stat-value">{n_layers}</div></div>
                <div class="stat-box"><div class="stat-label">X Range</div><div class="stat-value" style="font-size:0.8rem">{x_range}</div></div>
                <div class="stat-box"><div class="stat-label">Y Range</div><div class="stat-value" style="font-size:0.8rem">{y_range}</div></div>
                <div class="stat-box"><div class="stat-label">Z Range</div><div class="stat-value" style="font-size:0.8rem">{z_range}</div></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Controls ────────────────────────────────────────────
            ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 2, 1, 1])
            with ctrl_col1:
                color_mode = st.selectbox(
                    "Color by",
                    options=["layer", "speed", "type"],
                    format_func=lambda x: {
                        "layer": "🎨 Layer (Z height)",
                        "speed": "⚡ Speed (feedrate)",
                        "type":  "🔵 Move type",
                    }[x],
                    index=0,
                    key="color_mode",
                )
            with ctrl_col2:
                show_travel = st.checkbox("Show travel moves", value=False)
            with ctrl_col3:
                max_points = st.selectbox("Max points", [10_000, 50_000, 200_000, "All"], index=1)

            # ── Layer slider ────────────────────────────────────────
            if n_layers > 1:
                layer_idx = st.slider(
                    "Layer cutoff",
                    min_value=0, max_value=n_layers - 1,
                    value=n_layers - 1,
                    format="Layer %d",
                    help="Show toolpaths up to and including this layer (by Z height index)",
                )
                z_cutoff = layers[layer_idx]
                st.caption(f"Showing up to Z = {z_cutoff:.3f} mm  (layer {layer_idx + 1}/{n_layers})")
            else:
                z_cutoff = None

            # ── Build plot DataFrame ────────────────────────────────
            plot_df = df.copy()
            if z_cutoff is not None:
                plot_df = plot_df[plot_df["Z"] <= z_cutoff]
            if not show_travel:
                plot_df = plot_df[plot_df["G"] == "G01"]
            if max_points != "All" and len(plot_df) > int(max_points):
                step_s = max(1, len(plot_df) // int(max_points))
                plot_df = plot_df.iloc[::step_s]

            with st.spinner("Rendering…"):
                fig = build_toolpath_figure(plot_df, z_cutoff, color_mode=color_mode, show_travel=show_travel)

            st.plotly_chart(fig, width="stretch", config={"displayModeBar": True})

            # ── Raw GCode preview ───────────────────────────────────
            with st.expander("📄 Raw GCode preview (first 200 lines)"):
                preview_lines = gcode_content.splitlines()[:200]
                preview_text  = "\n".join(preview_lines)
                st.markdown(f'<div class="gcode-display">{preview_text}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Tab 3 — Extrusion Map
# ──────────────────────────────────────────────────────────
with tab_extrusion:
    st.markdown('<div class="panel-title">🌡 Extrusion Density Map</div>', unsafe_allow_html=True)
    st.caption("Upload or re-use the GCode parsed in the Viewer tab.")

    ext_gcode_up = st.file_uploader(
        "Upload GCode for extrusion map", type=["gcode", "g"],
        label_visibility="collapsed", key="ext_upload"
    )

    ext_content = None
    if ext_gcode_up:
        ext_content = ext_gcode_up.read().decode("utf-8", errors="replace")
    elif gcode_content:
        ext_content = gcode_content
        st.info("Using GCode from Viewer tab.")

    if ext_content:
        with st.spinner("Parsing…"):
            ext_df_raw = parse_gcode(ext_content)

        ext_coord_type = check_coord_type(ext_df_raw)

        if ext_coord_type == "core":
            st.markdown(
                '<div class="banner-core">⟳ Core-R-Theta — using same b_len as Viewer tab for transform</div>',
                unsafe_allow_html=True,
            )
            # Reuse b_len from viewer tab if set, else default
            ext_b_len = st.session_state.get("viewer_b_len", 41.5)
            ext_hash = str(hash(ext_content + str(ext_b_len)))
            with st.spinner("Applying kinematic transform…"):
                ext_df = core_to_xyz(ext_hash, ext_df_raw, ext_b_len)
        else:
            ext_df = ext_df_raw

        z_vals = sorted(ext_df["Z"].dropna().unique())
        if len(z_vals) > 1:
            z_lo, z_hi = st.select_slider(
                "Z range",
                options=z_vals,
                value=(z_vals[0], z_vals[-1]),
                format_func=lambda v: f"{v:.2f}",
            )
            ext_df = ext_df[(ext_df["Z"] >= z_lo) & (ext_df["Z"] <= z_hi)]

        fig2 = build_extrusion_figure(ext_df)
        if fig2:
            st.plotly_chart(fig2, width="stretch", config={"displayModeBar": True})
        else:
            st.warning("No extrusion moves with E values found in this GCode.")
    else:
        st.markdown("""
        <div style="color:#3a5060; font-family:'IBM Plex Mono',monospace; font-size:0.8rem;
                    text-align:center; padding:3rem 1rem; border:1px dashed #1a2a35; border-radius:4px">
            Load a GCode file in the Viewer tab, or upload one here.
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Tab 4 — Raw Config
# ──────────────────────────────────────────────────────────
with tab_rawconfig:
    st.markdown('<div class="panel-title">📋 Raw Configuration</div>', unsafe_allow_html=True)

    sub1, sub2 = st.columns(2, gap="large")

    with sub1:
        st.markdown("**easys4.ini**")
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                ini_text = f.read()
            edited_ini = st.text_area(
                "easys4.ini", value=ini_text, height=500,
                label_visibility="collapsed", key="raw_ini"
            )
            if st.button("💾 Write ini", key="write_ini"):
                with open(CONFIG_PATH, "w") as f:
                    f.write(edited_ini)
                st.success("Saved easys4.ini")
        else:
            st.info(f"{CONFIG_PATH} not found. Save settings from the sidebar first.")

    with sub2:
        st.markdown("**core.def.json** (read-only preview)")
        if os.path.exists(CORE_DEF_PATH):
            with open(CORE_DEF_PATH, "r") as f:
                json_text = f.read()
            st.text_area(
                "core.def.json", value=json_text[:8000] + ("…" if len(json_text) > 8000 else ""),
                height=500, label_visibility="collapsed", key="raw_json",
                disabled=True
            )
            st.caption(f"Full file: {len(json_text) // 1024} KB")
        else:
            st.info(f"{CORE_DEF_PATH} not found.")

        # Import from JSON
        st.markdown("**Import settings from JSON**")
        json_up = st.file_uploader(
            "Drop a Cura JSON export here", type=["json"],
            label_visibility="collapsed", key="json_import"
        )
        if json_up:
            try:
                imported = json.load(json_up)
                cfgp_imp = load_config()
                count = 0
                for k, v in imported.get("settings", {}).get("global", {}).get("all", {}).items():
                    if isinstance(v, dict):
                        val = v.get("value") or v.get("default_value", "")
                    else:
                        val = v
                    set_setting(cfgp_imp, k, val)
                    count += 1
                save_config(cfgp_imp)
                st.success(f"✓ Imported {count} settings from JSON")
            except Exception as e:
                st.error(f"Import error: {e}")