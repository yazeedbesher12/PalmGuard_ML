import os
import json
import base64
from datetime import datetime
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import matplotlib.pyplot as plt

from src.config import DATASET_DIR, LABELS_CSV, MODEL_DIR, BEST_MODEL_PATH, OUTPUTS_DIR
from src.dataset import load_labels, split_by_tree, SegmentDataset
from src.train_utils import train_model, evaluate_segment_level
from src.infer_utils import (
    load_model,
    infer_segments_for_file,
    infer_segment_probs,
    aggregate_tree_risk,
    tree_level_metrics,
    tree_confusion_matrix_df,
)
from src.audio import wav_load, logmel_from_waveform

# -----------------------
# Page config + style
# -----------------------
st.set_page_config(page_title="PalmGuard", page_icon="🌴", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      :root {
        /* Darker mint background (requested) */
        --pg-bg1: #97d1a4;
        --pg-bg2: #b3e2bc;
        --pg-card: rgba(255,255,255,0.78);
        --pg-border: rgba(15,23,42,0.10);
        --pg-shadow: 0 14px 40px rgba(15,23,42,0.08);
        --pg-text: #0f172a;
        --pg-muted: rgba(15,23,42,0.70);
        --pg-primary: #15803d;
        --pg-primary2: #16a34a;
        --pg-accent: #06b6d4;
      }

      html, body, [data-testid="stAppViewContainer"], .stApp {
        background:
          radial-gradient(circle at 18% 8%, rgba(34,197,94,0.26), transparent 42%),
          radial-gradient(circle at 85% 14%, rgba(6,182,212,0.14), transparent 42%),
          radial-gradient(circle at 55% 92%, rgba(245,158,11,0.10), transparent 48%),
          linear-gradient(180deg, var(--pg-bg1) 0%, var(--pg-bg2) 55%, #ffffff 100%);
        color: var(--pg-text);
      }

      /* Leave room for the fixed top-nav */
      .block-container { padding-top: 3.2rem !important; }

      /* Hide Streamlit toolbar/header (Deploy bar) */
      [data-testid="stToolbar"] { visibility: hidden; height: 0px; position: fixed; }
      [data-testid="stHeader"] { visibility: hidden; height: 0px; position: fixed; }
      header { visibility: hidden; height: 0px; }
      #MainMenu { visibility: hidden; }
      footer { visibility: hidden; }

      /* Top navigation bar (fixed at very top) */
      .pg-topnav {
        position: fixed;
        top: 0.20rem;
        left: 0;
        right: 0;
        z-index: 1000000;
        pointer-events: none;
      }
      .pg-topnav-inner {
        width: min(1200px, calc(100% - 2rem));
        margin: 0 auto;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid rgba(6,95,70,0.28);
        background: rgba(245,255,248,0.86);
        backdrop-filter: blur(12px);
        box-shadow: 0 18px 44px rgba(15,23,42,0.12);
        display: flex;
        gap: 6px;
        justify-content: center;
        align-items: center;
        pointer-events: auto;
      }
      .pg-navbtn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
        border-radius: 999px;
        padding: 0.34rem 0.75rem;
        font-weight: 850;
        font-size: 12px;
        border: 1px solid rgba(6,95,70,0.34);
        background: rgba(6,95,70,0.12);
        color: #064e3b;
        transition: transform 120ms ease, filter 120ms ease, background 120ms ease;
        white-space: nowrap;
      }
      .pg-navbtn:hover { transform: translateY(-1px); filter: brightness(1.02); background: rgba(6,95,70,0.16); }
      .pg-navbtn.active {
        background: #064e3b;
        color: #ffffff;
        border-color: rgba(6,95,70,0.60);
        box-shadow: 0 16px 30px rgba(6,95,70,0.24);
      }

      section[data-testid="stSidebar"] > div {
        background: rgba(255,255,255,0.72);
        border-right: 1px solid var(--pg-border);
        backdrop-filter: blur(10px);
      }
      section[data-testid="stSidebar"] .stMarkdown,
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] span,
      section[data-testid="stSidebar"] p {
        color: var(--pg-text) !important;
      }

      .pg-card {
        padding: 14px 16px;
        border-radius: 18px;
        border: 1px solid var(--pg-border);
        background: var(--pg-card);
        box-shadow: var(--pg-shadow);
      }


.pg-titlebox{
  max-width: 1100px;
  margin: 10px auto 16px auto;
  padding: 28px 28px;
  border-radius: 22px;
  border: 1px solid rgba(6,95,70,0.18);
  background: rgba(255,255,255,0.80);
  box-shadow: var(--pg-shadow);
  text-align: center;
}
.pg-titlebox-title{
  font-size: 72px;
  font-weight: 950;
  letter-spacing: -0.03em;
  color: #064e3b;
  line-height: 1.02;
}
.pg-titlebox-sub{
  font-size: 16px;
  color: var(--pg-muted);
  margin-top: 6px;
}

.pg-media { margin: 0 0 44px 0; }

.pg-media-frame{
  width: 100%;
  height: 460px;
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid var(--pg-border);
  box-shadow: var(--pg-shadow);
  background: rgba(255,255,255,0.55);
}
@media (min-width: 1100px){
  .pg-media-frame{ height: 620px; }
}

.pg-media-media{
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.pg-title { font-size: 34px; font-weight: 900; margin-bottom: 6px; color: #064e3b; }
      .pg-sub { color: var(--pg-muted); margin-top: -6px; }
      .pg-small { color: rgba(15,23,42,0.55); font-size: 12px; }

      .pg-hero { text-align: center; margin: 8px 0 12px 0; }
      .pg-hero-title { font-size: 54px; font-weight: 950; letter-spacing: -0.03em; margin-bottom: 6px; line-height: 1.02; color: #064e3b; }
      .pg-hero-sub { font-size: 15px; color: var(--pg-muted); margin-top: 2px; }

      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #064e3b; letter-spacing: -0.02em; }

      .stButton>button {
        border-radius: 14px;
        padding: 0.62rem 1.05rem;
        border: 1px solid rgba(22,163,74,0.35);
        background: linear-gradient(135deg, var(--pg-primary), #34d399);
        color: #ffffff;
        box-shadow: 0 12px 22px rgba(34,197,94,0.26);
      }
      .stButton>button:hover { transform: translateY(-1px); filter: brightness(1.02); }
      .stButton>button:active { transform: translateY(0px); }

      .stTextInput input, .stNumberInput input, .stTextArea textarea {
        border-radius: 14px !important;
      }
      div[data-baseweb="select"] > div { border-radius: 14px; }

      .pg-video-wrap video {
        width: 100% !important;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        border-radius: 20px;
        border: 1px solid var(--pg-border);
        box-shadow: var(--pg-shadow);
      }
      .pg-video-wrap { margin: 4px 0 14px 0; }

      .stImage img {
        border-radius: 20px;
        border: 1px solid var(--pg-border);
        box-shadow: var(--pg-shadow);
      }

      div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid var(--pg-border);
        box-shadow: var(--pg-shadow);
      }
</style>
    """,
    unsafe_allow_html=True,
)
assets_dir = "assets"

# Landing assets (shown only on the main page)
hero_img_1 = os.path.join(assets_dir, "WhatsApp_Image_2026-01-28_at_10.20.26_PM_(2).jpeg")
hero_img_2 = os.path.join(assets_dir, "WhatsApp_Image_2026-01-28_at_10.20.26_PM_(3).jpeg")
hero_video_path = os.path.join(assets_dir, "WhatsApp_Video_2026-01-28_at_10.20.29_PM.mp4")


# -----------------------
# Top navigation (replaces sidebar nav)
# -----------------------
PAGES = ["Home", "Train", "Test", "Live Inference (WAV)"]

# -----------------------
# Top navigation (NO URL changes)
# -----------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if st.session_state["page"] not in PAGES:
    st.session_state["page"] = "Home"

st.markdown(
    """
    <style>
      /* Style the FIRST radio (our top-nav) as fixed pill navigation */
      div[data-testid="stRadio"]:first-of-type{
        position: fixed;
        top: 0.35rem;
        left: 50%;
        transform: translateX(-50%);
        width: min(1200px, calc(100% - 2rem));
        z-index: 1000001;
        margin: 0 !important;
        padding: 0 !important;
      }
      div[data-testid="stRadio"]:first-of-type > div{
        border-radius: 999px;
        border: 1px solid rgba(6,95,70,0.28);
        background: rgba(245,255,248,0.86);
        backdrop-filter: blur(12px);
        box-shadow: 0 18px 44px rgba(15,23,42,0.12);
        padding: 10px 12px;
      }
      div[data-testid="stRadio"]:first-of-type [role="radiogroup"]{
        display:flex;
        gap: 14px;
        justify-content:center;
        align-items:center;
      }
      div[data-testid="stRadio"]:first-of-type label{ margin: 0 !important; }

      div[data-testid="stRadio"]:first-of-type label[data-baseweb="radio"]{
        border-radius: 999px !important;
        padding: 0.92rem 1.45rem !important;
        border: 1px solid rgba(6,95,70,0.34) !important;
        background: rgba(6,95,70,0.12) !important;
        transition: transform 120ms ease, filter 120ms ease, background 120ms ease;
      }
      div[data-testid="stRadio"]:first-of-type label[data-baseweb="radio"] *{
        font-weight: 900 !important;
        font-size: 16px !important;
        color: #064e3b !important;
      }
      div[data-testid="stRadio"]:first-of-type label[data-baseweb="radio"]:hover{
        transform: translateY(-1px);
        filter: brightness(1.02);
        background: rgba(6,95,70,0.16) !important;
      }
      div[data-testid="stRadio"]:first-of-type label[data-baseweb="radio"]:has(input:checked){
        background: #064e3b !important;
        border-color: rgba(6,95,70,0.60) !important;
        box-shadow: 0 16px 30px rgba(6,95,70,0.24);
      }
      div[data-testid="stRadio"]:first-of-type label[data-baseweb="radio"]:has(input:checked) *{
        color: #ffffff !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

nav_choice = st.radio(
    "Navigation",
    options=PAGES,
    index=PAGES.index(st.session_state["page"]),
    horizontal=True,
    label_visibility="collapsed",
    key="pg_nav_choice",
)

if nav_choice != st.session_state["page"]:
    st.session_state["page"] = nav_choice
    st.rerun()

page = st.session_state["page"]

def _b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _img_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return "image/png"
    return "image/jpeg"

def render_landing_hero():
    """Hero section shown ONLY on the main (Data Explorer) page."""
    st.markdown(
        """
        <div class="pg-titlebox">
          <div class="pg-titlebox-title">PalmGuard 🌴</div>
          <div class="pg-titlebox-sub">Early detection of Red Palm Weevil using audio sensors + Machine Learning.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Media row (hero): images + looping video
    c1, c2, c3 = st.columns(3, gap="large")

    # Left image
    if os.path.exists(hero_img_1):
        b64 = _b64_file(hero_img_1)
        mime = _img_mime(hero_img_1)
        with c1:
            st.markdown(
                f'<div class="pg-media"><div class="pg-media-frame"><img class="pg-media-media" src="data:{mime};base64,{b64}" /></div></div>',
                unsafe_allow_html=True,
            )

    # Center looping video (autoplay + loop + muted)
    if os.path.exists(hero_video_path):
        b64v = _b64_file(hero_video_path)
        with c2:
            st.markdown(
                f"""
                <div class="pg-media"><div class="pg-media-frame">
                  <video class="pg-media-media" autoplay loop muted playsinline preload="auto">
                    <source src="data:video/mp4;base64,{b64v}" type="video/mp4" />
                  </video>
                </div></div>
                """,
                unsafe_allow_html=True,
            )

    # Right image
    if os.path.exists(hero_img_2):
        b64 = _b64_file(hero_img_2)
        mime = _img_mime(hero_img_2)
        with c3:
            st.markdown(
                f'<div class="pg-media"><div class="pg-media-frame"><img class="pg-media-media" src="data:{mime};base64,{b64}" /></div></div>',
                unsafe_allow_html=True,
            )

    # Spacer between hero media and the rest of the page
    st.markdown('<div style="height:42px"></div>', unsafe_allow_html=True)

# Show hero ONLY on the main page (not on training/testing/inference pages)
if page == "Home":
    render_landing_hero()

def require_dataset():
    if not os.path.exists(LABELS_CSV):
        st.error("labels.csv not found. Put your dataset at: data/dataset/labels.csv")
        st.stop()

def load_metrics_if_any():
    mpath = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(mpath):
        try:
            return json.load(open(mpath, "r", encoding="utf-8"))
        except Exception:
            return None
    return None

def load_tree_locations():
    """Load farm layout positions (row/col) for each tree.

    Expected: data/dataset/tree_locations.csv with columns:
      tree_id, row, col (and optional lat/lon/block)
    """
    loc_path = os.path.join(DATASET_DIR, "tree_locations.csv")
    if not os.path.exists(loc_path):
        return None
    loc = pd.read_csv(loc_path)
    if "tree_id" not in loc.columns or "row" not in loc.columns or "col" not in loc.columns:
        return None
    loc["tree_id"] = loc["tree_id"].astype(str).str.zfill(4)
    loc["row"] = loc["row"].astype(int)
    loc["col"] = loc["col"].astype(int)
    return loc

def render_farm_grid(loc_df: pd.DataFrame, risk_df: pd.DataFrame, threshold: float, grid_rows: int = 10, grid_cols: int = 10):
    """Render a 10x10 farm layout as a colored grid (each cell = one tree)."""
    # Merge risk onto locations (risk_df must contain tree_id, risk, recommendation, label(optional))
    show_cols = ["tree_id", "risk", "recommendation"]
    if "label" in risk_df.columns:
        show_cols.append("label")
    merged = loc_df.merge(risk_df[show_cols], on="tree_id", how="left")

    # build quick lookup by (row, col)
    lookup = {}
    for r in merged.itertuples(index=False):
        lookup[(int(r.row), int(r.col))] = r

    # colors
    RED = "#ef4444"      # high-risk
    GREEN = "#22c55e"    # low-risk
    GREY = "#cbd5e1"     # not tested / no data

    # controls
    col1, col2, col3 = st.columns([2, 2, 2])
    show_ids = col1.checkbox("Show tree_id on cells", value=True)
    show_only_high = col2.checkbox("Show only high-risk (others grey)", value=False)
    show_untested = col3.checkbox("Grey out untested trees", value=True)

    # Legend
    st.markdown(
        f"""<div class="pg-card">
        <b>Legend</b><br>
        <span style="display:inline-block;width:14px;height:14px;background:{RED};border-radius:4px;margin-right:6px;"></span> High risk (risk ≥ {threshold:.2f}) &nbsp;&nbsp;
        <span style="display:inline-block;width:14px;height:14px;background:{GREEN};border-radius:4px;margin-right:6px;"></span> Low risk (risk &lt; {threshold:.2f}) &nbsp;&nbsp;
        <span style="display:inline-block;width:14px;height:14px;background:{GREY};border-radius:4px;margin-right:6px;"></span> No data / not in test split
        </div>""",
        unsafe_allow_html=True,
    )

    # CSS grid
    cell_size = 44
    gap = 6
    html = [f"""<div style="display:grid;grid-template-columns:repeat({grid_cols},{cell_size}px);gap:{gap}px;align-items:center;justify-content:flex-start;">"""]

    for rr in range(1, grid_rows + 1):
        for cc in range(1, grid_cols + 1):
            rec = lookup.get((rr, cc), None)

            # default cell
            tree_id = ""
            risk = None
            recommendation = ""
            label = None

            if rec is not None:
                tree_id = str(rec.tree_id)
                risk = None if (not hasattr(rec, "risk")) else getattr(rec, "risk")
                recommendation = "" if (not hasattr(rec, "recommendation")) else str(getattr(rec, "recommendation"))
                if hasattr(rec, "label"):
                    try:
                        label = int(getattr(rec, "label")) if pd.notna(getattr(rec, "label")) else None
                    except Exception:
                        label = None

            # decide color
            if risk is None or (isinstance(risk, float) and np.isnan(risk)):
                color = GREY
                status = "NO DATA"
            else:
                risk_f = float(risk)
                high = risk_f >= float(threshold)
                if show_only_high and not high:
                    color = GREY
                else:
                    color = RED if high else GREEN
                status = "HIGH" if high else "LOW"

            # grey out untested (if requested)
            if show_untested and (risk is None or (isinstance(risk, float) and np.isnan(risk))):
                color = GREY

            # tooltip
            tip = f"Row {rr}, Col {cc}"
            if tree_id:
                tip += f" | Tree {tree_id}"
            if risk is not None and not (isinstance(risk, float) and np.isnan(risk)):
                tip += f" | risk {float(risk):.3f} | {status}"
            if label is not None:
                tip += f" | label {label}"
            if recommendation:
                tip += f" | {recommendation}"

            # content
            content = tree_id if show_ids and tree_id else ""

            
            text_color = "rgba(15,23,42,0.78)" if color == GREY else "rgba(255,255,255,0.95)"
            border_color = "rgba(15,23,42,0.14)" if color == GREY else "rgba(255,255,255,0.22)"
            shadow = "0 8px 20px rgba(15,23,42,0.10)" if color == GREY else "0 10px 24px rgba(15,23,42,0.18)"

            html.append(
                f"""<div title="{tip}" style="
                    width:{cell_size}px;height:{cell_size}px;
                    background:{color};
                    border-radius:12px;
                    display:flex;align-items:center;justify-content:center;
                    font-weight:800;
                    color: {text_color};
                    border: 1px solid {border_color};
                    box-shadow: {shadow};
                    user-select:none;
                ">{content}</div>"""
            )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

# -----------------------
# Page: Data Explorer
# -----------------------
if page == "Home":
    require_dataset()
    st.markdown('<div class="pg-card">', unsafe_allow_html=True)

    df = load_labels(LABELS_CSV)

    c1, c2, c3, c4 = st.columns(4)
    n_trees = df["tree_id"].nunique()
    n_files = df.shape[0]
    infected_trees = int(df.groupby("tree_id")["label"].max().sum())
    infected_files = int(df["label"].sum())

    c1.metric("Trees", n_trees)
    c2.metric("Files", n_files)
    c3.metric("Infected trees", infected_trees)
    c4.metric("Infected files", infected_files)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Browse labels.csv")
    with st.expander("Filters", expanded=True):
        colf1, colf2, colf3 = st.columns([2, 2, 2])
        label_filter = colf1.selectbox("Label filter", ["all", "healthy (0)", "infected (1)"])
        tree_pick = colf2.selectbox("Tree", ["(all)"] + sorted(df["tree_id"].unique().tolist()))
        search_txt = colf3.text_input("Search file_path contains", value="")

    view = df.copy()
    if label_filter != "all":
        want = 0 if "0" in label_filter else 1
        view = view[view["label"] == want]
    if tree_pick != "(all)":
        view = view[view["tree_id"] == tree_pick]
    if search_txt.strip():
        view = view[view["file_path"].str.contains(search_txt.strip(), case=False, na=False)]

    st.dataframe(view, use_container_width=True, height=360)

    st.markdown("### Inspect one recording")
    if len(view) == 0:
        st.info("No rows match your filters.")
        st.stop()

    row_idx = st.number_input("Row index in filtered table", min_value=0, max_value=max(0, len(view)-1), value=0, step=1)
    row = view.iloc[int(row_idx)]
    wav_path = os.path.join(DATASET_DIR, row["file_path"])

    cols = st.columns([2, 3])
    with cols[0]:
        st.write("**Selected**")
        st.write(f"Tree: `{row['tree_id']}`")
        st.write(f"Label: `{row['label']}`")
        st.write(f"File: `{row['file_path']}`")
        if os.path.exists(wav_path):
            st.audio(wav_path)
        else:
            st.error("WAV file missing: " + wav_path)

    with cols[1]:
        if os.path.exists(wav_path):
            y, sr = wav_load(wav_path, target_sr=16000)
            logm = logmel_from_waveform(y, sr)

            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            ax.set_title("Log-Mel Spectrogram (visual cue for clicks/energy bursts)")
            ax.set_xlabel("Time frames")
            ax.set_ylabel("Mel bins")
            ax.imshow(logm, aspect="auto", origin="lower")
            st.pyplot(fig, clear_figure=True)

# -----------------------
# Page: Train Model
# -----------------------
elif page == "Train":
    require_dataset()

    st.markdown("### Training configuration")
    col1, col2, col3 = st.columns(3)
    train_ratio = col1.slider("Train ratio", 0.5, 0.9, 0.70, 0.01)
    val_ratio = col2.slider("Val ratio", 0.05, 0.3, 0.15, 0.01)
    test_ratio = col3.slider("Test ratio", 0.05, 0.3, 0.15, 0.01)
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        st.warning(f"Ratios sum to {s:.2f}. We will normalize automatically (same behavior as split_by_tree).")

    col4, col5, col6 = st.columns(3)
    seed = col4.number_input("Seed", value=42, step=1)
    epochs = col5.slider("Epochs", 3, 30, 12, 1)
    batch_size = col6.selectbox("Batch size", [16, 32, 64], index=1)

    lr = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)
    cache_audio = False  # UI option removed
    stratify = True      # UI option removed (always stratify)
    st.markdown("---")
    st.markdown("### Run training")

    if st.button("🚀 Train now", type="primary"):
        df = load_labels(LABELS_CSV)
        df_train, df_val, df_test = split_by_tree(
            df,
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            seed=int(seed),
            stratify=bool(stratify),
        )

        from src.config import AudioConfig
        audio_cfg = AudioConfig()

        with st.spinner("Building segment datasets (this can take a bit)…"):
            train_ds = SegmentDataset(df_train, audio_cfg=audio_cfg, cache_audio=cache_audio)
            val_ds = SegmentDataset(df_val, audio_cfg=audio_cfg, cache_audio=cache_audio)
            test_ds = SegmentDataset(df_test, audio_cfg=audio_cfg, cache_audio=cache_audio)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Training on device: **{device}**")

        with st.spinner("Training model…"):
            model, metrics = train_model(
                train_loader,
                val_loader,
                device=device,
                lr=float(lr),
                epochs=int(epochs),
                audio_cfg=audio_cfg,
            )
            test_metrics = evaluate_segment_level(model, test_loader, device=device)

        # Save merged metrics for UI
        os.makedirs(MODEL_DIR, exist_ok=True)
        merged = dict(metrics)
        merged["test_acc_segment"] = float(test_metrics["acc"])
        merged["test_loss_segment"] = float(test_metrics["loss"])
        merged["split"] = {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
            "seed": int(seed),
            "stratify": bool(stratify),
            "tree_level": True,
        }
        with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        st.success("Training done ✅ Model saved to models/best_cnn.pt")

        # Display metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Val Acc", round(float(metrics["best_val_acc"]), 4))
        c2.metric("Test Acc (segment)", round(float(test_metrics["acc"]), 4))
        c3.metric("Best Epoch", int(metrics["best_epoch"]))

        # Plot history
        hist_train = pd.DataFrame(metrics["history"]["train"])
        hist_val = pd.DataFrame(metrics["history"]["val"])

        left, right = st.columns(2)
        with left:
            st.markdown("**Accuracy per epoch**")
            chart_df = pd.DataFrame({
                "train_acc": hist_train["acc"].values,
                "val_acc": hist_val["acc"].values,
            })
            st.line_chart(chart_df, height=260)
        with right:
            st.markdown("**Loss per epoch**")
            chart_df2 = pd.DataFrame({
                "train_loss": hist_train["loss"].values,
                "val_loss": hist_val["loss"].values,
            })
            st.line_chart(chart_df2, height=260)

    st.markdown("---")
    st.markdown("### Last saved training run")
    metrics = load_metrics_if_any()
    if metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Val Acc", round(float(metrics.get("best_val_acc", 0.0)), 4))
        c2.metric("Test Acc (segment)", round(float(metrics.get("test_acc_segment", 0.0)), 4))
        c3.metric("Best Epoch", int(metrics.get("best_epoch", 0)))
        st.code(json.dumps(metrics.get("split", {}), indent=2, ensure_ascii=False))
    else:
        st.info("No metrics.json found yet. Train a model first.")

# -----------------------
# Page: Test & Evaluation
# -----------------------
elif page == "Test":
    require_dataset()

    st.markdown("### Evaluate the trained model on the test split")
    if not os.path.exists(BEST_MODEL_PATH):
        st.warning("No trained model found yet. Train first in Train.")
        st.stop()

    # Controls (these stay interactive; results persist via st.session_state)
    col1, col2, col3 = st.columns(3)
    top_k = col1.slider("Top-K segments for tree risk", 1, 12, 5, 1, key="eval_top_k")
    threshold = col2.slider("Risk threshold for infected", 0.0, 1.0, 0.50, 0.01, key="eval_threshold")
    batch_size = col3.selectbox("Inference batch size", [32, 64, 128], index=1, key="eval_batch")

    col4, col5, col6 = st.columns(3)
    seed = col4.number_input("Split seed", value=42, step=1, key="eval_seed")
    train_ratio = col5.slider("Train ratio (for split)", 0.5, 0.9, 0.70, 0.01, key="eval_train_ratio")
    val_ratio = col6.slider("Val ratio (for split)", 0.05, 0.3, 0.15, 0.01, key="eval_val_ratio")

    test_ratio = 1.0 - (float(train_ratio) + float(val_ratio))
    if test_ratio <= 0.0:
        st.error("Train+Val ratios too high, leaving no test data.")
        st.stop()
    st.caption(f"Computed test ratio: {test_ratio:.2f}")
    stratify = True  # UI option removed (always stratify)

    # Run evaluation button
    if st.button("🧪 Run evaluation", type="primary", key="eval_run_btn"):
        df = load_labels(LABELS_CSV)
        _, _, df_test = split_by_tree(
            df,
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            seed=int(seed),
            stratify=bool(stratify),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, audio_cfg, ckpt = load_model(BEST_MODEL_PATH, device=device)

        with st.spinner("Running inference on test split…"):
            seg_df = infer_segment_probs(
                model,
                df_files=df_test,
                dataset_dir=DATASET_DIR,
                audio_cfg=audio_cfg,
                device=device,
                batch_size=int(batch_size),
            )
            tree_df = aggregate_tree_risk(seg_df, top_k=int(top_k))

        # Save outputs
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tree_out = os.path.join(OUTPUTS_DIR, f"tree_risk_test_{ts}.csv")
        seg_out = os.path.join(OUTPUTS_DIR, f"segment_preds_test_{ts}.csv")
        tree_df.to_csv(tree_out, index=False)
        seg_df.to_csv(seg_out, index=False)

        # Persist results so the grid doesn't disappear on widget changes (Streamlit reruns)
        st.session_state["eval_tree_df"] = tree_df
        st.session_state["eval_seg_df"] = seg_df
        st.session_state["eval_paths"] = {"tree_out": tree_out, "seg_out": seg_out}
        st.session_state["eval_meta"] = {
            "ts": ts,
            "top_k": int(top_k),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "stratify": bool(stratify),
        }

        st.success("Evaluation complete ✅")
        st.caption(f"Saved: {tree_out}")
        st.caption(f"Saved: {seg_out}")

    # If we have results, keep showing them + keep them reactive
    if "eval_tree_df" in st.session_state and isinstance(st.session_state["eval_tree_df"], pd.DataFrame):
        tree_df = st.session_state["eval_tree_df"].copy()
        seg_df = st.session_state.get("eval_seg_df", pd.DataFrame()).copy()

        # Recompute metrics LIVE when threshold changes (no need to re-run inference)
        metrics = tree_level_metrics(tree_df, threshold=float(threshold))
        cm_df = tree_confusion_matrix_df(tree_df, threshold=float(threshold))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tree Acc", round(metrics["accuracy"], 4))
        c2.metric("Precision", round(metrics["precision"], 4))
        c3.metric("Recall", round(metrics["recall"], 4))
        c4.metric("F1", round(metrics["f1"], 4))

        st.markdown("#### Confusion matrix (tree-level)")
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("#### Top risky trees (test split)")
        st.dataframe(tree_df.sort_values("risk", ascending=False).head(20), use_container_width=True, height=320)

        # --- Farm layout grid (dynamic) ---
        st.markdown("## 🧩 Farm Layout Grid — each cell is a tree")
        loc_df = load_tree_locations()
        if loc_df is None:
            st.warning("tree_locations.csv not found or missing row/col. Put it at data/dataset/tree_locations.csv")
        else:
            # Default grid size from the CSV (max row/col)
            default_rows = int(loc_df["row"].max())
            default_cols = int(loc_df["col"].max())

            g1, g2, g3 = st.columns([1, 1, 2])
            grid_cols = g1.number_input("Grid columns", min_value=1, value=default_cols, step=1, key="grid_cols")
            grid_rows = g2.number_input("Grid rows", min_value=1, value=default_rows, step=1, key="grid_rows")
            auto_layout = g3.checkbox(
                "Auto-layout (override row/col) — place trees sequentially by tree_id",
                value=False,
                key="grid_auto_layout",
                help="Use this if you want a specific shape like 20×10 for 200 trees, regardless of saved row/col.",
            )

            grid_cols = int(grid_cols)
            grid_rows = int(grid_rows)

            if auto_layout:
                loc2 = loc_df.copy().sort_values("tree_id").reset_index(drop=True)
                idx = np.arange(len(loc2), dtype=int)
                loc2["row"] = (idx // grid_cols) + 1
                loc2["col"] = (idx % grid_cols) + 1
            else:
                loc2 = loc_df
                max_r = int(loc2["row"].max())
                max_c = int(loc2["col"].max())
                if grid_rows < max_r or grid_cols < max_c:
                    st.warning(
                        f"Your grid ({grid_rows}×{grid_cols}) is smaller than the positions in tree_locations.csv "
                        f"(needs at least {max_r}×{max_c}). Increase rows/cols or enable Auto-layout."
                    )

            render_farm_grid(loc2, tree_df, threshold=float(threshold), grid_rows=grid_rows, grid_cols=grid_cols)
    else:
        st.info("Click **Run evaluation** to generate results, then the farm grid will stay interactive.")


# -----------------------
# Page: Live Inference (single WAV)
# -----------------------
elif page == "Live Inference (WAV)":
    st.markdown("### Upload a WAV and get a risk estimate (prototype)")
    if not os.path.exists(BEST_MODEL_PATH):
        st.warning("No trained model found yet. Train first in Train.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    top_k = col1.slider("Top-K segments", 1, 12, 5, 1)
    threshold = col2.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)
    batch_size = col3.selectbox("Batch size", [32, 64, 128], index=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, audio_cfg, ckpt = load_model(BEST_MODEL_PATH, device=device)

    up = st.file_uploader("Upload WAV", type=["wav"])
    if up is not None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        tmp_path = os.path.join(OUTPUTS_DIR, "uploaded.wav")
        with open(tmp_path, "wb") as f:
            f.write(up.getbuffer())

        st.audio(tmp_path)

        with st.spinner("Running inference…"):
            seg_df = infer_segments_for_file(
                model,
                tmp_path,
                tree_id="uploaded",
                rel_path="uploaded.wav",
                audio_cfg=audio_cfg,
                device=device,
                batch_size=int(batch_size),
            )
            tree_df = aggregate_tree_risk(seg_df, top_k=int(top_k))
            risk = float(tree_df.iloc[0]["risk"]) if len(tree_df) else 0.0

        st.markdown("### Result")
        st.metric("Risk score", round(risk, 4))
        pred = "INFECTED" if risk >= float(threshold) else "HEALTHY"
        st.write("Decision:", f"**{pred}** (threshold={threshold})")
        st.write("Recommendation:", tree_df.iloc[0]["recommendation"] if len(tree_df) else "N/A")

        st.markdown("#### Segment probabilities")
        st.dataframe(seg_df[["seg_i", "p_infected"]], use_container_width=True, height=260)

        # Plot log-mel for visual explanation
        y, sr = wav_load(tmp_path, target_sr=audio_cfg.sr)
        logm = logmel_from_waveform(y, sr)
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.set_title("Log-Mel Spectrogram (uploaded WAV)")
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
        ax.imshow(logm, aspect="auto", origin="lower")
        st.pyplot(fig, clear_figure=True)

    # Demo video (centered) under the Live Inference page
    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
    demo_video_path = os.path.join(assets_dir, "live_demo.mp4")
    if os.path.exists(demo_video_path):
        b64v = _b64_file(demo_video_path)
        st.markdown(
            f"""
            <div style="display:flex;justify-content:center;align-items:center;">
              <div class="pg-media" style="max-width: 980px; width: 100%;">
                <div class="pg-media-frame">
                  <video class="pg-media-media" controls autoplay loop muted playsinline preload="auto">
                    <source src="data:video/mp4;base64,{b64v}" type="video/mp4" />
                  </video>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
