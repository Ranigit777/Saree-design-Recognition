"""
DeepLure / AIE-CASE: Color-Invariant Saree Design Recognition
Interactive Streamlit Application.

Demonstrates:
1. Gallery Retrieval: Top-K visual similarity search
2. Pairwise Verification: Pattern-family proxy matching with validation-selected threshold
3. Color Robustness: Real-time synthetic palette manipulation and embedding stability
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

# Ensure src/ is on sys.path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from augmentations import build_eval_transform
from config import (
    BEST_METRIC_MODEL_PATH,
    EMBEDDINGS_DIR,
    LABEL_TO_PATTERN_FAMILY,
    VERIFICATION_THRESHOLD_PATH,
)
from loaders import get_device
from model import SareeEmbeddingModel


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Saree Design Recognition | DeepLure AIE-CASE",
    page_icon="🥻",
    layout="wide",
)

st.title("🥻 Color-Invariant Saree Design Recognition — Prototype")
st.caption("Notice: Current evaluation operates on a pattern-family proxy because verified fine-grained design IDs are unavailable in the dataset.")


# ---------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_metric_model():
    device = get_device()
    if not BEST_METRIC_MODEL_PATH.is_file():
        return None, f"Model checkpoint not found at: {BEST_METRIC_MODEL_PATH}"
    try:
        ckpt = torch.load(BEST_METRIC_MODEL_PATH, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        emb_dim = int(cfg.get("embedding_dim", 256))
        dropout = float(cfg.get("dropout", 0.2))
        model = SareeEmbeddingModel(
            embedding_dim=emb_dim,
            pretrained=False,
            dropout=dropout,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model, None
    except Exception as exc:
        return None, f"Failed to load checkpoint: {str(exc)}"


@st.cache_resource
def load_gallery():
    emb_path = EMBEDDINGS_DIR / "train_embeddings.npy"
    meta_path = EMBEDDINGS_DIR / "train_metadata.csv"
    if not emb_path.is_file() or not meta_path.is_file():
        return None, None, f"Gallery files missing in {EMBEDDINGS_DIR}."
    try:
        embs = np.load(emb_path)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / norms
        meta = pd.read_csv(meta_path)
        return embs, meta, None
    except Exception as exc:
        return None, None, f"Error loading gallery: {str(exc)}"


@st.cache_data
def load_threshold():
    if VERIFICATION_THRESHOLD_PATH.is_file():
        try:
            with open(VERIFICATION_THRESHOLD_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return float(data.get("selected_threshold", 0.5)), data
        except Exception:
            pass
    return 0.5, {}


model, model_err = load_metric_model()
gallery_embs, gallery_meta, gallery_err = load_gallery()
threshold, thresh_data = load_threshold()
eval_tfm = build_eval_transform()


def embed_image(im: Image.Image) -> np.ndarray:
    device = get_device()
    t = eval_tfm(im.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t, return_logits=True)
        emb = out["embedding"].cpu().numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        logits = out.get("logits", None)
        pred_label = int(logits.argmax(dim=1).item()) if logits is not None else None
        pred_family = LABEL_TO_PATTERN_FAMILY.get(pred_label, "Unknown") if pred_label is not None else "Unknown"
    return emb.flatten(), pred_family


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Project Specification")
    st.markdown("**Architecture:** EfficientNet-B0")
    st.markdown("**Embedding:** 256 dimensions (L2-normalized)")
    st.markdown("**Similarity Metric:** Cosine similarity")
    st.markdown("**Training Task:** Pattern-family proxy metric learning")
    st.markdown("**Dataset Total:** 1,633 images")
    st.markdown("**Proxy Classes:** Banarasi, Bandhani, Ikat, Pichwai")
    st.markdown(f"**Verification Threshold:** `{threshold:.4f}` (Val-calibrated)")

    st.markdown("---")
    st.warning(
        "**Important Limitation:**\n"
        "No verified fine-grained design IDs or authentic same-design/different-color pairs "
        "exist in the dataset. Predictions describe coarse motif pattern families, not unique design identities."
    )


# ---------------------------------------------------------
# System Check
# ---------------------------------------------------------
if model_err:
    st.error(f"⚠️ {model_err}")
    st.stop()
if gallery_err:
    st.error(f"⚠️ {gallery_err}")
    st.stop()


# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Mode 1: Image Retrieval",
    "⚖️ Mode 2: Pairwise Verification",
    "🎨 Mode 3: Color Robustness",
])


# =========================================================
# TAB 1: IMAGE RETRIEVAL
# =========================================================
with tab1:
    st.subheader("Gallery Search & Identification")
    st.write("Upload a saree image to find the closest matches in the reference gallery.")

    col_q1, col_q2 = st.columns([1, 2])
    with col_q1:
        top_k = st.selectbox("Top-K Results:", options=[1, 3, 5, 10], index=2)
        uploaded_file = st.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png", "webp"], key="retrieval_upload")
        if uploaded_file is not None:
            query_im = Image.open(uploaded_file).convert("RGB")
            st.image(query_im, caption="Query Image", use_container_width=True)

    with col_q2:
        if uploaded_file is not None:
            with st.spinner("Extracting 256-D embedding and querying gallery..."):
                q_emb, pred_family = embed_image(query_im)
                sims = (q_emb @ gallery_embs.T).flatten()
                top_indices = np.argsort(-sims)[:top_k]

            st.success(f"Query processed. Predicted pattern family (proxy): **{pred_family}**")
            st.markdown(f"### Top {top_k} Retrieved Matches")

            cols = st.columns(top_k if top_k <= 5 else 5)
            for i, idx in enumerate(top_indices):
                col = cols[i % 5]
                row = gallery_meta.iloc[idx]
                sim_score = float(sims[idx])
                img_path = PROJECT_ROOT / row["image_path"]

                with col:
                    if img_path.is_file():
                        match_im = Image.open(img_path).convert("RGB")
                        st.image(match_im, use_container_width=True)
                    else:
                        st.write("Image file missing")
                    st.caption(
                        f"**Rank {i+1}**\n\n"
                        f"Similarity: `{sim_score:.4f}`\n\n"
                        f"Family: **{row.get('pattern_family', 'Unknown')}**"
                    )
        else:
            st.info("Please upload an image from the sidebar or select a sample image to begin search.")


# =========================================================
# TAB 2: PAIRWISE VERIFICATION
# =========================================================
with tab2:
    st.subheader("Pattern-Family Proxy Verification")
    st.caption("Determine whether two saree images represent the same pattern family.")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        file_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png", "webp"], key="verif_a")
        if file_a is not None:
            im_a = Image.open(file_a).convert("RGB")
            st.image(im_a, caption="Image A", use_container_width=True)

    with col_v2:
        file_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png", "webp"], key="verif_b")
        if file_b is not None:
            im_b = Image.open(file_b).convert("RGB")
            st.image(im_b, caption="Image B", use_container_width=True)

    if file_a is not None and file_b is not None:
        with st.spinner("Computing pairwise cosine similarity..."):
            emb_a, fam_a = embed_image(im_a)
            emb_b, fam_b = embed_image(im_b)
            cos_sim = float(np.dot(emb_a, emb_b))
            is_same = cos_sim >= threshold
            margin = cos_sim - threshold

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Cosine Similarity", f"{cos_sim:.4f}")
        res_col2.metric("Validation Threshold", f"{threshold:.4f}")
        res_col3.metric("Decision Margin", f"{margin:+.4f}")

        if is_same:
            st.success("✅ **Proxy Verification Result: SAME PATTERN FAMILY**")
        else:
            st.error("❌ **Proxy Verification Result: DIFFERENT PATTERN FAMILY**")

        st.info(
            f"Image A predicted family: **{fam_a}** | Image B predicted family: **{fam_b}**\n\n"
            "*Notice: This verifies pattern-family consistency only. True same-design verification "
            "requires fine-grained design identifiers.*"
        )


# =========================================================
# TAB 3: COLOR ROBUSTNESS
# =========================================================
with tab3:
    st.subheader("Color Robustness & Palette Perturbation")
    st.caption("Apply controlled synthetic color transformations to inspect embedding stability.")

    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        c_file = st.file_uploader("Upload Saree Image", type=["jpg", "jpeg", "png", "webp"], key="color_upload")
        if c_file is not None:
            base_img = Image.open(c_file).convert("RGB")
        else:
            # Load default test sample if available
            test_csv = PROJECT_ROOT / "outputs/results/test_manifest.csv"
            if test_csv.is_file():
                df_test = pd.read_csv(test_csv)
                base_img = Image.open(PROJECT_ROOT / df_test.iloc[0]["image_path"]).convert("RGB")
            else:
                base_img = Image.new("RGB", (224, 224), color=(180, 50, 50))

        st.write("#### Color Controls")
        b_val = st.slider("Brightness:", 0.2, 2.0, 1.0, 0.1)
        c_val = st.slider("Contrast:", 0.2, 2.0, 1.0, 0.1)
        s_val = st.slider("Saturation:", 0.0, 3.0, 1.0, 0.1)
        h_val = st.slider("Hue Shift:", -0.5, 0.5, 0.0, 0.05)
        to_gray = st.checkbox("Convert to Grayscale (Full Desaturation)")
        swap_channels = st.checkbox("Swap Color Channels (RGB -> BGR)")

    # Apply transformations
    tfm_img = base_img.copy()
    if to_gray:
        tfm_img = TF.to_grayscale(tfm_img, num_output_channels=3)
    if b_val != 1.0:
        tfm_img = TF.adjust_brightness(tfm_img, b_val)
    if c_val != 1.0:
        tfm_img = TF.adjust_contrast(tfm_img, c_val)
    if s_val != 1.0 and not to_gray:
        tfm_img = TF.adjust_saturation(tfm_img, s_val)
    if h_val != 0.0 and not to_gray:
        tfm_img = TF.adjust_hue(tfm_img, h_val)
    if swap_channels:
        r, g, b = tfm_img.split()
        tfm_img = Image.merge("RGB", (b, g, r))

    with col_c2:
        st.write("#### Comparison")
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            st.image(base_img, caption="Original Image", use_container_width=True)
        with cmp2:
            st.image(tfm_img, caption="Transformed Image", use_container_width=True)

        # Compute embedding similarity
        orig_emb, orig_fam = embed_image(base_img)
        tfm_emb, tfm_fam = embed_image(tfm_img)
        stability_sim = float(np.dot(orig_emb, tfm_emb))

        st.metric("Embedding Cosine Similarity (Stability)", f"{stability_sim:.4f}")
        st.write(f"Original predicted family: **{orig_fam}**")
        st.write(f"Transformed predicted family: **{tfm_fam}**")

        if orig_fam == tfm_fam:
            st.success("Representation is stable under this transformation.")
        else:
            st.warning("Pattern-family prediction changed under this severe color shift.")

    st.markdown("---")
    st.info(
        "ℹ️ **Scientific Disclaimer:**\n"
        "These are synthetic color transformations used to evaluate representation robustness to mathematical palette changes. "
        "They are not ground-truth same-design / different-colorway pairs."
    )
