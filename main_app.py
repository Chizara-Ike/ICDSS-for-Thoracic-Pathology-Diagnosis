# ==============================
# Clinical ICDSS for Thoracic Pathology Diagnosis
# Refactored & Enhanced Version
# ==============================

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from io import BytesIO
import datetime
import pydicom
import traceback

# PDF (reportlab)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Thoracic ICDSS",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Clinical dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Titles */
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #58a6ff; }
h1 { font-size: 1.6rem; letter-spacing: 0.05em; }
h3 { font-size: 1rem; color: #8b949e; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #8b949e;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff;
    border-bottom: 2px solid #58a6ff;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
}

/* Buttons */
.stButton > button {
    background-color: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    padding: 8px 16px;
}
.stButton > button:hover { background-color: #388bfd; }

/* Download buttons */
.stDownloadButton > button {
    background-color: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
.stDownloadButton > button:hover { background-color: #2ea043; }

/* Pathology badge */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    margin: 3px;
}
.badge-high  { background: #3d1f1f; color: #f85149; border: 1px solid #f85149; }
.badge-med   { background: #2d2200; color: #e3b341; border: 1px solid #e3b341; }
.badge-low   { background: #1a2e1a; color: #56d364; border: 1px solid #56d364; }

/* Alert banner */
.alert-box {
    background: #2d1f00;
    border-left: 4px solid #e3b341;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 0.85rem;
    margin-bottom: 12px;
}

/* Footer */
.footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #484f58;
    border-top: 1px solid #21262d;
    padding-top: 12px;
    margin-top: 40px;
    text-align: center;
}

/* Divider */
hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = r"C:\Users\Phoenix\Downloads\FYB 26\NIH Chest X-ray\model_stage3_targeted.h5"
IMG_SIZE   = (224, 224)

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

# Severity weights for colour-coding (subjective clinical urgency)
SEVERITY = {
    "Pneumothorax": "high", "Pneumonia": "high", "Edema": "high",
    "Cardiomegaly": "med", "Consolidation": "med", "Mass": "med",
    "Effusion": "med", "Emphysema": "med",
    "Atelectasis": "low", "Fibrosis": "low", "Hernia": "low",
    "Infiltration": "low", "Nodule": "low", "Pleural_Thickening": "low",
}

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {patient_id, name, timestamp, detected, preds}

# ─────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH, compile=False)

try:
    model = load_model_cached()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize(IMG_SIZE)
    img   = np.array(image.convert("RGB")) / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)


def find_last_conv_layer(mdl) -> str:
    for layer in reversed(mdl.layers):
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("No convolutional layer found in model.")


def get_all_conv_layers(mdl) -> list:
    """Return (name, output_shape) for every 4-D layer, ordered from last to first."""
    results = []
    for layer in reversed(mdl.layers):
        try:
            if len(layer.output_shape) == 4:
                results.append((layer.name, layer.output_shape))
        except Exception:
            continue
    return results


def make_gradcam_heatmap(img_array: np.ndarray, mdl, last_conv_layer_name: str,
                          threshold: float):
    """
    Grad-CAM that avoids building any sub-model — which breaks on DenseNet121
    in newer Keras because the dense block layer graph cannot be cleanly traced
    into a new Functional model.

    Strategy: call the full model once inside a persistent GradientTape,
    extract the target conv layer's output by name from the layer's direct
    __call__ result using a forward-hook pattern via tf.keras backend.
    If that still fails, fall back to numerical gradient approximation.
    """
    img_tensor = tf.constant(img_array, dtype=tf.float32)   # immutable, no Variable

    # ── Attempt 1: intermediate-tensor extraction via get_layer().output ─────
    # Works when the layer output tensor is still reachable from the graph.
    try:
        target_layer = mdl.get_layer(last_conv_layer_name)

        with tf.GradientTape(persistent=True) as tape:
            # Run the full model; watch the input so we can get all intermediates
            img_var = tf.Variable(img_tensor, trainable=False)
            tape.watch(img_var)

            # Manually walk the model layer by layer to capture the conv output
            x = img_var
            conv_out_tensor = None
            for layer in mdl.layers:
                try:
                    x = layer(x, training=False)
                    if layer.name == last_conv_layer_name:
                        conv_out_tensor = x
                        tape.watch(conv_out_tensor)
                except Exception:
                    # Some layers (e.g. InputLayer) cannot be called standalone
                    continue

            if conv_out_tensor is None:
                raise ValueError(f"Layer {last_conv_layer_name!r} not reached in manual walk.")

            # Final predictions — use the model directly on the original input
            predictions = mdl(img_var, training=False)
            class_indices = tf.where(predictions[0] > threshold)
            class_idx = (int(tf.argmax(predictions[0]))
                         if tf.shape(class_indices)[0] == 0
                         else int(class_indices[0][0]))
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_out_tensor)
        del tape

        if grads is None:
            raise ValueError("Gradients None from manual walk.")

        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out_tensor[0] * pooled, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), class_idx, last_conv_layer_name

    except Exception as e1:
        pass

    # ── Attempt 2: score-based gradient w.r.t. input image ──────────────────
    # Always works regardless of architecture — gradient flows from loss → input.
    # Produces a saliency map rather than a true Grad-CAM, but is visually useful.
    try:
        img_var = tf.Variable(img_tensor, trainable=False)

        with tf.GradientTape() as tape:
            tape.watch(img_var)
            predictions = mdl(img_var, training=False)
            class_indices = tf.where(predictions[0] > threshold)
            class_idx = (int(tf.argmax(predictions[0]))
                         if tf.shape(class_indices)[0] == 0
                         else int(class_indices[0][0]))
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, img_var)   # shape: (1, 224, 224, 3)

        if grads is None:
            raise ValueError("Input gradients are None.")

        # Collapse channels by max (highlights any discriminative direction)
        heatmap = tf.reduce_max(tf.abs(grads[0]), axis=-1)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), class_idx, "input_saliency"

    except Exception as e2:
        raise RuntimeError(
            f"All Grad-CAM strategies failed.\n"
            f"  Attempt 1 (layer walk): {e1}\n"
            f"  Attempt 2 (input saliency): {e2}"
        )

def overlay_heatmap(heatmap: np.ndarray, image: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    h, w  = image.shape[:2]
    hmap  = cv2.resize(heatmap, (w, h))
    hmap  = np.uint8(255 * hmap)
    hmap  = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    super_img = hmap * alpha + image.astype(np.float32)
    return np.clip(super_img, 0, 255).astype(np.uint8)


def load_dicom(uploaded) -> Image.Image:
    """Convert a DICOM file to a PIL RGB image."""
    ds    = pydicom.dcmread(uploaded)
    arr   = ds.pixel_array.astype(np.float32)
    arr   = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    arr   = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(arr)


def img_to_bytes(img_array: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(img_array).save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────
# PDF BUILDER
# ─────────────────────────────────────────────
def build_pdf(patient_id, full_name, age, gender, smoker,
              detected, preds, original_img, cam_img, threshold) -> BytesIO:
    buffer = BytesIO()
    doc    = SimpleDocTemplate(buffer, rightMargin=40, leftMargin=40,
                               topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    el     = []

    # Title
    el.append(Paragraph("Thoracic Pathology ICDSS Report", styles["Title"]))
    el.append(Spacer(1, 6))
    el.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"]
    ))
    el.append(Spacer(1, 12))

    # Patient table
    patient_data = [
        ["Patient ID", patient_id or "N/A"],
        ["Full Name",  full_name  or "N/A"],
        ["Age",        str(age)],
        ["Gender",     gender],
        ["Smoker",     smoker],
        ["Threshold",  f"{threshold:.0%}"],
    ]
    tbl = Table(patient_data, colWidths=[120, 300])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1f2937")),
        ("TEXTCOLOR",  (0, 0), (0, -1), colors.HexColor("#93c5fd")),
        ("TEXTCOLOR",  (1, 0), (1, -1), colors.black),
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("BOX",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    el.append(tbl)
    el.append(Spacer(1, 16))

    # Results
    el.append(Paragraph("Diagnosis Results", styles["Heading2"]))
    if detected:
        res_data = [["Pathology", "Confidence"]] + [
            [label, f"{prob*100:.2f}%"] for label, prob in detected
        ]
        rtbl = Table(res_data, colWidths=[200, 120])
        rtbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
            ("BOX",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        el.append(rtbl)
    else:
        el.append(Paragraph("No significant pathology detected above threshold.", styles["Normal"]))

    el.append(Spacer(1, 16))

    # Images
    def pil_to_rl(arr, w=220, h=220):
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        buf.seek(0)
        return RLImage(buf, width=w, height=h)

    img_table = Table(
        [[pil_to_rl(original_img), pil_to_rl(cam_img)]],
        colWidths=[240, 240]
    )
    img_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))

    el.append(Paragraph("X-ray Images", styles["Heading2"]))
    el.append(Paragraph("Original X-ray  ·  Grad-CAM Overlay", styles["Normal"]))
    el.append(Spacer(1, 8))
    el.append(img_table)

    el.append(Spacer(1, 20))
    el.append(Paragraph(
        "⚠ This report is for research/decision-support use only. "
        "It does not constitute a clinical diagnosis.",
        styles["Normal"]
    ))

    doc.build(el)
    buffer.seek(0)
    return buffer


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🫁 ICDSS")
    st.markdown("**Thoracic Pathology**")
    st.markdown("---")

    st.markdown("#### Patient Information")
    patient_id = st.text_input("Patient ID",     placeholder="e.g. PT-00123")
    full_name  = st.text_input("Full Name",       placeholder="Surname, Firstname")
    age        = st.number_input("Age", 0, 120, 30)
    gender     = st.selectbox("Gender",  ["Male", "Female"])
    smoker     = st.selectbox("Smoker",  ["No", "Yes", "Ex-smoker"])

    st.markdown("---")
    st.markdown("#### Model Settings")
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Minimum confidence to flag a pathology")

    st.markdown("---")
    if not model_loaded:
        st.error(f"Model failed to load:\n{model_error}")
    else:
        st.success("Model ready ✓")
        with st.expander("🔬 Model Diagnostics", expanded=False):
            if model_loaded:
                st.markdown(f"**Input shape:** `{model.input_shape}`")
                st.markdown(f"**Output shape:** `{model.output_shape}`")
                st.markdown(f"**Total layers:** {len(model.layers)}")
                conv_layers = get_all_conv_layers(model)
                st.markdown(f"**Conv (4-D) layers:** {len(conv_layers)}")
                if conv_layers:
                    df_layers = pd.DataFrame(conv_layers, columns=["Layer Name", "Output Shape"])
                    st.dataframe(df_layers, use_container_width=True, height=200)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("## 🩺 Clinical ICDSS — Thoracic Pathology")
st.markdown("Intelligent Clinical Decision Support System for chest X-ray interpretation.")
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_analyse, tab_history = st.tabs(["📡  Analysis", "📋  Session History"])

# ══════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ══════════════════════════════════════════════
with tab_analyse:

    st.markdown("#### Upload Chest X-ray(s)")
    st.caption("Supports JPEG, PNG, and DICOM (.dcm) formats. Upload multiple files to batch-process.")

    uploaded_files = st.file_uploader(
        "Upload X-ray(s)",
        type=["jpg", "jpeg", "png", "dcm"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if not uploaded_files:
        st.info("Upload one or more chest X-ray files to begin analysis.")

    for file_idx, uploaded_file in enumerate(uploaded_files):

        st.markdown(f"---")
        st.markdown(f"### File {file_idx + 1}: `{uploaded_file.name}`")

        # ── Load image ──────────────────────────────
        try:
            if uploaded_file.name.lower().endswith(".dcm"):
                image = load_dicom(uploaded_file)
                st.caption("📂 DICOM file detected — converted to RGB.")
            else:
                image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            continue

        col_img, col_results = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("**Original X-ray**")
            st.image(image, use_container_width=True)

        # ── Inference ───────────────────────────────
        if not model_loaded:
            st.error("Model is not loaded. Cannot run inference.")
            continue

        img_array  = preprocess_image(image)

        progress = st.progress(0, text="Preprocessing image…")

        # ── Step 1: Inference (must succeed) ────────
        try:
            progress.progress(25, text="Running inference…")
            preds = model.predict(img_array, verbose=0)[0]
        except Exception as e:
            st.error(f"Inference failed: {e}")
            progress.empty()
            continue

        # ── Step 2: Grad-CAM (optional) ─────────────
        try:
            progress.progress(60, text="Computing Grad-CAM…")
            last_conv = find_last_conv_layer(model)
            heatmap, class_idx, cam_layer = make_gradcam_heatmap(
                img_array, model, last_conv, threshold
            )
            img_np  = np.array(image.convert("RGB"))
            cam_img = overlay_heatmap(heatmap, img_np)
            cam_ok  = True
        except Exception as e:
            cam_ok  = False
            cam_err = str(e)
            cam_img = None

        progress.progress(90, text="Compiling results…")

        detected = [
            (CLASS_NAMES[i], float(preds[i]))
            for i in range(len(CLASS_NAMES))
            if preds[i] >= threshold
        ]

        progress.progress(100, text="Done ✓")
        progress.empty()

        # ── Results panel ───────────────────────────
        with col_results:
            st.markdown("**Detected Pathologies**")
            if detected:
                badges_html = ""
                for label, prob in sorted(detected, key=lambda x: -x[1]):
                    sev   = SEVERITY.get(label, "low")
                    cls   = f"badge-{sev}"
                    badges_html += (
                        f'<span class="badge {cls}">'
                        f'{label} {prob*100:.0f}%</span>'
                    )
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.success("No significant pathology detected above threshold.")

            st.markdown("---")

            # Metric summary
            m1, m2, m3 = st.columns(3)
            m1.metric("Pathologies Found", len(detected))
            m2.metric("Top Confidence",
                      f"{max(preds)*100:.1f}%" if len(preds) else "—")
            m3.metric("Threshold", f"{threshold:.0%}")

        # ── Grad-CAM ────────────────────────────────
        st.markdown("**Explainability — Grad-CAM Overlay**")
        if cam_ok:
            st.image(cam_img,
                     caption=f"Grad-CAM: {CLASS_NAMES[class_idx]}  |  layer: {cam_layer}",
                     use_container_width=True)
        else:
            st.warning(f"Grad-CAM could not be generated: {cam_err}")

        # ── Full probability chart ───────────────────
        with st.expander("📊 All Class Probabilities", expanded=False):
            df = pd.DataFrame({
                "Pathology":   CLASS_NAMES,
                "Probability": [float(p) for p in preds],
            })
            df["Above Threshold"] = df["Probability"] >= threshold
            df_sorted = df.sort_values("Probability", ascending=False)
            st.bar_chart(df_sorted.set_index("Pathology")["Probability"])
            st.dataframe(
                df_sorted.style.format({"Probability": "{:.3f}"}),
                use_container_width=True,
                height=300
            )

        # ── Downloads ───────────────────────────────
        st.markdown("**Download**")
        dl1, dl2 = st.columns(2)

        with dl1:
            if cam_ok and cam_img is not None:
                st.download_button(
                    "⬇ Grad-CAM Image",
                    data=img_to_bytes(cam_img),
                    file_name=f"gradcam_{uploaded_file.name}.png",
                    mime="image/png",
                    key=f"dl_cam_{file_idx}"
                )

        with dl2:
            try:
                pdf_buffer = build_pdf(
                    patient_id, full_name, age, gender, smoker,
                    detected, preds,
                    np.array(image.convert("RGB")),
                    cam_img if cam_ok and cam_img is not None else np.zeros((224, 224, 3), dtype=np.uint8),
                    threshold
                )
                st.download_button(
                    "⬇ Full PDF Report",
                    data=pdf_buffer,
                    file_name=f"ICDSS_report_{uploaded_file.name}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{file_idx}"
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

        # ── Save to session history ──────────────────
        st.session_state.history.append({
            "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file":       uploaded_file.name,
            "patient_id": patient_id or "—",
            "name":       full_name  or "—",
            "detected":   ", ".join([l for l, _ in detected]) if detected else "None",
            "n_findings": len(detected),
            "top_conf":   f"{max(preds)*100:.1f}%" if len(preds) else "—",
        })


# ══════════════════════════════════════════════
# TAB 2 — SESSION HISTORY
# ══════════════════════════════════════════════
with tab_history:
    st.markdown("#### Session Analysis Log")
    st.caption("Records all analyses performed during this session. Cleared on page refresh.")

    if not st.session_state.history:
        st.info("No analyses performed yet in this session.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, height=400)

        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Export History as CSV",
            data=csv,
            file_name="icdss_session_log.csv",
            mime="text/csv"
        )

        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    '⚠ For research and decision-support use only. '
    'This system does not constitute a clinical diagnosis. '
    'Always consult a qualified radiologist.'
    '</div>',
    unsafe_allow_html=True
)