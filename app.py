import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.keras", compile=False)
    meta = json.load(open("class_names.json"))
    return model, meta

model, meta = load_model()
CLASSES, IMG_SIZE = meta["classes"], tuple(meta["img_size"])

st.title("🧠 Brain Tumor MRI Classification")
st.caption(f"Model: {meta['best_model']} · Classes: {', '.join(c.replace('_',' ').title() for c in CLASSES)}")
st.warning("For educational use only. This is not a medical diagnosis.")

file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_container_width=True)

    x = np.array(img.resize(IMG_SIZE), dtype=np.float32)[None, ...]   # raw 0–255; model normalizes internally
    with st.spinner("Analyzing..."):
        probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASSES[idx].replace("_", " ").title()

    if CLASSES[idx] == "no_tumor":
        st.success(f"Prediction: **{label}** ({probs[idx]*100:.2f}% confidence)")
    else:
        st.error(f"Prediction: **{label}** ({probs[idx]*100:.2f}% confidence)")

    st.subheader("Confidence for each class")
    for c, p in sorted(zip(CLASSES, probs), key=lambda t: -t[1]):
        st.write(f"{c.replace('_',' ').title()}: {p*100:.2f}%")
        st.progress(float(p))