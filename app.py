import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="DeepFake Detection",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ DeepFake Image Detection System")

st.write(
    """
Upload a facial image to determine whether it is

- ✅ Real Image
- ⚠️ AI Generated (DeepFake)
"""
)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/deepfake_model.keras")
    return model


model = load_model()

class_names = ["Fake", "Real"]


def preprocess_image(image):

    image = image.resize((224, 224))

    image = np.array(image)

    image = tf.keras.applications.efficientnet.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image


uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:

        if st.button("Detect"):

            with st.spinner("Analyzing Image..."):

                img = preprocess_image(image)

                prediction = model.predict(img, verbose=0)[0][0]

                if prediction > 0.5:
                    label = "Real"
                    confidence = prediction
                else:
                    label = "Fake"
                    confidence = 1 - prediction

                st.subheader("Prediction")

                if label == "Real":

                    st.success(f"✅ {label}")

                else:

                    st.error(f"⚠️ {label}")

                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )

                st.progress(float(confidence))


st.divider()

st.header("Model Performance")

tab1, tab2, tab3 = st.tabs(
    [
        "Training Curve",
        "Confusion Matrix",
        "ROC Curve"
    ]
)

with tab1:

    if os.path.exists("images/training_curves.png"):
        st.image("images/training_curves.png")

with tab2:

    if os.path.exists("images/confusion_matrix.png"):
        st.image("images/confusion_matrix.png")

with tab3:

    if os.path.exists("images/roc_curve.png"):
        st.image("images/roc_curve.png")


st.sidebar.title("About")

st.sidebar.info(
    """
DeepFake Detection using

• TensorFlow

• EfficientNetB0

• Streamlit

Upload an image and the model predicts whether it is **Real** or **Fake**.
"""
)