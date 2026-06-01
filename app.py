import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="centered"
)

# =====================================
# LOAD MODEL
# =====================================

MODEL_PATH = "model/deepfake_detector.h5"

@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model

try:
    model = load_my_model()
    st.success("✅ Model Loaded Successfully")

except Exception as e:

    st.error(f"❌ Model Loading Error:\n{e}")
    st.stop()

# =====================================
# TITLE
# =====================================

st.title("🔍 Deepfake Detection System")

st.write(
    "Upload an image and the AI will detect whether it is REAL or FAKE."
)

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDICTION
# =====================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        img = np.array(image)

        # RGBA -> RGB
        if len(img.shape) == 3 and img.shape[2] == 4:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGBA2RGB
            )

        # Grayscale -> RGB
        if len(img.shape) == 2:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2RGB
            )

    except Exception as e:

        st.error(f"Image Error: {e}")
        st.stop()

    if st.button("Predict"):

        try:

            processed = cv2.resize(
                img,
                (224, 224)
            )

            processed = processed.astype(
                np.float32
            )

            processed = tf.keras.applications.efficientnet.preprocess_input(
                processed
            )

            processed = np.expand_dims(
                processed,
                axis=0
            )

            prediction = model.predict(
                processed,
                verbose=0
            )[0][0]

            st.write(
                f"Prediction Score: {prediction:.4f}"
            )

            # Dataset classes:
            # fake = 0
            # real = 1

            if prediction < 0.5:

                confidence = (1 - prediction) * 100

                st.error("❌ FAKE IMAGE")

                st.write(
                    f"Confidence: {confidence:.2f}%"
                )

            else:

                confidence = prediction * 100

                st.success("✅ REAL IMAGE")

                st.write(
                    f"Confidence: {confidence:.2f}%"
                )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )
