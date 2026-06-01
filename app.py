import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# =====================================
# LOAD MODEL
# =====================================

MODEL_PATH = r"model/deepfake_detector.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Deepfake Detection System")

st.write(
    "Upload an image and the model will predict whether it is REAL or FAKE."
)

# =====================================
# IMAGE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDICTION
# =====================================

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        # --------------------------
        # PREPROCESS
        # --------------------------

        processed = cv2.resize(
            img,
            (224, 224)
        )

        processed = cv2.cvtColor(
            processed,
            cv2.COLOR_BGR2RGB
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

        # --------------------------
        # PREDICT
        # --------------------------

        prediction = model.predict(
            processed,
            verbose=0
        )[0][0]

        st.write(
            f"Raw Prediction Score: {prediction:.4f}"
        )

        # TensorFlow labels:
        # fake = 0
        # real = 1

        if prediction < 0.5:

            confidence = (1 - prediction) * 100

            st.error(
                f"FAKE IMAGE ❌"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

        else:

            confidence = prediction * 100

            st.success(
                f"REAL IMAGE ✅"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )