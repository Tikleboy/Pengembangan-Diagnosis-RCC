import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Diagnosis Renal Cell Carcinoma")
st.subheader("Upload Gambar CT SCAN Ginjal")


# LOAD 2 MODEL
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model("model/Model_CNN.h5")
    resnet_model = tf.keras.models.load_model("model/Model_ResNet50.h5")
    return cnn_model, resnet_model

cnn_model, resnet_model = load_models()

# FUNGSI PREPROCESS
def preprocess(image_data):
    img = image_data.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# FUNGSI DIAGNOSIS
def predict(model, image_data):
    img_array = preprocess(image_data)

    prediction = model.predict(img_array, verbose=0)

    label = "Normal" if prediction[0][0] > 0.5 else "Tumor"

    confidence = (
        prediction[0][0]
        if prediction[0][0] > 0.5
        else 1 - prediction[0][0]
    )

    return label, confidence

# UPLOAD GAMBAR
uploaded_file = st.file_uploader(
    "Upload Gambar CT Scan Ginjal",
    type=["jpg", "jpeg", "png"]
)

# HASIL DIAGNOSIS
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang Diupload",
        use_column_width=True
    )

    with st.spinner("Menganalisis..."):

        # Diagnosis CNN
        cnn_label, cnn_conf = predict(cnn_model, image)

        # Diagnosis ResNet50
        resnet_label, resnet_conf = predict(resnet_model, image)

    st.markdown("## Hasil Diagnosis")

    # CNN
    st.success(f"**CNN Model : {cnn_label}**")
    st.write(f"Confidence Score CNN : **{cnn_conf:.2%}**")

    st.markdown("---")

    # ResNet50
    st.success(f"**ResNet50 Model : {resnet_label}**")
    st.write(f"Confidence Score ResNet50 : **{resnet_conf:.2%}**")

# DISCLAIMER
st.markdown("---")
st.info(
    "**Disclaimer:** Hasil diagnosis ini bersifat informatif "
    "dan bertujuan sebagai pendukung keputusan medis. "
    "Konsultasikan hasil diagnosis anda kepada tenaga medis profesional."
)