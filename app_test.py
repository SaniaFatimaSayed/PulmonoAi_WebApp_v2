import streamlit as st
from audio_pipeline import run_audio_pipeline

st.set_page_config(page_title="Pulmono.AI", page_icon="🫁", layout="wide")

st.markdown("""
<style>
.header-container {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}

.logo {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    object-fit: cover;
}

.title {
    font-size: 42px;
    font-weight: 700;
    margin: 0;
}

.subtitle{
    color:gray;
    font-size:18px;
    margin-top:-5px;
}
</style>
""", unsafe_allow_html=True)

import base64

with open("Pulmono.AI.png", "rb") as f:
    img = base64.b64encode(f.read()).decode()

st.markdown(
f"""
<div class="header-container">

<img src="data:image/png;base64,{img}"
class="logo">

<div>

<div class="title">
Pulmono.AI
</div>

<div class="subtitle">
Voice-based Lung Function Screening
</div>

</div>

</div>
""",
unsafe_allow_html=True
)


st.info("Research prototype only. This tool does not replace clinical spirometry.")

uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

noise_mode = st.radio(
    "Recording environment",
    ["Quiet environment", "Noisy environment"]
)

age = st.number_input("Age", 18, 100, 30)
height = st.number_input("Height (cm)", 50, 250, 165)
weight = st.number_input("Weight (kg)", 40, 250, 70)
sex = st.radio("Sex", ["Female", "Male"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Run analysis"):
        with st.spinner("Running Pulmono.AI pipeline..."):

            patient_info = {
                "age": age,
                "height": height,
                "weight": weight,
                "is_male": 1 if sex == "Male" else 0,
                "is_female": 1 if sex == "Female" else 0,
            }

            results = run_audio_pipeline(
                uploaded_file=uploaded_file,
                noise_mode=noise_mode,
                patient_info=patient_info
            )

        st.success("Analysis complete")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted FEV1", f"{results['fev1']:.1f}%")
            st.write("Status:", results["fev1_binary_label"])
            st.write("Severity:", results["fev1_multiclass_label"])
            st.caption(f"Binary confidence: {results['fev1_binary_confidence']:.1f}%")

        with col2:
            st.metric("Predicted FEV1/FVC", f"{results['fev1_fvc']:.1f}%")
            st.write("Status:", results["fev1_fvc_binary_label"])
            st.write("Severity:", results["fev1_fvc_multiclass_label"])
            st.caption(f"Binary confidence: {results['fev1_fvc_binary_confidence']:.1f}%")