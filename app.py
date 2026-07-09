import streamlit as st

st.set_page_config(
    page_title="Pulmono.AI",
    page_icon="🫁",
    layout="wide"
)

st.title("Pulmono.AI")
st.subheader("Voice-based lung function screening")

st.info(
    "This tool is for research and screening purposes only. "
    "It does not replace clinical spirometry."
)

uploaded_file = st.file_uploader(
    "Upload a voice recording",
    type=["wav", "mp3", "m4a"]
)

noise_mode = st.radio(
    "Recording environment",
    ["Quiet environment", "Noisy environment"]
)

age = st.number_input("Age", min_value=18, max_value=100, value=30)
height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
weight = st.number_input("Weight (kg)", min_value=20, max_value=250, value=70)

sex = st.radio("Sex", ["Female", "Male"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Run analysis"):
        with st.spinner("Processing audio and running models..."):

            # Later:
            # audio_path = save_uploaded_file(uploaded_file)
            # if noise_mode == "Noisy environment":
            #     audio_path = denoise_audio(audio_path)
            # segments = run_bigru_segmentation(audio_path)
            # features = extract_features(audio_path, segments)
            # results = predict_all_models(features)

            # Temporary demo results
            results = {
                "fev1": 52.9,
                "fev1_fvc": 48.1,
                "fev1_binary": "Abnormal",
                "fev1_binary_confidence": 97.7,
                "fev1_multiclass": "Reduced FEV1",
                "fev1_fvc_binary": "Abnormal",
                "fev1_fvc_binary_confidence": 95.4,
                "fev1_fvc_multiclass": "Moderate obstruction"
            }

        st.success("Analysis complete")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted FEV1", f"{results['fev1']:.1f}%")
            st.write("FEV1 status:", results["fev1_binary"])
            st.write("Severity:", results["fev1_multiclass"])
            st.progress(results["fev1_binary_confidence"] / 100)
            st.caption(f"Confidence: {results['fev1_binary_confidence']:.1f}%")

        with col2:
            st.metric("Predicted FEV1/FVC", f"{results['fev1_fvc']:.1f}%")
            st.write("Obstruction status:", results["fev1_fvc_binary"])
            st.write("Severity:", results["fev1_fvc_multiclass"])
            st.progress(results["fev1_fvc_binary_confidence"] / 100)
            st.caption(f"Confidence: {results['fev1_fvc_binary_confidence']:.1f}%")

        st.warning(
            "Predictions should be interpreted as screening outputs only. "
            "Clinical confirmation with spirometry is required."
        )