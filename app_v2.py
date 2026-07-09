import os
import uuid
import streamlit as st

from streamlit_mic_recorder import mic_recorder
from audio_pipeline_v2 import run_audio_pipeline_from_path

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

# =========================
# CONFIG
# =========================
AUDIO_DIR = "outputs/recorded_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


RAINBOW_TEXT = """
When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.
The rainbow is a division of white light into many beautiful colors. These take the
shape of a long round arch, with its path high above, and its two ends apparently
beyond the horizon. There is , according to legend, a boiling pot of gold at one end.
People look, but no one ever finds it. When a man looks for something beyond his
reach, his friends say he is looking for the pot of gold at the end of the rainbow.
Throughout the centuries people have explained the rainbow in various ways. Some
have accepted it as a miracle without physical explanation. To the Hebrews it was a
token that there would be no more universal floods. The Greeks used to imagine that
it was a sign from the gods to foretell war or heavy rain.
"""


# =========================
# HELPERS
# =========================
def save_audio(audio_bytes):
    path = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.wav")
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


def initialise_state():
    if "page" not in st.session_state:
        st.session_state.page = "record"

    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None

    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

    if "results" not in st.session_state:
        st.session_state.results = None


initialise_state()


# =========================
# PAGE 1: RECORDING
# =========================
if st.session_state.page == "record":

    #st.title("Pulmono.AI")
    #st.subheader("Voice-based lung function screening")

    st.info(
        "Please read the paragraph below aloud. "
        "The recording should be 30 seconds. "
        "Only the 15th to 25th second will be processed."
    )

    st.markdown(
        f"""
        <div style="
            font-size: 20px;
            line-height: 1.8;
            font-weight: 600;
            border: 2px;
            background: white;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">
        {RAINBOW_TEXT}
        """,
        unsafe_allow_html=True
    )

    st.warning("Click record, read for 30 seconds, then stop recording.")

    audio = mic_recorder(
        start_prompt="Start 30-second recording",
        stop_prompt="Stop recording",
        just_once=False,
        use_container_width=True,
        key="recorder"
    )

    if audio is not None and st.session_state.audio_path is None:
        audio_path = save_audio(audio["bytes"])
        st.session_state.audio_path = audio_path
        st.session_state.audio_bytes = audio["bytes"]
        st.success("Recording saved.")

    if st.session_state.audio_path is not None:
        st.audio(st.session_state.audio_bytes, format="audio/wav")

        if st.button("Next: Enter phenotype values"):
            st.session_state.page = "phenotype"
            st.rerun()


# =========================
# PAGE 2: PHENOTYPE
# =========================
elif st.session_state.page == "phenotype":

    st.title("Phenotype Information")

    st.info("Please enter the phenotype values. These are required for prediction.")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    height = st.number_input(
        "Height (cm)",
        min_value=50,
        max_value=250,
        value=165
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=20,
        max_value=250,
        value=70
    )

    sex = st.radio(
        "Sex",
        ["Female", "Male"]
    )

    noise_mode = st.radio(
        "Recording environment",
        ["Quiet environment", "Noisy environment"]
    )

    col_back, col_run = st.columns(2)

    with col_back:
        if st.button("Back to recording"):
            st.session_state.page = "record"
            st.rerun()

    with col_run:
        if st.button("Run analysis"):
            patient_info = {
                "age": age,
                "height": height,
                "weight": weight,
                "is_male": 1 if sex == "Male" else 0,
                "is_female": 1 if sex == "Female" else 0,
            }

            with st.spinner("Processing recording..."):
                results = run_audio_pipeline_from_path(
                    audio_path=st.session_state.audio_path,
                    noise_mode=noise_mode,
                    patient_info=patient_info,
                    crop_start=15,
                    crop_end=25
                )

            st.session_state.results = results
            st.session_state.page = "results"
            st.rerun()


# =========================
# PAGE 3: RESULTS
# =========================
elif st.session_state.page == "results":

    st.title("Results")

    results = st.session_state.results

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted FEV1", f"{results['fev1']:.1f}%")
        st.write("Status:", results["fev1_binary_label"])
        st.write("Severity:", results["fev1_multiclass_label"])
        st.caption(f"Confidence: {results['fev1_binary_confidence']:.1f}%")

    with col2:
        st.metric("Predicted FEV1/FVC", f"{results['fev1_fvc']:.1f}%")
        st.write("Status:", results["fev1_fvc_binary_label"])
        st.write("Severity:", results["fev1_fvc_multiclass_label"])
        st.caption(f"Confidence: {results['fev1_fvc_binary_confidence']:.1f}%")

    st.warning(
        "Research prototype only. This tool does not replace clinical spirometry."
    )

    if st.button("Start again"):
        st.session_state.page = "record"
        st.session_state.audio_path = None
        st.session_state.audio_bytes = None
        st.session_state.results = None
        st.rerun()