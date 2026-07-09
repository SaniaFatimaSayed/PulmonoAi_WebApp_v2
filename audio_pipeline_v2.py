import os
import librosa
import soundfile as sf

from denoiser_module import denoise_audio
from bigru_segmenter import run_bigru_segmentation
from feature_extraction import extract_features_from_audio
from inference import predict_all_models


def crop_audio(audio_path, start_sec=15, end_sec=25, output_dir="outputs/cropped_audio"):
    os.makedirs(output_dir, exist_ok=True)

    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    start = int(start_sec * sr)
    end = int(end_sec * sr)

    y_crop = y[start:end]

    output_path = os.path.join(
        output_dir,
        os.path.basename(audio_path).replace(".wav", "_15_25s.wav")
    )

    sf.write(output_path, y_crop, sr)

    return output_path


def run_audio_pipeline_from_path(
    audio_path,
    noise_mode,
    patient_info,
    crop_start=15,
    crop_end=25
):
    audio_path = crop_audio(
        audio_path,
        start_sec=crop_start,
        end_sec=crop_end
    )

    if noise_mode == "Noisy environment":
        audio_path = denoise_audio(audio_path)

    segments = run_bigru_segmentation(audio_path)

    features = extract_features_from_audio(
        audio_path=audio_path,
        segments=segments,
        patient_id="uploaded_recording"
    )

    features.update(patient_info)

    results = predict_all_models(features)

    return results