import numpy as np
import librosa


SR = 44100


def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else 0.0


def safe_std(values):
    return float(np.std(values)) if len(values) > 0 else 0.0


def get_mean_amplitude(y):
    if y is None or len(y) == 0:
        return 0.0
    return float(np.mean(np.abs(y)))


def safe_feature_mean(values):
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if len(values) > 0 else 0.0


def extract_single_segment_features(y, sr):
    if y is None or len(y) == 0:
        return {
            "roll85": 0.0,
            "roll95": 0.0,
            "zcr": 0.0,
            "rms": 0.0,
            "spectral_centroid": 0.0,
            "flatness": 0.0,
            "spectral_bandwidth": 0.0,
            "mean_amp": 0.0,
            "spectral_contrast": 0.0,
        }

    return {
        "zcr": float(np.mean(librosa.feature.zero_crossing_rate(
            y=y,
            frame_length=2048,
            hop_length=512
        ))),

        "roll85": float(np.mean(librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            roll_percent=0.85
        ))),

        "roll95": float(np.mean(librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            roll_percent=0.95
        ))),

        "rms": float(np.mean(librosa.feature.rms(y=y))),

        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        ))),

        "flatness": float(np.mean(librosa.feature.spectral_flatness(y=y))),

        "spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        ))),

        "mean_amp": get_mean_amplitude(y),

        "spectral_contrast": float(np.mean(librosa.feature.spectral_contrast(
            y=y,
            sr=sr
        ))),
    }


def aggregate_segment_features(segment_audios, sr):
    rolls85 = []
    rolls95 = []
    zcrs = []
    rms_values = []
    centroids = []
    flatness = []
    bandwidths = []
    mean_amps = []
    contrasts = []

    for segment in segment_audios:
        feats = extract_single_segment_features(segment, sr)

        rolls85.append(feats["roll85"])
        rolls95.append(feats["roll95"])
        zcrs.append(feats["zcr"])
        rms_values.append(feats["rms"])
        centroids.append(feats["spectral_centroid"])
        flatness.append(feats["flatness"])
        bandwidths.append(feats["spectral_bandwidth"])
        mean_amps.append(feats["mean_amp"])
        contrasts.append(feats["spectral_contrast"])

    return {
        "roll85": safe_feature_mean(rolls85),
        "roll95": safe_feature_mean(rolls95),
        "zcr": safe_feature_mean(zcrs),
        "rms": safe_feature_mean(rms_values),
        "spectral_centroid": safe_feature_mean(centroids),
        "flatness": safe_feature_mean(flatness),
        "spectral_bandwidth": safe_feature_mean(bandwidths),
        "mean_amp": safe_feature_mean(mean_amps),
        "spectral_contrast": safe_feature_mean(contrasts),
    }


def extract_mfcc_from_concatenated_segments(segment_audios, sr, prefix):
    mfcc_features = {}

    if len(segment_audios) == 0:
        for i in range(1, 14):
            mfcc_features[f"mfcc_{prefix}_{i}"] = 0.0
        return mfcc_features

    concatenated = np.concatenate(segment_audios)

    if len(concatenated) == 0:
        for i in range(1, 14):
            mfcc_features[f"mfcc_{prefix}_{i}"] = 0.0
        return mfcc_features

    mfcc = librosa.feature.mfcc(
        y=concatenated,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    for i, value in enumerate(mfcc_mean, start=1):
        mfcc_features[f"mfcc_{prefix}_{i}"] = float(value)

    return mfcc_features


def extract_segment_audios(y, sr, segments, segment_type):
    audios = []

    for seg in segments:
        if seg["segment_type"] != segment_type:
            continue

        start = int(seg["start_sec"] * sr)
        end = int(seg["end_sec"] * sr)

        clip = y[start:end]

        if len(clip) > 0:
            audios.append(clip)

    return audios


def get_breath_cycle_durations(breath_segments):
    """
    Approximation of calculate_breaths_cycle_durations().
    Uses distance between consecutive breath starts.
    """
    if len(breath_segments) < 2:
        return []

    starts = [seg["start_sec"] for seg in breath_segments]
    return [
        starts[i + 1] - starts[i]
        for i in range(len(starts) - 1)
    ]


def extract_features_from_audio(audio_path, segments, patient_id="uploaded_audio"):
    y, sr = librosa.load(
        audio_path,
        sr=SR,
        mono=True
    )

    if segments is None:
        segments = []

    breath_segments = [
        seg for seg in segments
        if seg["segment_type"] == "breath"
    ]

    speech_segments = [
        seg for seg in segments
        if seg["segment_type"] == "speech"
    ]

    breath_durations = [
        seg["duration_sec"]
        for seg in breath_segments
    ]

    breath_cycle_durations = get_breath_cycle_durations(breath_segments)

    breath_audios = extract_segment_audios(
        y=y,
        sr=sr,
        segments=segments,
        segment_type="breath"
    )

    speech_audios = extract_segment_audios(
        y=y,
        sr=sr,
        segments=segments,
        segment_type="speech"
    )

    breath_feats = aggregate_segment_features(
        segment_audios=breath_audios,
        sr=sr
    )

    speech_feats = aggregate_segment_features(
        segment_audios=speech_audios,
        sr=sr
    )

    breath_mfcc = extract_mfcc_from_concatenated_segments(
        segment_audios=breath_audios,
        sr=sr,
        prefix="breath"
    )

    speech_mfcc = extract_mfcc_from_concatenated_segments(
        segment_audios=speech_audios,
        sr=sr,
        prefix="speech"
    )

    features = {
        "patient": patient_id,

        "mean_bc": safe_mean(breath_cycle_durations),
        "std_bc": safe_std(breath_cycle_durations),

        "mean_breath_duration": safe_mean(breath_durations),
        "std_breath_duration": safe_std(breath_durations),
        "breath_number": len(breath_segments),

        "roll_off_breath_85": breath_feats["roll85"],
        "roll_off_breath_95": breath_feats["roll95"],
        "ZCR_breath": breath_feats["zcr"],
        "rms_breath": breath_feats["rms"],
        "spectral_centroid_breath": breath_feats["spectral_centroid"],
        "flatness_breath": breath_feats["flatness"],
        "spectral_bandwidth_breath": breath_feats["spectral_bandwidth"],
        "Mean_Amp_breath": breath_feats["mean_amp"],
        "SpectralContrast_breath": breath_feats["spectral_contrast"],

        "roll_off_speech85": speech_feats["roll85"],
        "roll_off_speech95": speech_feats["roll95"],
        "ZCR_speech": speech_feats["zcr"],
        "rms_speech": speech_feats["rms"],
        "spectral_centroid_speech": speech_feats["spectral_centroid"],
        "flatness_speech": speech_feats["flatness"],
        "spectral_bandwidth_speech": speech_feats["spectral_bandwidth"],
        "Mean_Amp_speech": speech_feats["mean_amp"],
        "SpectralContrast_speech": speech_feats["spectral_contrast"],

        "has_speech": 1 if len(speech_audios) > 0 else 0,
        "has_breath": 1 if len(breath_audios) > 0 else 0,
    }

    features.update(breath_mfcc)
    features.update(speech_mfcc)

    return features