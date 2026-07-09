import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)

import json
import joblib
import pandas as pd
import numpy as np

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def prepare_features(feature_dict, features_path, means_path):
    feature_cols = load_json(features_path)
    train_means = joblib.load(means_path)

    X = pd.DataFrame([feature_dict])

    for col in feature_cols:
        if col not in X.columns:
            X[col] = train_means.get(col, 0)

    X = X[feature_cols]
    X = X.fillna(train_means)

    return X

def prepare_merf_features(feature_dict, features_path, medians_path, scaler_path):
    feature_cols = load_json(features_path)
    train_medians = joblib.load(medians_path)
    scaler = joblib.load(scaler_path)

    X = pd.DataFrame([feature_dict])

    for col in feature_cols:
        if col not in X.columns:
            X[col] = train_medians.get(col, 0)

    X = X[feature_cols]
    X = X.fillna(train_medians)

    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_cols
    )

    return X_scaled

def predict_merf_regression(
    feature_dict,
    model_path,
    features_path,
    medians_path,
    scaler_path,
    cluster_id="new_patient"
):
    model = joblib.load(model_path)

    X = prepare_merf_features(
        feature_dict,
        features_path,
        medians_path,
        scaler_path
    )

    Z = np.ones((len(X), 1))
    clusters = pd.Series([cluster_id])

    pred = model.predict(
        X=X,
        Z=Z,
        clusters=clusters
    )

    return float(pred[0])


def predict_model(feature_dict, model_path, features_path, means_path):
    model = joblib.load(model_path)

    X = prepare_features(
        feature_dict,
        features_path,
        means_path
    )

    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
    else:
        prob = None

    return pred, prob

# sample_features = {
#     "height": 165,
#     "weight": 68,
#     "age": 28,
#     "gender": 1,
#     "spectral_bandwidth_breath": 1200,
#     "mfcc_speech_7": 0.5,
#     "has_speech": 1,
#     "has_breath": 1
# }

#ABNORMAL
sample_features = {
    "mean_bc": 2.031538462,
    "std_bc": 0.694205092,
    "mean_breath_duration": 0.947857143,
    "std_breath_duration": 0.453701893,
    "breath_number": 14,

    "roll_off_breath_85": 6434.367329,
    "roll_off_breath_95": 12246.81088,
    "ZCR_breath": 0.076147013,
    "rms_breath": 0.009735399,
    "spectral_centroid_breath": 3299.231521,
    "flatness_breath": 0.010850714,
    "spectral_bandwidth_breath": 3952.971948,
    "Mean_Amp_breath": 0.007508616,
    "SpectralContrast_breath": 17.30095044,

    "roll_off_speech85": 4660.86631,
    "roll_off_speech95": 9161.461114,
    "ZCR_speech": 0.058610969,
    "rms_speech": 0.025764592,
    "spectral_centroid_speech": 2532.245156,
    "flatness_speech": 0.004475505,
    "spectral_bandwidth_speech": 3175.831471,
    "Mean_Amp_speech": 0.02000012,
    "SpectralContrast_speech": 19.56745739,

    "mfcc_breath_1": -419.2065125,
    "mfcc_breath_2": 126.8537598,
    "mfcc_breath_3": -17.76580811,
    "mfcc_breath_4": 28.67396927,
    "mfcc_breath_5": 5.193836212,
    "mfcc_breath_6": 9.095412254,
    "mfcc_breath_7": -6.125859737,
    "mfcc_breath_8": 1.691509485,
    "mfcc_breath_9": -5.774842739,
    "mfcc_breath_10": -2.643042326,
    "mfcc_breath_11": 5.506739616,
    "mfcc_breath_12": -2.294797659,
    "mfcc_breath_13": 4.156300545,

    "mfcc_speech_1": -351.8068237,
    "mfcc_speech_2": 148.9514465,
    "mfcc_speech_3": -15.49184513,
    "mfcc_speech_4": 39.2689743,
    "mfcc_speech_5": 10.48425674,
    "mfcc_speech_6": 13.93920994,
    "mfcc_speech_7": -14.84525394,
    "mfcc_speech_8": 1.361973286,
    "mfcc_speech_9": -15.23270798,
    "mfcc_speech_10": -13.63517475,
    "mfcc_speech_11": 5.509619236,
    "mfcc_speech_12": -7.015918255,
    "mfcc_speech_13": 2.063240767,

    "is_male": 1,
    "is_female": 0,
    "height": 186,
    "weight": 90,
    "age": 44,

    "has_speech": 1,
    "has_breath": 1
}

pred = predict_merf_regression(
    sample_features,
    "deployment_models/FEV1_FVC/regression_merf/fev1_fvc_merf_model.pkl",
    "deployment_models/FEV1_FVC/regression_merf/fev1_fvc_merf_features.json",
    "deployment_models/FEV1_FVC/regression_merf/fev1_fvc_merf_train_medians.pkl",
    "deployment_models/FEV1_FVC/regression_merf/fev1_fvc_merf_scaler.pkl"
)

print("Predcited value: ", pred, "\n")

pred, prob = predict_model(
    sample_features,
    "deployment_models/FEV1_FVC/multiclass_xgb/fev1_fvc_multiclass_xgb_model.pkl",
    "deployment_models/FEV1_FVC/multiclass_xgb/fev1_fvc_multiclass_xgb_features.json",
    "deployment_models/FEV1_FVC/multiclass_xgb/fev1_fvc_multiclass_xgb_train_means.pkl"
)

print("Multiclass = \n FEV1: 0: Reduced FEV1, 1: Mildly reduced FEV1,2: Normal FEV1 \n FEV1/FVC: 0: Severe obstruction, 1: Moderate obstruction, 2: Mild obstruction, 3: Normal", pred)
print("Multiclass Prob= ", prob, "\n" )

pred, prob = predict_model(
    sample_features,
    "deployment_models/FEV1_FVC/binary_xgb/fev1_fvc_binary_xgb_model.pkl",
    "deployment_models/FEV1_FVC/binary_xgb/fev1_fvc_binary_xgb_features.json",
    "deployment_models/FEV1_FVC/binary_xgb/fev1_fvc_binary_xgb_train_means.pkl"
)

print("Binary (0: Abnormal, 1: Normal)= ", pred)
print("Binary Prob = ", prob, "\n" )
