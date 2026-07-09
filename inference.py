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

# def predict_all_models(feature_dict):
#     def debug_missing_features(feature_dict, features_path, model_name):
#         feature_cols = load_json(features_path)
#
#         missing = [c for c in feature_cols if c not in feature_dict]
#         present = [c for c in feature_cols if c in feature_dict]
#
#         print(f"\n{model_name}")
#         print("Expected features:", len(feature_cols))
#         print("Present features:", len(present))
#         print("Missing features:", missing)
#
#     debug_missing_features(
#         feature_dict,
#         "deployment_models/FEV1/regression_merf/fev1_merf_features.json",
#         "FEV1 MERF"
#     )
#
#     fev1 = predict_merf_regression(
#         feature_dict,
#         "deployment_models/FEV1/regression_merf/fev1_merf_model.pkl",
#         "deployment_models/FEV1/regression_merf/fev1_merf_features.json",
#         "deployment_models/FEV1/regression_merf/fev1_merf_train_medians.pkl",
#         "deployment_models/FEV1/regression_merf/fev1_merf_scaler.pkl"
#     )
#
#     fev1_binary_pred, fev1_binary_prob = predict_model(
#         feature_dict,
#         "deployment_models/FEV1/binary_xgb/fev1_binary_xgb_model.pkl",
#         "deployment_models/FEV1/binary_xgb/fev1_binary_xgb_features.json",
#         "deployment_models/FEV1/binary_xgb/fev1_binary_xgb_train_means.pkl"
#     )
#
#     fev1_multi_pred, fev1_multi_prob = predict_model(
#         feature_dict,
#         "deployment_models/FEV1/multiclass_xgb/fev1_multiclass_xgb_model.pkl",
#         "deployment_models/FEV1/multiclass_xgb/fev1_multiclass_xgb_features.json",
#         "deployment_models/FEV1/multiclass_xgb/fev1_multiclass_xgb_train_means.pkl"
#     )
#
#     fev1_fvc = predict_merf_regression(
#         feature_dict,
#         "deployment_models/FEV1_FVC/regression_merf/fev1_fvc_merf_model.pkl",
#         "deployment_models/FEV1_FVC/regression_merf/fev1_fvc_merf_features.json",
#         "deployment_models/FEV1_FVC/regression_merf/fev1_fvc_merf_train_medians.pkl",
#         "deployment_models/FEV1_FVC/regression_merf/fev1_fvc_merf_scaler.pkl"
#     )
#
#     fev1_fvc_binary_pred, fev1_fvc_binary_prob = predict_model(
#         feature_dict,
#         "deployment_models/FEV1_FVC/binary_rf/fev1_fvc_binary_rf_model.pkl",
#         "deployment_models/FEV1_FVC/binary_rf/fev1_fvc_binary_rf_features.json",
#         "deployment_models/FEV1_FVC/binary_rf/fev1_fvc_binary_rf_train_means.pkl"
#     )
#
#     fev1_fvc_multi_pred, fev1_fvc_multi_prob = predict_model(
#         feature_dict,
#         "deployment_models/FEV1_FVC/multiclass_xgb/fev1_fvc_multiclass_xgb_model.pkl",
#         "deployment_models/FEV1_FVC/multiclass_xgb/fev1_fvc_multiclass_xgb_features.json",
#         "deployment_models/FEV1_FVC/multiclass_xgb/fev1_fvc_multiclass_xgb_train_means.pkl"
#     )
#
#     return {
#         "fev1": fev1,
#         "fev1_binary_pred": int(fev1_binary_pred),
#         "fev1_binary_label": "Normal" if int(fev1_binary_pred) == 1 else "Abnormal",
#         "fev1_binary_confidence": float(max(fev1_binary_prob) * 100),
#
#         "fev1_multiclass_pred": int(fev1_multi_pred),
#         "fev1_multiclass_label": {
#             0: "Severely Abnormal FEV1",
#             1: "Mildly abnormal FEV1",
#             2: "Normal FEV1"
#         }.get(int(fev1_multi_pred), "Unknown"),
#         "fev1_multiclass_confidence": float(max(fev1_multi_prob) * 100),
#
#         "fev1_fvc": fev1_fvc,
#         "fev1_fvc_binary_pred": int(fev1_fvc_binary_pred),
#         "fev1_fvc_binary_label": "Normal" if int(fev1_fvc_binary_pred) == 1 else "Abnormal" ,
#         "fev1_fvc_binary_confidence": float(max(fev1_fvc_binary_prob) * 100),
#
#         "fev1_fvc_multiclass_pred": int(fev1_fvc_multi_pred),
#         "fev1_fvc_multiclass_label": {
#             0: "Severe obstruction",
#             1: "Moderate obstruction",
#             2: "Mild obstruction",
#             3: "Normal"
#         }.get(int(fev1_fvc_multi_pred), "Unknown"),
#         "fev1_fvc_multiclass_confidence": float(max(fev1_fvc_multi_prob) * 100),
#     }

def predict_all_models(feature_dict):
    def debug_missing_features(feature_dict, features_path, model_name):
        feature_cols = load_json(features_path)

        missing = [c for c in feature_cols if c not in feature_dict]
        present = [c for c in feature_cols if c in feature_dict]

        print(f"\n{model_name}")
        print("Expected features:", len(feature_cols))
        print("Present features:", len(present))
        print("Missing features:", missing)

    debug_missing_features(
        feature_dict,
        "deployment_models_without_pheno/FEV1/regression_merf/fev1_merf_features.json",
        "FEV1 MERF"
    )

    fev1 = predict_merf_regression(
        feature_dict,
        "deployment_models_without_pheno/FEV1/regression_merf/fev1_merf_model.pkl",
        "deployment_models_without_pheno/FEV1/regression_merf/fev1_merf_features.json",
        "deployment_models_without_pheno/FEV1/regression_merf/fev1_merf_train_medians.pkl",
        "deployment_models_without_pheno/FEV1/regression_merf/fev1_merf_scaler.pkl"
    )

    fev1_binary_pred, fev1_binary_prob = predict_model(
        feature_dict,
        "deployment_models_without_pheno/FEV1/binary_xgb/fev1_binary_xgb_model.pkl",
        "deployment_models_without_pheno/FEV1/binary_xgb/fev1_binary_xgb_features.json",
        "deployment_models_without_pheno/FEV1/binary_xgb/fev1_binary_xgb_train_means.pkl"
    )

    fev1_multi_pred, fev1_multi_prob = predict_model(
        feature_dict,
        "deployment_models_without_pheno/FEV1/multiclass_xgb/fev1_multiclass_xgb_model.pkl",
        "deployment_models_without_pheno/FEV1/multiclass_xgb/fev1_multiclass_xgb_features.json",
        "deployment_models_without_pheno/FEV1/multiclass_xgb/fev1_multiclass_xgb_train_means.pkl"
    )

    fev1_fvc = predict_merf_regression(
        feature_dict,
        "deployment_models_without_pheno/FEV1_FVC/regression_merf/fev1_fvc_merf_model.pkl",
        "deployment_models_without_pheno/FEV1_FVC/regression_merf/fev1_fvc_merf_features.json",
        "deployment_models_without_pheno/FEV1_FVC/regression_merf/fev1_fvc_merf_train_medians.pkl",
        "deployment_models_without_pheno/FEV1_FVC/regression_merf/fev1_fvc_merf_scaler.pkl"
    )

    fev1_fvc_binary_pred, fev1_fvc_binary_prob = predict_model(
        feature_dict,
        "deployment_models_without_pheno/FEV1_FVC/binary_xgb/fev1_fvc_binary_xgb_model.pkl",
        "deployment_models_without_pheno/FEV1_FVC/binary_xgb/fev1_fvc_binary_xgb_features.json",
        "deployment_models_without_pheno/FEV1_FVC/binary_xgb/fev1_fvc_binary_xgb_train_means.pkl"
    )

    fev1_fvc_multi_pred, fev1_fvc_multi_prob = predict_model(
        feature_dict,
        "deployment_models_without_pheno/FEV1_FVC/multiclass_xgb/fev1_fvc_multiclass_xgb_model.pkl",
        "deployment_models_without_pheno/FEV1_FVC/multiclass_xgb/fev1_fvc_multiclass_xgb_features.json",
        "deployment_models_without_pheno/FEV1_FVC/multiclass_xgb/fev1_fvc_multiclass_xgb_train_means.pkl"
    )

    return {
        "fev1": fev1,
        "fev1_binary_pred": int(fev1_binary_pred),
        "fev1_binary_label": "Normal" if int(fev1_binary_pred) == 1 else "Abnormal",
        "fev1_binary_confidence": float(max(fev1_binary_prob) * 100),

        "fev1_multiclass_pred": int(fev1_multi_pred),
        "fev1_multiclass_label": {
            0: "Severely Abnormal FEV1",
            1: "Mildly abnormal FEV1",
            2: "Normal FEV1"
        }.get(int(fev1_multi_pred), "Unknown"),
        "fev1_multiclass_confidence": float(max(fev1_multi_prob) * 100),

        "fev1_fvc": fev1_fvc,
        "fev1_fvc_binary_pred": int(fev1_fvc_binary_pred),
        "fev1_fvc_binary_label": "Normal" if int(fev1_fvc_binary_pred) == 1 else "Abnormal" ,
        "fev1_fvc_binary_confidence": float(max(fev1_fvc_binary_prob) * 100),

        "fev1_fvc_multiclass_pred": int(fev1_fvc_multi_pred),
        "fev1_fvc_multiclass_label": {
            0: "Severe obstruction",
            1: "Moderate obstruction",
            2: "Mild obstruction",
            3: "Normal"
        }.get(int(fev1_fvc_multi_pred), "Unknown"),
        "fev1_fvc_multiclass_confidence": float(max(fev1_fvc_multi_prob) * 100),
    }