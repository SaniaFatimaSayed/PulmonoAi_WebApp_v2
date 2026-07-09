import pandas as pd
from inference_without_pheno import predict_all_models

df = pd.read_csv("/Users/sania/Library/CloudStorage/Dropbox/PhD-sfs2/asthma_voice_recordings/features_asthma_patients.csv")

all_results = []

for _, row in df.iterrows():

    feature_dict = row.to_dict()

    patient = feature_dict.pop("patient", "Unknown")

    results = predict_all_models(feature_dict)

    results["patient"] = patient

    all_results.append(results)

results_df = pd.DataFrame(all_results)

print(results_df)

results_df.to_csv("voice_only_predictions.csv", index=False)

print("Saved!")