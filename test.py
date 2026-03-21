import pandas as pd
import numpy as np
import h2o
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import platform
import sklearn
import sys

from datetime import datetime
from h2o.automl import H2OAutoML
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize



# START H2O
h2o.init()

# CREATE TIMESTAMPED RESULTS FOLDER

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"

os.makedirs(results_dir, exist_ok=True)


# LOAD DATA


bone_train = pd.read_csv('Bone_train.csv')
adipose_train = pd.read_csv('Adipose_train.csv')
cartilage_train = pd.read_csv('Cartilage_train.csv')
tendon_train = pd.read_csv('Tendon_train.csv')
muscle_train = pd.read_csv('Muscle_train.csv')

bone_valid = pd.read_csv('Bone_valid.csv')
adipose_valid = pd.read_csv('Adipose_valid.csv')
cartilage_valid = pd.read_csv('Cartilage_valid.csv')
tendon_valid = pd.read_csv('Tendon_valid.csv')
muscle_valid = pd.read_csv('Muscle_valid.csv')

bone_test = pd.read_csv('Bone_test.csv')
adipose_test = pd.read_csv('Adipose_test.csv')
cartilage_test = pd.read_csv('Cartilage_test.csv')
tendon_test = pd.read_csv('Tendon_test.csv')
muscle_test = pd.read_csv('Muscle_test.csv')


# ASSIGN LABELS


for df, label in [
(bone_train,'Bone'),(adipose_train,'Adipose'),(cartilage_train,'Cartilage'),
(tendon_train,'Tendon'),(muscle_train,'Muscle'),
(bone_valid,'Bone'),(adipose_valid,'Adipose'),(cartilage_valid,'Cartilage'),
(tendon_valid,'Tendon'),(muscle_valid,'Muscle'),
(bone_test,'Bone'),(adipose_test,'Adipose'),(cartilage_test,'Cartilage'),
(tendon_test,'Tendon'),(muscle_test,'Muscle')
]:
    df['label'] = label


# CONCATENATE DATASETS


train_df = pd.concat([bone_train, adipose_train, cartilage_train, tendon_train, muscle_train])
valid_df = pd.concat([bone_valid, adipose_valid, cartilage_valid, tendon_valid, muscle_valid])
test_df = pd.concat([bone_test, adipose_test, cartilage_test, tendon_test, muscle_test])



# CONVERT TO H2O


train = h2o.H2OFrame(train_df)
valid = h2o.H2OFrame(valid_df)
test = h2o.H2OFrame(test_df)

train['label'] = train['label'].asfactor()
valid['label'] = valid['label'].asfactor()
test['label'] = test['label'].asfactor()

Y = 'label'
X = list(train.columns)
X.remove(Y)



# RUN AUTOML


print("\nStarting H2O AutoML training...\n")

aml = H2OAutoML(
    max_models=10,
    seed=1,
    nfolds=0,
    sort_metric="mean_per_class_error",
    exclude_algos=["DeepLearning"])

aml.train(
    x=X,
    y=Y,
    training_frame=train,
    validation_frame=valid
)

print("\nAutoML training completed\n")



# SAVE LEADERBOARD


lb = aml.leaderboard.as_data_frame()
lb_filtered = lb[['model_id', 'logloss', 'mse', 'rmse', 'mean_per_class_error']]
lb_filtered.to_csv(f"{results_dir}/leaderboard_filtered.csv", index=False)
print("\nAutoML Leaderboard:\n")
print(lb_filtered)



# BEST MODEL INFORMATION


best_model = aml.leader

print("\nBest Model:")
print(best_model.model_id)
print("Algorithm:", best_model.algo)


# PREDICTIONS


pred = best_model.predict(test)
pred_df = pred.as_data_frame()
y_true = test['label'].as_data_frame().values.flatten()
y_pred = pred_df['predict'].values
pred_df.to_csv(f"{results_dir}/test_predictions.csv", index=False)


# CLASSIFICATION METRICS

report = classification_report(y_true, y_pred)
print("\nClassification Report:\n")
print(report)
with open(f"{results_dir}/classification_report.txt","w") as f:
    f.write(report)



# VALIDATION CONFUSION MATRIX 


pred_valid = best_model.predict(valid).as_data_frame()

y_valid_true = valid['label'].as_data_frame().values.flatten()
y_valid_pred = pred_valid['predict'].values

cm_valid = confusion_matrix(y_valid_true, y_valid_pred)

labels = np.unique(y_valid_true)

plt.figure(figsize=(7,6))
sns.heatmap(cm_valid, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Validation Confusion Matrix")

plt.tight_layout()
plt.savefig(f"{results_dir}/validation_confusion_matrix.png", dpi=300)
plt.close()



# NORMALIZED VALIDATION CONFUSION MATRIX 

cm_valid_norm = cm_valid.astype(float) / cm_valid.sum(axis=1, keepdims=True)

plt.figure(figsize=(7,6))
sns.heatmap(cm_valid_norm, annot=True, cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Validation Confusion Matrix")

plt.tight_layout()
plt.savefig(f"{results_dir}/validation_confusion_matrix_normalized.png", dpi=300)
plt.close()



#  TEST CONFUSION MATRIX 

cm_test = confusion_matrix(y_true, y_pred)

labels_test = np.unique(y_true)

plt.figure(figsize=(7,6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_test, yticklabels=labels_test)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")

plt.tight_layout()
plt.savefig(f"{results_dir}/test_confusion_matrix.png", dpi=300)
plt.close()



#  NORMALIZED TEST CONFUSION MATRIX 


cm_test_norm = cm_test.astype(float) / cm_test.sum(axis=1, keepdims=True)

plt.figure(figsize=(7,6))
sns.heatmap(cm_test_norm, annot=True, cmap="Blues",
            xticklabels=labels_test, yticklabels=labels_test)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Test Confusion Matrix")

plt.tight_layout()
plt.savefig(f"{results_dir}/test_confusion_matrix_normalized.png", dpi=300)
plt.close()

# SAVE BEST MODEL 


model_path = h2o.save_model(best_model, path=results_dir, force=True)
print("\nBest model saved at:", model_path)
model_params = best_model.params

# EXTRACT BEST MODEL HYPERPARAMETERS 

best_model = aml.leader

# Store hyperparameters
best_params = {}

for param_name, param_info in best_model.params.items():
    
    try:
        # Extract actual value used by model
        best_params[param_name] = param_info["actual"]
    except:
        best_params[param_name] = None

# Convert to DataFrame
params_df = pd.DataFrame(list(best_params.items()),
                        columns=["Hyperparameter", "Value"])

# Save
params_df.to_csv(f"{results_dir}/best_model_all_hyperparameters.csv", index=False)

# Also save JSON
with open(f"{results_dir}/best_model_all_hyperparameters.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("\nBest model hyperparameters extracted and saved.")

# FILTER IMPORTANT GLM HYPERPARAMETERS

important_keys = [
    "family",
    "alpha",
    "lambda",
    "lambda_search",
    "nlambdas",
    "standardize",
    "max_iterations",
    "solver"
]

important_params = {k: best_params.get(k, None) for k in important_keys}

important_df = pd.DataFrame(list(important_params.items()),
                           columns=["Hyperparameter", "Value"])

important_df.to_csv(f"{results_dir}/best_model_key_hyperparameters.csv", index=False)

print("\nKey GLM hyperparameters:")
print(important_df)



# SAVE EXPERIMENT INFO

experiment_info = {

    "dataset_size_train": train.nrows,
    "dataset_size_validation": valid.nrows,
    "dataset_size_test": test.nrows,

    "number_of_features": len(X),

    "automl_max_models": aml.max_models,
    "automl_seed": aml.seed,

    "best_model_id": best_model.model_id,
    "best_model_algorithm": best_model.algo,

    "h2o_version": h2o.__version__,
    "python_version": platform.python_version(),
    "sklearn_version": sklearn.__version__

}

with open(f"{results_dir}/experiment_info.json", "w") as f:
    json.dump(experiment_info, f, indent=4)


# FINISHED


print("\nExperiment completed successfully")
print("Results saved in folder:", results_dir)




