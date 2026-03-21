import pandas as pd
import numpy as np
import h2o
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import platform

from h2o.automl import H2OAutoML
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# Start H2O
# ------------------------------

h2o.init()

# ------------------------------
# Load datasets
# ------------------------------

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

# ------------------------------
# Assign labels
# ------------------------------

for df, label in [
(bone_train,'Bone'),(adipose_train,'Adipose'),(cartilage_train,'Cartilage'),
(tendon_train,'Tendon'),(muscle_train,'Muscle'),
(bone_valid,'Bone'),(adipose_valid,'Adipose'),(cartilage_valid,'Cartilage'),
(tendon_valid,'Tendon'),(muscle_valid,'Muscle'),
(bone_test,'Bone'),(adipose_test,'Adipose'),(cartilage_test,'Cartilage'),
(tendon_test,'Tendon'),(muscle_test,'Muscle')
]:
    df['label'] = label

# ------------------------------
# Combine datasets
# ------------------------------

train_df = pd.concat([bone_train, adipose_train, cartilage_train, tendon_train, muscle_train])
valid_df = pd.concat([bone_valid, adipose_valid, cartilage_valid, tendon_valid, muscle_valid])
test_df = pd.concat([bone_test, adipose_test, cartilage_test, tendon_test, muscle_test])

# ------------------------------
# Convert to H2O
# ------------------------------

train = h2o.H2OFrame(train_df)
valid = h2o.H2OFrame(valid_df)
test = h2o.H2OFrame(test_df)

train['label'] = train['label'].asfactor()
valid['label'] = valid['label'].asfactor()
test['label'] = test['label'].asfactor()

Y = 'label'
X = list(train.columns)
X.remove(Y)

# ------------------------------
# Run AutoML
# ------------------------------

aml = H2OAutoML(
    max_models=5,
    seed=1
)

aml.train(
    x=X,
    y=Y,
    training_frame=train,
    validation_frame=valid
)

experiment_info = {
    "dataset_size_train": train.nrows,
    "dataset_size_validation": valid.nrows,
    "dataset_size_test": test.nrows,
    "number_of_features": len(X),

    "automl_max_models": aml.max_models,
    "automl_seed": 1,

    "best_model_id": aml.leader.model_id,
    "best_model_algorithm": aml.leader.algo,

    "h2o_version": h2o.__version__,
    "python_version": platform.python_version(),
    "sklearn_version": sklearn.__version__
}

with open("results/experiment_info.json", "w") as f:
    json.dump(experiment_info, f, indent=4)

# Extract feature importance from best model
importance = aml.leader.varimp(use_pandas=True)

# Save importance table
importance.to_csv("results/feature_importance.csv", index=False)

print(importance.head(20))

# ------------------------------
# Leaderboard
# ------------------------------

lb = aml.leaderboard.as_data_frame()
print(lb)

# ------------------------------
# Predictions
# ------------------------------

pred = aml.leader.predict(test)

y_true = test['label'].as_data_frame().values.flatten()
y_pred = pred['predict'].as_data_frame().values.flatten()

# ------------------------------
# Metrics
# ------------------------------

report = classification_report(y_true, y_pred)
print(report)

# ------------------------------
# Save results
# ------------------------------

os.makedirs("results", exist_ok=True)

lb.to_csv("results/leaderboard.csv")

pred.as_data_frame().to_csv("results/test_predictions.csv")

# ------------------------------
# Confusion Matrix
# ------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()

# ------------------------------
# Save classification report
# ------------------------------

with open("results/classification_report.txt","w") as f:
    f.write(report)

# ------------------------------
# Save best model
# ------------------------------
plt.figure(figsize=(10,6))

importance_top = importance.head(20)

plt.barh(importance_top['variable'], importance_top['relative_importance'])

plt.xlabel("Importance")
plt.ylabel("Raman Feature")
plt.title("Top Raman Features for Tissue Classification")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("results/feature_importance.png", dpi=300)

plt.close()
h2o.save_model(aml.leader, path="results", force=True)

print("Experiment completed")