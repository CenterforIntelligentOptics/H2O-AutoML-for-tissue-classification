# H2O AutoML-Based Classification of Raman Spectra for Tissue Differentiation

## Overview

This repository presents a machine learning pipeline based on the **H2O Automated Machine Learning (AutoML) framework** for classifying Raman spectra acquired from biological tissues relevant to laser osteotomy.

## Dataset

The repository includes Raman spectral datasets (unprocessed datasets) corresponding to five tissue classes:

- Bone  
- Adipose tissue  
- Cartilage  
- Tendon  
- Muscle  

The datasets are pre-structured into training, validation, and test sets to ensure reproducible model development and evaluation.

## Methodology

The pipeline consists of the following steps:

1. Data integration and labeling  
2. Conversion to H2O data structures  
3. Automated model training using H2O AutoML  
4. Best performing model selection based on mean per class error  
5. Performance evaluation on test datasets  

Deep learning models are excluded in the current configuration.

## 📊 Evaluation Metrics

The best performing model was selected based on the **mean per class error** classification metric, ensuring balanced performance across all tissue classes.

Model performance is assessed on test datasets using:

- Precision  
- Recall  
- F1-score  
- Confusion matrices (absolute and normalized)  

## Installation

Install the required dependencies:

```bash
python -m venv automl_env
source automl_env/bin/activate
pip install -r requirements.txt
python test.py
```
## Expected Output

Upon execution, the script generates a timestamped results directory (e.g., `results_YYYYMMDD_HHMMSS/`) containing all outputs from the experiment.

The output typically includes:

- **Model Leaderboard**  
  A CSV file summarizing the performance of all trained models (e.g., logloss, RMSE, mean per class error).

- **Best Model Information**  
  The selected model with the lowest mean per class error, along with its algorithm type.

- **Predictions**  
  A CSV file containing predicted labels for the test dataset.

- **Classification Report**  
  A text file reporting precision, recall, and F1-score for each tissue class.

- **Confusion Matrices**  
  High-resolution images illustrating classification performance:
  - Validation confusion matrix  
  - Normalized validation confusion matrix  
  - Test confusion matrix  
  - Normalized test confusion matrix  

- **Model File**  
  The trained best-performing model saved for future inference.

- **Hyperparameters**  
  CSV and JSON files listing all hyperparameters of the best model, including key parameters for interpretability.

- **Experiment Info**  
  A JSON file documenting dataset sizes, number of features, software versions, and training configuration.

