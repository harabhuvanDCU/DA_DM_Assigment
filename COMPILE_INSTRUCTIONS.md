
---

# How to Compile and Run the Notebook

This guide provides step-by-step instructions to compile and execute the `DA_DM_FINAL_SUBMIT.ipynb` file from this repository.

---

## 1. Requirements

Before running the notebook, make sure the following libraries are installed:

- `transformers`
- `datasets`
- `pandas`
- `numpy`
- `scikit-learn`
- `torch`
- `textblob`
- `wandb`
- `joblib`

Install all using:

```bash
pip install transformers datasets pandas numpy scikit-learn torch textblob wandb joblib
```

Or use:

```bash
pip install -r requirements.txt
```

---

## 2. Recommended Platform

Use [Google Colab](https://colab.research.google.com/) to execute the notebook:

- GPU: Recommended (T4 or similar)
- Runtime: Python 3.10+ and GPU enabled

---

## 3. Directory Structure (Expected)

The notebook assumes the following folder paths exist in your Google Drive:

```
/content/drive/MyDrive/new2/
├── final_balanced_dataset2.csv     # Preprocessed dataset
├── DA_DM_model/                    # Final saved model (used for inference)
└── further_finetuned_model/        # Output folder for training checkpoints
```

You should mount your Google Drive using:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 4. Running the Notebook

The notebook has the following key sections:

### 🔹 Data Cleaning & Preprocessing

- Loads `final_balanced_dataset2.csv`
- Drops duplicates and missing values
- Balances the dataset into 6 themes

### 🔹 Feature Engineering

- Generates `sentiment_numeric` using `TextBlob`
- Normalizes text length into `text_length_norm`
- Combines features into `estimated_engagement_score`

### 🔹 Model Setup and Training

- Loads `distilbert-base-uncased` from HuggingFace
- Modifies it for regression (`num_labels=1`)
- Fine-tunes on 80/20 train/test split
- Uses early stopping, R² as metric

### 🔹 Evaluation

- Evaluates using RMSE and R²
- Compares against a linear regression baseline
- Visualizes results and insights

### 🔹 Inference Example

- Loads saved model from `DA_DM_model`
- Predicts engagement for sample posts
- Returns float engagement scores (positive, neutral, negative)

---

## 5. Notes

- Make sure to restart the runtime and re-run all cells if interrupted.
- Use GPU for faster training.
- To avoid widget metadata issues when uploading to GitHub, strip metadata using:

```bash
!jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --clear-output --inplace DA_DM_FINAL_SUBMIT.ipynb
```

---

## 6. Inference Tips

You can use the `inference_examples.ipynb` file to:

- Load the trained model
- Enter your own post
- Get an estimated engagement score instantly

---
