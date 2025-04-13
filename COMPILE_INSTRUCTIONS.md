
---

HOW TO COMPILE AND RUN THE NOTEBOOK

This guide explains how to execute the `DA_DM_FINAL_SUBMIT.ipynb` notebook in Google Colab or any local Jupyter environment.

---

1. REQUIREMENTS

Make sure the following Python libraries are installed:

- transformers
- datasets
- pandas
- numpy
- scikit-learn
- torch
- textblob
- wandb
- joblib

Install them using the command:

pip install -r requirements.txt

Or install manually using:

pip install transformers datasets pandas numpy scikit-learn torch textblob wandb joblib

---

2. RECOMMENDED PLATFORM

Google Colab is recommended due to its GPU support.

Recommended setup:

- Runtime Type: Python 3.x
- Hardware Accelerator: GPU (T4 or equivalent)

---

3. REQUIRED DIRECTORY STRUCTURE

Ensure the following files are present in your Google Drive:

/content/drive/MyDrive/new2/
- final_balanced_dataset2.csv  (Cleaned and balanced dataset)
- DA_DM_model/                 (Folder with saved fine-tuned DistilBERT model)
- further_finetuned_model/     (Used during training for checkpoints)

You must mount your Google Drive with:

from google.colab import drive  
drive.mount('/content/drive')

---

4. RUNNING THE NOTEBOOK

The notebook follows a structured pipeline:

A. Data Preprocessing  
- Loads the dataset  
- Removes duplicates and nulls  
- Balances six thematic classes  

B. Feature Engineering  
- Computes sentiment using TextBlob  
- Normalizes word count  
- Creates engagement score using a weighted formula  

C. Model Setup  
- Loads DistilBERT (distilbert-base-uncased)  
- Converts it to regression (num_labels=1)  
- Applies early stopping and R²-based evaluation  

D. Training  
- Trains the model on 80% data  
- Validates on 20%  
- Saves the best model  

E. Evaluation  
- Calculates RMSE and R²  
- Compares with baseline linear regression  
- Plots performance results  

F. Inference  
- Loads trained model  
- Predicts engagement for custom text examples  

---

5. SAMPLE INFERENCE WORKFLOW

You can test the trained model using inference_examples in same noteboook

- Loading the saved model  
- Tokenizing new text  
- Outputting engagement scores  

---

7. TIPS

- Always restart and run all cells in order after a runtime disconnect.
- Ensure all file paths match the expected structure.
- Use GPU for faster training, especially for transformers.

---

