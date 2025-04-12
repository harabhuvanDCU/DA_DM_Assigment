# Predicting Social Media Engagement Using DistilBERT

This project presents an end-to-end Natural Language Processing (NLP) pipeline to predict the engagement score of social media posts using a fine-tuned DistilBERT model. It uses engineered features like sentiment polarity and text length to train a regression model capable of estimating the post's potential impact.

---

## Project Objective

- **What**: Build a model that predicts how engaging a post will be using only the text.
- **Why**: Engagement metrics guide marketing, trend detection, and content visibility. Most current models ignore semantic content.
- **How**: Fine-tune a lightweight transformer (DistilBERT) using engineered sentiment and verbosity features.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                    # Original datasets from HuggingFace
│   └── cleaned/                # Balanced and filtered dataset
├── model/
│   └── DA_DM_model/            # Final fine-tuned model
├── src/
│   └── fine_tune_distilbert.py # Training and evaluation pipeline
├── examples/
│   └── inference_examples.ipynb # Model testing with sample inputs
├── README.md
└── requirements.txt
```

---

## Dataset Summary

| Stage              | Rows     | Description                              |
|--------------------|----------|------------------------------------------|
| Raw Combined       | 71,189   | Aggregated from 4 open datasets          |
| After Cleaning     | 49,636   | Removed nulls and duplicates             |
| Final Balanced     | 40,086   | Six content themes, 6,681 entries each   |

Themes: Politics, Sports, People, Entertainment, Technology, Other  
Sources: Tweets, Reddit-style posts, LinkedIn content, marketing captions

---

## Feature Engineering

We created the following features:

- `sentiment_numeric`: From TextBlob, range [-0.96, +0.97]
- `text_length_norm`: Min-max normalized word count
- `estimated_engagement_score`:  
  `Score = 0.7 × text_length_norm + 0.3 × sentiment_numeric`

Why this formula?  
It balances verbosity and emotional tone. This heuristic aligns with literature like Asur & Huberman (2010), showing that longer, emotionally charged posts attract more interaction.

---

## Model Architecture

- Model: `distilbert-base-uncased` (6-layer transformer)
- Task: Regression (num_labels=1)
- Input: Social media text (max 128 tokens)
- Output: Continuous engagement score (float)
- Framework: Hugging Face Transformers

---

## Training Configuration

| Parameter               | Value            |
|-------------------------|------------------|
| Learning Rate           | 1e-5             |
| Batch Size              | 16               |
| Epochs                  | 5 (early stopping at 2) |
| Optimizer               | AdamW            |
| Metric                  | R² score         |
| Environment             | Google Colab (T4 GPU, <8 GB RAM)

---

## Model Performance

| Model           | R² Score | RMSE   |
|------------------|----------|--------|
| Linear Baseline  | 0.418    | 0.1184 |
| DistilBERT       | 0.7752   | 0.0618 |

- Standard Deviation (RMSE) across folds: ±0.007  
- Validation loss: 0.0038  
- Result: ~77.5% of engagement variance explained by text alone

---

## Inference Examples

Example Predictions:

1. "Just launched our new app! Try it out and share your feedback!" → 0.2078  
2. "Feeling lost today. Everything seems to be falling apart." → -0.1758  
3. "Huge congratulations to the team for reaching 10,000 followers!" → 0.2661  
4. "That awkward moment when WiFi dies during a Zoom call." → -0.0474  

Insight: Longer, optimistic or informative posts scored higher, while short or negative posts scored lower.

---

## Limitations

- Does not account for non-text features like author, timestamp, or media
- Limited to English textual posts (no multilingual support)
- Sensitive to training hyperparameters (e.g., learning rate, epochs)
- Engagement definition is a proxy, not platform-specific

---

## Real-World Applications

- Marketing dashboards: Rank draft posts by impact
- Influencer analysis: Score captions before posting
- Content moderation: Flag high-impact or harmful content
- Real-time trend tracking: Use in streaming platforms via API

---

## Future Enhancements

- Multimodal learning (add images, videos, hashtags)
- Hyperparameter optimization (GridSearch/Optuna)
- Deployment as REST or streaming API
- Recast as classification (low/med/high engagement)


