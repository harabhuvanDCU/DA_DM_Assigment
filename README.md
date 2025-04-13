---

# Predicting Social Media Engagement Using DistilBERT

This project presents an end-to-end Natural Language Processing (NLP) pipeline to predict the engagement score of social media posts using a fine-tuned DistilBERT model. It combines semantic information with engineered features like sentiment polarity and text length to train a regression model capable of estimating user interaction.

---

## Project Objective

- **What**: Build a model that predicts how engaging a post will be using only its text.
- **Why**: Engagement metrics are essential for marketing, trend detection, and platform visibility. Most models overlook deep semantic cues in language.
- **How**: Fine-tune a lightweight transformer (DistilBERT) using engineered sentiment and verbosity features to estimate a continuous engagement score.

---

## Repository Structure

```
Predicting-Social-Media-Engagement-Using-DistilBERT/
├── data/
│   ├── raw/
│   │   └── combined_social_media_dataset.csv        # Original combined dataset
│   └── cleaned/
│       └── final_balanced_dataset2.csv              # Cleaned and balanced dataset
│
├── model/
│   └── DA_DM_model/                                 # Final saved DistilBERT model and tokenizer files
│       ├── config.json
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_args.bin
│       └── vocab.txt
│
├── src/
│   ├── DA_DM_FINAL_SUBMIT.ipynb                     # Complete ML pipeline notebook
│   └── COMPILE_INSTRUCTIONS.md                      # Setup and compile/run instructions
│
├── docs/
│   └── README.md                                     # Project overview and documentation
│
├── requirements.txt                                  # Python dependencies

```

---

## Dataset Summary

| Stage              | Rows     | Description                              |
|--------------------|----------|------------------------------------------|
| Raw Combined       | 71,189   | Merged from 4 diverse Hugging Face sources |
| After Cleaning     | 49,636   | Duplicates removed, nulls filtered       |
| Final Balanced     | 40,086   | 6 themes × 6,681 entries each            |

**Themes**: Politics, Sports, People, Entertainment, Technology, Other  
**Sources**: Hate speech tweets, Reddit-style comments, marketing content, instructional prompts

---

## Feature Engineering

Engineered features:

- `sentiment_numeric`: Derived from TextBlob sentiment analysis (range: -0.96 to +0.97)
- `text_length_norm`: Word count scaled between 0 and 1 using min-max normalization
- `estimated_engagement_score`:  
  \[
  \text{Score} = 0.7 \times \text{text\_length\_norm} + 0.3 \times \text{sentiment\_numeric}
  \]

**Why this formula?**  
It reflects the observation that longer and more emotionally expressive posts gain higher engagement, supported by literature such as Asur & Huberman (2010) and Bhargava et al. (2023).

---

## Model Architecture

- **Model**: `distilbert-base-uncased` (6-layer transformer)
- **Objective**: Regression (`num_labels = 1`)
- **Input**: Text up to 128 tokens
- **Output**: Continuous engagement score
- **Library**: Hugging Face Transformers

---

## Training Configuration

| Parameter               | Value                    |
|-------------------------|--------------------------|
| Learning Rate           | 1e-5                     |
| Batch Size              | 16                       |
| Epochs                  | 5 (early stopping after 2) |
| Optimizer               | AdamW                    |
| Evaluation Metric       | $R^2$ Score              |
| Training Platform       | Google Colab (T4 GPU, <8 GB RAM) |

---

## Model Performance

| Model           | R² Score | RMSE   |
|------------------|----------|--------|
| Linear Regression | 0.418    | 0.1184 |
| DistilBERT        | 0.7752   | 0.0618 |

- Cross-validation RMSE Std Dev: ±0.007  
- Validation Loss: 0.0038  
- Interpretation: ~77.5% of engagement variance explained using text alone

---

## Inference Examples

**Examples with Predicted Engagement Scores:**

1. “Just launched our new app! Try it out and share your feedback!” → **0.2078**  
2. “Feeling lost today. Everything seems to be falling apart.” → **-0.1758**  
3. “Huge congratulations to the team for reaching 10,000 followers!” → **0.2661**  
4. “That awkward moment when WiFi dies during a Zoom call.” → **-0.0474**

**Insight**: Informative and optimistic posts tend to score higher. Posts with negative tone or short length tend to score lower.

---

## Limitations

- Does not account for non-textual cues (author profile, time of post, likes, images)
- Limited to English; no multilingual or code-mixed text support
- Sensitive to hyperparameter tuning (learning rate, token length)
- Engagement score is a heuristic metric, not tied to platform-specific interactions

---

## Real-World Applications

- **Marketing Dashboards**: Estimate engagement for draft campaigns
- **Influencer Analytics**: Score content before publication
- **Content Moderation**: Flag unusually high-impact or harmful posts
- **Streaming Analysis**: Predict trends in real-time using APIs

---

## Future Enhancements

- Integrate images/videos (multimodal learning)
- Convert to classification (Low / Medium / High engagement)
- Perform hyperparameter tuning (GridSearch, Optuna)
- Deploy via REST API or stream processor

---
