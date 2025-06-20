# Sentiment Analysis on Amazon Product Reviews

## Executive Summary

This project focuses on building a sentiment analysis pipeline for Amazon product reviews to classify each review as either **positive** or **negative**. The objective is to explore the effectiveness of both traditional machine learning models and transformer-based language models (e.g., DistilBERT) in sentiment classification tasks.

Using a dataset of 4,000,000 Amazon reviews with binary sentiment labels, the project compares **Logistic Regression** and **DistilBERT** models, with the transformer model significantly outperforming classical approaches in predictive accuracy.

---

## Dataset

- **Source**: Amazon product review dataset (binary sentiment: 1 for positive, 0 for negative)
- **Size**: 4,000,000 labeled reviews
  - **Training set:** 3,600,000 reviews  
    - Logistic Regression uses all 3.6 M  
    - DistilBERT uses a balanced 344,000‑review subset (~10%)
  - **Test set** 400,000 reviews  
- **Features**: `label`, `title`, `content`
- **Preprocessing**:
  - Lowercasing
  - Punctuation, stopword, and digit removal
  - Custom tokenization for n‑gram analysis (TF‑IDF)
  - Hugging Face tokenization & attention masks for DistilBERT

---

## Approach

### Exploratory Data Analysis (EDA)

- Analyzed class balance (equal positive/negative)
- Extracted and visualized top unigrams, bigrams, and trigrams
- Investigated review length distributions and potential signal from summary vs. body text

### Text Preprocessing

- Custom tokenization and cleaning
- Vectorization using:
  - **TF-IDF** for traditional models
  - **Token IDs** with attention masks for DistilBERT

### Modeling Techniques

| Model              | Notes                                  |
|-------------------|----------------------------------------|
| Logistic Regression | TF‑IDF vectorized uni‑/bi‑grams; baseline model |            |
| DistilBERT         | Pretrained transformer, fine‑tuned for 2 epochs |

---

## Evaluation Metrics
| Model               | Accuracy | F1‑Score | ROC‑AUC |
|---------------------|:--------:|:--------:|:-------:|
| Logistic Regression | `90.40%` | `0.90` | `0.96` |
| **DistilBERT**      | `95.97%` | `0.96` | `0.99` |

> **Note:** Classification reports are on the full 400,000‑row test set.  
> For DistilBERT, a balanced 200,000/200,000 support per class yielded precision, recall, and F1 ≈ 0.96.

---

## Conclusions

- Transformer models (like DistilBERT) provide superior performance on sentiment classification tasks, especially when fine-tuned on task-specific data.
- Traditional models are simpler and faster but may miss subtleties in language.
- Clean preprocessing, class balance, and robust evaluation are essential for reliable NLP models.
- Thoughtful subset sampling allowed large‑scale transformer fine‑tuning despite hardware limits.

---

## Future Work

- Expand to **multi‑class** (1–5 stars) sentiment.  
- Incorporate review metadata (helpfulness votes, product category).  
- Experiment with larger transformers (RoBERTa, DeBERTa).  
- Deploy via a real‑time API (FastAPI or Flask).
