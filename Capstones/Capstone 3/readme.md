# Sentiment Analysis on Amazon Product Reviews

## Executive Summary

This project focuses on building a sentiment analysis pipeline for Amazon product reviews to classify each review as either **positive** or **negative**. The objective is to explore the effectiveness of both traditional machine learning models and transformer-based language models (e.g., DistilBERT) in sentiment classification tasks.

Using a dataset of ~50,000 Amazon reviews with binary sentiment labels, the project compares models including **Logistic Regression**, **Random Forest**, and **DistilBERT**, with the transformer model significantly outperforming classical approaches in predictive accuracy.

---

## Project Structure

- `Sentiment Analysis on Amazon Product Reviews.ipynb`: Full exploratory analysis, preprocessing, modeling, and evaluation.
- `README.md`: This document.
- `models/`: (optional) Contains saved model files or fine-tuned weights.
- `data/`: (optional) Preprocessed or raw data files (excluded from GitHub if too large).

---

## Dataset

- **Source**: Amazon product review dataset (binary sentiment: 1 for positive, 0 for negative)
- **Size**: ~50,000 labeled reviews
- **Features**: `reviewText`, `summary`, `overall`, `sentiment`
- **Preprocessing**:
  - Lowercasing
  - Removing punctuation, stopwords, and digits
  - Custom tokenizer for n-gram analysis

---

## Approach

### 🔍 Exploratory Data Analysis (EDA)

- Analyzed class balance (roughly equal positive/negative)
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
| Logistic Regression | Baseline traditional model            |
| Random Forest      | Captures non-linear patterns           |
| DistilBERT         | Pretrained transformer fine-tuned for classification |

---

## Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

DistilBERT achieved the highest scores across all metrics, particularly improving recall and F1 over traditional models.

---

## Key Results

| Model           | Accuracy | F1 Score | ROC-AUC |
|----------------|----------|----------|---------|
| Logistic Regression | ~0.85     | ~0.84     | ~0.90    |
| Random Forest       | ~0.86     | ~0.85     | ~0.91    |
| **DistilBERT**       | **0.92** | **0.92** | **0.96** |

---

## Conclusions

- Transformer models (like DistilBERT) provide superior performance on sentiment classification tasks, especially when fine-tuned on task-specific data.
- Traditional models are simpler and faster but may miss subtleties in language.
- Clean preprocessing, class balance, and robust evaluation are essential for reliable NLP models.

---

## Future Work

- Incorporate multi-class sentiment (e.g., 1–5 stars)
- Use full review metadata (e.g., helpfulness votes)
- Experiment with larger transformer models like BERT or RoBERTa
- Deploy as a real-time sentiment classification API

---

## Author

**Justin Feathers**  
[LinkedIn](https://www.linkedin.com/in/justin-feathers/)  
[Portfolio](https://github.com/justinzane24)  
