# 📄 Project Report — Mail Spam Prediction

**Author:** Vaisak  
**Domain:** Machine Learning | Natural Language Processing  
**Tools:** Python, Scikit-learn, Streamlit  
**Dataset:** UCI SMS Spam Collection (via Kaggle)

---

## 1. Introduction

### 1.1 Background

Spam messages — unsolicited, often deceptive texts sent in bulk — remain a major nuisance across SMS and email platforms. Automated spam filters built on machine learning can classify incoming messages in milliseconds, protecting users without manual review. Text classification using the Bag of Words model is a well-established NLP approach for this problem.

### 1.2 Problem Statement

Build a binary text classifier that accurately labels SMS messages as **Spam** or **Ham (legitimate)**, and deploy the model as a lightweight web application for real-time single-message prediction.

### 1.3 Objectives

- Load, clean, and explore the SMS Spam Collection dataset
- Convert raw text into numerical features using CountVectorizer
- Train and compare three classification models
- Select the best model by accuracy and spam-specific metrics
- Save the model and vectorizer, then deploy via Streamlit

---

## 2. Dataset Description

**File:** `spam.csv`  
**Source:** UCI SMS Spam Collection Dataset

| Property | Value |
|----------|-------|
| Total Records | 5,572 |
| Features (after cleaning) | 2 — `Label`, `SMS` |
| Missing Values | None |
| Target Column | `Label` (`ham` / `spam`) |

### 2.1 Raw vs. Cleaned Columns

The original CSV contained five columns — two meaningful, three empty/unnamed artefacts:

| Raw Column | Action | Result |
|------------|--------|--------|
| `v1` | Renamed | `Label` |
| `v2` | Renamed | `SMS` |
| `Unnamed: 2` | Dropped | — |
| `Unnamed: 3` | Dropped | — |
| `Unnamed: 4` | Dropped | — |

### 2.2 Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Ham (Legitimate) | 4,825 | 86.6% |
| Spam | 747 | 13.4% |

The dataset has a moderate class imbalance (roughly 6.5:1 ham-to-spam ratio). This was factored into metric selection — spam-specific Precision, Recall, and F1 were tracked alongside overall accuracy.

---

## 3. Exploratory Data Analysis

### 3.1 Data Quality
- Zero missing values in both `Label` and `SMS` columns post-cleaning
- 5,163 unique messages out of 5,572 total — some messages repeated (e.g. *"Sorry, I'll call later"* appears 30 times)
- Both classes confirmed present; no label encoding needed (model handles string labels)

### 3.2 Class Distribution Visualisation
A bar chart of `Label` value counts confirmed the ham majority. This informed the decision to evaluate models on spam-specific metrics rather than relying solely on accuracy.

### 3.3 Sample Messages

| Label | Example |
|-------|---------|
| ham | *"Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."* |
| spam | *"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005..."* |

---

## 4. Methodology

### 4.1 Feature and Target Separation

```python
x = data["SMS"]    # Feature — raw message text
y = data["Label"]  # Target — 'ham' or 'spam'
```

### 4.2 Train-Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```

| Split | Records |
|-------|---------|
| Training set | 4,457 |
| Test set | 1,115 (965 ham, 150 spam) |

### 4.3 Text Vectorization — Bag of Words

`CountVectorizer` converts each SMS into a sparse vector representing word frequencies across the full training vocabulary.

```python
count_vector = CountVectorizer()
x_train_data = count_vector.fit_transform(x_train)  # fit + transform on train
x_test_data  = count_vector.transform(x_test)       # transform only on test
```

The vectorizer is fitted **only on training data** to prevent vocabulary leakage from the test set into the feature space.

---

## 5. Model Training & Evaluation

### 5.1 Models Trained

| Model | Notes |
|-------|-------|
| Multinomial Naive Bayes | Probabilistic; well-suited for word-count features |
| Logistic Regression | Linear classifier; `max_iter=1000` |
| Support Vector Machine | Linear kernel (`SVC(kernel='linear')`) |

### 5.2 Evaluation Metrics

All metrics below use `pos_label='spam'` — treating spam as the positive class:

| Metric | Relevance |
|--------|-----------|
| **Accuracy** | Overall correctness across both classes |
| **Precision** | Of all messages flagged as spam, how many were actually spam? |
| **Recall** | Of all actual spam messages, how many did the model catch? |
| **F1 Score** | Harmonic mean of Precision and Recall — balance metric |

### 5.3 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Naive Bayes** | **98.30%** | **98.52%** | **88.67%** | **93.33%** |
| SVM (Linear) | 98.03% | 97.76% | 87.33% | 92.25% |
| Logistic Regression | 97.94% | 100.00% | 84.67% | 91.70% |

### 5.4 Key Observations

- **Logistic Regression** achieved perfect Precision (0 false positives) but the lowest Recall — it misses more spam than the other models
- **SVM** is close to Naive Bayes across all metrics but slightly weaker on both Accuracy and F1
- **Naive Bayes** delivers the best Accuracy and highest F1 Score, making it the most balanced choice for this task
- All three models perform comfortably above 97% accuracy, reflecting the learnable structure of spam language patterns

### 5.5 Confusion Matrix (Best Model — Naive Bayes)

|  | Predicted: Ham | Predicted: Spam |
|--|----------------|-----------------|
| **Actual: Ham** | True Negatives (high) | False Positives (very low) |
| **Actual: Spam** | False Negatives (~17 missed) | True Positives (~133 caught) |

The confusion matrix confirmed that the model correctly classifies the vast majority of both classes, with a small number of spam messages going undetected (false negatives).

---

## 6. Best Model Selection

**Winner: Multinomial Naive Bayes**

Selected on the basis of:
- Highest overall accuracy: **98.30%**
- Highest F1 Score for spam class: **93.33%**
- Best balance between catching spam (Recall) and avoiding false positives (Precision)

The trained model and fitted vectorizer were serialised:

```python
pickle.dump(best_model,   open("model.pkl",     "wb"))
pickle.dump(count_vector, open("vectorizer.pkl", "wb"))
```

---

## 7. Streamlit Web Application

### 7.1 Overview

`app.py` wraps the saved model and vectorizer in a single-page web app for real-time message classification.

### 7.2 Application Flow

```
User types a message
        ↓
Click "Predict"
        ↓
Message vectorized using saved CountVectorizer
        ↓
Prediction made using saved Naive Bayes model
        ↓
Result displayed: 🚨 SPAM  or  ✅ HAM
```

### 7.3 Implementation Notes

- Both `model.pkl` and `vectorizer.pkl` are loaded at app startup using `pickle`
- The same `CountVectorizer.transform()` (not `fit_transform`) is applied to user input — ensuring vocabulary consistency with the training data
- Empty input is caught with a warning before prediction is attempted

### 7.4 Running the App

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## 8. Key Findings

- Naive Bayes is highly effective for spam detection — its probabilistic word-frequency approach aligns naturally with the task
- The Bag of Words representation, despite being simple, captures enough signal in spam language patterns to achieve 98%+ accuracy
- Spam-specific metrics (Precision, Recall, F1) paint a clearer picture than accuracy alone given the class imbalance
- The vectorizer must be saved and reused consistently — any mismatch between training vocabulary and inference vocabulary would break predictions

---

## 9. Limitations & Future Work

| Limitation | Suggested Improvement |
|------------|-----------------------|
| Bag of Words ignores word order and context | Use TF-IDF or word embeddings (Word2Vec, BERT) |
| No text preprocessing (stopwords, stemming) | Add NLTK preprocessing pipeline |
| Only 3 models compared | Extend to Random Forest, XGBoost, or fine-tuned BERT |
| Single-message prediction only | Add batch CSV upload for bulk classification |
| No confidence score shown in app | Display spam probability alongside label |

---

## 10. Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Pandas / NumPy | Data loading and manipulation |
| Scikit-learn | CountVectorizer, ML models, metrics |
| Pickle | Model and vectorizer serialisation |
| Matplotlib | Visualisation (class distribution, model comparison) |
| Streamlit | Web application deployment |

---

## 11. Project Files

| File | Description |
|------|-------------|
| `Mail_Spam_Prediction.ipynb` | End-to-end pipeline — EDA, vectorization, training, evaluation, export |
| `app.py` | Streamlit web app for real-time spam prediction |
| `model.pkl` | Serialised Multinomial Naive Bayes classifier |
| `vectorizer.pkl` | Serialised CountVectorizer (fitted on training data) |
| `spam.csv` | Source dataset — 5,572 labelled SMS messages |
| `README.md` | Quick-start guide and project summary |
| `PROJECT_REPORT.md` | This document — detailed methodology and findings |

---

*Report prepared for GitHub portfolio submission.*
