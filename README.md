# 📧 Mail Spam Prediction-Machine-Learning
 Natural Language Processing | Python, Scikit-learn, Streamlit  


A machine learning project to classify SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing and classification algorithms, deployed as an interactive web application with Streamlit.

---

## 📌 Project Overview

Spam messages are a persistent problem in digital communication. This project builds and compares multiple text classification models on an SMS dataset, selects the best performer, and deploys it through a Streamlit app where users can type any message and get an instant spam verdict.

---

## 🗂️ Repository Structure

```
mail-spam-prediction/
│
├── Mail_Spam_Prediction.ipynb   # Full ML pipeline notebook
├── app.py                       # Streamlit web application
├── model.pkl                    # Saved best model (Naive Bayes)
├── vectorizer.pkl               # Saved CountVectorizer
├── spam.csv                     # SMS dataset (5,572 messages)
└── README.md                    # Project documentation
```

---

## 📁 Dataset

**File:** `spam.csv`  
**Records:** 5,572 SMS messages  
**Source:** [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

| Column | Description |
|--------|-------------|
| `Label` | Target — `ham` (legitimate) or `spam` |
| `SMS` | Raw message text |

**Class Distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| Ham (Legitimate) | 4,825 | 86.6% |
| Spam | 747 | 13.4% |

---

## 🔬 ML Pipeline

### 1. Data Cleaning
- Dropped three unnamed/empty columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`)
- Renamed `v1` → `Label` and `v2` → `SMS`
- Confirmed zero missing values

### 2. Feature Extraction — Bag of Words
Text was converted to numerical features using `CountVectorizer`, which builds a vocabulary from the training corpus and represents each message as a word-frequency vector.

```python
count_vector = CountVectorizer()
x_train_data = count_vector.fit_transform(x_train)
x_test_data  = count_vector.transform(x_test)
```

### 3. Train-Test Split
- **80% training / 20% testing** — `random_state=42`
- Test set: 1,115 messages (965 ham, 150 spam)

### 4. Models Compared

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Naive Bayes** | **98.30%** | **98.52%** | **88.67%** | **93.33%** |
| SVM (Linear) | 98.03% | 97.76% | 87.33% | 92.25% |
| Logistic Regression | 97.94% | 100.00% | 84.67% | 91.70% |

> Metrics calculated with `pos_label='spam'`.

### 5. Best Model
**Multinomial Naive Bayes** — selected on highest accuracy (98.30%) and best overall balance of Precision and Recall for spam detection.

### 6. Model Export
Both the trained model and vectorizer were saved for use in the app:

```python
pickle.dump(best_model,   open("model.pkl",      "wb"))
pickle.dump(count_vector, open("vectorizer.pkl",  "wb"))
```

---

## 🖥️ Streamlit App

`app.py` provides a minimal, easy-to-use interface for real-time single-message classification.

**Features:**
- Free-text input area for any message
- Instant spam/ham prediction on button click
- Clear visual result — 🚨 SPAM or ✅ HAM
- Input validation (warns on empty submission)

**Sample Predictions:**
> *"Congratulations! You won a free ticket!"* → 🚨 **SPAM**  
> *"Thanks"* → ✅ **HAM**

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn streamlit
```

### Run the Notebook

Open `Mail_Spam_Prediction.ipynb` and run all cells. This trains all three models, prints the comparison table, and saves `model.pkl` and `vectorizer.pkl`.

### Launch the Web App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | NLP vectorization, ML models, metrics |
| Pickle | Model & vectorizer serialisation |
| Streamlit | Web application deployment |

---

## 👤 Author

**Vaisak**  
Data Analyst | MSc Data Science  
*Skills: Python, Machine Learning, NLP, Streamlit, SQL, Tableau, Power BI*

---

## 📝 License

This project is intended for portfolio and educational purposes.  
Dataset source: [UCI SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
