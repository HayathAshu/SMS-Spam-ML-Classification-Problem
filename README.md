# 📩 SMS Spam Classification using Machine Learning

A Machine Learning project that classifies SMS messages as **Spam** or **Ham** using **Natural Language Processing (NLP)** techniques, **TF-IDF vectorization**, and multiple classification models.  
This project uses the popular **SMS Spam Collection Dataset** from Kaggle.

---

## 📌 Project Overview

The goal of this project is to automatically detect whether a given SMS message is spam or not.  
To achieve this, the text data is cleaned, transformed into numerical form, and used to train different ML models.

The project compares four machine learning algorithms:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Naive Bayes (MultinomialNB)  
- Random Forest Classifier  

**Naive Bayes achieved the best performance** for this dataset.

---

## 🧠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  
- Matplotlib / Seaborn (for visualization)  
- Jupyter Notebook  

---

## 🧹 Text Preprocessing Steps

Before training the model, each SMS message goes through the following NLP steps:

1. Convert to lowercase  
2. Remove punctuation  
3. Remove stopwords  
4. Apply stemming  
5. Transform text using **TF-IDF Vectorization**

---

## ⚙️ Machine Learning Models Used

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear classification model |
| **SVM** | Strong classifier for high-dimensional data |
| **Naive Bayes** | Best for text classification |
| **Random Forest** | Ensemble model for non-linear patterns |

---

## 🏆 Best Performing Model

### ✅ **Naive Bayes Classifier**

Naive Bayes performed best because:

- Works extremely well with text data  
- Efficient with high-dimensional TF-IDF vectors  
- Fast and accurate for spam detection tasks  

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

---

## 📦 Dataset Source

The dataset used in this project is taken from Kaggle:

📌 **Kaggle Dataset:**  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

It contains a collection of SMS messages labeled as **spam** or **ham**, making it ideal for text-based classification.

---

## 📥 Download Dataset Using KaggleHub

You can automatically download the dataset using the following code:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")

print("Path to dataset files:", path)
