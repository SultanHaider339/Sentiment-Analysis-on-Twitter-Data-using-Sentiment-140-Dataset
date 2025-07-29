# 🧠 Sentiment Analysis on Twitter Data using Sentiment140 Dataset

This project performs **sentiment analysis** on Twitter data using the **Sentiment140 dataset**, which contains 1.6 million labeled tweets. It includes a complete machine learning pipeline: data preprocessing, feature engineering, model training, and evaluation using ensemble methods.

---

## 📂 Dataset Overview
- **Dataset Link**: https://www.kaggle.com/datasets/kazanova/sentiment140
- **Dataset**: Sentiment140 (CSV format)
- **Size**: 1.6 million tweets
- **Columns Used**:
  - **Sentiment Label**: `0` = Negative, `4` = Positive
  - **Tweet Text**: Raw tweet content

---

## 🧹 Data Preprocessing

The preprocessing pipeline cleans the raw tweets to prepare them for modeling:

- Remove Twitter handles (`@username`)
- Remove hashtags (`#hashtag`)
- Strip URLs (`http://...`)
- Remove emojis and special characters
- Convert text to lowercase
- Remove newlines and quotation marks
- Tokenization using `nltk`
- Stopword removal
- Lemmatization using `WordNetLemmatizer`

### Sample Preprocessing Function

```python
def preprocess_article(text: str) -> str:
    text = re.sub('@[^\\s]+', '', text)
    text = re.sub(r'\\B#\\S+', '', text)
    text = re.sub(r"http\\S+", "", text)
    text = text.lower()
    ...
    return cleaned_text
