import os
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# --- Download NLTK resources if not already ---
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# --- 1. Load dataset ---
script_dir = os.path.dirname(__file__)
dataset_path = "../data/processed/dataset.csv"
df = pd.read_csv(os.path.abspath(os.path.join(script_dir, dataset_path)))  # columns: 'rating' and 'review'

# --- 2. Map ratings to sentiments ---

def map_sentiment(rating):
    if rating <= 4:
        return 0  # Negative
    elif rating >= 7:
        return 1  # Positive
    else:
        return None  # Neutral reviews ignored

df['sentiment'] = df['rating'].apply(map_sentiment)
df = df.dropna(subset=['sentiment'])
df['sentiment'] = df['sentiment'].astype(int)

# --- 3. Text cleaning function ---

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # URLs
    text = re.sub(r'<.*?>', '', text)  # HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation))  # punctuation
    text = re.sub(r'\d+', '', text)  # numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

print("Cleaning reviews...")
df['clean_review'] = df['review'].apply(clean_text)

# --- 4. Split into train and test sets ---

X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. Create and train the model pipeline ---

model = make_pipeline(TfidfVectorizer(max_features=10000), MultinomialNB())

print("Training the model...")
model.fit(X_train, y_train)

# --- 6. Evaluate the model ---

print("Testing the model...")
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
