import pandas as pd
import re
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('wordnet')

# Load dataset
df = pd.read_csv("Reviews.csv", nrows=80000)
df = df.dropna(subset=["Text", "Score"])
df = df[["Text", "Score"]]

# Convert rating to sentiment (for training labels only)
def convert_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score <= 2:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Score"].apply(convert_sentiment)

# Clean text
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

df["Cleaned_Text"] = df["Text"].apply(clean_text)

# Improved TF-IDF (with bigrams)
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["Cleaned_Text"])
y = df["Sentiment"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Accuracy:", accuracy)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Text-based NLP model trained successfully.")