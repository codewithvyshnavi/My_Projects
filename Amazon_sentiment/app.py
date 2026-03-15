import streamlit as st
import pickle
import re
import numpy as np
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="NLP Sentiment Analyzer")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

st.title("Text-Based Sentiment Analysis (NLP)")
review_text = st.text_area("Enter Review Text")

if st.button("Predict Sentiment"):

    if review_text.strip() == "":
        st.error("Please enter review text.")
    else:
        cleaned = clean_text(review_text)
        X = vectorizer.transform([cleaned])

        probs = model.predict_proba(X)[0]
        classes = model.classes_

        index = np.argmax(probs)
        sentiment = classes[index]
        confidence = probs[index]

        # Color Box
        if sentiment == "Positive":
            color = "#28a745"
            emoji = "😊"
        elif sentiment == "Negative":
            color = "#dc3545"
            emoji = "😡"
        else:
            color = "#ffc107"
            emoji = "😐"

        st.markdown(f"""
            <div style="
                background-color:{color};
                padding:20px;
                border-radius:12px;
                text-align:center;
                color:white;
                font-size:22px;
                font-weight:bold;">
                {emoji} Predicted Sentiment: {sentiment}
            </div>
        """, unsafe_allow_html=True)

        st.write(f"### Confidence: {round(confidence*100,2)}%")