import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================
# 1. DATASET (Improved)
# ==============================
data = {
    "text": [
        "I love this product", "This is amazing", "I am very happy",
        "Fantastic experience", "Best purchase ever",
        "I hate this", "Very bad experience", "I am sad",
        "Worst product", "Not good", "Terrible service",
        "Absolutely wonderful", "Superb quality",
        "Awful", "Waste of money"
    ],
    "label": [1,1,1,1,1,0,0,0,0,0,0,1,1,0,0]
}

df = pd.DataFrame(data)

# ==============================
# 2. TRAIN MODEL
# ==============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

# ==============================
# 3. STREAMLIT UI
# ==============================
st.title("💬 Sentiment Analysis App")

user_input = st.text_input("Enter your sentence:")

if user_input:
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]

    if prediction == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😠")

    # ==============================
    # 4. GRAPH
    # ==============================
    labels = ['Positive', 'Negative']
    values = [prediction, 1 - prediction]

    fig, ax = plt.subplots()
    ax.bar(labels, values)

    st.pyplot(fig)