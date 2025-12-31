import os
import pandas as pd
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model

model = load_model("artifacts/imdb_review_model.keras")

with open("artifacts/tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)

def load_review(base_path):
    data=[]
    labels=[]
    for path in ['pos', 'neg']:
        fold_path = os.path.join(base_path, path)
        for file in os.listdir(fold_path):
            if file.endswith('.txt'):
                file_path = os.path.join(fold_path, file)
                with open(file_path, 'r', encoding="utf-8") as f:
                    review = f.read()
                    data.append(review)
                    labels.append(1 if path=='pos' else 0)
    return pd.DataFrame({
        "reviews" : data,
        "sentiments" : labels
    })
    
    
def clean_text(t):
    text = t.lower()
    text = re.sub(r'<.*?>', '', text)      # remove HTML
    text = re.sub(r'[^a-z\s]', '', text)   # remove numbers & punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def predict(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=200, padding='post')
    pred = model.predict(pad)[0][0]
    # sentiment ="Positive" if pred > 0.5 else "Negative"
    if pred >= 0.6:
        sentiment = "Positive"
    elif pred >= 0.3 and pred < 0.6:
        sentiment = "Neutral"
    else:
        sentiment = "Negative"
    return pred, sentiment