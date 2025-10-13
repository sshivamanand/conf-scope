import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ---------------------------
# 1️⃣ Load dataset
# ---------------------------
df = pd.read_csv("conf-scope/server/data/parsed_20251013_115823.csv")

# Combine title + introduction
def combine_text(row):
    title = str(row.get("title", ""))
    intro = str(row.get("introduction", ""))
    return f"{title} {intro}"

df["text"] = df.apply(combine_text, axis=1)

# ---------------------------
# 2️⃣ Clean text
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# ---------------------------
# 3️⃣ Encode labels
# ---------------------------
label_map = {
    "acl": "100",
    "conll": "010",
    "iclr": "001",
    "rejected": "000",
}

df["label_str"] = df["folder_label"].map(label_map)
df = df.dropna(subset=["label_str"])

# Map each label to integer
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label_str"])

# Label encoder mapping
print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# ---------------------------
# 4️⃣ Split data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ---------------------------
# 5️⃣ TF-IDF features
# ---------------------------
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Convert to dense arrays for neural net
X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()

# One-hot encode labels
num_classes = len(label_encoder.classes_)
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# ---------------------------
# 6️⃣ Build Neural Network
# ---------------------------
model = Sequential([
    Dense(512, input_shape=(X_train_dense.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  # Output probabilities
])

optimizer = Adam(learning_rate=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

# ---------------------------
# 7️⃣ Train the model
# ---------------------------
history = model.fit(
    X_train_dense, y_train_cat,
    validation_data=(X_test_dense, y_test_cat),
    epochs=10,
    batch_size=32,
    verbose=1
)

# ---------------------------
# 8️⃣ Evaluate model
# ---------------------------
loss, acc = model.evaluate(X_test_dense, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {acc:.3f}")

# ---------------------------
# 9️⃣ Predict probabilities for new input
# ---------------------------
def predict_conference(title, intro):
    text = clean_text(f"{title} {intro}")
    vec = vectorizer.transform([text]).toarray()
    probs = model.predict(vec, verbose=0)[0]
    
    labels = label_encoder.classes_
    result = {labels[i]: float(probs[i]) for i in range(len(labels))}
    
    # Normalize to percentages (optional)
    for key in result:
        result[key] = round(result[key], 3)
    
    return result

# Example usage
example = predict_conference(
    "Neural Morphological Inflection",
    "We propose a neural sequence model for morphological inflection..."
)
print("\nPrediction Example:")
print(example)
