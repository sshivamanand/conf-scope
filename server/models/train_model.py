import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import joblib

df = pd.read_csv("../data/parsed_20251013_115823.csv")
print(f"Dataset loaded: {df.shape[0]} samples")

def combine_text(row):
    title = str(row.get("title", ""))
    intro = str(row.get("introduction", ""))
    return f"{title} {intro}"

df["text"] = df.apply(combine_text, axis=1)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

label_map = {
    "acl": "100",
    "conll": "010",
    "iclr": "001",
    "rejected": "000",
}

df["label_str"] = df["folder_label"].map(label_map)
df = df.dropna(subset=["label_str"])

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label_str"])
num_classes = len(label_encoder.classes_)
print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

vectorizer = TfidfVectorizer(max_features=7500, ngram_range=(1, 2), stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights}")

y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

model = Sequential([
    Dense(256, input_shape=(X_train_dense.shape[1],), activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    X_train_dense, y_train_cat,
    validation_data=(X_test_dense, y_test_cat),
    epochs=15,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

train_loss, train_acc = model.evaluate(X_train_dense, y_train_cat, verbose=0)
test_loss, test_acc = model.evaluate(X_test_dense, y_test_cat, verbose=0)

print("\n================= RESULTS =================")
print(f"Training Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

y_pred = np.argmax(model.predict(X_test_dense), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

model.save("conf_scope_model.h5")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("\nModel, vectorizer, and label encoder saved successfully.")

def predict_conference(title, intro):
    text = clean_text(f"{title} {intro}")
    vec = vectorizer.transform([text]).toarray()
    probs = model.predict(vec, verbose=0)[0]
    
    labels = label_encoder.classes_
    result = {labels[i]: round(float(probs[i]), 3) for i in range(len(labels))}
    return result

example = predict_conference(
    "Measures of Distributional Similarity",
    '''In this work, Lee explores various distributional similarity measures to improve the estimation of probabilities for unseen co-occurrences. The study offers three main contributions: Empirical Comparison: An extensive evaluation of a broad range of similarity measures. Classification Framework: A categorization of similarity functions based on the information they incorporate.
    Introduction of a Novel Function: The proposal of a new similarity measure that outperforms existing ones in evaluating potential proxy distributions. These contributions aim to enhance the accuracy of probability estimations for word pairs that are not directly observed in training data, thereby improving various NLP tasks that rely on such estimations.'''
)

print("\nPrediction Example:")
print(example)
