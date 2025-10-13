# Conference Scope Predictor

This is a **full-stack web application** that predicts the likelihood of a research project being accepted by different conferences based on its **title** and **introduction**. The model returns probabilities for each class along with a predicted label.

---

## 🎯 Features

- Input a **project title** and **introduction**.
- Returns **predicted probabilities** for each class:
  - `'000'` → Rejected
  - `'001'` → ICLR
  - `'010'` → CoNLL
  - `'100'` → ACL
- Predicts the **most likely category**.
- Built with **Python (Flask/TensorFlow)** for the backend and any frontend framework for UI.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/conference-scope-predictor.git
cd conference-scope-predictor
```

### 2. Create a virtual environment and activate it

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / MacOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place your trained model

Place your trained model and vectorizer in the appropriate folder (e.g., `models/`).

---

## 🧠 Usage

### 1️⃣ Start the backend server

```bash
python app.py
```

### 2️⃣ Access the frontend

Open your browser at `http://localhost:5000` and input:
- Title of the project
- Introduction of the project

### 3️⃣ Prediction

The backend will return:

```json
{
  "000": 0.12,
  "001": 0.68,
  "010": 0.05,
  "100": 0.15
}
```

Where each key is a class, and the value is the predicted probability. The highest probability is the predicted class.

---

## 🧩 Example

```python
from predictor import predict_conference

result = predict_conference(
    "Neural Morphological Inflection",
    "We propose a neural sequence model for morphological inflection..."
)

print(result)
# Output: {'000': 0.12, '001': 0.68, '010': 0.05, '100': 0.15}
```

---

## 📊 Model Details

- **Architecture**: Dense Neural Network (512 → 256 → output)
- **Input**: TF-IDF vectorized text (title + introduction)
- **Output**: Probabilities for 4 classes (softmax)
- **Metrics**:
  - Training Accuracy: ~0.98
  - Test Accuracy: ~0.86
  - Additional metrics: Precision, Recall, F1-score

---

## ⚡ Notes

- Currently, the model uses title + introduction. If you only input the title, it still works but predictions may be slightly less accurate.
- For deployment, ensure the frontend sends the inputs as JSON to the backend API.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, TensorFlow/Keras
- **Frontend**: HTML, CSS, JS (or your preferred frontend framework)
- **Data Processing**: pandas, scikit-learn
- **Model**: TF-IDF + Dense Neural Network

---
