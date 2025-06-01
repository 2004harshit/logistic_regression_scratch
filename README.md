# Logistic Regression From Scratch 🧠

This repository contains a clean and modular implementation of **Logistic Regression** from scratch using **NumPy**. It includes preprocessing, model training, evaluation metrics, and unit testing — all without relying on machine learning libraries like Scikit-learn for core logic.

---

## 📦 Features

- 🔍 Train a binary classifier using gradient descent
- 📊 Custom `Metrics` class to evaluate model accuracy
- 🧪 Unit tests using **pytest**
- 🔁 Simple preprocessing pipeline for scaling features
- ⚙️ Logging setup to debug and trace execution
- 🤖 Uses `sklearn.datasets` for testing on real-world data
- ✅ GitHub Actions ready for CI automation

---

## 📁 Project Structure

```text
logistic_regression_scratch/
├── src/
│   ├── main.py               # Driver code
│   ├── preprocessing.py      # Preprocessor class
├── models/
│   └── logistic_regression.py # Model implementation
├── utils/
│   ├── metrics.py            # Accuracy metric
│   └── test_metrics.py       # Pytest test cases
├── logs/
│   └── debug.log             # Debug logs
├── requirements.txt
└── README.md

--

##  Project Journey & Roadmap

For detailed progress, milestones, and future plans, see [PROJECT_JOURNEY.md](./PROJECT_JOURNEY.md).

git clone https://github.com/your-username/logistic_regression_scratch.git
cd logistic_regression_scratch
pip install -r requirements.txt
