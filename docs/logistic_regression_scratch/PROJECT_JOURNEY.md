# 🧠 Project Journey: Logistic Regression from Scratch

> A self-driven implementation of Logistic Regression with a focus on clean code, testing, automation, and best practices.

---

## 📅 Project Timeline

- **Start Date:** 18 April 2025  
- **Current Phase:** Enhancing functionality & preparing for implementing KNN  
- **Tools Used:** Python, NumPy, Matplotlib, Scikit-learn (for benchmarking), Pytest, GitHub Actions, Black, Flake8, isort, pre-commit

---

## 🚀 Goals

- Implement logistic regression from scratch using NumPy  
- Follow clean coding and reproducibility practices  
- Apply code automation (CI, linting, formatting)  
- Build comparisons with `scikit-learn`  
- Visualize performance metrics  
- Simulate real-world practices (like testing, CI/CD, and version control)  

---

## ✅ Milestones

### 🔹 Week 1: Define project structure and file organization

- Designed folder structure (`src`, `utils`, `models`, etc.)  
- Implemented basic functions: sigmoid, cost function, gradient descent  
- Created the main LogisticRegression class  
- Created `main.py` for testing predictions  

### 🔹 Week 2: Implement preprocessing utility

- Implement standard and min-max scaling  
- Fixed a bug causing negative cost  
- Wrote unit tests using `pytest`  

### -- Examination break for 15 days --

### 🔹 Week 3: Automate test execution, code quality, and pre-commit hooks

- Created `.github/workflows/python.yml` to automate test execution on `push` and `pull_request`  
- Fixed bugs in gradient logic after test failures  
- Used GitHub Actions to simulate professional testing workflows  
- Installed and configured `black`, `flake8`, `isort` for formatting and linting  
- Set up `.pre-commit-config.yaml` and integrated pre-commit hooks  
- Ensured code formatting and linting run before any `git commit`  

### 🔹 Week 4: Compare custom model and sklearn model, identify future improvements

- Implemented comparison notebook (`comparison.ipynb`)  
- Explored future enhancement opportunities  

---

## 📂 Project Structure (as of now)

*(You may fill this section later as needed)*

---

## ✅ What I Have Done So Far

### Project Setup & Structure
- Designed and implemented a clear, modular folder structure (`src`, `models`, `utils`, etc.)  
- Created core files: `main.py`, `logistic_regression.py`, `metrics.py`, and preprocessing utilities  

### Model Implementation
- Built Logistic Regression model from scratch using NumPy  
- Implemented key components: sigmoid function, cost function, gradient descent  
- Developed model training (`fit`) and prediction (`predict`) methods  

### Preprocessing Pipeline
- Created a Preprocessor class for feature scaling (standardization)  
- Handled scaling bugs and ensured data integrity during transformation  

### Testing & Quality Assurance
- Wrote comprehensive unit tests for the Metrics class using `pytest`  
- Set up CI pipeline with GitHub Actions for automated test runs on every commit  
- Integrated code quality tools: `black`, `flake8`, `isort` and pre-commit hooks for linting and formatting  

### Experimentation & Benchmarking
- Tested model on real-world dataset (`breast_cancer` from sklearn)  
- Compared model predictions with ground truth labels using custom accuracy metric  

### Documentation & Automation
- Created detailed `README.md` and project journey documentation  
- Configured logging for debugging and traceability during training and evaluation  

---

## 🚧 Backlog & Future Work

To keep track of the next features and improvements to implement for a robust logistic regression from scratch project:

- **Preprocessing Enhancements:**  
  - Null value handling  
  - Outlier detection/removal  
  - Encoding categorical features  
  - Complete transform pipeline  

- **Model Improvements:**  
  - Loss tracking during training  
  - Probability prediction (`predict_proba`)  
  - Custom thresholding in prediction  

- **Metrics & Visualization:**  
  - Precision, recall, F1-score  
  - Confusion matrix & visualizations  
  - Benchmark plots comparing with sklearn  

- **Usability & Features:**  
  - CLI/config-driven parameters  
  - Model persistence (save/load)  
  - Robust exception handling  

- **Testing & Automation:**  
  - Unit tests for model and preprocessing  
  - Automated CI testing  
  - Code linting and formatting setup  

---

## 🔜 Upcoming Plans

- Add more metrics: F1, ROC AUC, precision/recall curve  
- Implement `visualize.py` for real-time performance plots  
- Expand to other supervised models: SVM, Decision Tree, KNN  
- Deploy on Streamlit or FastAPI as part of MLOps integration  
- Eventually build a dashboard comparing different models  

---

## ✍️ Reflections

- Learned how to integrate **industry-grade tooling** in a small ML project  
- Understood how **CI/CD, unit testing, and code formatting** matter, even for solo projects  
- Realized the value of **modular, readable, and scalable code** early in the ML journey  
- Tracking micro-goals in Jira helped stay clear-headed and focused  

---

## 💡 Inspiration

This project is designed to help me:  
- Prepare for machine learning internships  
- Build a compelling GitHub profile  
- Connect my academic learning with real-world engineering practices  

---
