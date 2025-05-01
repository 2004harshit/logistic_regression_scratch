import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from models.logistic_regression import LogisticRegression
from src.preprocessing import Preprocessor
import logging
import pandas as pd
from sklearn.datasets import load_breast_cancer
from utils.metrics import Metrics


logging.basicConfig(level = logging.DEBUG,
                    filename='logs/debug.log',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    )
# Dummy data

data = load_breast_cancer()
X, y = data.data, data.target

pr = Preprocessor(X)

x_scaled = pr.transform(X)



# Train
model = LogisticRegression(learning_rate=0.1, n_iters=1000)
model.fit(x_scaled, y)

# Predict
preds = model.predict(x_scaled)
print("Predictions:", preds)

mr = Metrics()

print("Accuracy : ",mr.accuracy(y, preds))