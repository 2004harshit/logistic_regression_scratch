import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from models.logistic_regression import LogisticRegression
import logging

logging.basicConfig(level = logging.DEBUG,
                    filename='logs/debug.log',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    )
# Dummy data
X = np.array([[1, 2], [1, 3], [2, 3], [3, 4]])
y = np.array([0, 0, 1, 1])

# Train
model = LogisticRegression(learning_rate=0.1, n_iters=1000)
model.fit(X, y)

# Predict
preds = model.predict(X)
print("Predictions:", preds)
