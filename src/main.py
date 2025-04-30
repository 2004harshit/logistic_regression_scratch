import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from models.logistic_regression import LogisticRegression
from src.preprocessing import Preprocessor
import logging

logging.basicConfig(level = logging.DEBUG,
                    filename='logs/debug.log',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    )
# Dummy data
X = np.array([[99, 2], [1, 33], [12, 3], [3, 4]])
y = np.array([0, 0, 1, 1])

pr = Preprocessor(X)

x_scaled = pr.transform(X)

print("Original X: ",X)
print("Transformed X:",x_scaled)

# Train
model = LogisticRegression(learning_rate=0.1, n_iters=1000)
model.fit(x_scaled, y)

# Predict
preds = model.predict(x_scaled)
print("Predictions:", preds)
