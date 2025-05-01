import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import Metrics
import pytest

def test_correct_accuracy():
    mr = Metrics()
    actual_y = [0, 0, 1, 1]
    predicted_y = [0, 0, 1, 0]

    assert mr.accuracy(actual_y , predicted_y) == 75.00

def test_all_correct():
    mr = Metrics()
    assert mr.accuracy([1,0,1,0],[1,0,1,0])==100.00

def test_all_wrong():
    mr = Metrics()
    assert mr.accuracy([1,0],[0,1])==0.0

def test_mismatch_length():
    mr = Metrics()
    with pytest.raises(ValueError):
        mr.accuracy([1,0,1,1], [1,0])

def test_empty_input():
    mr = Metrics()
    with pytest.raises(ValueError):
        mr.accuracy([],[])

def test_non_array_input():
    mr = Metrics()
    with pytest.raises(ValueError):
        mr.accuracy("abc" , [1,0])