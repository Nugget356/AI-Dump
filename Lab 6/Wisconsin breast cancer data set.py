


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

columns ="ID Diagnosis radius texture perimeter area smoothness compactness concavity concave_points symmetry fracta dimension".split
data = load_breast_cancer()
print(data)

