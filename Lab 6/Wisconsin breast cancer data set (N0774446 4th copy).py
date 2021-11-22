

# sklearn.datasets.load_breast_cancer(return_X_y=False)
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

columns ="ID Diagnosis radius texture perimeter area smoothness compactness concavity concave_points symmetry fracta dimension".split
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=columns)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.4)

print (X_train.shape, Y_train.shape)
print (X_test.shape, Y_test.shape)
