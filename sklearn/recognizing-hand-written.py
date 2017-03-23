import tkinter
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

digits = datasets.load_digits()

for key, value in digits.items():
    try:
        print (key, value.shape)
    except:
        print (key)
