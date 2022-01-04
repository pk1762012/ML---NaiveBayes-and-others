import pandas as pd
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sympy.physics.quantum.circuitplot import matplotlib
digits = load_digits()

print(digits.data[0])
plt.gray()
#for i in range(5):
 #   plt.matshow(digits.images[i])
  #  matplotlib.pyplot.show()

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

model.fit(X_train, y_train)
model.predict(digits.data[0:5])

y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')