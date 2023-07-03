import numpy as np
import matplotlib.pyplot as plt
import plotext as plx
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

raw_data = pd.read_csv('Dataset_Trabalho.csv', sep=';')
print(raw_data.describe())

X = np.array(raw_data.values[1:, :10])
Y = raw_data.values[1:, -1]

# splitting X and y into training and testing sets
X_train, X_test,\
	y_train, y_test = train_test_split(X, Y,
									test_size=0.8,
									random_state=1)

ax = sns.countplot(x='Target', data=raw_data)

# Extract the counts from the countplot
x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
counts = [rect.get_height() for rect in ax.patches]
print("Distribution [Dropout, Graduate, Enrolled]: ", counts)

# Create an ASCII bar plot using Plotext
plx.bar(x_labels, counts)
plx.title('Classes distribution')
plx.xlabel('Classes')
plx.ylabel('Count')

# Display the countplot in the terminal
plx.show()

# cov_matrix = np.corrcoef(X)

# plx.matrix_plot(cov_matrix)
# plx.plotsize(np.size(cov_matrix, 0), np.size(cov_matrix, 1))
# plx.title("Covariance Matrix")
# plx.show()

# Plot correlation matrix
# plt.figure(figsize=(8, 8))
# plt.imshow(cov_matrix, interpolation='nearest')
# plt.title('Covariance Matrix')
# plt.colorbar()
# plt.show()

# Create an instance of Logistic Regression Classifier and fit the data.
logreg = LogisticRegression(C=1e5, max_iter=1000)
logreg.fit(X_train, y_train)

y_pred=logreg.predict(X_test) 

cnf_matrix = metrics.confusion_matrix(y_test, y_pred) 

print(cnf_matrix) 

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) 

print("Precision:",metrics.precision_score(y_test, y_pred, average='weighted', zero_division=1)) 

print("Recall:",metrics.recall_score(y_test, y_pred, average='weighted', zero_division=1))