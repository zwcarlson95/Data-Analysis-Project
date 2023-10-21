import numpy as np
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------------------------
# Preprocessing the data
# ---------------------------------------------------------------------------------------------


# Importing the dataset
dataset = pd.read_csv('/Users/zachcarlson/Downloads/Social_Network_Ads.csv')
print(dataset)
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ---------------------------------------------------------------------------------------------
# Perform data analysis
# ---------------------------------------------------------------------------------------------


avg_age = dataset['Age'].mean()
avg_salary = dataset['EstimatedSalary'].mean()

print('\nAverage Age: ' + str(round(avg_age, 2)))
print('Average Salary: ' + str(round(avg_salary)))


describe_age = dataset['Age'].describe()
print("\nAge Analysis:")
print(describe_age)

describe_salary = dataset['EstimatedSalary'].describe()
print("\nSalary Analysis:")
print(describe_salary)


dataset.hist(column='Age')
plt.show()

dataset.hist(column='EstimatedSalary')
plt.show()


# ---------------------------------------------------------------------------------------------
# Create a decision tree
# ---------------------------------------------------------------------------------------------


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("\nDecision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred))

tree.plot_tree(clf)
plt.show()


# ---------------------------------------------------------------------------------------------
# Create  K-NN model
# ---------------------------------------------------------------------------------------------


# Training the K-NN model on the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print("\nKNN Accuracy: ", ac)


# ---------------------------------------------------------------------------------------------
# Create a Naive Bayes classifier
# ---------------------------------------------------------------------------------------------


print("\nNumber of observations and dimensions in 'X':", X.shape)
print("Number of observations in 'y':", y.shape)


print("Number of observations and dimensions in training set:", X_train.shape)
print("Number of observations and dimensions in test set:", X_test.shape)
print("Number of observations in training set:", y_train.shape)
print("Number of observations in test set:", y_test.shape)


nbModel = GaussianNB()
nbModel.fit(X_train, y_train)

y_pred = nbModel.predict(X_test)

print("")
print(confusion_matrix(y_test, y_pred), ": is the confusion matrix")
print(accuracy_score(y_test, y_pred), ": is the accuracy score")
print(precision_score(y_test, y_pred), ": is the precision score")
print(recall_score(y_test, y_pred), ": is the recall score")
print(f1_score(y_test, y_pred), ": is the f1 score")


probs = nbModel.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()








