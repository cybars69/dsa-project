# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %%
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.interpolate import make_interp_spline
# from sklearn.impute import SimpleImputer


# %%
# In the previous project, we compared the accuracies by training
# the classifier on different classification models
# (like Logistic Regression, SVM etc.). In this project, we will be
# evaluating model performance using XG Boost on the 'Stroke Prediction Dataset'
# from kaggle.

df = pd.read_csv(
    './data/heart.csv')

df.head()


# %%
df.columns


# %%
# x is independent variables
# y consists of dependent variable

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# %%
# This shows a list of features that consist of null values

df[df.columns].isnull().sum()

# Luckily we have no null values


# %%
# THIS BLOCK IS USED INCASE WE HAVE ANY NULL VALUES.

# Replacing the null value for that feature with the mean of all other values

# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(x[:, [9]])
# x[:, [9]] = imputer.transform(x[:, [9]])


# %%
# Encoding categorical data

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [
                       1, 2, 6, 8, 10])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


# %%
def trainAndTest(testSize, x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=testSize, random_state=0)

    sc = StandardScaler()
    x_train[:, 14:20] = sc.fit_transform(x_train[:, 14:20])
    x_test[:, 14:20] = sc.transform(x_test[:, 14:20])

    classifier = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    accuracy_score_ = accuracy_score(y_test, y_pred)

    return("{:.0f}-{:.4f}".format((1-testSize)*100, accuracy_score_))


# %%
accuracy = []
trainingSize = []
for tt in range(10, 40):
    res = trainAndTest(tt*1.0/100, x, y)
    size, accuracy_score_ = res.split('-')
    trainingSize.append(float(size))
    accuracy.append(float(accuracy_score_))
    print("Accuracy for training on {}% of the data - {}".format(size, accuracy_score_))

trainingSize = np.array(trainingSize)
accuracy = np.array(accuracy)

# %%
trainingSizeNew = np.copy(trainingSize)
accuracyNew = np.array(accuracy)

plt.figure(figsize=(10, 10))
plt.plot(trainingSizeNew, accuracyNew*100)
plt.show()
