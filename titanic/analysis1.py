# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
from pandas import DataFrame, Series
import matplotlib.pyplot as plot

data=pd.read_csv("./titanic/train.csv")
sur_and_age = data[['Survived','Age']]
# mice
# mean_age = round(sur_and_age.Age.mean())
# sur_and_age.Age = sur_and_age.Age.replace(np.nan, mean_age)

percent_of_live =sur_and_age.Survived[sur_and_age.Survived == 1].count() / sur_and_age.Survived.count()
live_and_age = sur_and_age[sur_and_age.Survived == 1]
live_and_age.describe()
ax = sns.distplot(live_and_age.Age, kde=False)
ax.figure.savefig("./images/live_and_age.png")

dead_and_age = sur_and_age[sur_and_age.Survived == 0]
dead_and_age.describe()
plot.cla()
ax = sns.distplot(dead_and_age.Age, kde=False)
ax.figure.savefig("./images/dead_and_age.png")

man = sur_and_age[(sur_and_age.Age >= 20)&(sur_and_age.Age <= 35)]
percent_of_man = len(man) / len(sur_and_age)
print (percent_of_man)

teen = sur_and_age[(sur_and_age.Age < 18)]
man = sur_and_age[(sur_and_age.Age >= 18)&(sur_and_age.Age < 40)]
old = sur_and_age[(sur_and_age.Age >= 40)]

teen_live_ratio = len(teen[teen.Survived ==1]) / len(teen)
man_live_ratio = len(man[man.Survived ==1]) / len(man)
old_live_ratio = len(old[old.Survived ==1]) / len(old)

print ("小孩：%f" % teen_live_ratio)
print ("青壯年：%f" % man_live_ratio)
print ("年長者：%f" % old_live_ratio)
data.count()
# Using Imputer to replace the NaN value in Age.
from sklearn.preprocessing import Imputer

imp = Imputer(strategy='most_frequent', axis=0)
age_notnull = data[pd.isnull(data.Age)!= True][['Age']]
imp.fit(age_notnull[:500]['Age'])
result = imp.transform(data.Age[:500])
result2 = imp.transform(data.Age[-500:])
result = np.append(result[0],result2[0][109:])
# Before
print(pd.isnull(data.Age).value_counts())
data['Age'] = pd.Series(result, index=data.index)
# After
print(pd.isnull(data.Age).value_counts())

data.Sex[data.Sex=='male'] =1
data.Sex[data.Sex=='female']=0
from sklearn.decomposition import PCA

pca = PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,svd_solver='auto', tol=0.0, whiten=False)
care_data =data[['Sex','Pclass','Age','SibSp','Parch','Fare','Survived']] 
pca.fit(care_data)
# The ratios of those components can be repesent the data.
# First component
print(pca.explained_variance_ratio_[0])
# 0.921072239423

# Second component
print(pca.explained_variance_ratio_[1])
# 0.0779527221514

# We can just focus on the first component.
print(pca.components_[0])
selected_feature = ['Sex','Pclass','Age','SibSp','Parch','Fare','Survived']
for i in range(0,7):
    print ("%10s: %10f" % (selected_feature[i], pca.components_[0][i]))

#       Sex:  -0.001747
#    Pclass:  -0.009275
#       Age:   0.028186
#     SibSp:   0.003504
#     Parch:   0.003484
#      Fare:   0.999543
#  Survived:   0.002512

# We can observe Fare which is the most important feature.

# Using LogisticRegression
from sklearn.cross_validation import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
care_data =data[['Sex','Pclass','Age','SibSp','Parch','Fare']] 
x_train, x_test, y_train, y_test = tts(care_data.values, data[['Survived']].values, test_size=0.7, random_state=5)

lr = LogisticRegression()
lr.fit(x_train, y_train)
test_predict = lr.predict(x_test)
report = classification_report(y_test,test_predict, digits=5)
print (report)

#              precision    recall  f1-score   support
#
#           0    0.80361   0.92708   0.86094       384
#           1    0.84530   0.63750   0.72684       240
#
# avg / total    0.81965   0.81571   0.80937       624

# Using SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
tuned_parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10,100]}
svr = SVC()
gscv = GridSearchCV(svr,tuned_parameters)
gscv.fit(x_train, y_train[:,0])
test_predict = gscv.predict(x_test)

report = classification_report(y_test, test_predict,digits=5)
print(report)

#              precision    recall  f1-score   support
#
#           0    0.82452   0.89323   0.85750       384
#           1    0.80288   0.69583   0.74554       240
#
# avg / total    0.81620   0.81731   0.81444       624