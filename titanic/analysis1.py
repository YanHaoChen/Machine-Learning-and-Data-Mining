# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
from pandas import DataFrame, Series
import matplotlib.pyplot as plot

data=pd.read_csv("./train.csv")
sur_and_age = data[['Survived','Age']]

mean_age = round(sur_and_age.Age.mean())
sur_and_age.Age = sur_and_age.Age.replace(np.nan, mean_age)

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