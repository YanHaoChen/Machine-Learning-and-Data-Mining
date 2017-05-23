# Titanic (Survived vs Age)

分析 Titanic 資料集中，`Survived`與`Age`間的關係。

### 引入需要的模組及取得資料

```python
import pandas as pd
import numpy as np
import seaborn as sns
from pandas import DataFrame, Series
import matplotlib.pyplot as plot

data=pd.read_csv("./train.csv")
sur_and_age = data[['Survived','Age']]
```

### 資料前處理

因年齡（Age）此項目中，很多是沒有正確記錄的（`NaN`）。所以在此，將不知道年齡的紀錄，替換成平均值（`mean`）：

```python
mean_age = round(sur_and_age.Age.mean())
# mean_age = 30.0
sur_and_age.Age = sur_and_age.Age.replace(np.nan, mean_age)
```

### 計算存活率

首先探究整體的存活率，瞭解大致狀況。

```python
percent_of_live =sur_and_age.Survived[sur_and_age.Survived == 1].count() / sur_and_age.Survived.count()
# percent_of_live = 0.38383838383838381
```
可以發現，超過六成的人再見了。

接下來，分析**存活**跟**再見**的年齡分佈是如何。

### 存活年齡分佈
透過統計數據及直條圖觀察分佈狀況。

```python
live_and_age = sur_and_age[sur_and_age.Survived == 1]
live_and_age.describe()
#        Survived         Age
# count     342.0  342.000000
# mean        1.0   28.595526
# std         0.0   13.776751
# min         1.0    0.420000
# 25%         1.0   21.000000
# 50%         1.0   30.000000
# 75%         1.0   35.000000
# max         1.0   80.000000

ax = sns.distplot(live_and_age.Age, kde=False)
ax.figure.savefig("./images/live_and_age.png")
```
![live\_and\_age](https://raw.githubusercontent.com/YanHaoChen/Machine-Learning-and-Data-Mining/master/titanic/images/live_and_age.png)

其中可以發現 21 歲至 35 歲的區間就佔了一半。也因此，判斷青壯年存佔存活人口的大宗。

接下來，探究再見的年齡分佈。

### 再見年齡分佈
在此一樣透過統計數據及直條圖觀察分佈狀況。

```python
dead_and_age = sur_and_age[sur_and_age.Survived == 0]
dead_and_age.describe()
#        Survived         Age
# count     549.0  549.000000
# mean        0.0   30.483607
# std         0.0   12.454065
# min         0.0    1.000000
# 25%         0.0   23.000000
# 50%         0.0   30.000000
# 75%         0.0   35.000000
# max         0.0   74.000000

plot.cla()
ax.sns.distplot(dead_and_age.Age, kde=False)
ax.figure.savefig("./images/dead_and_age.png")
```
![dead\_and\_age](https://raw.githubusercontent.com/YanHaoChen/Machine-Learning-and-Data-Mining/master/titanic/images/dead_and_age.png)

結果顯示，23 歲至 35 歲的區間也是一樣佔了一半。青壯年也是再見人口的大宗。

兩個分析結果，都顯示青壯年的人數皆為大宗。因此，進一步確認青壯年在資料中的比例為何。

### 青壯年在資料中比率

藉由得知比率，進一步解釋上述的狀況。這此將青壯年的區間假設為 20-35 歲。

```python
man = sur_and_age[(sur_and_age.Age >= 20)&(sur_and_age.Age <= 35)]
percent_of_man = len(man) / len(sur_and_age)
print (percent_of_man)
# percent_of_man = 0.5723905723905723
```

結果顯示，青壯年原本就佔了超過一半的比例。因此，上述的兩個分析，我的解釋方式為：青壯年的比例本皆偏高的狀況，是因為青壯年的人口本來就屬於大宗。

在電影中，小孩跟年長者可以先上救生艇。也因此，在下一個階段，將比較**小孩**、**青壯年**、**年長者**之間的存活率。

### 存活率比較

這此將假設小孩、青壯年、年長者的年齡區間，並進一步比較存活率。假設條件如下：

* 小孩 < 18
*  18 <= 青壯年 <= 40
*  40 <= 年長者

> 在此已重新假設青壯年所在年齡區間。

```python
teen = sur_and_age[(sur_and_age.Age < 18)]
man = sur_and_age[(sur_and_age.Age >= 18)&(sur_and_age.Age < 40)]
old = sur_and_age[(sur_and_age.Age >= 40)]

teen_live_ratio = len(teen[teen.Survived ==1]) / len(teen)
man_live_ratio = len(man[man.Survived ==1]) / len(man)
old_live_ratio = len(old[old.Survived ==1]) / len(old)

print ("小孩：%f" % teen_live_ratio)
print ("青壯年：%f" % man_live_ratio)
print ("年長者：%f" % old_live_ratio)
# 小孩：0.539823
# 青壯年：0.357724
# 年長者：0.374233
```

結果顯示，**小孩**的存活率最高。**青壯年**與**年長者**間的比率相近，但還是年長者的比率較高。
 
### 總結

整體資料因在`Age`項目中，有不少缺失，因此利用平均值填補。但也因為這樣，造成可能與實際狀況有所落差。

年齡跟存活率間最大的關聯在於年齡被分屬於哪一個階段（小孩、青壯年、年長者）。但也因當時在做區分時，是以目測（覺得是老弱婦孺先上救生艇），再加上每個年代的老化狀況不同，很難確認真正的區間定義為何，也因此本次分析假設的區間也不能保證是正確的區分。