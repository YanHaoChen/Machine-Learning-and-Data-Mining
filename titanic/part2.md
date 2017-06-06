# Titanic (Who can survive?)

延續 [Part1](https://github.com/YanHaoChen/Machine-Learning-and-Data-Mining/blob/master/titanic/part1.md)，在此更改了填補 Age 遺失資料的方式（Imputer）。接下來，以主成份分析（PCA）觀察 Feature，最後以 LogisticRegression 及SVM 來進行機器學習，判斷存活率。

> PCA 跟後續的機器學習在此沒有太大的關聯，僅屬於資料分析的一部份。

### 使用 Imputer 填補遺失資料
Imputer 提供了三個方案，來填補遺失的資料：

* mean：平均值
* median：中位數
* most_frequent：出現頻率

在此使用**出現頻率**作為本次填補的方式，因為認為旅遊會有特定年紀的客群，所以用此方式，希望能趨近真實情況。

```python
from sklearn.preprocessing import Imputer

imp = Imputer(strategy='most_frequent', axis=0)
age_notnull = data[pd.isnull(data.Age)!= True][['Age']]
imp.fit(age_notnull[:500]['Age'])
result = imp.transform(data.Age[:500])
result2 = imp.transform(data.Age[-500:])
result = np.append(result[0],result2[0][109:])

# Before
print(pd.isnull(data.Age).value_counts())

# False    714
# True     177
# Name: Age, dtype: int64

data['Age'] = pd.Series(result[0], index=data.index)
# After
print(pd.isnull(data.Age).value_counts())

# False    891
# Name: Age, dtype: int64
```
### 主成份分析（PCA）

透過主成份分析，發掘有潛力的關鍵特徵。

在此，先建立好 PCA 的模型`pca`。因基本上 2 個主成份就能涵蓋大部份分析對象的資料意義，所以在參數`n_components`的部分，設定為 2。

```python
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
```


```python
# We can just focus on the first component.
print(pca.components_[0])
selected_feature = ['Sex','Pclass','Age','SibSp','Parch','Fare','Survived']
for i in range(0,7):
    print ("%10s: %10f" % (selected_feature[i], pca.components_[0][i]))

#       Sex:  -0.001754
#    Pclass:  -0.009246
#       Age:  -0.000000
#     SibSp:   0.003544
#     Parch:   0.003508
#      Fare:   0.999940
#  Survived:   0.002520
```
票價（Fare）的佔比會那麼高，


