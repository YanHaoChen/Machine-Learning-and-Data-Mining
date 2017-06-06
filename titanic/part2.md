# Titanic (Who can survive?)

延續 [Part1](https://github.com/YanHaoChen/Machine-Learning-and-Data-Mining/blob/master/titanic/part1.md)，在此更改了填補 Age 遺失資料的方式（Imputer）。接下來，以主成份分析（PCA）觀察 Feature，最後以 LogisticRegression 及SVM 來進行機器學習，判斷存活率。

> PCA 跟後續的機器學習在此沒有太大的關聯，僅屬於資料分析的一部份。

### 使用 Imputer 填補遺失資料
Imputer 提供了三個方案，來填補遺失的資料：

* mean：平均值
* median：中位數
* most_frequent：出現頻率

在此使用**出現頻率**作為本次填補的方式，因為認為旅遊會有特定年紀的客群，所以用此方式，希望能趨近真實情況。

在使用 Imputer 上要注意一件事情，`transform`的陣列大小要與`fit`時用的陣列相同。所以在此，取出前 500 筆資料訓練 Imputer，再對資料分兩個部分進行 Transform，最後合併兩個結果。

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

在此，先建立好 PCA 的模型`pca`。基本上 2 個主成份就能涵蓋大部份分析對象的資料特徵，所以在參數`n_components`的部分，設定為 2。

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

第一主成份跟第二主成份分別佔了整體資料特徵的 92% 及 7%。為了更進一步的探索各主成份所隱含的意義，我們將觀察這些主成份內，各特徵的比例為何：

```python
# Features in first component.
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

# Features in second component.
print(pca.components_[1])
for i in range(0,7):
    print ("%10s: %10f" % (selected_feature[i], pca.components_[1][i]))

#       Sex:   0.002789
#    Pclass:  -0.013480
#       Age:   0.999319
#     SibSp:  -0.016513
#     Parch:  -0.009696
#      Fare:  -0.028200
#  Survived:  -0.003033
```
在第一主成份上（92%），顯示了票價對資料來說，有大幅度的影響。也合理的預估票價其他項目有一定的關聯，如不同的艙等、孩童票跟成人票等等。在不確定鐵達尼當初的售票標準的前提下，假設第一主成份的解釋為此。

在第二主成份上（7%），雖然能解釋的程度不到 10%，但還有它的意義存在。其中顯示 Age 為主要特徵項目，在年齡上對其他項目也是有一定的影響，例如在第一主成份所提到的票價就很有可能受年齡影響。

以上的結果，可推估**票價**及**年齡**為主要會對資料有所影響的項目。

### 預測模型（LogisticRegression and SVM）


### LogisticRegression
```python
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
```

### SVM

```python
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
```


