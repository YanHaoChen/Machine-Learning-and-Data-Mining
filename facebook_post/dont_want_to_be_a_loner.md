# 試著脫離邊緣人
希望利用機器學習及 FB 貼文資料集，找出如何撰寫貼文會引人注目的方式。

## 分析步驟

* 資料前處理
	* 分出可控制特徵
	* 處理 Page Category (供建立模型使用)
* 分群
	* 利用 K-mean 以 Total Comment 進行分群 
* 建立模型
	* Decision Tree
	* Logical Regression
	* SVM

## 資料前處理
### 分出可控制特徵
```python
care_data = data[['total_comments','have_been_there','post_length','page_category',\
'Sun','Mon','Tue','Wed','Thu','Fri','Sat']]
```
### 處理 Page Category

因 Page Category 屬於類別型資料，所以我們讓 Page Category 一個類別就當作一個欄位，以正確進行模型建立。

```python
category_list = []
with open('./facebook_post/category_list.txt','r') as f:
   for category in f:
       category_list.append(str(category.strip()))

for category in category_list:
    care_data[category] = pd.Series(np.zeros(care_data.count()[0]), index=care_data.index)


for i in range(1, len(category_list)+1):
    care_data[category_list[i-1]][care_data.page_category==i]= 1


care_data.drop('page_category', axis=1, inplace=True)
```

## 分群

因為貼文的特徵分佈，都是屬於`長尾分佈`。因此，如果沒有事先分群，網路名人的特徵，可能就會影響一般人該怎麼做會成為`Key Man`的判斷。我們以`Total comment`進行分群：

```python
from sklearn.cluster import KMeans
cluster_num = 10

training_data = care_data[['total_comments']]
kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(training_data)
kmeans.cluster_centers_

pred = kmeans.predict(training_data)
care_data['cluster'] = pd.Series(pred, index=care_data.index)
```
### Before
![](https://github.com/YanHaoChen/Machine-Learning-and-Data-Mining/blob/master/facebook_post/images/scatter_of_non_clustering.png?raw=true)
### After
![](https://github.com/YanHaoChen/Machine-Learning-and-Data-Mining/blob/master/facebook_post/images/scatter_of_clustering.png?raw=true)
> 上圖皆為分五群後的呈現結果。實際情況，為分十群。

## 建立模型

我們使用三種方式建立模式。試圖找出怎麼做，才能不邊緣。

### Decision Tree

```python
from sklearn import tree
import pydotplus
import graphviz
clf = tree.DecisionTreeClassifier()
clf = clf.fit(lv1_cluster, lv1_target)

header_ar = lv1_cluster.columns

dot_data=tree.export_graphviz(clf,out_file=None,\
                                max_depth=4,\
                         feature_names=header_ar,\
                         class_names=['loner','key man'],\
                         filled=True, rounded=True,\
                         special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("tree.pdf")
```
[tree.pdf](https://github.com/YanHaoChen/Machine-Learning-and-Data-Mining/blob/master/facebook_post/images/tree.pdf)

### Logical Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import classification_report

logreg = LogisticRegression(C=1000, solver='newton-cg',tol=1e-5)
logreg.fit(lv1_cluster, lv1_target)
```
##### Result

```python
# Training
           precision     recall   f1-score    support
0.0          0.79022    0.98245    0.87591      20739 1.0          0.53393    0.07158    0.12623       5826 avg / total  0.73401    0.78268    0.71150      26565

# Test (10 TIMES)
accuracy : 0.754428 precision : 0.373175 recall : 0.166079 f1_score : 0.222837
```

### SVM

```python
from sklearn.svm import SVC
svc = SVC(C=123, gamma=0.0153)
svc.fit(lv1_cluster, lv1_target)
```
##### Result

```python
# Training
           precision     recall   f1-score    support 0.0          0.86593    0.97507    0.91726      20739 1.0          0.83904    0.46258    0.59637       5826avg / total  0.86003    0.86268    0.84689      26565

# Test (10 TIMES)
accuracy : 0.734037precision : 0.410684recall : 0.380749f1_score : 0.375054
```

## 結論

光靠人可以控制的特徵，並無法找出如何撰寫出引人注目的貼文。認為可歸咎在，群體的產生有很多原因，因此要在各自的群體間有引人注目的貼文的方式皆不相同。因此無法有效歸納出，會引人注目的貼文撰寫方式。