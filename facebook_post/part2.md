# 視覺化數據（以視覺方式感受數據）
以視覺化的方式，可以在短時間發現資料的特性，更快瞭解資料狀況。

## 從個別項目開始

### 首先，取出個別資料，以便後續分析：

```python
data_likes = first_analysis_data['likes']
data_visited = first_analysis_data['visited']
data_visited_and_like = first_analysis_data['visited_and_like']
data_total_comments = first_analysis_data['total_comments']
```
### 透過 seaborn 繪製統計圖表（以 likes 為例）
第一個圖表，直方圖（Histgram）：

```python
ax = sns.distplot(data_likes, kde=False)
ax.fig.savefig('./facebook_post/images/likes_histogram.png')

```
> distplot 預設會運算、顯示密度曲線（kde）。在這次的範例中不進行顯示，所以將參數設定為`False`。

製作出的圖表如下：

![likes_histgram](https://github.com/YanHaoChen/Machine-Learning-and-Data-Mining/tree/master/facebook_post/images/likes_histogram.png?raw=true)

從圖表中，能發現此資料是屬於 Power Law Distribution（冪次現象）。也應此可以對此資料多一些見解，例如：可能符合**二八法則**或跟**長尾理論**有關。

第二個圖表，箱型圖（Box）：

```python
ax = sns.boxplot(data_likes)
ax.fig.savefig('./facebook_post/images/visited_boxplot.png')
```
製作出的圖表如下：

![likes_boxplot](https://github.com/YanHaoChen/Machine-Learning-and-Data-Mining/tree/master/facebook_post/images/likes_boxplot.png?raw=true)

也因為此資料的特性，可以發現**上下鄰近值**、**上下樞紐**及**中位值**幾乎就一條線的狀況，而且還有數值高很多的離群值。此圖也彰顯了 Power Law Distribution。

其它的三個屬性，也能用相同的方式觀察，但能查到的狀況都跟此屬性類似，所以在此不多作介紹。現在結束了對單一屬性的觀察，在下一個部分，將進入一次多屬性的觀察。

## 視覺化彼此的關係

再做彼此之間的觀察前，先將屬性進行正規化，以提高觀察效果。此時就需要使用另一個套件 `sklearn`：

```python
from sklearn import preprocessing
# 初始化 MinMaxScaler() 準備進行資料正規化
normal_scaler = preprocessing.MinMaxScaler()

# 餵入資料，並得到正規化後的資料（陣列）
normalized_data = normal_scaler.fit_transform(first_analysis_data)

# 建立正規化後的 DataFrame 
first_analysis_data_normalized = pd.DataFrame(normalized_data, columns= first_analysis_data.columns)
```

正規化後，可以先觀察資料彼此間的散佈狀況，進而判斷接下來該怎麼分析此份資料（例如：散佈狀況集中，則較有價值進行相關係數運算，如果是分散的，則否）。利用 **seaborn** 可以很容易達成此目的，方法如下：

```python
ax = sns.pairplot(first_analysis_data_normalized)
ax.fig.savefig('./facebook_post/images/first_analysis_data_scatter.png')
```
製作出的圖表如下：

![likes_boxplot](https://github.com/YanHaoChen/Machine-Learning-and-Data-Mining/tree/master/facebook_post/images/first_analysis_data_scatter.png?raw=true)

從中可以發現：

* **total comments** 跟 **visited** 之間的分佈較為分散。很直覺的可以想到，**曝光率（visited）跟會產生的留言（total comments）關係可能沒有那麼強烈**。
* 其他資料間，**因為有很誇裝的離群職值，很難好好評估**。