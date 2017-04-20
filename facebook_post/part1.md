# 資料前處理＋敘述統計

## 分析前的準備

### 資料準備

[資料集](https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset)

### python 套件準備

* pandas：讀取、解析資料
* matplotlib：視覺化圖表
* seaborn：同用來處理視覺化的部分

## 資料前處理

利用 pandas  讀入資料：

```python
import pandas as pd

data = pd.read_csv('the file path of dataset')
```

其中的屬性項目總共有 54 項，資料及文件中，都有大概說明一下。因為數據中只有資料，並沒有項目名稱，為了方便處理，將有興趣的項目名稱加入其中：

```python
# 1 此文被按讚的次數
# 2 此文被看過的次數
# 3 看過再按讚的次數
# 4 文章種類

headers = ['likes', 'visited','visited_and_like','page_category']
for i in range(0,25):
    headers.extend([str(i)])
# 30 總共的留言數
headers.extend(['total_comments'])
# 31 最新 24 小時內的留言數
headers.extend(['last24_comments'])
# 32 最新 24 小時至 48 小時，期間內的留言數（-24 ~ -48）
headers.extend(['last24to48_comments'])
# 33 發文後 24 小時內的留言數
headers.extend(['posted24_comments'])
# 34
headers.extend([str(25)])
# 35
headers.extend([str(26)])
# 36 貼文長度
headers.extend(['post_length'])
# 37 貼文被分享的次數
headers.extend(['post_share_count'])
# 38 此文有沒有被貼廣
headers.extend(['post_promotion_status'])
# 39 H 小時後，會到達預期的留言數
headers.extend(['at_hr_got_target'])
# 40 - 46 星期幾發文
# 40 
headers.extend(['Sun'])
# 41
headers.extend(['Mon'])
# 42
headers.extend(['Tue'])
# 43
headers.extend(['Wed'])
# 44 
headers.extend(['Thu'])
# 45
headers.extend(['Fri'])
# 46 
headers.extend(['Sat'])

for i in range(28,35):
    headers.extend([str(i)])


# 54 H 小時後（H 為項目 39 ），會獲得的留言數
headers.extend(['target_var'])

data.columns = headers
```

處理完後，會得到這樣的數據：

```python
print (data)
         likes  visited  visited_and_like  page_category  ...       
0       634995        0               463              1  ...
1       634995        0               463              1  ...
...
```
這樣的項目還是太多了，所以先選擇部分的項目進行分析。在這裡選擇：

* （No.1）此文被按讚的次數
* （No.2）此文被看過的次數
* （No.3）看過再按讚的次數
* （No.30）總共的留言數

此四項進行分析，希望找出他們之間的關聯性：

```python
first_analysis_data = data[['likes','visited','visited_and_like','total_comments']]
```

此時的資料就簡單很多了。第一步，從敘述統計開始：

```python
print (first_analysis_data.describe())

              likes        visited  visited_and_like  total_comments
count  4.094800e+04   40948.000000      4.094800e+04    40948.000000
mean   1.313830e+06    4676.247949      4.480133e+04       55.721745
std    6.785834e+06   20593.423357      1.109349e+05      136.977101
min    3.600000e+01       0.000000      0.000000e+00        0.000000
25%    3.673400e+04       0.000000      6.980000e+02        2.000000
50%    2.929110e+05       0.000000      7.141000e+03       11.000000
75%    1.204214e+06      99.000000      5.026400e+04       46.000000
max    4.869723e+08  186370.000000      6.089942e+06     2341.000000
```
從中的觀察：

* 從各項目的```std```（標準差）中，可以發現資料的離散的程度都很大。
* 因為離散程度大，`mean` （平均值）的參考價值應該不高。
* 大部分的貼文（超過一半），是沒有人會去讀的。
* 至少 25% 的貼文只有 2 則留言。 