import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import copy
from sklearn import preprocessing

data = pd.read_csv('./facebook_post/Dataset/Training/Features_Variant_1.csv', header=None)
headers = ['likes', 'visited','visited_and_like','page_category']
for i in range(0,25):
    headers.extend([str(i)])
# 30
headers.extend(['total_comments'])
# 31
headers.extend(['last24_comments'])
# 32
headers.extend(['last24to48_comments'])
# 33
headers.extend(['posted24_comments'])
# 34
headers.extend([str(25)])
# 35
headers.extend([str(26)])
# 36
headers.extend(['post_length'])
# 37
headers.extend(['post_share_count'])
# 38
headers.extend(['post_promotion_status'])
# 39
headers.extend(['at_hr_got_target'])
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

# 54
headers.extend(['target_var'])

data.columns = headers
print (data)
# first analysis: like, visit, total_comments
first_analysis_data = data[['likes','visited','visited_and_like','total_comments']]
print (first_analysis_data.describe())

data_likes = first_analysis_data['likes']
data_visited = first_analysis_data['visited']
data_visited_and_like = first_analysis_data['visited_and_like']
data_total_comments = first_analysis_data['total_comments']

data_likes.head(5)
data_likes.describe()
ax = sns.distplot(data_likes, kde=False)
ax.fig.savefig('./facebook_post/images/likes_histogram.png')
ax = sns.boxplot(data_likes)
ax.fig.savefig('./facebook_post/images/likes_boxplot.png')

data_visited.head(5)
data_visited.describe()
ax = sns.distplot(data_visited, kde=False)
ax.fig.savefig('./facebook_post/images/visited_histogram.png')
ax = sns.boxplot(data_visited)
ax.fig.savefig('./facebook_post/images/visited_boxplot.png')

data_visited_and_like.head(5)
data_visited_and_like.describe()
ax = sns.distplot(data_visited_and_like, kde=False)
ax.fig.savefig('./facebook_post/images/data_visited_and_like_histogram.png')
ax = sns.boxplot(data_visited_and_like)
ax.fig.savefig('./facebook_post/images/data_visited_and_like_boxplot.png')


data_total_comments.head(5)
data_total_comments.describe()
ax = sns.distplot(data_total_comments, kde=False)
ax.fig.savefig('./facebook_post/images/data_total_comments_histogram.png')
ax = sns.boxplot(data_total_comments)
ax.fig.savefig('./facebook_post/images/data_total_comments_boxplot.png')

normal_scaler = preprocessing.MinMaxScaler()
normalized_data = normal_scaler.fit_transform(first_analysis_data)
first_analysis_data_normalized = pd.DataFrame(normalized_data, columns= first_analysis_data.columns)

ax = sns.pairplot(first_analysis_data_normalized)
ax.fig.savefig('./facebook_post/images/first_analysis_data_scatter.png')

correlation = first_analysis_data.corr()

sns.set(style="white")
ax = plot.axes()
sns.heatmap(correlation, ax = ax)
plot.show()

plot.xticks([1,2],['visited','total_comments'])
plot.scatter(data_visited_and_like, data_total_comments, color = 'r')
plot.scatter(data_visited, data_visited_and_like, color = 'r')