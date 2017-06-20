import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import copy
from sklearn import preprocessing

data = pd.read_csv('./facebook_post/Dataset/Training/Features_Variant_1.csv', header=None)
headers = ['likes', 'have_been_there','interested','page_category']
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
first_analysis_data = data[['likes','have_been_there','interested','total_comments']]
print (first_analysis_data.describe())

data_likes = first_analysis_data['likes']
data_have_been_there = first_analysis_data['have_been_there']
data_interested = first_analysis_data['interested']
data_total_comments = first_analysis_data['total_comments']

data_likes.head(5)
data_likes.describe()
ax = sns.distplot(data_likes, kde=False)
ax.figure.savefig('./facebook_post/images/likes_histogram.png')
ax = sns.boxplot(data_likes)
ax.figure.savefig('./facebook_post/images/likes_boxplot.png')

data_have_been_there.head(5)
data_have_been_there.describe()
ax = sns.distplot(data_have_been_there, kde=False)
ax.figure.savefig('./facebook_post/images/have_been_there_histogram.png')
ax = sns.boxplot(data_have_been_there)
ax.figure.savefig('./facebook_post/images/have_been_there_boxplot.png')

data_interested.head(5)
data_interested.describe()
ax = sns.distplot(data_interested, kde=False)
ax.figure.savefig('./facebook_post/images/interested_histogram.png')
ax = sns.boxplot(data_interested)
ax.figure.savefig('./facebook_post/images/interested_boxplot.png')

plot.cla()
data_total_comments.head(5)
data_total_comments.describe()
ax = sns.distplot(data_total_comments, kde=False)
ax.figure.savefig('./facebook_post/images/total_comments_histogram.png')
ax = sns.boxplot(data_total_comments)
ax.figure.savefig('./facebook_post/images/total_comments_boxplot.png')

normal_scaler = preprocessing.MinMaxScaler()
normalized_data = normal_scaler.fit_transform(first_analysis_data)
first_analysis_data_normalized = pd.DataFrame(normalized_data, columns= first_analysis_data.columns)

ax = sns.pairplot(first_analysis_data_normalized)
ax.fig.savefig('./facebook_post/images/first_analysis_data_scatter.png')

correlation = first_analysis_data.corr()
print (correlation)
sns.set(style="white")
ax = plot.axes()
sns.heatmap(correlation, ax = ax)
ax.figure.savefig('./facebook_post/images/first_analysis_data_corr.png')