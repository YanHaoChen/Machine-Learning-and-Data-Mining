import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot

data = pd.read_csv('./facebook_post/Dataset/Training/Features_Variant_1.csv')
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
)
correlation = first_analysis_data.corr()

sns.set(style="white")
ax = plot.axes()
sns.heatmap(correlation, ax = ax)
plot.show()


data_likes = first_analysis_data['likes']
data_visited = first_analysis_data['visited']
data_visited_and_like = first_analysis_data['visited_and_like']
data_total_comments = first_analysis_data['total_comments']

data_likes.head(5)
data_likes.describe()
plot.plot(data_likes)
plot.boxplot([data_likes])
plot.xticks([1],['likes'])
plot.hist(data_likes)
plot.ylabel("count")
plot.xlabel("likes")

data_visited.head(5)
data_visited.describe()
plot.plot(data_visited)
plot.boxplot([data_visited])
plot.xticks([1],['visited'])
plot.hist(data_visited)
plot.ylabel("count")
plot.xlabel("visited")

data_visited_and_like.head(5)
data_visited_and_like.describe()
plot.plot(data_visited_and_like)
plot.boxplot([data_visited_and_like])
plot.xticks([1],['visited_and_like'])
plot.hist(data_visited_and_like)
plot.ylabel("count")
plot.xlabel("visited_and_like")

data_total_comments.head(5)
data_total_comments.describe()
plot.plot(data_total_comments)
plot.boxplot([data_total_comments])
plot.xticks([1],['total_comments'])
plot.hist(data_total_comments)
plot.ylabel("count")
plot.xlabel("total_comments")


plot.xticks([1,2],['visited','total_comments'])
plot.scatter(data_visited_and_like, data_total_comments, color = 'r')
plot.scatter(data_visited, data_visited_and_like, color = 'r')
