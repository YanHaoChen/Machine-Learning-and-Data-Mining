import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plot
import copy
from sklearn import preprocessing
import numpy as np

## Preprocess

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
# 38 All of data is zero
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

# Selects columns what I need.
care_data = data[['total_comments','have_been_there','post_length','page_category',\
'Sun','Mon','Tue','Wed','Thu','Fri','Sat']]

#le = preprocessing.LabelEncoder()
#le.fit(category_list)
#care_data.page_category = le.inverse_transform(care_data.page_category)

category_list = []
with open('./facebook_post/category_list.txt','r') as f:
   for category in f:
       category_list.append(str(category.strip()))

for category in category_list:
    care_data[category] = pd.Series(np.zeros(care_data.count()[0]), index=care_data.index)


for i in range(1, len(category_list)+1):
    care_data[category_list[i-1]][care_data.page_category==i]= 1


care_data.drop('page_category', axis=1, inplace=True)
# Clusters the data by total_comments with K-Mean
from sklearn.cluster import KMeans
cluster_num = 10

training_data = care_data[['total_comments']]
kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(training_data)
kmeans.cluster_centers_

pred = kmeans.predict(training_data)
care_data['cluster'] = pd.Series(pred, index=care_data.index)
# Observes the result with scatter diagram.

result_list = []
count_set = set()
colors = ["r","g","b","y","c"]
for data in training_data.total_comments:
    if data not in count_set:
        pd = colors[kmeans.predict(data)[0]]
        count = training_data[training_data.total_comments == data].count()[0] 
        result_list.append([data,count,pd])
        count_set.add(data)

result_list =np.asarray(result_list)

ax = plot.scatter(result_list[:,0],result_list[:,1])
ax.figure.savefig("./facebook_post/images/scatter_of_non_clustering.png")


ax = plot.scatter(result_list[:,0],result_list[:,1],c=result_list[:,2])
ax.figure.savefig("./facebook_post/images/scatter_of_clustering.png")

# Defines the levels of clusters and uses the variables to represent that.

lv1_cluster = care_data[care_data.cluster==7]

lv1_cluster.drop('cluster', axis=1, inplace=True)
lv1_cluster.total_comments.describe()
lv1_q80 = lv1_cluster[['total_comments']].quantile(q=0.7)[0]

import pandas as pd
lv1_cluster['label'] = pd.Series(np.zeros(lv1_cluster.count()[0]), index=lv1_cluster.index)
lv1_cluster.label[lv1_cluster.total_comments >= lv1_q80]= 1

lv1_target = lv1_cluster.label.values
lv1_cluster.drop('total_comments', axis=1, inplace=True)
lv1_cluster.drop('label', axis=1, inplace=True)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import classification_report

logreg = LogisticRegression(C=1000, solver='newton-cg',tol=1e-5)
logreg.fit(lv1_cluster, lv1_target)

test_predict = logreg.predict(lv1_cluster)
report = classification_report(lv1_target,test_predict, digits=5)
print(report)

# cluster 10
# 107 0.0017 73 41 31 33
# 137 0.0097 72 38 32 34
# 137 0.0177 73 39 35 35
# 137 0.0377 73 36 30 30
# 117 0.0197 73 39 34 34
# 127 0.0197 72 38 33 33
# 127 0.0137 73 40 37 36
# 123 0.0137 73 40 37 36
# 123 0.0117 72 39 33 34
# 127 0.0147 73 40 38 37
# 127 0.0153 73 41 38 37

# cluster 5
# 123 0.0137 71 46 23 30
from sklearn.svm import SVC
svc = SVC(C=123, gamma=0.0153)
svc.fit(lv1_cluster, lv1_target)

test_predict = svc.predict(lv1_cluster)
report = classification_report(lv1_target,test_predict, digits=5)
print(report)



#-------------Test data---------------
from sklearn.metrics import accuracy_score, recall_score,precision_score, f1_score
lr_accuracy_scores = []
lr_precision_scores = []
lr_recall_scores = []
lr_f1_scores = []
lr_reports = []

svm_accuracy_scores = []
svm_precision_scores = []
svm_recall_scores = []
svm_reports = []
svm_f1_scores = []

for i in range(1,11):
    data_name = str('./facebook_post/Dataset/Testing/TestSet/Test_Case_%d.csv' % i)
    test_data = pd.read_csv(data_name, header=None)
    test_data.columns = headers

    test_care_data = test_data[['total_comments','have_been_there','post_length','page_category',\
'Sun','Mon','Tue','Wed','Thu','Fri','Sat']]

    test_care_data.page_category.astype('object')
    test_cluster_data = test_care_data[['total_comments']]

    for category in category_list:
        test_care_data[category] = pd.Series(np.zeros(test_care_data.count()[0]), index=test_care_data.index)

    for i in range(1, len(category_list)+1):
        test_care_data[category_list[i-1]][test_care_data.page_category==i]= 1

    test_care_data.drop('page_category', axis=1, inplace=True)

    pred = kmeans.predict(test_cluster_data)
    test_care_data['cluster'] = pd.Series(pred, index=test_care_data.index)
    test_lv1_cluster = test_care_data[test_care_data.cluster == 7]
    test_lv1_cluster.drop('cluster', axis=1, inplace=True)

    test_lv1_cluster['label'] = pd.Series(np.zeros(test_lv1_cluster.count()[0]), index=test_lv1_cluster.index)
    test_lv1_cluster.label[test_lv1_cluster.total_comments >= lv1_q80]= 1

    test_target = test_lv1_cluster.label.values
    test_lv1_cluster.drop('label', axis=1, inplace=True)
    test_lv1_cluster.drop('total_comments', axis=1, inplace=True)

    ## logical
    test_predict = logreg.predict(test_lv1_cluster)
    report = classification_report(test_target,test_predict, digits=5)
    lr_reports.append(report)
    # accuracy
    lr_accuracy_score = accuracy_score(test_target,test_predict)
    lr_accuracy_scores.append(lr_accuracy_score)
    # precision
    lr_precision_score = precision_score(test_target,test_predict)
    lr_precision_scores.append(lr_precision_score)
    # recall
    lr_recall_score = recall_score(test_target,test_predict)
    lr_recall_scores.append(lr_recall_score)
    # f1_score
    lr_f1_score = f1_score(test_target,test_predict)
    lr_f1_scores.append(lr_f1_score)

    ## svm
    test_predict = svc.predict(test_lv1_cluster)
    report = classification_report(test_target, test_predict,digits=5)
    svm_reports.append(report)
    # accuracy
    svm_accuracy_score = accuracy_score(test_target,test_predict)
    svm_accuracy_scores.append(svm_accuracy_score)
    # precision
    svm_precision_score = precision_score(test_target,test_predict)
    svm_precision_scores.append(svm_precision_score)
    # recall
    svm_recall_score = recall_score(test_target,test_predict)
    svm_recall_scores.append(svm_recall_score)
    # f1_score
    svm_f1_score = f1_score(test_target,test_predict)
    svm_f1_scores.append(svm_f1_score)


print("---Logical Regression---")
print ("lr_accuracy   : %20f" % (sum(lr_accuracy_scores) / len(lr_accuracy_scores)))
print ("lr_precision  : %20f" % (sum(lr_precision_scores) / len(lr_precision_scores)))
print ("lr_recall     : %20f" % (sum(lr_recall_scores) / len(lr_recall_scores)))
print ("f1_score      : %20f" % (sum(lr_f1_scores) / len(lr_f1_scores)))
print("---SVM---")
print ("svm_accuracy  : %20f" % (sum(svm_accuracy_scores) / len(svm_accuracy_scores)))
print ("svm_precision : %20f" % (sum(svm_precision_scores) / len(svm_precision_scores)))
print ("svm_recall    : %20f" % (sum(svm_recall_scores) / len(svm_recall_scores)))
print ("f1_score      : %20f" % (sum(svm_f1_scores) / len(svm_f1_scores)))

for report in svm_reports:
    print(report)
#-------------------------------------
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

