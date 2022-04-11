# encoding = 'utf-8'
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kmodes.kmodes import KModes
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# CUSTOMER SEGMENTATION ANALYSIS:

# Read CSV into pandas dataframe 
filename = 'future-merchants-data.csv'
df = pd.read_csv(filename, encoding='utf-8')
#rename as dummy then drop
df.rename(columns={'Unnamed: 0': 't_id'}, inplace=True)
df.drop('t_id', axis=1, inplace=True) 
# df.reset_index(drop=True, inplace=True) 
df['amount_dollars'] = df['amount_usd_in_cents'] / 100
dd = df.copy() # for churn analysis df later, before splitting time
df['time']= pd.to_datetime(df['time'],format='%Y-%m-%d') 
df['year']= df['time'].dt.year
df['month']= df['time'].dt.month
df['day']= df['time'].dt.day
df['hour'] = df['time'].dt.hour
df.drop('time', axis=1, inplace=True)
df.drop('amount_usd_in_cents', axis=1, inplace=True)
# df

# df.describe()
# df.dtypes
# df.shape
# df.isnull() 

lst = []
merch = df['merchant'].values.tolist()
for i in merch:
  if i not in lst: # want distinct merchants, no duplicates
    lst.append(i)
  else:
    continue

print(len(lst))

# Standardize data to get numerical variables on same scale
col_names = ['amount_dollars', 'year', 'month', 'day', 'hour'] 
features = df[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features = pd.DataFrame(features, columns = col_names)
# scaled_features.head()

# since merchant is categorical and this isnt a binary problem -> convert to numerical by subbing in dummy arrays
merchant = df['merchant']
newdf = scaled_features.join(merchant)
newdf = pd.get_dummies(newdf, prefix=None,  sparse=False, drop_first=False, dtype=None)
# newdf

# PCA to reduce dimensionality of dataset and check for largest variance in model
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(newdf)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA Features')
plt.ylabel('Variance (%)')
plt.xticks(features)
PCA_components = pd.DataFrame(principalComponents)

# first 2 PCA components explain *roughly* 35% of the dataset variance

# feed these components into model
# building again with first 2 principal components, find k
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(PCA_components.iloc[:,:2])
    inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# using k = 3 from above:

# K-means with 3 merchant clusters
model = KMeans(n_clusters=3)
model.fit(PCA_components.iloc[:,:2])

# silhouette score
print(silhouette_score(PCA_components.iloc[:,:2], model.labels_, metric='euclidean'))
# around 0.35 - not that great

# array of centroid locations
# print(kmeans.cluster_centers_) 

# add cluster label column to newdf
model = KMeans(n_clusters=3)
clusters = model.fit_predict(PCA_components.iloc[:,:2])
newdf['label'] = clusters # last column 'label' represents cluster number
#columns for clusters and merchants were added, cna check with newdf.shape
# map back clusters to df 
pred = model.predict(PCA_components.iloc[:,:2])
frame = pd.DataFrame(df)
frame['cluster'] = pred
# frame

# Histogram of amount by cluster
plt.hist('amount_dollars', data=frame[frame['cluster'] == 0], alpha=0.5, label='Cluster0')
plt.hist('amount_dollars', data=frame[frame['cluster'] == 1], alpha=0.5, label='Cluster1')
plt.hist('amount_dollars', data=frame[frame['cluster'] == 2], alpha=0.5, label='Cluster2')
plt.title('Distribution of Transaction Size by Cluster')
plt.xlabel('Transaction Amount ($)')
plt.legend()
# plt.show()

# Heatmap to look for correlations
# sns.heatmap(frame.corr(), annot=True)
# not much here worth noting...

# Looking at largest share of big transactions across clusters 
m_df = pd.DataFrame(frame.groupby(['cluster','amount_dollars'])['merchant'].max()).sort_values(by=['amount_dollars'],ascending=False)
big_trans = m_df
# big_trans.head(30)
# --> cluster 1 has the top 28 largest transactions 

# Looking at largest share of small transactions across clusters
m_df2 = pd.DataFrame(frame.groupby(['cluster','amount_dollars'])['merchant'].max()).sort_values(by=['amount_dollars'],ascending=True)
small_trans = m_df2
# small_trans.head(30)
# no real pattern here across clusters and the smallest transactions

# scatterplot to see if we can learn anything about these clusters
scat2 = sns.scatterplot('hour', 'amount_dollars', hue = 'cluster', data=frame, palette=['green','blue','red'])
plt.title('Transaction Sizes by Hour, Colored by Cluster')

# plot 2D clusters to look for separation
centroids = kmeans.cluster_centers_
for i in range(3):
   plt.scatter(newdf.values[label == i , 0] , newdf.values[label == i , 1] , label = i)

plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.title('Cluster and Respective Centroids')
plt.show() 

# using this dataframe only for rank / statistics about merchants:
m_df = frame.groupby(['merchant'], as_index=False).sum()
m_df.drop('cluster', axis=1, inplace=True)
m_df.drop('year', axis=1, inplace=True)
m_df.drop('month', axis=1, inplace=True)
m_df.drop('day', axis=1, inplace=True)
m_df.drop('hour', axis=1, inplace=True)
m_df['rank'] = df['amount_dollars'].rank()
m_df['pct_rank'] = df['amount_dollars'].rank(pct=True)
# m_df

# looking at largest share of big transactions across clusters 
m_df = pd.DataFrame(frame.groupby(['cluster','amount_dollars'])['merchant'].max()).sort_values(by=['amount_dollars'],ascending=False)
# m_df.head(30)
#graph this to show cluster 1 has most of top largest transactions

#To compare attributes of the different clusters:
# find the average of all variables across clusters
avg_df = frame.groupby(['cluster'], as_index=True).mean()
# avg_df
sns.barplot(x='cluster',y='amount_dollars',data=avg_df)

# Look at the distribution of transactions across different time granularities 
sns.countplot(x='year', data=frame)
plt.title('Distribution of Transactions by Year')
# plt.show()

sns.countplot(x='month', data=df)
plt.title('Distribution of Transactions by Month')
# plt.show()

sns.countplot(x='day', data=df)
plt.title('Distribution of Transactions by Day of Month')
# plt.show()

sns.countplot(x='hour', data=df)
plt.title('Distribution of Transactions by Hour of Day')
# plt.show()

##############################################

# CHURN ANALYSIS:

# binary classification  - churn (1) or not (0)
# split up to look at last 3 months of data (Oct through Dec. 2034) for local analysis on busiest months
# using copy of df before splitting time, dd, from beginning of code:
oct = dd[(dd['time'] >= '2034-10-01') & (dd['time'] <= '2034-10-31')]
nov = dd[(dd['time'] >= '2034-11-01') & (dd['time'] <= '2034-11-30')]
dec = dd[(dd['time'] >= '2034-12-01') & (dd['time'] <= '2034-12-31')]

oct_lst = list(set(oct['merchant'].values.tolist())) 
print(len(oct_lst)) # 749 merchants actively using service during Oct. 2034 

nov_lst = list(set(nov['merchant'].values.tolist())) 
print(len(nov_lst)) # 838 merchants actively using service during Nov. 2034 

dec_lst = list(set(dec['merchant'].values.tolist())) 
print(len(dec_lst)) # 885 merchants actively using service during Dec. 2034

# drop duplicates with list(set(lst))
# find how many churned between Nov and Dec 2034
ch_in_dec = list(set(nov_lst).difference(dec_lst))
print(str(len(ch_in_dec)) + ' merchants churned before Dec. 2034.')
print(ch_in_dec)
print('\n')
# find how many churned between Oct and Nov 2034
ch_in_nov = list(set(oct_lst).difference(nov_lst))
print(str(len(ch_in_nov)) + ' merchants churned before Nov. 2034.')
print(ch_in_nov)
print('\n')
# find how many merchants acquired in Dec 2034
new_dec = list(set(dec_lst).difference(nov_lst))
print(str(len(new_dec)) + ' new merchants started in Dec. 2034.')
print(new_dec)
print('\n')
# find how many merchants acquired in Nov 2034
new_nov = list(set(nov_lst).difference(oct_lst))
print(str(len(new_nov)) + ' new merchants started in Nov. 2034.')
print(new_nov)

# Churn - between Oct and EoY
months = d[(d['time'] >= '2034-10-01') & (d['time'] <= '2034-12-31')]
# months.drop('amount_dollars', axis=1, inplace=True)
months[['churn']] = None #use .loc instead apparently?
# months

# make new df that contains all days from oct-dec 31
# add binary 'churn' column to months dataframe
lst =[]
for row in months['merchant']:
  if row in ch_in_dec or row in ch_in_nov:
    lst.append(1) # churned within Oct-Eoy 2034
  else:
    lst.append(0) # active merchant still

# print(len(lst)) 

months['churn'] = lst
months['year'] = months['time'].dt.year
months['month'] = months['time'].dt.month
months['day'] = months['time'].dt.day
months['hour'] = months['time'].dt.hour
months.drop('time', axis=1, inplace=True)
# months

# use get_dummies() again for categorical -> numerical 
mer = pd.get_dummies(months.merchant).iloc[:,1:]
# mer

# append dummy merchants to Oct-EoY dataset:
months_final = pd.concat([months,mer], axis=1)
# drop merchant now, redundant
months_final.drop('merchant', axis=1, inplace=True) #redundant
# drop churn, since we're predicting this now 
# months_final.drop('amount_usd_in_cents', axis=1, inplace=True)
# months_final

# Use Random Forest - good for bagging with subset, no under/overfitting issues
# prep data for random forest:
# X is feature set of all cols except one we want to predict class (churn)
X = months_final.drop(['churn'], axis=1) 
y = months_final['churn']
# print(X)
# print(y)

# train random forest with 80% / 20% split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)
classifier = RandomForestClassifier(n_estimators=200, random_state=0) 
classifier.fit(X_train, y_train) 
predictions = classifier.predict(X_test)

# check accuracy scores:
pd.crosstab(y_test, predictions, rownames=['Actual Result'], colnames=['Predicted Result'])
print(classification_report(y_test,predictions)) 
print(accuracy_score(y_test, predictions))
# 89% accuracy, recall and precision both good
# think about what FP/FN mean for business

# feature evaluation for top 3 features:
# 'what impacts churn the most?'
feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(3).plot(kind='barh')

# for slides: 

# Total churned = 350+325 = 675 merchants
# Total at beginning (Oct.) = 749 merchants
# Churn rate = 675/749 = 90% churn – investigate why so high w/ more time
# Retention Rate = 10 % - how many stayed throughout Q4

# Total acquired = 383 + 361 = 744 merchants
# Total at beginning = 749 merchants
# Acquisition Rate = 744/749 = 99.3% acquisition 
# both churn and acqusition high -> 'leaky bucket problem'