from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

data = load_iris()
# print(data.data[0:5])

X = data.data
y = data.target

# select top 2 features
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new.shape)

# select top 10% features
X_new_percent = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
print(X_new_percent.shape)