# #http://www.dummies.com/how-to/content/data-science-using-python-to-perform-factor-and-pr.html
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
# import pandas as pd

# #basic example
# iris = load_iris()
# X,y = iris.data,iris.target
# pca = PCA().fit(X)
# df = pd.DataFrame(pca.components_,columns=iris.feature_names)
# print(iris.feature_names)
# print(pca.explained_variance_ratio_)

#more realistic example
from sklearn.decomposition import PCA
import pandas as pd
import math

def complexify(num):
	return math.pow(num,2)*17+4

df = pd.DataFrame()
for i in range(1000):
	df = df.append({"a":i,"b":i+2,"c":complexify(i)},ignore_index=True)

pca = PCA().fit(df.as_matrix())
new_df = pd.DataFrame(pca.components_,columns=df.columns)

#choosing which columns stay:

print(df.columns) #Index(['a', 'b', 'c'], dtype='object')
print(pca.explained_variance_ratio_) #[  1.00000000e+00   4.07015772e-10   1.23061438e-39]

#The ones with the highest explained variance ratio are the ones you keep
#every thing else is removed:

df = df["a"]
