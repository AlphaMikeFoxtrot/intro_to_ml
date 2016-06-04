#Matrix techniques

There are a number of techniques in the "family" of matrix factorization.  Before going to deep into the full set of techniques, we need to cover a few prerequisites from linear algebra.  From there well be able to go into a full set of techniques.

##Matrix Multiplication

The first general requirement for any matrix techniques will be matrix multiplication.  We'll understand this mathematically for the 2x2 case, algorithmically for the NxN case and then via a library (numpy or scikit learn).  

##Mathematical 2 - Case

```
[a, b]      [e, f]     [a*e+b*g  a*f+b*h]
[c, d]  X   [g, h]  =  [c*e+d*g  c*f+d*h] 
```

This formula may appear confusing, or how you might generalize it, but it's actually quiet simple.  You form the resultant matrix by taking the dot product of the ith row of the left matrix and jth column of the right matrix.  

The dot product:

We'll use python to define it:

```
def dot_product(a,b):
	return sum([a[index]*b[index] for index in range(len(a))])
```

##Programmatic N - Case


```
class Matrix:
    def __init__(self,matrix):
        self.matrix = matrix
            
    def get_size(self):
        return len(self.matrix),len(self.matrix[0])

    def pprint(self):
        for row in self.matrix:
            print(row)
                
    def to_array(self):
        return self.matrix
        
    def get_elem(self,row,col):
        return self.matrix[row][col]    
        
    def __add__(self,other):
        row_size,col_size = self.get_size()
        new_matrix = []
        for row in range(row_size):
            new_matrix.append([elem+other.matrix[row][ind] for ind,elem in enumerate(self.matrix[row])])
        return Matrix(new_matrix)
    
    def __sub__(self,other):
        row_size,col_size = self.get_size()
        new_matrix = []
        for row in range(row_size):
            new_matrix.append([elem-other.matrix[row][ind] for ind,elem in enumerate(self.matrix[row])])
        return Matrix(new_matrix)

    def simple_multiplication(self,A,B):
        row_size,col_size = self.get_size()
        new_matrix = [[0 for i in range(row_size)] for j in range(row_size)]
        for i in range(row_size):
            for k in range(row_size):
                for j in range(row_size):
                    new_matrix[i][j] += A[i][k] * B[k][j]
        return Matrix(new_matrix)
    
    def __mul__(self,other):
        row_size,col_size = self.get_size()
        if row_size <= 2:
            return self.simple_multiplication(self.matrix,other.matrix)
        else:
            new_size = row_size//2
            A = [[0 for j in range(new_size)] for i in range(new_size)]
            B = [[0 for j in range(new_size)] for i in range(new_size)]
            C = [[0 for j in range(new_size)] for i in range(new_size)]
            D = [[0 for j in range(new_size)] for i in range(new_size)]

            E = [[0 for j in range(new_size)] for i in range(new_size)]
            F = [[0 for j in range(new_size)] for i in range(new_size)]
            G = [[0 for j in range(new_size)] for i in range(new_size)]
            H = [[0 for j in range(new_size)] for i in range(new_size)]

            for i in range(new_size):
                for j in range(new_size):
                    A[i][j] = self.matrix[i][j]
                    B[i][j] = self.matrix[i][j+new_size]
                    C[i][j] = self.matrix[i + new_size][j]
                    D[i][j] = self.matrix[i + new_size][j + new_size]

                    E[i][j] = other.matrix[i][j]
                    F[i][j] = other.matrix[i][j+new_size]
                    G[i][j] = other.matrix[i + new_size][j]
                    H[i][j] = other.matrix[i + new_size][j + new_size]

            A = Matrix(A)
            B = Matrix(B)
            C = Matrix(C)
            D = Matrix(D)
            E = Matrix(E)
            F = Matrix(F)
            G = Matrix(G)
            H = Matrix(H)
            
            p1 = A*(F-H)
            p2 = (A+B)*H
            p3 = (C+D)*E
            p4 = D*(G-E)
            p5 = (A+D)*(E+H)
            p6 = (B-D)*(G+H)
            p7 = (A -C)*(E+F)

            c11 = p5 + p4 - p2 + p6
            c12 = p1 + p2
            c21 = p3 + p4
            c22 = p1+ p5 - p3 - p7

            final = [[0 for j in range(row_size)] for i in range(row_size)]
            for i in range(new_size):
                for j in range(new_size):
                    final[i][j] = c11.matrix[i][j]
                    final[i][j+new_size] = c12.matrix[i][j]
                    final[i + new_size][j] = c21.matrix[i][j]
                    final[i + new_size][j + new_size] = c22.matrix[i][j]
            return Matrix(final)
            
A = Matrix([])
B = Matrix([])
(A*B).pprint()
```

This multiplication makes use of a divide and conquer algorithmic approach to compute the matrix multiplication.  For a deeper understand of this check out the [stanford algorithms class](https://www.coursera.org/course/algo).  

##Conceptual N Case

We can think of matrix multiplication as something known as a linear transformation.  Where A,B are matrices - A*B is A transformed by the matrix B.  

##To Add

* The idea of linear transformations
* invertibility
* Finding a basis
* eigen values and eigen vectors

##Principal Components Analysis

The central idea in Principal Components Analysis (PCA) is similar to the idea of finding a basis.  The central assumption of PCA is that there are vectors in the matrix representation of the data set that are correlated enough that they are redundant.  In such a circumstance, you can perform more or less the same analysis by removing the redudant data.  

For those of you coming from a statistics background PCA is similar to model selection in multivariate linear regression.  The main difference being, PCA let's you test the t-statistic and F-statistic and find the right variables without checking each possible model representation explicitly.

To understand this in more detail let's look at a real example:

```
from sklearn.decomposition import PCA
import pandas as pd
import math

def complexify(num):
	return math.pow(num,2)*17+4

df = pd.DataFrame()
for i in range(1000):
	df = df.append({"a":i,"b":i+2,"c":complexify(i)},ignore_index=True)

pca = PCA(df.as_matrix()).fit()
new_df = pd.DataFrame(pca.components_,columns=df.columns)

#choosing which columns stay:

print(df.columns) #Index(['a', 'b', 'c'], dtype='object')
print(pca.explained_variance_ratio_) #[  1.00000000e+00   4.07015772e-10   1.23061438e-39]

#The ones with the highest explained variance ratio are the ones you keep
#every thing else is removed:

df = df["a"]
```

We can also use PCA in combination with regression analysis, so that any unnecessary model parameters are already taken out:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.cross_decomposition import PLSRegression
mse = []

kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=2)

for i in np.arange(1, 6):
    pls = PLSRegression(n_components=i, scale=False)
    pls.fit(scale(X_reduced),y)
    score = cross_validation.cross_val_score(pls, X_reduced, y, cv=kf_10, scoring='mean_squared_error').mean()
    mse.append(-score)

plt.plot(np.arange(1, 6), np.array(mse), '-v')
plt.xlabel('Number of principal components in PLS regression')
plt.ylabel('MSE')
plt.xlim((-0.2, 5.2))
```

Some words of caution - Automation can aid in a lot of decision making, however it removes the statistician from having deep, sometimes fatal understanding of what they are doing.  For this reason, I strongly advise that at least some classical techniques like linear regression with F-statistics and t-statistics should be used and the R^2 should be studied in detail.  PCA will not always work, and some times for subtle reasons.  Some times non-linear transformations will be applied to data, and yet there will be correlation between the two data sets.  Other times the algorithm will fail to pick up on the right model selection parameters.  For this reason, I think of PCA and any matrix Factorization techniques as important to have in the tool box, but not right for all problems.  

At the end of the day, it's always going to come down to the machine you are using, the data you are working on, and lots and lots of domain specific knowledge.  But having more tools in the tool box is always better than fewer.  In fact, dimensionality reduction is a vibrant field.  Within the last few years, a number of techniques from topology have been gaining ground in the data science space to address the nuances of non-linear data sets.  At some point dimensionality reduction techniques will be mature enough to replace classical statistical tests, intended for small data sets and those with lots of constraints.  

One of the major drawbacks of OLS and techniques like it is the normality assumption.  It also requires variables be independent and identically distributed.  This assumption falls apart for most modern data sets - especially those coming from unstructured data.  Only through the flexibility of algebra and topology can we truly begin to relax such assumptions, allowing for greater flexibility in our models.  However, we must not allow the simplicity of the interfaces to our code, undermine the deep understanding required to practice such techniques.  

PCA can easily go wrong, without a clear understanding of domain expertise of your data.  Perhaps it will reduce away quantities that aren't actually correlated at all, but appear as such because of a combination of differing units, transformations that occurred during cleaning and normalization.  One might be tempted to then draw real conclusions about ones domain based on spurious connections which do not occur, and miss something important that would have been evident under another application of data transformation, or by removing some other technique.

There are many techniques out there that don't require deep understanding - perhaps ironically, many of these techniques are called deep learning techniques - especially given that no deep knowledge is really needed.  Your feature transformations are applied for you.  

References:

* http://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/
* https://plot.ly/ipython-notebooks/principal-component-analysis/
* http://www.dummies.com/how-to/content/data-science-using-python-to-perform-factor-and-pr.html
* http://stats.stackexchange.com/questions/82050/principal-component-analysis-and-regression-in-python
