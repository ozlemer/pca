import os
import numpy as np
import pandas as pd
import select

from numpy import linalg as LA
from scipy.linalg import svd

with open("new-thyroid.data", "r") as f:
	lines = f.readlines()
	newlines = []
	for line in lines:
		newline = map(float, line.strip('\n').split(','))
		newlines.append(newline)

print(newlines)
df = pd.DataFrame(newlines, columns = ['class','T3-resin','Thyroxin','Triiodothyronine','basal-thyroid','TSH'])
#print(df.head())
M = (df.drop(columns=['class'])).mean(axis=0)
print(M)
print(M[0])
C = df.drop(columns=['class']) - M
#print(df)
#print(C)
V = np.cov(C.T)
values, vectors = LA.eig(V)
print(V)
print(vectors)
print(values)
P =vectors.T.dot(C.T)
print(P.T)