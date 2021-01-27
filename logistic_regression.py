import sys
import os

import pandas as pd
import numpy as np


from pandas.core.index import CategoricalIndex, RangeIndex, Index, MultiIndex

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline


"""
download csv full_df.csv describing the metadata: https://drive.google.com/file/d/1-XEKHT-EVWx2M-PmVYV1m4buBLW-QGqa/view?usp=sharing

"""

#please type in the path of the file full_df.csv, including the filename

full_df_path = ""

df = pd.read_csv(full_df_path)
df_gen = pd.get_dummies(df['Patient Sex'], prefix=None)
df["female"] = df_gen.Female
df["male"] = df_gen.Male

X = df[["Patient Age", "female", "male"]]
# X = np.array(df["Patient Age"]).reshape(-1,1)
y = df["labels"]
clf = LogisticRegression(random_state=0, max_iter = 10000).fit(X, y)
clf_pipe = pipeline.make_pipeline(StandardScaler(), clf)

clf_pipe.fit(X,y)
clf_pipe.predict_proba(X)
print("prediction score of logistic regression on the dataset is: {}".format(clf_pipe.score(X, y)))