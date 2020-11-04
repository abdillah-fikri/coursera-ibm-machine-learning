# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %%
import pandas as pd 
import numpy as np
import category_encoders as ce
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os

os.chdir("..")
from utils import null_checker, countplot_annot_hue

# %%
os.chdir("data")
df = pd.read_csv("Automobile_data.csv", na_values="?")
df.info()

# %%
df.head()

# %%
null_checker(df)

# %%
# Drop row with missing value of price
df = df[~df["price"].isna()]
null_checker(df)

# %%
df.drop(columns="normalized-losses", inplace=True)

# %%
df.describe()

# %%
t_encoder = ce.TargetEncoder()
df_temp = t_encoder.fit_transform(df.drop(columns=["price"]), df["price"])
df_temp = pd.concat([df_temp, df["price"]], axis=1)

plt.figure(figsize=(14*1.5, 10*1.5))
sns.heatmap(df_temp.corr(), annot=True, linewidths=0.5, fmt=".2f")
plt.show()

# %%
df_temp.corr()["price"].sort_values(ascending=False).index

# %%
df.describe()

# %%
df.select_dtypes("object")

# %%
# Create a list of float colums to check for skewing
mask = df.dtypes == np.float
float_cols = df.columns[mask]

skew_limit = 1 # define a limit above which we will log transform
skew_vals = df[float_cols].skew()

# Showing the skewed columns
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))

skew_cols

# %%
sns.set(style='darkgrid', palette='muted')

# %%
plt.figure(figsize=(17,8))
for i, col in enumerate(skew_cols.index):
    plt.subplot(2,2,i+1)
    sns.histplot(data=df, x=col)

# %%
plt.figure(figsize=(17,8))
for i, col in enumerate(skew_cols.index):
    plt.subplot(2,2,i+1)
    sns.histplot(data=np.log1p(df.select_dtypes("float64")), x=col)

# %%
df.select_dtypes("object").columns

# %%
cat_cols = ['fuel-type', 'aspiration', 'num-of-doors', 'drive-wheels']
plt.figure(figsize=(17,8))
for i, col in enumerate(cat_cols):
    plt.subplot(2,2,i+1)
    sns.barplot(data=df, x=col, y="price") 

# %%
diesel_car = df[df["fuel-type"]=="diesel"]
gas_car = df[df["fuel-type"]=="gas"]

# %%
import scipy.stats as st

ttest = st.ttest_ind(a = diesel_car['price'], b = gas_car['price'])
p_value = ttest.pvalue
print('P-Value :',p_value)
if p_value >= 0.05:
    print('Car with a fuel-type diesel has the same average price as a gas car.')
else:
    print('Car with a fuel-type diesel has an average price that is different from gas car.')

# %%
