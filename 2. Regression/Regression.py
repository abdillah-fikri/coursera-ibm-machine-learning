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
import plotly.express as p
import plotly.io as pio
import os

os.chdir("..")
from utils import null_checker, countplot_annot_hue
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV

# %%
sns.set(style='darkgrid', palette='muted')

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

# %% [markdown]
# # EDA

# %%
df.describe()

# %%
df.describe()

# %%
df.info()

# %%
int_cols = list(df.select_dtypes("int64"))
float_cols = list(df.select_dtypes("float64"))
obj_cols = list(df.select_dtypes("object"))

# %%
float_cols

# %%
plt.figure(figsize=(25,8))
for i, col in enumerate(float_cols[:5]):
    plt.subplot(2,5,i+1)
    sns.histplot(data=df, x=col)
    plt.title(col.capitalize())
for i, col in enumerate(float_cols[:5]):
    plt.subplot(2,5,i+6)
    sns.boxplot(data=df, x=col)

# %%
plt.figure(figsize=(25,8))
for i, col in enumerate(float_cols[5:]):
    plt.subplot(2,5,i+1)
    sns.histplot(data=df, x=col)
    plt.title(col.capitalize())
for i, col in enumerate(float_cols[5:]):
    plt.subplot(2,5,i+6)
    sns.boxplot(data=df, x=col)

# %%
plt.figure(figsize=(25,8))
for i, col in enumerate(int_cols):
    plt.subplot(2,5,i+1)
    sns.histplot(data=df, x=col)
    plt.title(col.capitalize())
for i, col in enumerate(int_cols):
    plt.subplot(2,5,i+6)
    sns.boxplot(data=df, x=col)

# %%
t_encoder = ce.TargetEncoder()
df_temp = t_encoder.fit_transform(df.drop(columns=["price"]), df["price"])
df_temp = pd.concat([df_temp, df["price"]], axis=1)

plt.figure(figsize=(14*1.5, 10*1.5))
sns.heatmap(df_temp.corr(), annot=True, linewidths=0.5, fmt=".2f")
plt.show()

# %%
obj_cols

# %%
plt.figure(figsize=(25,8))
for i, col in enumerate(obj_cols):
    plt.subplot(2,5,i+1)
    sns.barplot(data=df, x=col, y="price")
    plt.title(col.capitalize())
    plt.xticks(rotation=90)
plt.tight_layout()

# %% [markdown]
# # Preprocessing

# %%
from sklearn.model_selection import train_test_split
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
skew_limit = 0.85 # define a limit above which we will log transform
skew_vals = df[float_cols+int_cols].drop(columns="price").skew()

# Showing the skewed columns
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))

skew_cols

# %%
for col in skew_cols.index:
    X_train[col] = np.log(X_train[col])
    X_test[col] = np.log(X_test[col])

# %%
miss_cols = list(null_checker(X_train).head(5).index)
miss_cols

# %%
from sklearn.impute import SimpleImputer
mean_imp = SimpleImputer(strategy="mean")
mode_imp = SimpleImputer(strategy="most_frequent")

for col in miss_cols:
    if X_train[col].dtype=="object":
        X_train[col] = mode_imp.fit_transform(X_train[[col]])
        X_test[col] = mode_imp.transform(X_test[[col]])
    else:
        X_train[col] = mean_imp.fit_transform(X_train[[col]])
        X_test[col] = mean_imp.transform(X_test[[col]])

# %%
# Categorical encoding
import category_encoders as ce
oh_encoder = ce.OneHotEncoder(cols=obj_cols,
                              use_cat_names=True)
oh_encoder.fit(X_train)

# Encoding train set
X_train = oh_encoder.transform(X_train)
# Encoding test set
X_test = oh_encoder.transform(X_test)

# %%
# Scaling
scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# %%
# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_sc_pf = poly.fit_transform(X_train_sc)
X_test_sc_pf = poly.fit_transform(X_test_sc)

# %% [markdown]
# # Modeling

# %%
from sklearn import metrics

def evaluate_model(models, X_train, X_test, y_train, y_test):

    summary = {
        "Model": [],
        "Train R2": [],
        "Test R2": [],
        "Train RMSE": [],
        "Test RMSE": [],
    }

    for label, model in models.items():
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        summary["Model"].append(label)

        summary["Train R2"].append(metrics.r2_score(y_train, y_train_pred))
        summary["Train RMSE"].append(
            np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
        )

        summary["Test R2"].append(metrics.r2_score(y_test, y_test_pred))
        summary["Test RMSE"].append(
            np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
        )

    summary = pd.DataFrame(summary)
    summary.set_index("Model", inplace=True)

    return round(summary.sort_values(by="Test RMSE"), 4)



# %%
lr = LinearRegression()
evaluate_model({'Linear': lr}, X_train, X_test, y_train, y_test)

# %%
lasso = Lasso(max_iter=9999)
ridge = Ridge(max_iter=9999)

models = {'Lasso': lasso,
          'Ridge': ridge}

evaluate_model(models, X_train_sc_pf, X_test_sc_pf, y_train, y_test)

# %% tags=["outputPrepend"]
import optuna
from sklearn.model_selection import cross_val_score

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_sc_pf = poly.fit_transform(X_train_sc)
X_test_sc_pf = poly.fit_transform(X_test_sc)

def objective(trial):

    alpha = trial.suggest_uniform('alpha', 0.001, 10)
    max_iter = 9999

    model = Lasso(alpha=alpha, max_iter=max_iter)
    model.fit(X_train_sc_pf, y_train)

    return metrics.r2_score(y_test, model.predict(X_test_sc_pf))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %% tags=["outputPrepend"]
import optuna
from sklearn.model_selection import cross_val_score

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_sc_pf = poly.fit_transform(X_train_sc)
X_test_sc_pf = poly.fit_transform(X_test_sc)

def objective(trial):

    alpha = trial.suggest_uniform('alpha', 0.001, 10)
    max_iter = 9999

    model = Ridge(alpha=alpha, max_iter=max_iter)
    model.fit(X_train_sc_pf, y_train)

    return metrics.r2_score(y_test, model.predict(X_test_sc_pf))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %%
lasso = Lasso(alpha=4.192572864421096, max_iter=9999)
ridge = Ridge(alpha=1.732333139228904, max_iter=9999)

models = {'Lasso': lasso,
          'Ridge': ridge}

evaluate_model(models, X_train_sc_pf, X_test_sc_pf, y_train, y_test)

# %%
