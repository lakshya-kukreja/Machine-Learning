from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

import pickle

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "module_3_classification" / "datasets" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "pipeline.bin"


# 1. Load and clean data
df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(" ", "_")

df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges = df.totalcharges.fillna(0)

df['churn'] = (df.churn == 'yes').astype(int)

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)

df_full_train = df_full_train.reset_index(drop = True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.churn.values
y_test = df_test.churn.values

categorical = list(df.dtypes[(df.dtypes == 'str') & (df.columns != 'customerid')].index)
numerical = list(df.dtypes[(((df.dtypes == 'int64') | (df.dtypes == 'float64')) & (df.columns != 'churn'))].index)


train_dicts = df_full_train[categorical+numerical].to_dict(orient='records')

test_dicts = df_test[categorical + numerical].to_dict(orient='records')

pipeline = make_pipeline(
    DictVectorizer(sparse=False),
    LogisticRegression(
        C = 0.5,
        max_iter = 5000
    )
)

pipeline.fit(train_dicts, y_full_train)

y_pred = pipeline.predict_proba(test_dicts)[:, 1]

auc = roc_auc_score(y_test, y_pred)

print(f'AUC = {auc:.3f}')

with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved to " + str(MODEL_PATH))
