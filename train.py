import random

import pandas as pd
import boto3
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import mlflow
import mlflow.sklearn
import numpy as np
#fetched the cleaned data from s3 bucket
s3=boto3.client('s3')
BUCKET="mlops-prediction-56"
KEY="processed/2026-09-29/Mlops_house_predication_clean_v1.csv"

def fetch_data():
    obj=s3.get_object(Bucket=BUCKET,Key=KEY)
    df=pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    return df
df=fetch_data()
print(f"Fetched shape:{df.shape}")

#fetures/targets
X=df[['bedrooms',"bathrooms",'age_years','garage','location_score']]
y=df['price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("mlops-prediction-56")

with mlflow.start_run():
    n_estimators=150
    max_depth=8


    model=RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
        )

    model.fit(X_train,y_train)

    preds = model.predict(X_test)
    mae=mean_absolute_error(y_test,preds)
    rmse= np.sqrt(mean_squared_error(y_test,preds))
    r2=r2_score(y_test,preds)


    mlflow.log_param("n_estimators",n_estimators)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("data_source",f"s3://{BUCKET}/{KEY}")
    mlflow.log_metric("mae",mae)
    mlflow.log_metric("rmse",rmse)
    mlflow.log_metric("r2_score",r2)
    
    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

print(f"\nMAE: {mae:2f}| RMSE: {rmse:2f}| R2 Score: {r2:4f}")
