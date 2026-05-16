import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv("Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

engine = create_engine(
    "postgresql://postgres:Prabhas%40123@localhost:5432/CUSTOMER_CHURN_PROJECT"
)

df.to_sql(
    "customer_churn",
    engine,
    if_exists="replace",
    index=False
)

print("Dataset loaded successfully!")