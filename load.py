import pandas as pd
from sqlalchemy import create_engine

# Load dataset
df = pd.read_csv("Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# PostgreSQL connection
engine = create_engine(
    "postgresql://postgres:charan%4014@localhost:5433/CUSTOMER_CHURN_PROJECT"
)

# Upload dataset
df.to_sql(
    "customer_churn",
    engine,
    if_exists="replace",
    index=False
)

print("Dataset loaded successfully!")