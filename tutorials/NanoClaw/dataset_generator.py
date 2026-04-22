import pandas as pd
import numpy as np

np.random.seed(42)

n = 100

df = pd.DataFrame({
    "user_id": range(1, n + 1),
    "age": np.random.normal(30, 8, n).round(0),
    "income": np.random.normal(60000, 15000, n).round(0),
    "spend_score": np.random.uniform(1, 100, n).round(2),
    "country": np.random.choice(["US", "UK", "IN", "DE"], n),
    "signup_date": pd.date_range("2024-01-01", periods=n, freq="D")
})

# Add missing values
df.loc[np.random.choice(n, 10, replace=False), "income"] = np.nan
df.loc[np.random.choice(n, 8, replace=False), "age"] = np.nan

# Add outliers
df.loc[np.random.choice(n, 2), "income"] = 500000
df.loc[np.random.choice(n, 2), "spend_score"] = 1000

# SAVE IN SAME FOLDER
df.to_csv("dummy_users.csv")

print("Saved dummy_users.csv in current folder")