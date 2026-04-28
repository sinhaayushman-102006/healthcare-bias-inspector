import pandas as pd

# Load dataset (update path if needed)
df = pd.read_csv("data/heart.csv")

# Preview
print("Original Columns:", df.columns)

# ------------------ RENAME COLUMNS ------------------
# Standardize names for your app
df.rename(columns={
    "sex": "gender",   # IMPORTANT for bias detection
    "target": "target"
}, inplace=True)

# ------------------ SELECT REQUIRED COLUMNS ------------------
# Keep only useful columns
df = df[[
    "age",
    "gender",
    "chol",        # cholesterol
    "trestbps",    # blood pressure
    "thalach",     # max heart rate
    "target"
]]

# Rename for consistency with your app
df.rename(columns={
    "chol": "cholesterol",
    "trestbps": "bp",
    "thalach": "glucose"  # (proxy feature)
}, inplace=True)

# ------------------ HANDLE MISSING VALUES ------------------
df.fillna(df.median(numeric_only=True), inplace=True)

# ------------------ OPTIONAL: CREATE SLIGHT BIAS ------------------
# (Helps your demo — makes bias visible)
male_df = df[df["gender"] == 1]
female_df = df[df["gender"] == 0]

# Reduce female samples slightly to create imbalance
female_df = female_df.sample(frac=0.6, random_state=42)

df = pd.concat([male_df, female_df])

# ------------------ SAVE CLEAN DATASET ------------------
df.to_csv("data/processed_heart.csv", index=False)

print("✅ Processed dataset saved as data/processed_heart.csv")
