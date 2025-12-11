# %% [markdown]
# # Cable Failure Prediction using 15-KV XLPE Underground Cable Dataset
# 
# This notebook covers the end-to-end data processing pipeline for predicting cable failures.
# 

# %% [markdown]
# ## 🔵 STEP 1 — LOAD DATA & INITIAL EXPLORATION
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure plotting
%matplotlib inline
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load Dataset
file_path = '15-KV XLPE Cable.xlsx'
if os.path.exists(file_path):
    df = pd.read_excel(file_path)
    print("Dataset loaded successfully.")
else:
    print(f"File not found: {file_path}")



# %%
# Show basic info
print("Shape:", df.shape)
display(df.head())
display(df.tail())
print(df.info())
display(df.describe())



# %%
# Identify columns
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
target_col = 'Health Index'

# Adjust lists if target is in numerical
if target_col in numerical_features:
    numerical_features.remove(target_col)

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)
print("Target Column:", target_col)



# %% [markdown]
# ## 🔵 STEP 2 — DATA CLEANING
# 

# %%
# Check missing values
missing_summary = df.isnull().sum().to_frame(name='Missing Values')
missing_summary['Percentage'] = (missing_summary['Missing Values'] / len(df)) * 100
display(missing_summary)



# %%
# Handle missing values
# Impute numerical with median, categorical with mode
for col in numerical_features:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

for col in categorical_features:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

print("Missing values after handling:")
print(df.isnull().sum().sum())



# %%
# Detect Outliers
outlier_cols = ['Age', 'Partial Discharge', 'Neutral Corrosion', 'Loading']
# Check which columns actually exist
cols_to_plot = [c for c in outlier_cols if c in df.columns]

if cols_to_plot:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(cols_to_plot, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()
else:
    print("Columns for outlier detection not found.")



# %% [markdown]
# ## 🔵 STEP 3 — TARGET ENGINEERING
# 

# %%
def map_health_index(hi):
    if hi <= 2:
        return 0 # Safe
    elif hi == 3:
        return 1 # Medium
    else:
        return 2 # High Risk

if 'Health Index' in df.columns:
    df['health_class'] = df['Health Index'].apply(map_health_index)
    print("Target Class Distribution:")
    print(df['health_class'].value_counts())
else:
    print("Health Index column not found.")



# %% [markdown]
# ## 🔵 STEP 4 — FEATURE ENGINEERING
# 

# %%
from sklearn.preprocessing import StandardScaler

# 1. One-hot encode Visual Condition
if 'Visual Condition' in df.columns:
    df = pd.get_dummies(df, columns=['Visual Condition'], drop_first=True)
    print("One-hot encoding applied to Visual Condition.")

# 3. (Optional) Create engineered features
# Note: Creating these BEFORE scaling to avoid issues with scaled values (e.g. division by zero or negative age)
# load_age_ratio = Loading / (Age + 1)
# pd_corrosion = Partial_Discharge * Neutral_Corrosion

# Check for columns
cols = df.columns
if 'Loading' in cols and 'Age' in cols:
    df['load_age_ratio'] = df['Loading'] / (df['Age'] + 1)

if 'Partial Discharge' in cols and 'Neutral Corrosion' in cols:
    df['pd_corrosion'] = df['Partial Discharge'] * df['Neutral Corrosion']

# Update numerical features list to include new features
new_features = ['load_age_ratio', 'pd_corrosion']
for f in new_features:
    if f in df.columns:
        numerical_features.append(f)

# 2. Scale numerical features
scaler = StandardScaler()
# Scale all numerical features
# Ensure we only scale columns that exist and are numerical
valid_numerical = [c for c in numerical_features if c in df.columns]
df[valid_numerical] = scaler.fit_transform(df[valid_numerical])
print("Numerical features scaled.")
display(df.head())



# %% [markdown]
# ## 🔵 STEP 5 — EDA VISUALIZATION
# 

# %%
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Histograms
valid_numerical = [c for c in numerical_features if c in df.columns]
df[valid_numerical].hist(figsize=(15, 10), bins=20)
plt.suptitle('Histograms of Numerical Features')
plt.savefig('plots/histograms.png')
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 10))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('plots/correlation_heatmap.png')
plt.show()

# 3. Relationship Plots
relationships = [
    ('Age', 'health_class'),
    ('Partial Discharge', 'health_class'),
    ('Neutral Corrosion', 'health_class'),
    ('Loading', 'health_class')
]

for x_col, y_col in relationships:
    if x_col in df.columns and y_col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=y_col, y=x_col, data=df)
        plt.title(f'{x_col} vs {y_col}')
        plt.savefig(f'plots/{x_col.replace(" ", "_").lower()}_vs_{y_col}.png')
        plt.show()



# %% [markdown]
# ## 🔵 STEP 6 — TRAIN / TEST SPLIT
# 

# %%
from sklearn.model_selection import train_test_split
import pickle

# Define X and y
# Drop original target and created target from X
drop_cols = ['Health Index', 'health_class']
X = df.drop([c for c in drop_cols if c in df.columns], axis=1)
y = df['health_class']

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Save to pickle
with open('X_train.pkl', 'wb') as f: pickle.dump(X_train, f)
with open('X_test.pkl', 'wb') as f: pickle.dump(X_test, f)
with open('y_train.pkl', 'wb') as f: pickle.dump(y_train, f)
with open('y_test.pkl', 'wb') as f: pickle.dump(y_test, f)

print("Data split and saved.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)




