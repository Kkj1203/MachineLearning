
#!/usr/bin/env python3
\"\"\"clean_employees_pipeline.py

A safe, reusable pipeline to:
 - load an 'employees.csv' file (relative path)
 - one-hot encode the 'Gender' column (keeps clear column names)
 - optionally label-encode Gender instead (demo)
 - detect & remove salary outliers using IQR (1.5*IQR by default)
 - scale numeric features using StandardScaler
 - save cleaned CSV and plots to the working directory

Usage:
  - Place employees.csv in the same folder as this script, then run:
      python clean_employees_pipeline.py
  - The script will write:
      - employees_cleaned.csv
      - salary_before_boxplot.png
      - salary_after_boxplot.png

Notes:
  - The script uses a safe single read of the CSV and avoids SettingWithCopy warnings.
  - If your dataset has different column names, update the constants at the top.
\"\"\"

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# ----- CONFIG -----
INPUT_CSV = "employees.csv"   # place employees.csv in same directory
OUTPUT_CSV = "employees_cleaned.csv"
SALARY_COL = "Salary"
GENDER_COL = "Gender"
IQR_FACTOR = 1.5              # standard IQR multiplier for outlier detection
# ------------------

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f\"Input file not found: {path}\\nPlease put '{path}' in the script folder or update INPUT_CSV.\")
    df = pd.read_csv(path)
    return df

def one_hot_encode_gender(df: pd.DataFrame, gender_col: str = GENDER_COL) -> pd.DataFrame:
    if gender_col not in df.columns:
        raise KeyError(f\"Column '{gender_col}' not found in dataframe.\")
    ohe = OneHotEncoder(sparse=False, dtype=int)
    arr = ohe.fit_transform(df[[gender_col]])
    cols = ohe.get_feature_names_out([gender_col])
    df_ohe = df.copy()
    df_ohe = df_ohe.join(pd.DataFrame(arr, columns=cols, index=df_ohe.index))
    df_ohe = df_ohe.drop(columns=[gender_col])
    return df_ohe

def label_encode_gender(df: pd.DataFrame, gender_col: str = GENDER_COL) -> pd.DataFrame:
    if gender_col not in df.columns:
        raise KeyError(f\"Column '{gender_col}' not found in dataframe.\")
    le = LabelEncoder()
    df_le = df.copy()
    df_le[gender_col] = le.fit_transform(df_le[gender_col].astype(str))
    return df_le

def inject_outliers(df: pd.DataFrame, col: str = SALARY_COL, injections: dict | None = None) -> pd.DataFrame:
    \"\"\"Optional: safely set given index->value pairs for demonstration.
       injections should be a dict like {0:200000, 10:250000}
    \"\"\"
    df2 = df.copy()
    if injections:
        for idx, val in injections.items():
            if idx in df2.index:
                df2.loc[idx, col] = val
            else:
                print(f\"Warning: index {idx} not in dataframe; skipping injection.\")
    return df2

def remove_outliers_iqr(df: pd.DataFrame, col: str = SALARY_COL, factor: float = IQR_FACTOR) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f\"Column '{col}' not found in dataframe.\")
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    cleaned = df[(df[col] >= lower) & (df[col] <= upper)].copy()
    return cleaned, lower, upper

def scale_numeric(df: pd.DataFrame, exclude: list[str] = None) -> pd.DataFrame:
    exclude = exclude or []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [c for c in numeric_cols if c not in exclude]
    if not cols_to_scale:
        return df.copy()
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    return df_scaled

def plot_boxplots(before_series: pd.Series, after_series: pd.Series, before_path: str, after_path: str):
    plt.figure(figsize=(8,4))
    plt.boxplot(before_series.dropna().values)
    plt.title('Salary - before outlier removal')
    plt.savefig(before_path, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.boxplot(after_series.dropna().values)
    plt.title('Salary - after outlier removal')
    plt.savefig(after_path, bbox_inches='tight')
    plt.close()

def main():
    try:
        df = load_csv(INPUT_CSV)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(\"Loaded dataframe: rows=\", len(df), \"cols=\", len(df.columns))
    if SALARY_COL not in df.columns:
        print(f\"Warning: '{SALARY_COL}' column not found. Current columns: {df.columns.tolist()}\")
    else:
        # Demonstration: create plots before/after outlier removal
        before_plot = \"salary_before_boxplot.png\"
        after_plot = \"salary_after_boxplot.png\"

        # Optional: inject a few extreme salaries only if you know the dataset size.
        # injections = {0: 200184, 2: 230000}   # <-- uncomment & edit if you want to simulate outliers
        injections = None
        df_injected = inject_outliers(df, col=SALARY_COL, injections=injections)

        try:
            cleaned_df, lower, upper = remove_outliers_iqr(df_injected, col=SALARY_COL, factor=IQR_FACTOR)
            print(f\"IQR bounds: lower={lower:.2f}, upper={upper:.2f}\")
            if SALARY_COL in df_injected.columns:
                plot_boxplots(df_injected[SALARY_COL], cleaned_df[SALARY_COL], before_plot, after_plot)
                print(f\"Saved boxplots: {before_plot}, {after_plot}\")
        except KeyError:
            print(f\"Skipping outlier removal because '{SALARY_COL}' is missing.\")

    # Encoding: prefer one-hot for 'Gender' if it's categorical; fallback safely if missing
    if GENDER_COL in df.columns:
        try:
            df_ohe = one_hot_encode_gender(df, gender_col=GENDER_COL)
            print(f\"One-hot encoded: new columns added -> {[c for c in df_ohe.columns if c.startswith(GENDER_COL)]}\")
        except Exception as e:
            print(\"One-hot encoding failed:\", e)
            df_ohe = label_encode_gender(df, gender_col=GENDER_COL)
            print(\"Fell back to label encoding for Gender.\")
    else:
        print(f\"Gender column '{GENDER_COL}' not present; skipping encoding.\")
        df_ohe = df.copy()

    # Scale numeric columns but exclude Salary if you want to keep original scale for business reasons
    exclude_from_scaling = []  # e.g., ['Salary'] to keep salary unscaled
    df_scaled = scale_numeric(df_ohe, exclude=exclude_from_scaling)

    # Save the cleaned dataframe
    df_scaled.to_csv(OUTPUT_CSV, index=False)
    print(f\"Saved cleaned dataframe to: {OUTPUT_CSV} (rows={len(df_scaled)})\")

if __name__ == '__main__':
    main()
