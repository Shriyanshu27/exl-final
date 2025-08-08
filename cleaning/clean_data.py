import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  

# Function to handle missing values, clean categorical columns, and handle binary conversion
def handle_missing_and_binary_conversion(df, binary_columns=['HasCrCard', 'IsActiveMember', 'Churn']):
    
    # Handle missing values in 'Gender' and fill with mode (most frequent value)
    if 'Gender' in df.columns:
        gender_mode = df['Gender'].mode()[0]
        df['Gender'] = df['Gender'].fillna(gender_mode)
        
        # Convert 'Gender' to lowercase
        df['Gender'] = df['Gender'].str.lower()

    # Handle missing values in numeric columns and fill with mean value
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            mean_value = df[col].mean()  
            df[col] = df[col].fillna(mean_value)

    # Drop rows where 'CustomerID' and 'Churn' is missing
    df = df.dropna(subset=['CustomerID', 'Churn'])

    # 3. Convert binary columns to int (0 or 1)
    for col in binary_columns:
        # Convert the column to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df[col] = df[col].fillna(df[col].mode()[0])
        
        df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)
        
        df[col] = df[col].astype(int)

    print("Missing data handled and binary conversion complete.")
    return df

# Function to handle outliers using the IQR method
def handle_outliers(df):
    
    print(f"Shape before outlier handling: {df.shape}")
    
    # Identify numeric columns for outlier detection
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Outlier detection and removal using IQR method
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove the rows with outliers (filter the data to exclude outliers)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    print(f"Shape after outlier handling: {df.shape}")
    return df

# Function to apply Min-Max Scaling
def apply_min_max_scaling(df):
    
    # Check if DataFrame is empty after cleaning and outlier removal
    if df.empty:
        print("Error: The dataset is empty after cleaning and outlier removal.")
        return df

    print(f"Shape before scaling: {df.shape}")

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Apply scaling to the selected columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(f"Shape after scaling: {df.shape}")
    return df


def main():
    # Load the dataset
    df = pd.read_csv('../exl_credit_card_churn_data.csv')  
    
    # Handle missing data
    df = handle_missing_and_binary_conversion(df)
    print("After cleaning:")
    print(df.shape)
    
    # Handle outliers
    df = handle_outliers(df)
    print("After outlier removal:")
    print(df.shape)

    # Apply Min-Max Scaling
    df = apply_min_max_scaling(df)
    print("After scaling:")
    print(df.shape)

    df.to_csv('clean_data.csv', index=False)
    print("Data cleaning complete, scaling applied, and saved to 'clean_data.csv'.")

if __name__ == "__main__":
    main()
