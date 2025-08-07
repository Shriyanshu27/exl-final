import pandas as pd

# Function to handle missing values, convert gender to lowercase, and handle binary conversion
def handle_missing_and_binary_conversion(df, binary_columns=['HasCrCard', 'IsActiveMember', 'Churn']):
    """Handle missing values and binary conversion in the dataset"""
    
    # Handle missing values in 'Gender' and fill with mode (most frequent value)
    if 'Gender' in df.columns:
        # Fill missing values in 'Gender' with the mode (most frequent value)
        gender_mode = df['Gender'].mode()[0]
        df['Gender'] = df['Gender'].fillna(gender_mode)
        
        # Convert 'Gender' to lowercase
        df['Gender'] = df['Gender'].str.lower()

    # Handle missing values in numeric columns and fill with mean value
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            mean_value = df[col].mean()  # Replace with mean
            df[col] = df[col].fillna(mean_value)

    # Drop rows where 'CustomerID' is missing (if any)
    df = df.dropna(subset=['CustomerID'])

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
    """Detect and remove outliers in the dataset"""
    
    # Identify numeric columns for outlier detection
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Outlier detection and removal using IQR method
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Replace outliers with the median value of the column
        median_value = df[col].median()
        df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = median_value

    print("Outliers handled by replacing them with the median.")
    return df


# Example of how to use the functions:
def main():
    # Load the dataset
    df = pd.read_csv('exl_credit_card_churn_data.csv')  # Use your file path
    
    # Handle missing data and binary conversion
    df = handle_missing_and_binary_conversion(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Optionally, save the cleaned data to a new file
    df.to_csv('cleaned_churn_data.csv', index=False)
    print("Data cleaning complete and saved to 'cleaned_churn_data.csv'.")

if __name__ == "__main__":
    main()
