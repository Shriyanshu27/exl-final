import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load both the original and cleaned datasets
df_original = pd.read_csv('exl_credit_card_churn_data.csv')  # Original dataset (before cleaning)
df_cleaned = pd.read_csv('cleaned_churn_data.csv')    # Cleaned dataset

# Define a custom color palette
custom_palette = ['#1f77b4', '#ff7f0e']  # Example: blue and orange

# 1. Plot the Gender Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Gender', data=df_original, palette=custom_palette)
plt.title('Gender Distribution - Original')
plt.subplot(1, 2, 2)
sns.countplot(x='Gender', data=df_cleaned, palette=custom_palette)
plt.title('Gender Distribution - Cleaned')
plt.tight_layout()
plt.savefig('gender_distribution_comparison.png')  # Save the plot
plt.show()

# 2. Plot the Age Distribution (with KDE)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_original['Age'], kde=True, bins=20, color='#1f77b4')
plt.title('Age Distribution - Original')
plt.xlabel('Age')
plt.subplot(1, 2, 2)
sns.histplot(df_cleaned['Age'], kde=True, bins=20, color='#ff7f0e')
plt.title('Age Distribution - Cleaned')
plt.xlabel('Age')
plt.tight_layout()
plt.savefig('age_distribution_comparison.png')  # Save the plot
plt.show()

# 3. Plot the Tenure Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Tenure', data=df_original, palette=custom_palette)
plt.title('Tenure Distribution - Original')
plt.xlabel('Tenure (Years)')
plt.subplot(1, 2, 2)
sns.countplot(x='Tenure', data=df_cleaned, palette=custom_palette)
plt.title('Tenure Distribution - Cleaned')
plt.xlabel('Tenure (Years)')
plt.tight_layout()
plt.savefig('tenure_distribution_comparison.png')  # Save the plot
plt.show()

# 4. Plot the Balance Distribution (with KDE)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_original['Balance'], kde=True, bins=20, color='#1f77b4')
plt.title('Balance Distribution - Original')
plt.xlabel('Balance')
plt.subplot(1, 2, 2)
sns.histplot(df_cleaned['Balance'], kde=True, bins=20, color='#ff7f0e')
plt.title('Balance Distribution - Cleaned')
plt.xlabel('Balance')
plt.tight_layout()
plt.savefig('balance_distribution_comparison.png')  # Save the plot
plt.show()

# 5. Plot the Distribution of the Number of Products
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='NumOfProducts', data=df_original, palette=custom_palette)
plt.title('Number of Products Distribution - Original')
plt.xlabel('NumOfProducts')
plt.subplot(1, 2, 2)
sns.countplot(x='NumOfProducts', data=df_cleaned, palette=custom_palette)
plt.title('Number of Products Distribution - Cleaned')
plt.xlabel('NumOfProducts')
plt.tight_layout()
plt.savefig('numofproducts_distribution_comparison.png')  # Save the plot
plt.show()

# 6. Plot the Credit Card Ownership Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='HasCrCard', data=df_original, palette=custom_palette)
plt.title('Credit Card Ownership - Original')
plt.xlabel('HasCrCard (0=No, 1=Yes)')
plt.subplot(1, 2, 2)
sns.countplot(x='HasCrCard', data=df_cleaned, palette=custom_palette)
plt.title('Credit Card Ownership - Cleaned')
plt.xlabel('HasCrCard (0=No, 1=Yes)')
plt.tight_layout()
plt.savefig('hascrcard_distribution_comparison.png')  # Save the plot
plt.show()

# 7. Plot the Active Membership Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='IsActiveMember', data=df_original, palette=custom_palette)
plt.title('Active Membership - Original')
plt.xlabel('IsActiveMember (0=No, 1=Yes)')
plt.subplot(1, 2, 2)
sns.countplot(x='IsActiveMember', data=df_cleaned, palette=custom_palette)
plt.title('Active Membership - Cleaned')
plt.xlabel('IsActiveMember (0=No, 1=Yes)')
plt.tight_layout()
plt.savefig('isactivemember_distribution_comparison.png')  # Save the plot
plt.show()

# 8. Plot the Estimated Salary Distribution (with KDE)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_original['EstimatedSalary'], kde=True, bins=20, color='#1f77b4')
plt.title('Estimated Salary Distribution - Original')
plt.xlabel('EstimatedSalary')
plt.subplot(1, 2, 2)
sns.histplot(df_cleaned['EstimatedSalary'], kde=True, bins=20, color='#ff7f0e')
plt.title('Estimated Salary Distribution - Cleaned')
plt.xlabel('EstimatedSalary')
plt.tight_layout()
plt.savefig('estimated_salary_distribution_comparison.png')  # Save the plot
plt.show()

# 9. Plot the Churn Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Churn', data=df_original, palette=custom_palette)
plt.title('Churn Distribution - Original')
plt.xlabel('Churn (0=No, 1=Yes)')
plt.subplot(1, 2, 2)
sns.countplot(x='Churn', data=df_cleaned, palette=custom_palette)
plt.title('Churn Distribution - Cleaned')
plt.xlabel('Churn (0=No, 1=Yes)')
plt.tight_layout()
plt.savefig('churn_distribution_comparison.png')  # Save the plot
plt.show()

# Visualize the relationship between features and churn

# 1. Gender vs Churn
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Gender', hue='Churn', data=df_original, palette=custom_palette)
plt.title('Gender vs Churn - Original')
plt.subplot(1, 2, 2)
sns.countplot(x='Gender', hue='Churn', data=df_cleaned, palette=custom_palette)
plt.title('Gender vs Churn - Cleaned')
plt.tight_layout()
plt.savefig('gender_vs_churn_comparison.png')  # Save the plot
plt.show()

# 2. Age vs Churn (Box plot to show the spread)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Churn', y='Age', data=df_original, palette=custom_palette)
plt.title('Age Distribution by Churn - Original')
plt.subplot(1, 2, 2)
sns.boxplot(x='Churn', y='Age', data=df_cleaned, palette=custom_palette)
plt.title('Age Distribution by Churn - Cleaned')
plt.tight_layout()
plt.savefig('age_vs_churn_comparison.png')  # Save the plot
plt.show()

# 3. Tenure vs Churn (Box plot to show the spread)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Churn', y='Tenure', data=df_original, palette=custom_palette)
plt.title('Tenure vs Churn - Original')
plt.subplot(1, 2, 2)
sns.boxplot(x='Churn', y='Tenure', data=df_cleaned, palette=custom_palette)
plt.title('Tenure vs Churn - Cleaned')
plt.tight_layout()
plt.savefig('tenure_vs_churn_comparison.png')  # Save the plot
plt.show()

# 4. Balance vs Churn (Box plot to show the spread)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Churn', y='Balance', data=df_original, palette=custom_palette)
plt.title('Balance vs Churn - Original')
plt.subplot(1, 2, 2)
sns.boxplot(x='Churn', y='Balance', data=df_cleaned, palette=custom_palette)
plt.title('Balance vs Churn - Cleaned')
plt.tight_layout()
plt.savefig('balance_vs_churn_comparison.png')  # Save the plot
plt.show()

# 5. Number of Products vs Churn (Categorical comparison)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='NumOfProducts', hue='Churn', data=df_original, palette=custom_palette)
plt.title('Number of Products vs Churn - Original')
plt.subplot(1, 2, 2)
sns.countplot(x='NumOfProducts', hue='Churn', data=df_cleaned, palette=custom_palette)
plt.title('Number of Products vs Churn - Cleaned')
plt.tight_layout()
plt.savefig('numofproducts_vs_churn_comparison.png')  # Save the plot
plt.show()

# 6. Credit Card Ownership vs Churn (Categorical comparison)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='HasCrCard', hue='Churn', data=df_original, palette=custom_palette)
plt.title('Credit Card Ownership vs Churn - Original')
plt.xlabel('Has Credit Card (0 = No, 1 = Yes)')
plt.subplot(1, 2, 2)
sns.countplot(x='HasCrCard', hue='Churn', data=df_cleaned, palette=custom_palette)
plt.title('Credit Card Ownership vs Churn - Cleaned')
plt.xlabel('Has Credit Card (0 = No, 1 = Yes)')
plt.tight_layout()
plt.savefig('hascrcard_vs_churn_comparison.png')  # Save the plot
plt.show()

# 7. Active Membership vs Churn (Categorical comparison)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='IsActiveMember', hue='Churn', data=df_original, palette=custom_palette)
plt.title('Active Membership vs Churn - Original')
plt.xlabel('Is Active Member (0 = No, 1 = Yes)')
plt.subplot(1, 2, 2)
sns.countplot(x='IsActiveMember', hue='Churn', data=df_cleaned, palette=custom_palette)
plt.title('Active Membership vs Churn - Cleaned')
plt.xlabel('Is Active Member (0 = No, 1 = Yes)')
plt.tight_layout()
plt.savefig('isactivemember_vs_churn_comparison.png')  # Save the plot
plt.show()

# 8. Estimated Salary vs Churn (Box plot to show the spread)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Churn', y='EstimatedSalary', data=df_original, palette=custom_palette)
plt.title('Estimated Salary vs Churn - Original')
plt.subplot(1, 2, 2)
sns.boxplot(x='Churn', y='EstimatedSalary', data=df_cleaned, palette=custom_palette)
plt.title('Estimated Salary vs Churn - Cleaned')
plt.tight_layout()
plt.savefig('estimatedsalary_vs_churn_comparison.png')  # Save the plot
plt.show()
