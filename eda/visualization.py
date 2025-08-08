import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load both the original and cleaned datasets
df_original = pd.read_csv('../exl_credit_card_churn_data.csv')  # Original dataset (before cleaning)
df_cleaned = pd.read_csv('../cleaning/clean_data.csv')  # Cleaned dataset

# Define a custom color palette
custom_palette = ['#1f77b4', '#ff7f0e']  # Example: blue and orange

# Scatter plot: Age vs Balance
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(df_original['Age'], df_original['Balance'], color='#1f77b4', alpha=0.5)
plt.title('Scatter: Age vs Balance - Original')
plt.xlabel('Age')
plt.ylabel('Balance')

plt.subplot(1, 2, 2)
plt.scatter(df_cleaned['Age'], df_cleaned['Balance'], color='#ff7f0e', alpha=0.5)
plt.title('Scatter: Age vs Balance - Cleaned')
plt.xlabel('Age')
plt.ylabel('Balance')

plt.tight_layout()
plt.savefig('scatter_age_vs_balance_comparison.png')  # Save the plot
plt.show()

# Pie Chart: Churn Distribution
churn_counts_cleaned = df_cleaned['Churn'].value_counts()

# Ensure the labels match the churn values for the cleaned dataset
labels_cleaned = churn_counts_cleaned.index.map({0: 'No Churn (0)', 1: 'Churned (1)'})

plt.figure(figsize=(12, 6))
plt.pie(churn_counts_cleaned, labels=labels_cleaned, autopct='%1.1f%%', colors=custom_palette, startangle=90)
plt.title('Churn Distribution - Cleaned')

plt.tight_layout()
plt.savefig('churn_distribution_comparison_pie.png')  # Save the plot
plt.show()

# Correlation Heatmap
numeric_columns_cleaned = df_cleaned.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
corr_cleaned = numeric_columns_cleaned.corr()  # Correlation matrix for the cleaned dataset

# Plot the heatmap for the correlation matrix
plt.figure(figsize=(8, 6))

# Cleaned dataset correlation heatmap
sns.heatmap(corr_cleaned, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap - Cleaned')

plt.tight_layout()
plt.savefig('correlation_heatmap_cleaned.png')  # Save the plot for the cleaned dataset
plt.show()

#  Plot the Balance Distribution (with KDE)
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

#  Plot the Credit Card Ownership Distribution
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

#  Plot the Active Membership Distribution
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

#  Plot the Churn Distribution
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

#  Age vs Churn
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

#  Tenure vs Churn 
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

#  Balance vs Churn (
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

#  Credit Card Ownership vs Churn 
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

#  Active Membership vs Churn
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

#  Estimated Salary vs Churn
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
