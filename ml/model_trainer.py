import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the dataset
def load_data(filepath):
    return pd.read_csv(filepath)

# Function to one-hot encode the 'Gender' column
def one_hot_encode(df):
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True) 
    return df

# Function to train the model make predictions
def train_and_predict_model(filepath, target_column='Churn'):
    df = load_data(filepath)
    
    # One-hot encode the 'Gender' column
    df = one_hot_encode(df)
    
    # Split the data into features (X) and target (y)
    X = df.drop(columns=[target_column, 'CustomerID'], axis=1)
    y = df[target_column]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and train the RandomForest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model performance
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)


# Function to plot the confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    """Plot the confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')  # Save the plot
    plt.show()



# Main function
def main():
    train_and_predict_model('../cleaning/clean_data.csv')

if __name__ == "__main__":
    main()
