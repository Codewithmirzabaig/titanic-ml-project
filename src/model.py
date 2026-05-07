# --- IMPORT LIBRARIES ---
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# --- LOAD DATA ---
def load_data(path):
    df = pd.read_csv(path)
    return df


# --- PREPROCESS DATA ---
def preprocess_data(df):
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Drop unnecessary columns safely
    cols_to_drop = ['Cabin', 'Name', 'Ticket', 'PassengerId']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

    # Encode categorical variables
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    return df


# --- SPLIT DATA ---
def split_data(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    return train_test_split(X, y, test_size=0.2, random_state=42)


# --- TRAIN MODEL ---
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


# --- EVALUATE MODEL ---
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, cm, report


# --- SAVE RESULTS ---
def save_results(accuracy, cm, report, path="../outputs/results.txt"):
    with open(path, "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)


# --- MAIN FUNCTION ---
def main():
    # Load
    df = load_data("../data/Titanic-Dataset.csv")

    # Preprocess
    df = preprocess_data(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    accuracy, cm, report = evaluate_model(model, X_test, y_test)

    print("Accuracy:", accuracy)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    # Save results
    save_results(accuracy, cm, report)


# --- RUN SCRIPT ---
if __name__ == "__main__":
    main()