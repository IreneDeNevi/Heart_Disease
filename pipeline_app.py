import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load Data
def load_data(file_path, target_column):
    """
    Load the preprocessed dataset.
    """
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

# Create Pipeline
def create_pipeline(algorithm, **kwargs):
    """
    Create an ML pipeline with scaling and the specified algorithm.
    """
    if algorithm == "LogisticRegression":
        model = LogisticRegression(**kwargs)
    elif algorithm == "RandomForestClassifier":
        model = RandomForestClassifier(**kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    return pipeline

# Train and Evaluate Pipeline
def train_and_evaluate_pipeline(file_path, target_column, algorithm, test_size=0.2, random_state=42, cv=5):
    """
    Train and evaluate the ML pipeline.
    """
    # Load the dataset
    X, y = load_data(file_path, target_column)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create the pipeline
    pipeline = create_pipeline(algorithm)
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate the pipeline on the test set
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Cross-validation scores
    cv_scores = cross_val_score(pipeline, X, y, cv=cv)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.2f}")

# Main Function
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a machine learning pipeline for heart disease data")
    parser.add_argument("file_path", help="Path to the cleaned CSV file")
    parser.add_argument("target_column", help="Target column name")
    parser.add_argument("algorithm", help="ML algorithm to use (LogisticRegression or RandomForestClassifier)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test data")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    args = parser.parse_args()

    # Run the pipeline
    train_and_evaluate_pipeline(
        file_path=args.file_path,
        target_column=args.target_column,
        algorithm=args.algorithm,
        test_size=args.test_size,
        random_state=args.random_state,
        cv=args.cv
    )
