import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from logger_manager import get_logger

from main import AutoML

# Logger configuration
logger = get_logger('PipelineApp')

automl = AutoML()



# Hyperparameter Tuning
def tune_hyperparameters(pipeline, X, y, param_grid):
    try:
      logger.info("Starting hyperparameter tuning...")
      grid_search = GridSearchCV(pipeline, param_grid)
      grid_search.fit(X, y)
      logger.info("Hyperparameter tuning completed. Best parameters: %s", grid_search.best_params_)
      return grid_search.best_estimator_
    except Exception as e:
      logger.error("Error during hyperparameter tuning: %s", str(e))
      raise


# Save Model
def save_model(model, filename):
  try:
    joblib.dump(model, filename)
    logger.info("Model saved to %s", filename)
  except Exception as e:
    logger.error("Error saving model: %s", str(e))
    raise



# Visualize Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
  from sklearn.metrics import confusion_matrix
  try:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
  except Exception as e:
    logger.error("Error plotting confusion matrix: %s", str(e))
    raise

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

  try:
    # Run the pipeline
    pipeline, X, y = train_and_evaluate_pipeline(
      file_path=args.file_path,
      target_column=args.target_column,
      algorithm=args.algorithm,
      test_size=args.test_size,
      random_state=args.random_state,
      cv=args.cv
    )
    
    # Save the model
    save_model(pipeline, 'heart_disease_model.pkl')
    
    # tune hyperparameters
    param_grid = AutoML.create_param_grid(args.algorithm)

    best_pipeline = tune_hyperparameters(pipeline, X, y, param_grid)

  except Exception as e:
    logger.critical("Critical error in main: %s", str(e))

