from abc import ABC, abstractmethod

class BaseClassifier(ABC):
  """
  Abstract base class for machine learning predictors.
  Defines a common interface for ML models.
  """

  @abstractmethod
  def fit(self, X, y):
    """
    Trains the model on the given data.

    Parameters:
    X: array-like of shape (n_samples, n_features)
      Training data.
    y: array-like of shape (n_samples,)
      Target values.
    """
    pass

  @abstractmethod
  def predict(self, X):
    """
    Predicts the target values for the given input data.

    Parameters:
    X: array-like of shape (n_samples, n_features)
      Input data.

    Returns:
    y_pred: array-like of shape (n_samples,)
      Predicted target values.
    """
    pass

  @abstractmethod
  def evaluate(self, X, y):
    """
    Evaluates the model on the given test data and target values.

    Parameters:
    X: array-like of shape (n_samples, n_features)
      Test data.
    y: array-like of shape (n_samples,)
      True target values.

    Returns:
    score: float
      Evaluation metric score.
    """
    pass
