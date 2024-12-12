from BaseClassifier import BaseClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from logger_manager import get_logger

# Example subclass implementing the BaseClassifier interface
class ClassificationPredictor(BaseClassifier):

  def __init__(self, algorithm: str, **kwargs):
    self.logger = get_logger('ClassificationPredictor')
    self.logger.info(f"Initializing ClassificationPredictor with algorithm: {algorithm}")
    self.model = self.choose_model(algorithm, **kwargs)
    self.algorithm = algorithm

  def fit(self, X, y):
    self.logger.info("Fitting the model.")
    self.model.fit(X, y)

  def predict(self, X):
    self.logger.info("Making predictions.")
    return self.model.predict(X)

  def evaluate(self, X, y):
    from sklearn.metrics import accuracy_score
    self.logger.info("Evaluating the model.")
    y_pred = self.model.predict(X)
    return accuracy_score(y, y_pred)
  
  def choose_model(self, algorithm: str, **kwargs):
    """
    Select and return a ML model based on the input algorithm.

    Parameters:
    algorithm: str
      Name of the algorithm (LogisticRegression, RandomForestClassifier, DecisionTreeClassifier).
    kwargs: dict
      Optional parameters passed to the constructor of the model.

    Returns:
    model: Scikit-learn estimator
    """
    if algorithm == "LogisticRegression":
      return LogisticRegression(**kwargs)
    elif algorithm == "RandomForestClassifier":
      return RandomForestClassifier(**kwargs)
    elif algorithm == "DecisionTreeClassifier":
      return DecisionTreeClassifier(**kwargs)
    else:
      self.logger.error(f"Unsupported algorithm: {algorithm}.")
      raise ValueError(f"Unsupported algorithm: {algorithm}")
