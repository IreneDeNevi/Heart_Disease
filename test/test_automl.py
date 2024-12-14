import pytest
import os
import sys
import pandas as pd
# include the project root directory (Heart_Disease) in the Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from automl import AutoML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


@pytest.fixture(scope="module")
def setup_csv_file():
    # Create a temporary csv file
    csv_path = "test_dataset.csv"
    df = pd.DataFrame({
      'feature1': [1, 2, 3, 4, 100],
      'feature2': [5, 6, None, 8, 1000],
      'target': [0, 1, 0, 1, 0]
    })
    df.to_csv(csv_path, index=False)
    yield csv_path
    # Remove the file after the tests
    os.remove(csv_path)

@pytest.fixture
def automl_instance():
  return AutoML()

def test_initialize_model_logistic_regression(automl_instance):
  automl_instance.algorithm = "LogisticRegression"
  automl_instance.initialize_model()
  assert automl_instance.task_type == "Classification"
  assert automl_instance.model is not None
  assert isinstance(automl_instance.model, LogisticRegression)


def test_initialize_model_random_forest_classifier(automl_instance):
  automl_instance.algorithm = "RandomForestClassifier"
  automl_instance.initialize_model()
  assert automl_instance.task_type == "Classification"
  assert automl_instance.model is not None
  assert isinstance(automl_instance.model, RandomForestClassifier)


def test_initialize_model_decision_tree_classifier(automl_instance):
  automl_instance.algorithm = "DecisionTreeClassifier"
  automl_instance.initialize_model()
  assert automl_instance.task_type == "Classification"
  assert automl_instance.model is not None
  assert isinstance(automl_instance.model, DecisionTreeClassifier)


def test_initialize_model_linear_regression(automl_instance):
  automl_instance.algorithm = "LinearRegression"
  automl_instance.initialize_model()
  assert automl_instance.task_type == "Regression"
  assert automl_instance.model is not None
  assert isinstance(automl_instance.model, LinearRegression)


def test_initialize_model_decision_tree_regressor(automl_instance):
  automl_instance.algorithm = "DecisionTreeRegressor"
  automl_instance.initialize_model()
  assert automl_instance.task_type == "Regression"
  assert automl_instance.model is not None
  assert isinstance(automl_instance.model, DecisionTreeRegressor)


def test_initialize_model_random_forest_regressor(automl_instance):
  automl_instance.algorithm = "RandomForestRegressor"
  automl_instance.initialize_model()
  assert automl_instance.task_type == "Regression"
  assert automl_instance.model is not None
  assert isinstance(automl_instance.model, RandomForestRegressor)

def test_load_dataset(automl_instance, setup_csv_file):
  automl_instance.load_dataset(setup_csv_file)
  assert automl_instance.data is not None
  assert not automl_instance.data.empty

def test_check_if_dataset_contains_target(automl_instance, setup_csv_file):
  automl_instance.load_dataset(setup_csv_file)
  automl_instance.check_if_dataset_contains_target("target")
  assert automl_instance.target == "target"

def test_preprocess_data(automl_instance, setup_csv_file):
  automl_instance.load_dataset(setup_csv_file)
  automl_instance.check_if_dataset_contains_target("target")
  automl_instance.features = automl_instance.data.drop(columns=["target"])
  automl_instance.preprocess_data()
  assert not automl_instance.features.isnull().values.any()

def test_is_dataset_empty(automl_instance):
  with pytest.raises(ValueError):
    automl_instance.is_dataset_empty()
