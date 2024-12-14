from typing import Dict, Any
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
import joblib
import util
import importlib
importlib.reload(util)

# Logger configuration
logger = util.get_logger('AutoML', log_file='application.log')


class AutoML:
    def __init__(self):
        self.data = None
        self.target = None
        self.task_type = None
        self.features = None
        self.pipeline = None
        self.label_encoder = None
        self.algorithm = None
        self.model = None
        self.param_grid = None
        self.best_score = float('-inf')
        self.outlier_info = {}
    
    def init(self, csv_path: str, target_name: str, algorithm: str , random_state: int, model_name: str) -> None:
        logger.info(f"CSV path: [{csv_path}], Target Column: [{target_name}], Algorithm: [{algorithm}], Random State [{random_state}]")

        #Set algorithm
        self.algorithm = algorithm

        # Initialize models based on task type
        self.initialize_model()

        # Load and prepare the data
        self.load_dataset(csv_path)

        # Check if target column exists
        self.check_if_dataset_contains_target(target_name)

        # Split features and target
        self.target = self.data[target_name]
        self.features = self.data.drop(target_name, axis=1)

        # Clean and preprocess the data
        self.preprocess_data()

        # Maybe redundant
        self.label_encoder = LabelEncoder()
        self.target = pd.Series(self.label_encoder.fit_transform(self.target))

        # Save the model
        self.save_model(model_name)


    def preprocess_data(self) -> None:
        categorical_columns = self.features.select_dtypes(include=['object']).columns
        numerical_columns = self.features.select_dtypes(exclude=['object']).columns
        logger.info(f"Categorical columns:\n{categorical_columns}")
        logger.info(f"Numerical columns:\n{numerical_columns}")
        
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', LabelEncoder())
        ])
        
        logger.info(f"Processing numerical columns and detecting outliers...")
        # Process numerical columns and detect outliers
        if len(numerical_columns) > 0:
            for column in numerical_columns:
                # Calculate outlier boundaries
                Q1 = self.features[column].quantile(0.25)
                Q3 = self.features[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers_below = sum(self.features[column] < lower_bound)
                outliers_above = sum(self.features[column] > upper_bound)
                
                # Store outlier information
                self.outlier_info[column] = {
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outliers_below': outliers_below,
                    'outliers_above': outliers_above,
                    'total_outliers': outliers_below + outliers_above
                }
                logger.info(f"Total outliers found in {column}: {outliers_below + outliers_above}")
                
                # Cap the outliers
                self.features[column] = self.features[column].clip(lower_bound, upper_bound)
            
            # Apply the numeric pipeline after handling outliers
            self.features[numerical_columns] = numeric_pipeline.fit_transform(
                self.features[numerical_columns]
            )
        
        # Process categorical columns
        for column in categorical_columns:
            self.features[column] = categorical_pipeline.fit_transform(
                self.features[column].values.reshape(-1, 1)
            )

    def initialize_model(self) -> None:
        """
        Select anad initialize a ML model based on the algorithm.
        
        Assign and store the task_type
        """
        logger.info("Choosing Model...")
        algorithms = [
            "LogisticRegression",
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "LinearRegression",
            "DecisionTreeRegressor",
            "RandomForestRegressor"
        ]
        if self.algorithm not in algorithms:
            logger.error(f"Unsupported algorithm: {self.algorithm}.")
            raise ValueError(f"Unsupported algorithm: {self.algorithm}.")

        if self.algorithm == "LogisticRegression":
            self.task_type = 'Classification'
            self.model = LogisticRegression()
        elif self.algorithm == "RandomForestClassifier":
            self.task_type = 'Classification'
            self.model = RandomForestClassifier()
        elif self.algorithm == "DecisionTreeClassifier":
            self.task_type = 'Classification'
            self.model = DecisionTreeClassifier()
        elif self.algorithm == "LinearRegression":
            self.task_type = 'Regression'
            self.model = LinearRegression()
        elif self.algorithm == "DecisionTreeRegressor":
            self.task_type = 'Regression'
            self.model = DecisionTreeRegressor()
        elif self.algorithm == "RandomForestRegressor":
            self.task_type = 'Regression'
            self.model = RandomForestRegressor()

        # setup the param_grip to use in GridSearchCV as a parameter
        self.create_param_grid()
        logger.info(f"Model Initialized: {self.model}")

    def train_and_evaluate(self, test_size: float = 0.2, random_state: int = 42, cv_folds: int = 5) -> Dict[str, Any]:
        logger.info(f"Test Size: {test_size}, Random State: {random_state}, CV Folds: {cv_folds}")
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=random_state
        )

        results = {}

        try:
            self.tune_hyperparameters(X_train, y_train, cv_folds)

            logger.info(f"Predicting values of the best model...")

            y_pred = self.model.predict(X_test)
        except Exception as e:
            logger.info(f"Something went wrong while predicting values")
            raise ValueError(f"Something went wrong while predicting values: {str(e)}")
                
        if self.task_type == 'Classification':
            cv_scores = cross_val_score(self.model, self.features, self.target, 
                                    cv=cv_folds, scoring='accuracy')
            
            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            if results['accuracy'] > self.best_score:
                self.best_score = results['accuracy']
                
        else:  # regression
            cv_scores = cross_val_score(self.model, self.features, self.target, 
                                        cv=cv_folds, scoring='r2')
            
            results = {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            if results['r2_score'] > self.best_score:
                self.best_score = results['r2_score']

        formatted_results = json.dumps({key: value for key, value in results.items() if key != 'classification_report'}, indent=4)
        logger.info(f"Results of {self.task_type} task:\n{formatted_results}")
        logger.info(f"Results of {self.task_type} task:\n{results}")
        return results

    def create_param_grid(self):
        """
        Return a parameter grid for the specified algorithm.
        """
        if self.algorithm == 'LogisticRegression':
            self.param_grid = {
                'C': np.logspace(-2, 1, 3).tolist(),
                'solver': ['liblinear', 'saga']
            }
        elif self.algorithm == 'DecisionTreeClassifier':
            self.param_grid = {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [None] + list(range(5, 35, 10)),  # [None, 5, 10, 20, 30]
                'min_samples_split': np.arange(2, 11, 3).tolist(),  # [2, 5, 10]
                'min_samples_leaf': np.array([1, 2, 5, 10]).tolist(),
                'max_features': [None, 'sqrt', 'log2'],
                'ccp_alpha': np.logspace(-2, -1, 3).tolist()  # [0.0, 0.01, 0.1]
            }
        elif self.algorithm == 'RandomForestClassifier':
            self.param_grid = {
                'n_estimators': np.arange(50, 250, 50).tolist(),  # [50, 100, 150, 200]
                'max_depth': [None] + list(range(10, 30, 10)),  # [None, 10, 20]
                'min_samples_split': np.array([2, 5]).tolist(),
                'min_samples_leaf': np.array([1, 2]).tolist(),
                'bootstrap': [True, False]
            }
        elif self.algorithm == 'LinearRegression':
            self.param_grid = {
                'fit_intercept': [True, False],
                'positive': [True, False]
            }
        elif self.algorithm == 'DecisionTreeRegressor':
            self.param_grid = {
                'criterion': ['squared_error', 'absolute_error', 'poisson'],
                'splitter': ['best', 'random'],
                'max_depth': [None] + list(range(5, 35, 10)),
                'min_samples_split': np.arange(2, 11, 3).tolist(),
                'min_samples_leaf': np.array([1, 2, 5, 10]).tolist(),
                'max_features': [None, 'sqrt', 'log2'],
                'ccp_alpha': np.logspace(-2, -1, 3).tolist(),
            }
        elif self.algorithm == 'RandomForestRegressor':
            self.param_grid = {
                'n_estimators': np.arange(50, 250, 50).tolist(),
                'max_depth': [None] + list(range(10, 30, 10)),
                'min_samples_split': np.array([2, 5]).tolist(),
                'min_samples_leaf': np.array([1, 2]).tolist(),
                'bootstrap': [True, False]
            }

    def get_outlier_summary(self) -> dict:
        """
        Get summary of outliers detected in numerical columns.
        """
        # Initialize outlier_info dictionary
        self.outlier_info = {}
        
        # Get numerical columns
        numerical_columns = self.features.select_dtypes(exclude=['object']).columns
        
        # Calculate outlier information for each numerical column
        for column in numerical_columns:
            Q1 = self.features[column].quantile(0.25)
            Q3 = self.features[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_below = sum(self.features[column] < lower_bound)
            outliers_above = sum(self.features[column] > upper_bound)
            
            self.outlier_info[column] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_below': outliers_below,
                'outliers_above': outliers_above,
                'total_outliers': outliers_below + outliers_above
            }
        
        return self.outlier_info

    def get_task_type(self) -> str:
        logger.info(f"Retrieving task_type...")
        # if task_type is None then throw an error
        if self.task_type is None:
            raise ValueError("Must call init() first")
        
        logger.info(f"Processed data successfully retrieved: {self.task_type}.")
        return self.task_type
    
    def get_processed_data(self) -> tuple:
        logger.info(f"Retrieving processed data...")
        # if features or target is None then throw an error
        if self.features is None or self.target is None:
            raise ValueError("Must call init() first")
        
        logger.info(f"Processed data successfully retrieved")
        return self.features, self.target

    def load_dataset(self, csv_path: str):
        logger.info(f"Loading dataset...")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")
        
        self.data = pd.read_csv(csv_path)
        logger.info(f"Dataset loaded successfully")
        
    def check_if_dataset_contains_target(self, target_name: str):
        logger.info(f"Checking target column...")
        # check if dataset is empty before checking target column
        self.is_dataset_empty()
        
        if target_name == '':
            logger.error(f"target_name cannot be empty or None.")
            raise ValueError(f"target_name cannot be empty or None.")

        # if target columns is not contained in the columns then throw an error
        if target_name not in self.data.columns:
            logger.error(f"Target column '{target_name}' not found in the dataset.")
            raise ValueError(f"Target column '{target_name}' not found in the dataset.")
        
        self.target = target_name
        logger.info(f"Target column named: {self.target}")

    def is_dataset_empty(self):
        """
        Check if the dataset is empty or not initialized.

        This method verifies whether the dataset (self.data) has been properly
        initialized and contains data. If the dataset is empty or None, it raises
        a ValueError.

        Raises:
            ValueError: If the dataset is None or empty.

        Returns:
            None

        Logs:
            Error: If the dataset is not initialized.
            Info: If the dataset is not empty.
        """
        # if dataset is empty then throw an error
        if self.data is None or self.data.empty:
            logger.error(f"The dataset has not been initialized yet.")
            raise ValueError(f"The dataset has not been initialized yet.")

        logger.info(f"The dataset is not empty")


    def tune_hyperparameters(self, X, y, cv):
        """
        Perform hyperparameter tuning using GridSearchCV.

        This function uses GridSearchCV to find the best hyperparameters for the model
        based on the provided parameter grid. It updates the model with the best estimator found.

        Parameters:
        X (array-like): The input samples for training.
        y (array-like): The target values for training.
        cv (int): The number of cross-validation folds to use in GridSearchCV.

        Raises:
        Exception: If an error occurs during the hyperparameter tuning process.

        Returns:
        None
        """
        try:
            logger.info("Starting hyperparameter tuning...")
            grid_search = GridSearchCV(self.model, self.param_grid, cv=cv)
            grid_search.fit(X, y)
            logger.info(f"Hyperparameter tuning completed. Best parameters: {grid_search.best_params_}")
            self.model = grid_search.best_estimator_
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise
    
    # Save Model
    def save_model(self, model_name):
        try:
            logger.info(f"Saving model...")
            joblib.dump(self.model, model_name+'.pkl')
            logger.info("Model saved to %s", model_name+'.pkl')
        except Exception as e:
            logger.error("Error saving model: %s", str(e))
            raise


