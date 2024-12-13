from typing import Optional, Union, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
import joblib

class AutoML:
    def __init__(self):
        self.data = None
        self.target = None
        self.task_type = None
        self.features = None
        self.pipeline = None
        self.label_encoder = None
        self.algorithm = None
        self.models = {}
        self.best_model = None
        self.best_score = float('-inf')
        self.outlier_info = {}

    def init(self, csv_path: str, target_name: str, algorithm) -> None:
        #Set algorithm
        self.algorithm = algorithm
   
        # Load and prepare the data
        self.data = pd.read_csv(csv_path)
        if target_name not in self.data.columns:
            raise ValueError(f"Target column '{target_name}' not found in dataset")
        
        # Split features and target
        self.target = self.data[target_name]
        self.features = self.data.drop(target_name, axis=1)
        
        # Determine task type
        self._determine_task_type(algorithm)
        
        # Clean and preprocess the data
        self._preprocess_data()

        # Initialize models based on task type
        self.model = self._initialize_models(self.algorithm)

        # Maybe redundant
        self.label_encoder = LabelEncoder()
        self.target = pd.Series(self.label_encoder.fit_transform(self.target))


    def _determine_task_type(self, algorithm) -> None:
        """Automatically determine if this is a classification or regression task."""

        # Determine class type based on algorithm passed through
        if self.algorithm == "LinearRegression" or "DecisionTreeRegressor" or "RandomForestRegressor":
            self.task_type = 'Regression'
        elif self.algorithm == "LogisticRegression" or "DecisionTreeClassifier" or "RandomForestClassifier":
            self.task_type = 'Classification'
        else:
            raise ValueError(f"algorithm '{self.algorithm}' not found in types")
        return self.task_type

    def _preprocess_data(self) -> None:
        categorical_columns = self.features.select_dtypes(include=['object']).columns
        numerical_columns = self.features.select_dtypes(exclude=['object']).columns
        
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', LabelEncoder())
        ])
        
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

    def _initialize_models(self, algorithm: str):
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
            return LogisticRegression(max_iter=1000)
        elif algorithm == "RandomForestClassifier":
            return RandomForestClassifier(random_state=42)
        elif algorithm == "DecisionTreeClassifier":
            return DecisionTreeClassifier(random_state=42, n_estimators=100)
        elif algorithm == "LinearRegression":
            return LinearRegression()
        elif algorithm == "DecisionTreeRegressor":
            return DecisionTreeRegressor(random_state=42)
        elif algorithm == "RandomForestRegressor":
            return RandomForestRegressor(random_state=42, n_estimators=100)
        else:
            self.logger.error(f"Unsupported algorithm: {algorithm}.")
            raise ValueError(f"Unsupported algorithm: {algorithm}")


    def train_and_evaluate(self, test_size: float = 0.2, cv_folds: int = 5) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=42
        )
        
        self.model

        results = {}
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
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
    
        return results



    def create_param_grid(algorithm):
        """
        Return a parameter grid for the specified algorithm.
        """
        if algorithm == 'LogisticRegression':
            return {
                'model__C': [0.1, 1, 10],
                'model__solver': ['liblinear', 'saga']
            }
        elif algorithm == 'DecisionTreeClassifier':
            return {
                'model__criterion': ['gini', 'entropy'],
                'model__splitter': ['best', 'random'],
                'model__max_depth': [None, 5, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 5, 10],
                'model__max_features': [None, 'sqrt', 'log2'],
                'model__ccp_alpha': [0.0, 0.01, 0.1],
            }
        elif algorithm == 'RandomForestClassifier':
            return {
                'model__n_estimators': [50, 100],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2]
            }
        elif algorithm == 'LinearRegression':
            return {
                'model__fit_intercept': [True, False],
                'model__normalize': [True, False]
            }
        elif algorithm == 'DecisionTreeRegressor':
            return {
                'model__criterion': ['squared_error', 'absolute_error', 'poisson'],
                'model__splitter': ['best', 'random'],
                'model__max_depth': [None, 5, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 5, 10],
                'model__max_features': [None, 'sqrt', 'log2'],
                'model__ccp_alpha': [0.0, 0.01, 0.1],
            }
        elif algorithm == 'RandomForestRegressor':
            return {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2],
                'model__bootstrap': [True, False]
            }
        else:
            # self.logger.error("Unsupported algorithm: %s", algorithm)
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        



    def get_outlier_summary(self) -> dict:
        """
        Get summary of outliers detected in numerical columns.
        
        Returns:
            dict: Dictionary containing outlier information for each numerical column
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
        if self.task_type is None:
            raise ValueError("Must call init() first")
        return self.task_type
    

    def get_processed_data(self) -> tuple:
        if self.features is None or self.target is None:
            raise ValueError("Must call init() first")
        return self.features, self.target


