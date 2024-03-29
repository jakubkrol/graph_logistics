import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVMClassifier:
    def __init__(self, mode='Drones', svm=None, param_grid=None, mlflow_config=None):
        self.mode = mode
        self.svm = svm if svm else SVC()
        self.param_grid = param_grid if param_grid else {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf', 'linear']
        }
        self.mlflow_config = mlflow_config if mlflow_config else {'enabled': False}

    def preprocess_data(self, data):
        X = data[['MedDist', 'MeanHosp', 'Cluster', 'Centre']]
        y = data[f'Bool{self.mode}']
        return train_test_split(X, y, test_size=0.15, random_state=42)

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        cv = GridSearchCV(self.svm, self.param_grid, cv=5)
        if self.mlflow_config.get('enabled', False):
            mlflow.set_experiment(f'SVM_Classification_{self.mode}')
        cv.fit(X_train, y_train)
        if self.mlflow_config.get('enabled', False):
            # Add MLflow logging here
            mlflow.log_params(cv.best_params_)
            mlflow.sklearn.log_model(cv.best_estimator_, "best_svm_model")
            mlflow.log_metric("accuracy", cv.score(X_test, y_test))
            # mlflow.log_text(report, "classification_report.txt")
        self.model = cv.best_estimator_
        self.evaluate_model(X_test, y_test)

    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        print(classification_report(y_test, predictions))
        self._plot_confusion_matrix(y_test, predictions)

    def _plot_confusion_matrix(self, y_test, predictions):
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    def calculate_SVM_classifier(self, data):
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        self.train_and_evaluate(X_train, y_train, X_test, y_test)


class RandomForestClassifierWithImportance:
    def __init__(self, mode='Drones', rf=None, param_grid=None, mlflow_config=None):
        self.mode = mode
        self.rf = rf if rf else RandomForestClassifier()
        self.param_grid = param_grid if param_grid else {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [False]
        }
        self.mlflow_config = mlflow_config if mlflow_config else {'enabled': True, 'experiment_name': f'RF_Classification_{self.mode}'}

    def preprocess_data(self, data):
        X = data[['MedDist', 'MeanHosp', 'Cluster', 'Centre']]
        y = data[f'Bool{self.mode}']
        return train_test_split(X, y, test_size=0.15, random_state=42)

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        if self.mlflow_config.get('enabled', False):
            mlflow.set_experiment(self.mlflow_config.get('experiment_name', f'RF_Classification_{self.mode}'))
            
        with mlflow.start_run():
            cv = GridSearchCV(self.rf, self.param_grid, cv=5)
            cv.fit(X_train, y_train)
            
            if self.mlflow_config.get('enabled', False):
                mlflow.log_params(cv.best_params_)
                mlflow.log_metric("best_score", cv.best_score_)
                mlflow.sklearn.log_model(cv.best_estimator_, "model")
                
            self.model = cv.best_estimator_
            self.evaluate_model(X_test, y_test)
            self.display_feature_importance(X_train.columns)

             # Save the best model to a .pkl file
            best_model_filename = f"best_rf_model_{self.mode}.pkl"
            joblib.dump(self.model, best_model_filename)
            print(f"Best model saved as {best_model_filename}")

    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        report = classification_report(y_test, predictions)
        print(report)

        # Save the classification report to a text file
        report_file = "classification_report.txt"
        with open(report_file, "w") as f:
            f.write(report)

        # Check if MLflow logging is enabled, then log the report file as an artifact
        if self.mlflow_config.get('enabled', False):
            mlflow.log_artifact(report_file)
        
        self._plot_confusion_matrix(y_test, predictions)

    def _plot_confusion_matrix(self, y_test, predictions):
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    def display_feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig("feature_importances.png")
        plt.show()
        
        if self.mlflow_config.get('enabled', False):
            mlflow.log_artifact("feature_importances.png")

    def calculate_RF_classifier(self, data):
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        self.train_and_evaluate(X_train, y_train, X_test, y_test)

class GaussianProcessRegressorModel:
    def __init__(self, mode='NumericTarget', gpr=None, mlflow_config=None):
        self.mode = mode
        self.gpr = gpr if gpr else GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)))
        self.mlflow_config = mlflow_config if mlflow_config else {'enabled': True, 'experiment_name': f'GPR_{self.mode}'}
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.predictions = None
        self.y_test_original = None

    def preprocess_data(self, data):
        X = data[['MedDist', 'MeanHosp', 'Cluster', 'Centre']]
        y = data[self.mode].values.reshape(-1, 1)  # Reshape y to 2D array for scaler

        # Normalize X
        X_scaled = self.x_scaler.fit_transform(X)

        # Normalize y
        y_scaled = self.y_scaler.fit_transform(y)

        return train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=42)

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        self.y_test_original = self.y_scaler.inverse_transform(y_test)  # Inverse transform for original scale plotting
        
        if self.mlflow_config.get('enabled', False):
            mlflow.set_experiment(self.mlflow_config.get('experiment_name', f'GPR_{self.mode}'))

        with mlflow.start_run():
            self.gpr.fit(X_train, y_train.ravel())  # Flatten y_train for fitting
            
            if self.mlflow_config.get('enabled', False):
                mlflow.log_params({"kernel": str(self.gpr.kernel_)})
            
            predictions_scaled = self.gpr.predict(X_test)
            self.predictions = self.y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))  # Inverse transform predictions

            mse = mean_squared_error(self.y_test_original, self.predictions)
            r2 = r2_score(self.y_test_original, self.predictions)
            
            print(f'MSE: {mse}, R²: {r2}')
            
            if self.mlflow_config.get('enabled', False):
                mlflow.log_metrics({"MSE": mse, "R2": r2})
                
            self.plot_predictions()
        
        # Save the best model to a .pkl file
        best_model_filename = f"best_gpr_model_{self.mode}.pkl"
        joblib.dump(self.gpr, best_model_filename)
        print(f"Best model saved as {best_model_filename}")

    def plot_predictions(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test_original, self.predictions, edgecolor='k', label='Predictions vs. True Values')
        plt.plot([self.y_test_original.min(), self.y_test_original.max()], 
                 [self.y_test_original.min(), self.y_test_original.max()], 'r--', lw=2, label='Identity Line')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs. True Values')
        plt.legend()
        plt.show()

    def calculate_GPR(self, data):
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        self.train_and_evaluate(X_train, y_train, X_test, y_test)


class XGBoostRegressorModel:
    def __init__(self, mode='NumericTarget', mlflow_config=None):
        self.mode = mode
        self.model = xgb.XGBRegressor(objective ='reg:squarederror')
        self.mlflow_config = mlflow_config if mlflow_config else {'enabled': True, 'experiment_name': f'XGB_{self.mode}'}
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.predictions = None
        self.y_test_original = None

    def preprocess_data(self, data):
        X = data[['MedDist', 'MeanHosp', 'Cluster', 'Centre']]
        y = data[self.mode].values.reshape(-1, 1)  # Reshape y to 2D array for scaler

        # Normalize X
        X_scaled = self.x_scaler.fit_transform(X)

        # Normalize y
        y_scaled = self.y_scaler.fit_transform(y).ravel()  # Flatten for XGBoost

        return train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=42)

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        self.y_test_original = self.y_scaler.inverse_transform(y_test.reshape(-1, 1))  # Inverse transform for original scale
        
        if self.mlflow_config.get('enabled', False):
            mlflow.set_experiment(self.mlflow_config.get('experiment_name', f'XGB_{self.mode}'))

        with mlflow.start_run():
            self.model.fit(X_train, y_train)
            
            if self.mlflow_config.get('enabled', False):
                # Log model parameters
                mlflow.log_params(self.model.get_params())
            
            predictions_scaled = self.model.predict(X_test)
            self.predictions = self.y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

            mse = mean_squared_error(self.y_test_original, self.predictions)
            r2 = r2_score(self.y_test_original, self.predictions)
            
            print(f'MSE: {mse}, R²: {r2}')
            
            if self.mlflow_config.get('enabled', False):
                mlflow.log_metrics({"MSE": mse, "R2": r2})
                
            self.plot_predictions()

    def plot_predictions(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test_original, self.predictions, edgecolor='k', label='Predictions vs. True Values')
        plt.plot([self.y_test_original.min(), self.y_test_original.max()], 
                 [self.y_test_original.min(), self.y_test_original.max()], 'r--', lw=2, label='Identity Line')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs. True Values')
        plt.legend()
        plt.show()

    def calculate_XGB(self, data):
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        self.train_and_evaluate(X_train, y_train, X_test, y_test)
