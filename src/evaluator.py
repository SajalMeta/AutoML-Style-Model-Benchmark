from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Using AnyBaseClassifier for type hinting would be ideal, but 'Any' suffices for simple portability
from sklearn.base import BaseEstimator

class Evaluator:
    """
    Handles training and evaluation of machine learning models.
    """
    
    def train_and_evaluate(
        self, 
        models: Dict[str, BaseEstimator], 
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Trains a dictionary of models and calculates performance metrics.

        Args:
            models (Dict[str, BaseEstimator]): Dictionary of model name -> model instance.
            X_train (np.ndarray): Training features.
            X_test (np.ndarray): Testing features.
            y_train (np.ndarray): Training labels.
            y_test (np.ndarray): Testing labels.

        Returns:
            pd.DataFrame: A DataFrame containing Accuracy, Precision, Recall, and F1 Score for each model.
        """
        results: List[Dict[str, Any]] = []

        for name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions on the test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            # zero_division=0 handles cases where a class is not predicted at all
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            })
            
        # Return results sorted by Accuracy for easier comparison
        return pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
