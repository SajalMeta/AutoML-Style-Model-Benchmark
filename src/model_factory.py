from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

class ModelFactory:
    """
    Factory class to create and configure machine learning models.
    """
    
    @staticmethod
    def get_models() -> Dict[str, Any]:
        """
        Returns a dictionary of initialized models with default or reasonable parameters.
        
        Returns:
            Dict[str, Any]: A dictionary where keys are model names and values are model instances.
        """
        # Dictionary of models to be trained
        # Random states are fixed for reproducibility
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss', 
                random_state=42,
                verbosity=0
            ),
            "MLP Neural Network": MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=500, 
                random_state=42
            )
        }
        return models
