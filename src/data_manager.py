from typing import Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
import seaborn as sns

class DataManager:
    """
    Handles data loading, preprocessing, and splitting for machine learning tasks.
    """
    def __init__(self):
        """Initialize the data manager with a scaler and label encoder."""
        self.scaler = StandardScaler()
        # We will use separate encoders for features if needed, simple approach here
        self.label_encoder = LabelEncoder()

    def load_data(self, file: Optional[Any] = None, sample_dataset: str = 'iris') -> pd.DataFrame:
        """
        Loads data from an uploaded CSV file or a sample dataset.

        Args:
            file: File-like object (CSV) uploaded by user.
            sample_dataset (str): Name of the sample dataset. 
                                  Options: 'iris', 'wine', 'breast_cancer', 'digits', 'titanic', 'penguins'.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if file is not None:
            try:
                df = pd.read_csv(file)
                return df
            except Exception as e:
                raise ValueError(f"Error reading CSV file: {e}")
        
        # Load sample dataset if no file provided
        if sample_dataset == 'iris':
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df
        elif sample_dataset == 'wine':
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df
        elif sample_dataset == 'breast_cancer':
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df
        elif sample_dataset == 'digits':
            data = load_digits()
            df = pd.DataFrame(data.data, columns=[f"pixel_{i}" for i in range(data.data.shape[1])])
            df['target'] = data.target
            return df
        elif sample_dataset == 'titanic':
            # Seaborn datasets come as DataFrames appropriately
            # Drop rows with minimal info if needed, but we'll handle in preprocess
            df = sns.load_dataset('titanic')
            # Titanic has 'survived' as target. 'alive' is duplicate.
            # Select subset of useful columns to keep it simple for this demo
            keep_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'adult_male', 'survived']
            return df[keep_cols]
        elif sample_dataset == 'penguins':
            df = sns.load_dataset('penguins')
            return df.dropna() # Penguins has very few NaNs, safe to drop for demo
            
        else:
            # Fallback
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df

    def preprocess_data(
        self, 
        df: pd.DataFrame, 
        target_column: str = 'target', 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocesses the data: handles missing values, encodes target, scales features, and splits.
        Automatically handles categorical feature columns by LabelEncoding them.

        Args:
            df (pd.DataFrame): The raw dataframe.
            target_column (str): The name of the target variable column.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed for reproducibility.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
        """
        # Validate target column
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in dataset.")

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 1. Handle Missing Values
        # Numeric: Fill with mean
        # Categorical: Fill with "Missing" or mode
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Missing")

        # 2. Encode Feature Columns (if categorical)
        # We use simple LabelEncoding for all object/category columns for this baseline
        # In a real app, OneHotEncoding might be better for nominal data, but LE is robust for tree models (RF/XGB)
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # 3. Encode Target
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = self.label_encoder.fit_transform(y)

        # 4. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 5. Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
