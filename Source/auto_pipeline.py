import pandas as pd 
import numpy as np 
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

class auto_pipeline(BaseEstimator):
    def __init__(self, df: pd.DataFrame, target: str, model= None):
        self.df = df.copy()
        self.target = target
        self.model_algarithm = model
        self.model = None
        self.preprocessor = None 

    def _prepare_features(self):
        x = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        
        num_col = x.select_dtypes(include=[np.number]).columns.tolist()
        cat_col = x.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, num_col),
            ('cat', categorical_pipeline, cat_col)
        ])

        return x, y
    
    def fit(self):
        x, y = self._prepare_features()

        if self.model_algarithm is None:
            raise ValueError("Model mos emas, Algaritmni tekshiring")
        
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('estimator', self.model_algarithm)
        ])

        self.model.fit(x,y)
        return self 
    

    def predict(self, x=None):
        if self.model is None:
            raise ValueError("Model hali train qilinmagan, Avval Train qiling")
        if x is None:
            x = self.df.drop(columns=[self.target])
        return self.model.predict(x)
    
    def score(self, x=None, y=None):
        if self.model is None:
            raise ValueError("Model hali train qilinmagan, Avval Train qiling")
        if x is None and y is None:
            x = self.df.drop(columns=[self.target])
            y = self.df[self.target]
        return self.model.score(x, y)
    
    

    