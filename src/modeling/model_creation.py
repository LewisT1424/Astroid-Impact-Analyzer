import polars as pl
import pandas as pd
import xgboost as xgb
from src.utils.utils import logger, save_model
from typing import Dict, NotRequired
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.utils import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Model:
    def __init__(self, data: pl.DataFrame):
        self.data = data
        self.target_recall = 0.9
        self.best_params = {'objective': 'binary:logistic',
                            'n_estimators': 1200,
                            'learning_rate': 0.04,
                            'max_depth': 8,
                            'min_child_weight': 1, # Lower for more sensitivity
                            'subsample': 0.8,
                            'reg_alpha': 0,
                            'reg_lambda': 0.5, # Lower regularization
                            'scale_pos_weight': 25, # Heavier weighting for hazardous class
                            'random_state': 42,
                            'early_stopping_rounds': 100,
                            'eval_metric': 'logloss'
                            }
        
    def optimise_threshold_for_recall(self, model, X_val, y_val):
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        best_threshold = 0.5
        best_f1 = 0

        for threshold in np.arange(0.05, 0.8, 0.02):
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)

            current_recall = recall_score(y_val, y_pred_thresh)
            current_f1 = f1_score(y_val, y_pred_thresh)

            # Only consider thresholds that meet recall requirement
            if current_recall >= self.target_recall and current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        logger.info(f"Optimal threshold for {self.target_recall:.1%} recall {best_threshold:.3f}")
        return best_threshold
    
    def build_model(self, X_train, X_test, X_val, y_train, y_test, y_val):
        model = xgb.XGBClassifier(**self.best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )

        optimal_threshold = self.optimise_threshold_for_recall(
            model, X_val, y_val
        )

        results = model.evals_result()

        # Plot training vs validation
        plt.figure( figsize=(15,5))
        # Loss curves
        plt.plot(results['validation_0']['logloss'], label='Training Loss', alpha=0.8)
        plt.plot(results['validation_1']['logloss'], label='Validation Loss', alpha=0.8)
        plt.title('Model Loss During Training')
        plt.xlabel('Boosting Rounds')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.grid(True, alpha=0.3);

        plt.tight_layout();


        test_pred_proba = model.predict_proba(X_test)[:, 1]
        test_pred_optimal = (test_pred_proba >= optimal_threshold).astype(int)
        test_pred_default = model.predict(X_test)

        metrics_default = {
            'threshold': 0.5,
            'f1': f1_score(y_test, test_pred_default),
            'roc_auc': roc_auc_score(y_test, test_pred_default),
            'precision': precision_score(y_test, test_pred_default),
            'recall': recall_score(y_test, test_pred_default),
            'accuracy': accuracy_score(y_test, test_pred_default)
        }

        metrics_optimal = {
            'threshold': optimal_threshold,
            'f1': f1_score(y_test, test_pred_optimal),
            'roc_auc': roc_auc_score(y_test, test_pred_proba),
            'precision': precision_score(y_test, test_pred_optimal),
            'recall': recall_score(y_test, test_pred_optimal),
            'accuracy': accuracy_score(y_test, test_pred_optimal)
        }

        # Log both results
        logger.info("=== DEFAULT THRESHOLD (0.5) ===")
        for key, value in metrics_default.items():
            if key != 'threshold':
                logger.info(f"{key.capitalize()}: {value:.4f}")
        
        logger.info("=== OPTIMIZED THRESHOLD ===")
        for key, value in metrics_optimal.items():
            logger.info(f"{key.capitalize()}: {value:.4f}")


        # Mission readiness assessment
        optimal_recall = metrics_optimal['recall']
        if optimal_recall >= 0.95:
            logger.info("🟢 MISSION READY: 95%+ recall achieved!")
        elif optimal_recall >= 0.90:
            logger.info("🟡 DEPLOYMENT READY: 90%+ recall achieved")
        else:
            logger.info("🔴 NEEDS IMPROVEMENT: <90% recall")
        
        # Store optimal threshold in model for future use
        model.optimal_threshold = optimal_threshold
        
        return metrics_optimal, model

    def train_test_split(self, strategy):
        df_pandas = self.data.to_pandas()

        # Split data into X and y 
        X = df_pandas.drop(['is_potentially_hazardous'], axis=1)
        y = df_pandas['is_potentially_hazardous'].astype(int)

        # Create train and test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, stratify=y, random_state=42, test_size=0.4
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        if strategy == 'undersample':
            train_df = pd.concat([X_train, y_train], axis=1)
            majority =  train_df[train_df['is_potentially_hazardous'] == 0]
            minority =  train_df[train_df['is_potentially_hazardous'] == 1]
        
            majority_undersampled = resample(
                majority, replace=False, n_samples=len(minority), random_state=42
            )

            train_balanced = pd.concat([majority_undersampled, minority])
            X_train = train_balanced.drop(['is_potentially_hazardous'], axis=1)
            y_train = train_balanced['is_potentially_hazardous']
        elif strategy == 'smote':
            from imblearn.over_sampling import SMOTE 
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        return X_train, X_test, X_val, y_train, y_test, y_val



    def run_model_creation_pipeline(self):     
        # Get datasets for trainined
        X_train, X_test, X_val, y_train, y_test, y_val = self.train_test_split(strategy='undersample')
        # Build model
        model_metrics, model = self.build_model(X_train, X_test, X_val, y_train, y_test, y_val)

        # Display model metrics
        for metric, score in model_metrics.items():
            logger.info(f"{metric.capitalize()}: {score:.3f}")
        
        # Return model
        return model