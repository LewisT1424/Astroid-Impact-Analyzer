import polars as pl
from src.utils.utils import logger, save_model
from typing import Dict, NotRequired
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self, data: pl.DataFrame):
        self.data = data
        self.best_params = {'objective': 'binary:logistic',
                            'base_score': None,
                            'booster': None,
                            'callbacks': None,
                            'colsample_bylevel': 1.0,
                            'colsample_bynode': None,
                            'colsample_bytree': 0.8,
                            'device': None,
                            'early_stopping_rounds': None,
                            'enable_categorical': False,
                            'eval_metric': None,
                            'feature_types': None,
                            'feature_weights': None,
                            'gamma': 0,
                            'grow_policy': None,
                            'importance_type': None,
                            'interaction_constraints': None,
                            'learning_rate': 0.15,
                            'max_bin': None,
                            'max_cat_threshold': None,
                            'max_cat_to_onehot': None,
                            'max_delta_step': None,
                            'max_depth': 7,
                            'max_leaves': None,
                            'min_child_weight': 2,
                            'missing': nan,
                            'monotone_constraints': None,
                            'multi_strategy': None,
                            'n_estimators': 1200,
                            'n_jobs': None,
                            'num_parallel_tree': None,
                            'random_state': 42,
                            'reg_alpha': 0,
                            'reg_lambda': 1,
                            'sampling_method': None,
                            'scale_pos_weight': None,
                            'subsample': 0.9,
                            'tree_method': None,
                            'validate_parameters': None,
                            'verbosity': None}
    
    def build_model(self, X_train, X_test, X_val, y_train, y_test, y_val):
        model = xgb.Classifier(**self.best_params)
        model.fit(
            X_train, y_train,
            eval_set[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )

        results = model.eval_results()

        # Plot training vs validation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        # Loss curves
        ax1.plot(results['validation_0']['logloss'], label='Training Loss', alpha=0.8)
        ax1.plot(results['validation_1']['logloss'], label='Validation Loss', alpha=0.8)
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Boosting Rounds')
        ax1.set_ylabel('Log Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add early stoppping line
        if hasattr(model, 'best_iteration'):
            ax1.axvline(x=model.best_iteration, color='red', linestyle='--', label=f'Early Stop ({model.best_iteration})')
            ax1.legend()

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        test_pred_proba = model.predict_proba(X_test)[:, 1]

        scores = {
            'Training': f1_score(y_train, train_pred),
            'Validation': f1_score(y_val, val_pred),
            'Test': f1_score(y_test, test_pred)
        }

        ax2.bar(scores.keys(), scores.values(), alpha=0.7)
        ax2.set_title("F1 Score Comparison")
        ax2.set_ylabel("F1 Score")
        ax2.set_ylim(0, 1)

        # Add value labels on bars
        for i, (k, v) in enumerate(scores.items()):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')

        plt.tight_layout()
        plt.show()

        metrics = {
            'model': model_name,
            'f1': f1_score(y_test, test_pred),
            'roc_auc': roc_auc_score(y_test, test_pred_proba),
            'precision': average_precision_score(y_test, test_pred),
            'recall': recall_score(y_test, test_pred),
            'accuracy': accuracy_score(y_test, test_pred)
        }
        print(model_name)
        return metrics, model

    def get_balanced_train_test(self):
        df_pandas = self.data.to_pandas()

        majority = df_pandas[df_pandas['is_potentially_hazardous'] == 0]
        minority = df_pandas[df_pandas['is_potentially_hazardous'] == 1]

        minority_undersampled = resample(
            minority, 
            replace=True,
            n_samples=len(majority),
            random_state=42
        )

        df_balanced = pd.concat([majority, minority_undersampled])
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

        print(f"Balanced dataset size: {len(df_balanced)}")
        print(f"Balanced class distribution:")
        print(df_balanced['is_potentially_hazardous'].value_counts())

        # Now use this balanced dataset
        X_balanced = df_balanced.drop(['is_potentially_hazardous'], axis=1)
        y_balanced = df_balanced['is_potentially_hazardous']

        y_balanced = y_balanced.astype(int)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X_balanced, y_balanced, stratify=y_balanced, random_state=42, test_size=0.4
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        return X_train, X_test, X_val, y_train, y_test, y_val



    def run_model_creation_pipeline(self):     
        # Get datasets for trainined
        X_train, X_test, X_val, y_train, y_test, y_val = self.get_balanced_train_test()
        # Build model
        model_metrics, model = self.build_model(X_train, X_test, X_val, y_train, y_test, y_val)
        # Display performance to user
        for key, item in model_metrics
            logger.info(f"{key.capitalize()}: {item}")
        # Return model to user
        return model
