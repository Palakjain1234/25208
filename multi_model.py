# """
# multi_model_pipeline.py

# Multi-Model LightGBM pipeline for:
# - selected_for_rake_label   (binary classification)
# - choose_rail_label         (binary classification)
# - on_time_label             (binary classification)
# - rail_cost_total / road_cost_total (regression)

# Uses shared preprocessing and label-encoders, saves models and encoders using joblib.
# """

# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import (
#     accuracy_score, f1_score, roc_auc_score,
#     mean_squared_error, mean_absolute_error
# )
# import os
# from datetime import datetime


# class MultiModelPipeline:
#     """Multi-model pipeline for training and predicting on multiple targets."""
    
#     def __init__(self, csv_path, model_dir, random_state=42, test_size=0.2):
#         self.csv_path = csv_path
#         self.model_dir = model_dir
#         self.random_state = random_state
#         self.test_size = test_size
        
#         # Create model directory
#         os.makedirs(self.model_dir, exist_ok=True)
        
#         # Configuration
#         self.targets = {
#             'selected_for_rake': 'selected_for_rake_label',
#             'choose_rail': 'choose_rail_label',
#             'on_time': 'on_time_label',
#             'rail_cost': 'rail_cost_total',
#             'road_cost': 'road_cost_total'
#         }
        
#         self.feature_cols = None
#         self.categorical_cols = None
#         self.label_encoders = None
        
#         # Model parameters
#         self.params_reg = {
#             'objective': 'regression',
#             'metric': 'rmse',
#             'boosting_type': 'gbdt',
#             'learning_rate': 0.05,
#             'num_leaves': 31,
#             'feature_fraction': 0.9,
#             'bagging_fraction': 0.8,
#             'bagging_freq': 5,
#             'min_data_in_leaf': 20,
#             'verbose': -1,
#             'seed': self.random_state
#         }

#         self.params_clf = {
#             'objective': 'binary',
#             'metric': 'binary_logloss',
#             'boosting_type': 'gbdt',
#             'learning_rate': 0.05,
#             'num_leaves': 31,
#             'feature_fraction': 0.9,
#             'bagging_fraction': 0.8,
#             'bagging_freq': 5,
#             'min_data_in_leaf': 20,
#             'verbose': -1,
#             'seed': self.random_state
#         }

#     def load_and_preprocess_data(self):
#         """Load and preprocess the dataset."""
#         print("Loading data...")
#         df = pd.read_csv(self.csv_path)
#         print(f"Loaded {len(df)} records")
        
#         # Date feature engineering
#         df = self._engineer_date_features(df)
        
#         # Prepare features and targets
#         df = self._prepare_features(df)
        
#         # Handle missing values
#         df = self._handle_missing_values(df)
        
#         # Encode categorical variables
#         df = self._encode_categorical_features(df)
        
#         return df

#     def _engineer_date_features(self, df):
#         """Create temporal features from date columns."""
#         if 'plan_date' in df.columns:
#             try:
#                 df['plan_date'] = pd.to_datetime(df['plan_date'], dayfirst=True)
#                 df['plan_dayofweek'] = df['plan_date'].dt.dayofweek
#                 df['plan_month'] = df['plan_date'].dt.month
#                 df['plan_day'] = df['plan_date'].dt.day
#             except Exception as e:
#                 print(f"Date processing warning: {e}")
#         return df

#     def _prepare_features(self, df):
#         """Identify feature and target columns."""
#         # Drop identifier columns
#         drop_cols = []
#         for col in ['plan_id', 'plan_date', 'plan_id.1']:
#             if col in df.columns:
#                 drop_cols.append(col)
        
#         # Build feature list
#         self.feature_cols = [c for c in df.columns 
#                            if c not in list(self.targets.values()) + drop_cols]
        
#         print(f"Using {len(self.feature_cols)} feature columns")
#         return df

#     def _handle_missing_values(self, df):
#         """Handle missing values in feature columns."""
#         for col in self.feature_cols:
#             if df[col].dtype == 'object':
#                 df[col] = df[col].fillna('Unknown')
#             else:
#                 df[col] = df[col].fillna(df[col].median())
#         return df

#     def _encode_categorical_features(self, df):
#         """Encode categorical features using label encoding."""
#         # Identify categorical columns
#         self.categorical_cols = self._identify_categorical_columns(df)
#         print(f"Detected {len(self.categorical_cols)} categorical columns")
        
#         # Apply label encoding
#         self.label_encoders = {}
#         for col in self.categorical_cols:
#             le = LabelEncoder()
#             df[col] = df[col].astype(str)
#             try:
#                 df[col] = le.fit_transform(df[col])
#                 self.label_encoders[col] = le
#             except Exception:
#                 # Fallback to manual mapping
#                 vals = df[col].unique().tolist()
#                 mapping = {v: i for i, v in enumerate(vals)}
#                 df[col] = df[col].map(mapping)
#                 self.label_encoders[col] = mapping
        
#         # Save encoders
#         joblib.dump(self.label_encoders, 
#                    os.path.join(self.model_dir, "label_encoders.pkl"))
#         print("Saved label encoders")
        
#         return df

#     def _identify_categorical_columns(self, df):
#         """Identify categorical columns in the dataset."""
#         categorical_cols = [c for c in self.feature_cols if df[c].dtype == 'object']
        
#         # Add low-cardinality numeric columns
#         for col in self.feature_cols:
#             if (df[col].dtype in [np.int64, np.int32] or 
#                 (pd.api.types.is_float_dtype(df[col]) and df[col].nunique() < 20)):
#                 if df[col].nunique() < 50 and col not in categorical_cols:
#                     categorical_cols.append(col)
        
#         return sorted(list(set(categorical_cols)))

#     def _prepare_targets(self, df):
#         """Prepare target variables for training."""
#         targets = {}
        
#         for target_name, target_col in self.targets.items():
#             if target_col not in df.columns:
#                 print(f"Target '{target_col}' not found; skipping {target_name}")
#                 targets[target_name] = None
#                 continue
                
#             y = df[target_col].copy()
            
#             # Convert binary string targets to numeric
#             if y.dtype == 'object' and target_name in ['selected_for_rake', 'choose_rail', 'on_time']:
#                 y = y.map({'Yes': 1, 'No': 0}).astype(float)
#                 if y.isna().any():
#                     le_tmp = LabelEncoder()
#                     y = le_tmp.fit_transform(df[target_col])
            
#             targets[target_name] = y
        
#         return targets

#     def train_regression_model(self, X, y, model_name):
#         """Train a regression model."""
#         notnull_idx = y.notna()
#         X_train_full = X.loc[notnull_idx, :]
#         y_train_full = y.loc[notnull_idx]

#         Xtr, Xval, ytr, yval = train_test_split(
#             X_train_full, y_train_full, 
#             test_size=self.test_size, 
#             random_state=self.random_state
#         )

#         train_data = lgb.Dataset(Xtr, label=ytr, 
#                                categorical_feature=self.categorical_cols, 
#                                free_raw_data=False)
#         val_data = lgb.Dataset(Xval, label=yval, 
#                              categorical_feature=self.categorical_cols, 
#                              reference=train_data, free_raw_data=False)

#         model = lgb.train(
#             self.params_reg, 
#             train_data, 
#             valid_sets=[val_data],
#             num_boost_round=1000, 
#             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
#         )

#         # Evaluate model
#         preds = model.predict(Xval, num_iteration=model.best_iteration)
#         rmse = np.sqrt(mean_squared_error(yval, preds))
#         mae = mean_absolute_error(yval, preds)
#         print(f"[{model_name}] RMSE: {rmse:.2f}, MAE: {mae:.2f}")

#         # Save model
#         joblib.dump(model, os.path.join(self.model_dir, f"{model_name}.pkl"))
#         print(f"Saved {model_name}.pkl")
        
#         return model

#     def train_classification_model(self, X, y, model_name):
#         """Train a classification model."""
#         notnull_idx = y.notna()
#         X_train_full = X.loc[notnull_idx, :]
#         y_train_full = y.loc[notnull_idx]

#         Xtr, Xval, ytr, yval = train_test_split(
#             X_train_full, y_train_full, 
#             test_size=self.test_size, 
#             random_state=self.random_state, 
#             stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None
#         )

#         train_data = lgb.Dataset(Xtr, label=ytr, 
#                                categorical_feature=self.categorical_cols, 
#                                free_raw_data=False)
#         val_data = lgb.Dataset(Xval, label=yval, 
#                              categorical_feature=self.categorical_cols, 
#                              reference=train_data, free_raw_data=False)

#         model = lgb.train(
#             self.params_clf, 
#             train_data, 
#             valid_sets=[val_data],
#             num_boost_round=1000, 
#             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
#         )

#         # Evaluate model
#         preds_prob = model.predict(Xval, num_iteration=model.best_iteration)
#         preds = (preds_prob > 0.5).astype(int)

#         acc = accuracy_score(yval, preds)
#         f1 = f1_score(yval, preds, zero_division=0)
#         try:
#             auc = roc_auc_score(yval, preds_prob)
#         except Exception:
#             auc = None

#         print(f"[{model_name}] Accuracy: {acc:.3f}, F1: {f1:.3f}, AUC: {auc if auc is not None else 'NA'}")

#         joblib.dump(model, os.path.join(self.model_dir, f"{model_name}.pkl"))
#         print(f"Saved {model_name}.pkl")
        
#         return model

#     def train_all_models(self):
#         """Train all models for the specified targets."""
#         # Load and preprocess data
#         df = self.load_and_preprocess_data()
        
#         # Prepare features and targets
#         X = df[self.feature_cols].copy()
#         targets = self._prepare_targets(df)
        
#         models = {}
        
#         # Train classification models
#         classification_targets = ['selected_for_rake', 'choose_rail', 'on_time']
#         for target_name in classification_targets:
#             if targets[target_name] is not None:
#                 print(f"\nTraining {target_name} model...")
#                 models[target_name] = self.train_classification_model(
#                     X, targets[target_name], f"model_{target_name}"
#                 )
        
#         # Train regression models
#         regression_targets = ['rail_cost', 'road_cost']
#         for target_name in regression_targets:
#             if targets[target_name] is not None:
#                 print(f"\nTraining {target_name} model...")
#                 models[target_name] = self.train_regression_model(
#                     X, targets[target_name], f"model_{target_name}"
#                 )
        
#         print("\nTraining completed!")
#         return models

#     def predict_all(self, new_df):
#         """Make predictions using all trained models."""
#         # Load encoders
#         encoders = joblib.load(os.path.join(self.model_dir, "label_encoders.pkl"))
#         nd = new_df.copy()
        
#         # Validate input features
#         missing = [c for c in self.feature_cols if c not in nd.columns]
#         if missing:
#             raise ValueError(f"Missing feature columns in input: {missing}")
        
#         # Preprocess input data
#         nd = self._preprocess_prediction_data(nd, encoders)
        
#         # Load models and make predictions
#         return self._make_predictions(nd)

#     def _preprocess_prediction_data(self, df, encoders):
#         """Preprocess data for prediction."""
#         for col, le in encoders.items():
#             if col in df.columns:
#                 if isinstance(le, dict):
#                     df[col] = df[col].astype(str).map(le).fillna(-1).astype(int)
#                 else:
#                     df[col] = df[col].astype(str)
#                     known_classes = set(le.classes_)
#                     df[col] = df[col].apply(lambda x: x if x in known_classes else "__unknown__")
#                     mapping = {c: i for i, c in enumerate(le.classes_)}
#                     df[col] = df[col].map(lambda x: mapping[x] if x in mapping else -1)
#         return df

#     def _make_predictions(self, df):
#         """Make predictions using loaded models."""
#         X_input = df[self.feature_cols].copy()
        
#         loaded_models = {}
#         predictions = {}
        
#         # Load models
#         for name in ['selected_for_rake', 'choose_rail', 'on_time', 'rail_cost', 'road_cost']:
#             path = os.path.join(self.model_dir, f"model_{name}.pkl")
#             if os.path.exists(path):
#                 loaded_models[name] = joblib.load(path)
        
#         # Make predictions
#         if 'selected_for_rake' in loaded_models:
#             p = loaded_models['selected_for_rake'].predict(
#                 X_input, num_iteration=loaded_models['selected_for_rake'].best_iteration
#             )
#             predictions['selected_for_rake_prob'] = p
#             predictions['selected_for_rake'] = (p > 0.5).astype(int)
        
#         # Add predictions for other models as needed...
        
#         return predictions


# def main():
#     """Main execution function."""
#     # Configuration
#     CSV_PATH = r"C:\Users\INDIA\Desktop\sihPS2\bokaro_to_cmo_customers.csv"
#     MODEL_DIR = r"C:\Users\INDIA\Desktop\sihPS2\saved_models"
    
#     # Initialize and run pipeline
#     pipeline = MultiModelPipeline(CSV_PATH, MODEL_DIR)
#     pipeline.train_all_models()
    
#     print("Models trained successfully!")


# if __name__ == "__main__":
#     main()


"""
multi_model_pipeline.py

Multi-Model LightGBM pipeline for:
- selected_for_rake_label   (binary classification)
- choose_rail_label         (binary classification)
- on_time_label             (binary classification)
- rail_cost_total / road_cost_total (regression)

Uses shared preprocessing and label-encoders, saves models and encoders using joblib.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
import os
from datetime import datetime


class MultiModelPipeline:
    """Multi-model pipeline for training and predicting on multiple targets."""
    
    def __init__(self, csv_path, model_dir, random_state=42, test_size=0.2):
        self.csv_path = csv_path
        self.model_dir = model_dir
        self.random_state = random_state
        self.test_size = test_size
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.targets = {
            'selected_for_rake': 'selected_for_rake_label',
            'choose_rail': 'choose_rail_label',
            'on_time': 'on_time_label',
            'rail_cost': 'rail_cost_total',
            'road_cost': 'road_cost_total'
        }
        
        self.feature_cols = None
        self.categorical_cols = None
        self.label_encoders = None
        
        self.params_reg = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'seed': self.random_state
        }

        self.params_clf = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'seed': self.random_state
        }

    # ---------------------------------------------------------------------
    # Data Preparation
    # ---------------------------------------------------------------------
    def load_and_preprocess_data(self):
        print("Loading data...")
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} records")
        
        df = self._engineer_date_features(df)
        df = self._prepare_features(df)
        df = self._handle_missing_values(df)
        df = self._encode_categorical_features(df)
        return df

    def _engineer_date_features(self, df):
        if 'plan_date' in df.columns:
            try:
                df['plan_date'] = pd.to_datetime(df['plan_date'], dayfirst=True)
                df['plan_dayofweek'] = df['plan_date'].dt.dayofweek
                df['plan_month'] = df['plan_date'].dt.month
                df['plan_day'] = df['plan_date'].dt.day
            except Exception as e:
                print(f"Date processing warning: {e}")
        return df

    def _prepare_features(self, df):
        drop_cols = []
        for col in ['plan_id', 'plan_date', 'plan_id.1']:
            if col in df.columns:
                drop_cols.append(col)
        self.feature_cols = [c for c in df.columns 
                             if c not in list(self.targets.values()) + drop_cols]
        print(f"Using {len(self.feature_cols)} feature columns")
        return df

    def _handle_missing_values(self, df):
        for col in self.feature_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
        return df

    def _encode_categorical_features(self, df):
        self.categorical_cols = self._identify_categorical_columns(df)
        print(f"Detected {len(self.categorical_cols)} categorical columns")
        
        self.label_encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            try:
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            except Exception:
                vals = df[col].unique().tolist()
                mapping = {v: i for i, v in enumerate(vals)}
                df[col] = df[col].map(mapping)
                self.label_encoders[col] = mapping
        
        joblib.dump(self.label_encoders, os.path.join(self.model_dir, "label_encoders.pkl"))
        print("Saved label encoders")
        return df

    def _identify_categorical_columns(self, df):
        categorical_cols = [c for c in self.feature_cols if df[c].dtype == 'object']
        for col in self.feature_cols:
            if (df[col].dtype in [np.int64, np.int32] or 
                (pd.api.types.is_float_dtype(df[col]) and df[col].nunique() < 20)):
                if df[col].nunique() < 50 and col not in categorical_cols:
                    categorical_cols.append(col)
        return sorted(list(set(categorical_cols)))

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def _prepare_targets(self, df):
        targets = {}
        for target_name, target_col in self.targets.items():
            if target_col not in df.columns:
                print(f"Target '{target_col}' not found; skipping {target_name}")
                targets[target_name] = None
                continue
            y = df[target_col].copy()
            if y.dtype == 'object' and target_name in ['selected_for_rake', 'choose_rail', 'on_time']:
                y = y.map({'Yes': 1, 'No': 0}).astype(float)
                if y.isna().any():
                    le_tmp = LabelEncoder()
                    y = le_tmp.fit_transform(df[target_col])
            targets[target_name] = y
        return targets

    def train_regression_model(self, X, y, model_name):
        notnull_idx = y.notna()
        X_train_full = X.loc[notnull_idx, :]
        y_train_full = y.loc[notnull_idx]

        Xtr, Xval, ytr, yval = train_test_split(
            X_train_full, y_train_full, 
            test_size=self.test_size, random_state=self.random_state
        )

        train_data = lgb.Dataset(Xtr, label=ytr, 
                                 categorical_feature=self.categorical_cols, free_raw_data=False)
        val_data = lgb.Dataset(Xval, label=yval, 
                               categorical_feature=self.categorical_cols, reference=train_data, free_raw_data=False)

        model = lgb.train(
            self.params_reg, train_data, valid_sets=[val_data],
            num_boost_round=1000, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        preds = model.predict(Xval, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(yval, preds))
        mae = mean_absolute_error(yval, preds)
        print(f"[{model_name}] RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        joblib.dump(model, os.path.join(self.model_dir, f"{model_name}.pkl"))
        print(f"Saved {model_name}.pkl")
        return model

    def train_classification_model(self, X, y, model_name):
        notnull_idx = y.notna()
        X_train_full = X.loc[notnull_idx, :]
        y_train_full = y.loc[notnull_idx]

        Xtr, Xval, ytr, yval = train_test_split(
            X_train_full, y_train_full, 
            test_size=self.test_size, random_state=self.random_state,
            stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None
        )

        train_data = lgb.Dataset(Xtr, label=ytr, 
                                 categorical_feature=self.categorical_cols, free_raw_data=False)
        val_data = lgb.Dataset(Xval, label=yval, 
                               categorical_feature=self.categorical_cols, reference=train_data, free_raw_data=False)

        model = lgb.train(
            self.params_clf, train_data, valid_sets=[val_data],
            num_boost_round=1000, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        preds_prob = model.predict(Xval, num_iteration=model.best_iteration)
        preds = (preds_prob > 0.5).astype(int)

        acc = accuracy_score(yval, preds)
        f1 = f1_score(yval, preds, zero_division=0)
        try:
            auc = roc_auc_score(yval, preds_prob)
        except Exception:
            auc = None

        print(f"[{model_name}] Accuracy: {acc:.3f}, F1: {f1:.3f}, AUC: {auc if auc is not None else 'NA'}")

        joblib.dump(model, os.path.join(self.model_dir, f"{model_name}.pkl"))
        print(f"Saved {model_name}.pkl")
        return model

    def train_all_models(self):
        df = self.load_and_preprocess_data()
        X = df[self.feature_cols].copy()
        targets = self._prepare_targets(df)

        # Save feature columns for later prediction
        joblib.dump(self.feature_cols, os.path.join(self.model_dir, "feature_cols.pkl"))
        print("Saved feature columns list.")

        models = {}
        for target_name in ['selected_for_rake', 'choose_rail', 'on_time']:
            if targets[target_name] is not None:
                print(f"\nTraining {target_name} model...")
                models[target_name] = self.train_classification_model(X, targets[target_name], f"model_{target_name}")

        for target_name in ['rail_cost', 'road_cost']:
            if targets[target_name] is not None:
                print(f"\nTraining {target_name} model...")
                models[target_name] = self.train_regression_model(X, targets[target_name], f"model_{target_name}")

        print("\nTraining completed!")
        return models

    # ---------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------
    def predict_all(self, new_df):
        encoders = joblib.load(os.path.join(self.model_dir, "label_encoders.pkl"))
        nd = new_df.copy()

        # Load feature columns if missing
        if self.feature_cols is None:
            feature_path = os.path.join(self.model_dir, "feature_cols.pkl")
            if os.path.exists(feature_path):
                self.feature_cols = joblib.load(feature_path)
                print(f"Loaded {len(self.feature_cols)} feature columns for prediction.")
            else:
                raise ValueError("Feature columns not found. Please run training first.")

        missing = [c for c in self.feature_cols if c not in nd.columns]
        if missing:
            raise ValueError(f"Missing feature columns in input: {missing}")

        nd = self._preprocess_prediction_data(nd, encoders)
        return self._make_predictions(nd)

    def _preprocess_prediction_data(self, df, encoders):
        for col, le in encoders.items():
            if col in df.columns:
                if isinstance(le, dict):
                    df[col] = df[col].astype(str).map(le).fillna(-1).astype(int)
                else:
                    df[col] = df[col].astype(str)
                    known_classes = set(le.classes_)
                    df[col] = df[col].apply(lambda x: x if x in known_classes else "__unknown__")
                    mapping = {c: i for i, c in enumerate(le.classes_)}
                    df[col] = df[col].map(lambda x: mapping[x] if x in mapping else -1)
        return df

    def _make_predictions(self, df):
        X_input = df[self.feature_cols].copy()
        loaded_models = {}
        predictions = {}

        for name in ['selected_for_rake', 'choose_rail', 'on_time', 'rail_cost', 'road_cost']:
            path = os.path.join(self.model_dir, f"model_{name}.pkl")
            if os.path.exists(path):
                loaded_models[name] = joblib.load(path)

        if 'selected_for_rake' in loaded_models:
            p = loaded_models['selected_for_rake'].predict(X_input, num_iteration=loaded_models['selected_for_rake'].best_iteration)
            predictions['selected_for_rake_prob'] = p
            predictions['selected_for_rake_label'] = (p > 0.5).astype(int)

        if 'choose_rail' in loaded_models:
            p = loaded_models['choose_rail'].predict(X_input, num_iteration=loaded_models['choose_rail'].best_iteration)
            predictions['choose_rail_prob'] = p
            predictions['choose_rail_label'] = (p > 0.5).astype(int)

        if 'on_time' in loaded_models:
            p = loaded_models['on_time'].predict(X_input, num_iteration=loaded_models['on_time'].best_iteration)
            predictions['on_time_prob'] = p
            predictions['on_time_label'] = (p > 0.5).astype(int)

        if 'rail_cost' in loaded_models:
            predictions['pred_rail_cost_total'] = loaded_models['rail_cost'].predict(X_input)

        if 'road_cost' in loaded_models:
            predictions['pred_road_cost_total'] = loaded_models['road_cost'].predict(X_input)

        return predictions


def main():
    CSV_PATH = r"C:\Users\INDIA\Desktop\sihPS2\bokaro_to_cmo_customers.csv"
    MODEL_DIR = r"C:\Users\INDIA\Desktop\sihPS2\saved_models"
    pipeline = MultiModelPipeline(CSV_PATH, MODEL_DIR)
    pipeline.train_all_models()
    print("Models trained successfully!")


if __name__ == "__main__":
    main()
