"""
Advanced Ensemble Processing, Hyperparameter Tuning & ML Serialization Engine
Author: Custom Analytics Pipeline Platform
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# Mute minor warnings
warnings.filterwarnings('ignore')

class AdvancedEnsemblePipeline:
    def __init__(self, data_path: str):
        self.path = data_path
        self.df = None
        self.X = None
        self.y_clf = None
        
        # Split references
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.X_train_scaled, self.X_test_scaled = None, None
        
        # Saved models & outputs
        self.rf_feature_importances = None
        self.best_tuned_pipeline = None

    def load_and_preprocess_data(self):
        print("=== Step 1: Loading & Partitions Engineering ===")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Cleaned dataset missing at: {self.path}")
            
        self.df = pd.read_csv(self.path)
        
        # Drop columns with missing entries and drop transaction identifiers
        cols_to_drop = ['transaction_id', 'timestamp', 'missing_heavy_col']
        feat_cols = [c for c in self.df.columns if c not in cols_to_drop + ['revenue']]
        
        self.X = self.df[feat_cols].copy()
        
        # Perform categorical mappings
        if 'customer_segment' in self.X.columns:
            segment_mapping = {'Discount': 0, 'Standard': 1, 'Premium': 2}
            self.X['customer_segment'] = self.X['customer_segment'].map(segment_mapping)
            
        nominal_cols = ['repetitive_string_col']
        valid_nominals = [c for c in nominal_cols if c in self.X.columns]
        if valid_nominals:
            self.X = pd.get_dummies(self.X, columns=valid_nominals, drop_first=True, dtype=int)
            
        # Target assignment
        y_reg = self.df['revenue'].copy()
        self.y_clf = (y_reg > y_reg.median()).astype(int)
        
        # Split sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_clf, test_size=0.20, random_state=42, stratify=self.y_clf
        )
        
        # Pre-scale splits for baseline models that require manual matrix feeds
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        print(f"Matrix partitions created successfully. Target balance split: \n{self.y_train.value_counts(normalize=True)}")

    def run_decision_tree_benchmarks(self):
        print("\n=== Step 2: Evaluation of Decision Tree Variations ===")
        
        # 1. Unconstrained Baseline Model
        dt_raw = DecisionTreeClassifier(random_state=42)
        dt_raw.fit(self.X_train_scaled, self.y_train)
        print(f"Unconstrained Tree -> Train Acc: {accuracy_score(self.y_train, dt_raw.predict(self.X_train_scaled)):.4f} | Test Acc: {accuracy_score(self.y_test, dt_raw.predict(self.X_test_scaled)):.4f}")
        
        # 2. Constrained Structural Variance Model
        dt_fixed = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
        dt_fixed.fit(self.X_train_scaled, self.y_train)
        print(f"Constrained Tree   -> Train Acc: {accuracy_score(self.y_train, dt_fixed.predict(self.X_train_scaled)):.4f} | Test Acc: {accuracy_score(self.y_test, dt_fixed.predict(self.X_test_scaled)):.4f}")
        
        # 3. Gini vs Entropy Contrast Profiles
        dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
        dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)
        dt_gini.fit(self.X_train_scaled, self.y_train)
        dt_entropy.fit(self.X_train_scaled, self.y_train)
        print(f"Gini Split Criterion Test Acc   : {accuracy_score(self.y_test, dt_gini.predict(self.X_test_scaled)):.4f}")
        print(f"Entropy Split Criterion Test Acc: {accuracy_score(self.y_test, dt_entropy.predict(self.X_test_scaled)):.4f}")

    def run_advanced_ensembles(self):
        print("\n=== Step 3: Ensemble Architectures & Feature Ablation ===")
        
        # 1. Random Forest Training Base
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(self.X_train_scaled, self.y_train)
        rf_auc = roc_auc_score(self.y_test, rf.predict_proba(self.X_test_scaled)[:, 1])
        print(f"Random Forest -> Train Acc: {accuracy_score(self.y_train, rf.predict(self.X_train_scaled)):.4f} | Test Acc: {accuracy_score(self.y_test, rf.predict(self.X_test_scaled)):.4f} | Test AUC: {rf_auc:.4f}")
        
        # Store Feature Importance rankings
        self.rf_feature_importances = pd.Series(rf.feature_importances_, index=self.X.columns).sort_values(ascending=False)
        print("\nRandom Forest Top Features Importance Weights:")
        print(self.rf_feature_importances.head(5))
        
        # 2. Gradient Boosting Execution Model
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gb.fit(self.X_train_scaled, self.y_train)
        gb_auc = roc_auc_score(self.y_test, gb.predict_proba(self.X_test_scaled)[:, 1])
        print(f"Gradient Boosting -> Train Acc: {accuracy_score(self.y_train, gb.predict(self.X_train_scaled)):.4f} | Test Acc: {accuracy_score(self.y_test, gb.predict(self.X_test_scaled)):.4f} | Test AUC: {gb_auc:.4f}")

        # 3. Feature Ablation Experiment Logic
        # Pad features dynamically if column metrics drop below 5 to prevent array failures
        lowest_features = self.rf_feature_importances.tail(5).index.tolist()
        remaining_features = [c for c in self.X.columns if c not in lowest_features]
        
        if len(remaining_features) > 0:
            X_tr_reduced = pd.DataFrame(self.X_train_scaled, columns=self.X.columns)[remaining_features]
            X_te_reduced = pd.DataFrame(self.X_test_scaled, columns=self.X.columns)[remaining_features]
            
            rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf_reduced.fit(X_tr_reduced, self.y_train)
            reduced_auc = roc_auc_score(self.y_test, rf_reduced.predict_proba(X_te_reduced)[:, 1])
            print(f"\n[Ablation Analysis] Full Features AUC: {rf_auc:.4f} | Reduced Features (Dropped {len(lowest_features)}) AUC: {reduced_auc:.4f}")

    def execute_cross_validation_matrix(self):
        print("\n=== Step 4: Cross-Validation Benchmarking ===")
        
        models_dict = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
            'Constrained Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for label, model in models_dict.items():
            scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=skf, scoring='roc_auc')
            print(f"{label:<20} -> Mean CV AUC: {scores.mean():.4f} | Std CV AUC: {scores.std():.4f}")

    def run_grid_search_tuning_pipeline(self):
        print("\n=== Step 5: GridSearchCV Pipeline Tuning Architecture ===")
        
        # Build reproducible operational Pipeline
        modeling_pipeline = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            RandomForestClassifier(random_state=42)
        )
        
        param_grid = {
            'randomforestclassifier__n_estimators': [50, 100, 200],
            'randomforestclassifier__max_depth': [5, 10, None],
            'randomforestclassifier__min_samples_leaf': [1, 5]
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(modeling_pipeline, param_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Optimal Parameters Profile: {grid_search.best_params_}")
        print(f"Highest Grid Search CV Target Score (AUC): {grid_search.best_score_:.4f}")
        
        self.best_tuned_pipeline = grid_search.best_estimator_
        
        # Export production payload model asset to disk
        joblib.dump(self.best_tuned_pipeline, 'best_model.pkl')
        print("Production pipeline artifact successfully serialized and written to disk as 'best_model.pkl'.")

    def run_manual_learning_curve(self):
        print("\n=== Step 6: Learning Curve Fraction Assessment ===")
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        print(f"{'Training Fraction':<18} | {'Training AUC':<14} | {'Test AUC':<12}")
        print("-" * 52)
        
        for f in fractions:
            subset_size = int(f * len(self.X_train))
            X_tr_sub = self.X_train.iloc[:subset_size]
            y_tr_sub = self.y_train.iloc[:subset_size]
            
            # Skip evaluations if sub-samples don't contain both classes
            if len(np.unique(y_tr_sub)) < 2:
                print(f"{f:<18.1f} | Insufficient Class Subsamples | Skipped")
                continue
                
            self.best_tuned_pipeline.fit(X_tr_sub, y_tr_sub)
            
            auc_train = roc_auc_score(y_tr_sub, self.best_tuned_pipeline.predict_proba(X_tr_sub)[:, 1])
            auc_test = roc_auc_score(self.y_test, self.best_tuned_pipeline.predict_proba(self.X_test)[:, 1])
            
            print(f"{f:<18.1f} | {auc_train:<14.4f} | {auc_test:<12.4f}")

    def execute_reload_verification(self):
        print("\n=== Step 7: Execution Validation of Reloaded Model Asset ===")
        # Load object directly from binary state storage
        reloaded_pipeline = joblib.load('best_model.pkl')
        
        # Assemble two distinct, mock production test records
        mock_production_payload = pd.DataFrame([
            {c: (self.X[c].median() if self.X[c].dtype != 'object' else self.X[c].mode()[0]) for c in self.X.columns},
            {c: (self.X[c].mean() if self.X[c].dtype != 'object' else self.X[c].mode()[0]) for c in self.X.columns}
        ])
        
        predictions = reloaded_pipeline.predict(mock_production_payload)
        probabilities = reloaded_pipeline.predict_proba(mock_production_payload)[:, 1]
        
        print("Reload Verification Matrix Metrics Summary:")
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"Record #{idx+1} -> Predicted Label Target: {pred} | Assigned Predictive Confidence Probability: {prob:.4f}")

if __name__ == "__main__":
    # Point directly across to the clean dataset saved in the Part1 directory
    pipeline_manager = AdvancedEnsemblePipeline(data_path="../Part1/cleaned_data.csv")
    pipeline_manager.load_and_preprocess_data()
    pipeline_manager.run_decision_tree_benchmarks()
    pipeline_manager.run_advanced_ensembles()
    pipeline_manager.execute_cross_validation_matrix()
    pipeline_manager.run_grid_search_tuning_pipeline()
    pipeline_manager.run_manual_learning_curve()
    pipeline_manager.execute_reload_verification()