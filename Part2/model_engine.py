"""
Predictive Machine Learning Engineering Engine
Author: Custom Analytics Pipeline Platform
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

class MachineLearningEngine:
    def __init__(self, data_path: str):
        self.path = data_path
        self.df = None
        self.X = None
        self.y_reg = None
        self.y_clf = None
        
        # Data partitions
        self.X_train, self.X_test = None, None
        self.y_reg_train, self.y_reg_test = None, None
        self.y_clf_train, self.y_clf_test = None, None
        
        # Scaled partitions
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        # Model performance storage
        self.prob_baseline = None
        self.prob_regularized = None

    def execute_ingestion_and_split(self):
        print("=== Step 1: Data Ingestion & Target Definition ===")
        self.df = pd.read_csv(self.path)
        
        # Drop columns with any remaining missing values or identifiers
        cols_to_drop = ['transaction_id', 'timestamp', 'missing_heavy_col']
        feature_cols = [c for c in self.df.columns if c not in cols_to_drop + ['revenue']]
        
        self.X = self.df[feature_cols].copy()
        
        # Target definitions
        self.y_reg = self.df['revenue'].copy()
        median_pivot = self.y_reg.median()
        self.y_clf = (self.y_reg > median_pivot).astype(int)
        
        print(f"Continuous Label Median Pivot Point: {median_pivot:.4f}")
        print(f"Features Loaded (Safely Purged of NaNs): {list(self.X.columns)}")

    def encode_categorical_matrix(self):
        print("\n=== Step 2: Adaptive Categorical Encoding Architecture ===")
        
        # 1. Natural Ordinal Column Handling
        if 'customer_segment' in self.X.columns:
            print("Encoding Natural Ordinal Vector: 'customer_segment'")
            segment_mapping = {'Discount': 0, 'Standard': 1, 'Premium': 2}
            self.X['customer_segment'] = self.X['customer_segment'].map(segment_mapping)
            
        # 2. Nominal Structural Feature Handling
        nominal_cols = ['repetitive_string_col']
        valid_nominals = [c for c in nominal_cols if c in self.X.columns]
        
        if valid_nominals:
            print(f"Encoding Nominal Feature Array via One-Hot Representation: {valid_nominals}")
            self.X = pd.get_dummies(self.X, columns=valid_nominals, drop_first=True, dtype=int)
            
        print(f"Post-Encoding Feature Schema Dimensions: {self.X.shape}")

    def execute_leakage_free_partition(self):
        print("\n=== Step 3: Leakage-Free Train-Test Partitioning & Feature Scaling ===")
        
        # Combine labels temporarily to preserve multi-target random indexing splits
        indices = np.arange(self.X.shape[0])
        idx_tr, idx_te = train_test_split(indices, test_size=0.20, random_state=42)
        
        self.X_train, self.X_test = self.X.iloc[idx_tr].copy(), self.X.iloc[idx_te].copy()
        self.y_reg_train, self.y_reg_test = self.y_reg.iloc[idx_tr], self.y_reg.iloc[idx_te]
        self.y_clf_train, self.y_clf_test = self.y_clf.iloc[idx_tr], self.y_clf.iloc[idx_te]
        
        # Shield transform using isolated training metrics fitments
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        print("Scaling arrays processed successfully without global statistical leakage.")

    def run_regression_suite(self):
        print("\n=== Step 4: Continuous Target Regression Suite ===")
        
        # Ordinary Least Squares Execution
        ols_model = LinearRegression()
        ols_model.fit(self.X_train_scaled, self.y_reg_train)
        preds_ols = ols_model.predict(self.X_test_scaled)
        
        mse_ols = mean_squared_error(self.y_reg_test, preds_ols)
        r2_ols = r2_score(self.y_reg_test, preds_ols)
        
        print(f"OLS Plain Linear Regression Performance Metrics:")
        print(f"Mean Squared Error (MSE): {mse_ols:.4f} | R-Squared (R²): {r2_ols:.4f}")
        
        print("\nOLS Coefficient Profiles Matrix:")
        coef_summary = pd.DataFrame({
            'Feature': self.X.columns,
            'Coefficient': ols_model.coef_,
            'AbsVal': np.abs(ols_model.coef_)
        }).sort_values(by='AbsVal', ascending=False)
        print(coef_summary)
        
        # Ridge Regularized Model Implementation
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(self.X_train_scaled, self.y_reg_train)
        preds_ridge = ridge_model.predict(self.X_test_scaled)
        
        mse_ridge = mean_squared_error(self.y_reg_test, preds_ridge)
        r2_ridge = r2_score(self.y_reg_test, preds_ridge)
        
        print(f"\nRidge Regularized Regression Performance Metrics:")
        print(f"Mean Squared Error (MSE): {mse_ridge:.4f} | R-Squared (R²): {r2_ridge:.4f}")

    def run_classification_suite(self):
        print("\n=== Step 5: Binary Classification Framework ===")
        
        print("Checking class distribution of the training set:")
        distribution = self.y_clf_train.value_counts(normalize=True)
        print(distribution)
        
        # Adjusting for class imbalance inside constructor to maintain safety boundaries
        print("Handling potential class imbalances using balanced weight profiles.")
        
        # Baseline Classification Engine Initialization (C=1.0)
        clf_baseline = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, random_state=42)
        clf_baseline.fit(self.X_train_scaled, self.y_clf_train)
        
        preds_clf = clf_baseline.predict(self.X_test_scaled)
        self.prob_baseline = clf_baseline.predict_proba(self.X_test_scaled)[:, 1]
        
        print("\nConfusion Matrix Output:")
        print(confusion_matrix(self.y_clf_test, preds_clf))
        print("\nClassification Report Diagnostics Summary:")
        print(classification_report(self.y_clf_test, preds_clf, zero_division=0))
        
        # ROC Calculation and Visual Asset Generation
        fpr, tpr, _ = roc_curve(self.y_clf_test, self.prob_baseline)
        auc_score = roc_auc_score(self.y_clf_test, self.prob_baseline)
        
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color='#c53030', lw=2, label=f"Baseline Model AUC = {auc_score:.4f}")
        plt.plot([0, 1], [0, 1], color='#4a5568', linestyle='--')
        plt.title("Receiver Operating Characteristic (ROC Curve Interface)")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("asset_classification_roc.png", dpi=200)
        plt.close()
        print(f"Receiver Operating Curve asset saved successfully. AUC Metrics calculated at: {auc_score:.4f}")

    def process_threshold_sensitivity(self):
        print("\n=== Task B: Decision Threshold Sensitivity Analysis ===")
        threshold_steps = np.arange(0.30, 0.80, 0.10)
        
        print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print("-" * 52)
        
        for thresh in threshold_steps:
            modified_preds = (self.prob_baseline >= thresh).astype(int)
            p = precision_score(self.y_clf_test, modified_preds, zero_division=0)
            r = recall_score(self.y_clf_test, modified_preds, zero_division=0)
            f = f1_score(self.y_clf_test, modified_preds, zero_division=0)
            print(f"{thresh:<12.2f} | {p:<10.4f} | {r:<10.4f} | {f:<10.4f}")

    def process_regularization_experiment(self):
        print("\n=== Step 6 & Task B Extended: Regularization Contrast Suite ===")
        
        clf_regularized = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.01, random_state=42)
        clf_regularized.fit(self.X_train_scaled, self.y_clf_train)
        
        preds_reg = clf_regularized.predict(self.X_test_scaled)
        self.prob_regularized = clf_regularized.predict_proba(self.X_test_scaled)[:, 1]
        
        p_r = precision_score(self.y_clf_test, preds_reg, zero_division=0)
        r_r = recall_score(self.y_clf_test, preds_reg, zero_division=0)
        auc_r = roc_auc_score(self.y_clf_test, self.prob_regularized)
        
        print("Highly Regularized Framework Model Outputs (C=0.01 Parameters Applied):")
        print(f"Precision Score: {p_r:.4f} | Recall Score: {r_r:.4f} | Area Under Curve (AUC): {auc_r:.4f}")

    def process_bootstrap_delta_validation(self):
        print("\n=== Guided Extension: Bootstrap Confidence Interval Delta Evaluation ===")
        
        iterations = 500
        auc_differences_array = []
        y_test_arr = np.array(self.y_clf_test)
        
        np.random.seed(42) # Explicitly seed the resampling generator to preserve repeatability benchmarks
        
        for _ in range(iterations):
            sampled_indices = np.random.choice(len(y_test_arr), size=len(y_test_arr), replace=True)
            
            # Check if sample contains both binary structural targets to protect downstream calculations
            if len(np.unique(y_test_arr[sampled_indices])) < 2:
                continue
                
            auc_c1 = roc_auc_score(y_test_arr[sampled_indices], self.prob_baseline[sampled_indices])
            auc_c01 = roc_auc_score(y_test_arr[sampled_indices], self.prob_regularized[sampled_indices])
            
            auc_differences_array.append(auc_c1 - auc_c01)
            
        mean_delta = np.mean(auc_differences_array)
        percentile_lower = np.percentile(auc_differences_array, 2.5)
        percentile_upper = np.percentile(auc_differences_array, 97.5)
        
        print(f"Bootstrap Resampling Engine Output Metrics ({iterations} Validation Cycles Run):")
        print(f"Mean Area Under Curve (AUC) Delta Difference: {mean_delta:.4f}")
        print(f"95% Confidence Bounds Interval Range Profile: [{percentile_lower:.4f}, {percentile_upper:.4f}]")

if __name__ == "__main__":
    engine = MachineLearningEngine(data_path="cleaned_data.csv")
    engine.execute_ingestion_and_split()
    engine.encode_categorical_matrix()
    engine.execute_leakage_free_partition()
    engine.run_regression_suite()
    engine.run_classification_suite()
    engine.process_threshold_sensitivity()
    engine.process_regularization_experiment()
    engine.process_bootstrap_delta_validation()