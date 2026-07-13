"""
Automated Data Engineering & Exploratory Discovery Engine
Author: Custom Analytics Pipeline Platform
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AssetPipeline:
    def __init__(self, data_path: str):
        self.path = data_path
        self.frame = None
        self.original_shape = None
        self.skew_registry = {}
        
    def execute_ingestion(self):
        print("=== Step 1: Ingesting Raw Data Asset ===")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Target dataset file missing at: {self.path}")
            
        self.frame = pd.read_csv(self.path)
        self.original_shape = self.frame.shape
        
        print(f"Schema Fingerprint Summary:")
        print(f"Observations: {self.original_shape[0]} | Attributes: {self.original_shape[1]}")
        print("\nStructural Architecture:")
        print(self.frame.dtypes)
        print("\nPreviewing Heading Rows:")
        print(self.frame.head(5))

    def evaluate_missingness(self):
        print("\n=== Step 2: System Missingness Profile ===")
        total_records = self.frame.shape[0]
        abs_nulls = self.frame.isnull().sum()
        pct_nulls = (abs_nulls / total_records) * 100
        
        missing_matrix = pd.DataFrame({'AbsNull': abs_nulls, 'PctNull': pct_nulls})
        print(missing_matrix)
        
        dropped_pool = pct_nulls[pct_nulls > 20.0].index.tolist()
        print(f"Attributes breaching 20% volatility threshold: {dropped_pool}")
        
        numeric_features = self.frame.select_dtypes(include=[np.number]).columns
        for element in numeric_features:
            if 0 < pct_nulls[element] <= 20.0:
                calc_median = self.frame[element].median()
                self.frame[element] = self.frame[element].fillna(calc_median)

    def scrub_redundancies(self):
        print("\n=== Step 3: Resolving Row Redundancies ===")
        dup_count = self.frame.duplicated().sum()
        print(f"Identified redundant record blocks: {dup_count}")
        
        self.frame = self.frame.drop_duplicates()
        post_scrub_shape = self.frame.shape
        print(f"Purged rows count: {self.original_shape[0] - post_scrub_shape[0]}")

    def optimize_types(self, misclassified_numeric: str, high_frequency_string: str):
        print("\n=== Step 4: Typing Scheme & Footprint Optimization ===")
        bytes_init = self.frame.memory_usage(deep=True).sum()
        
        if misclassified_numeric in self.frame.columns:
            self.frame[misclassified_numeric] = pd.to_numeric(self.frame[misclassified_numeric], errors='coerce')
            
        if high_frequency_string in self.frame.columns:
            self.frame[high_frequency_string] = self.frame[high_frequency_string].astype('category')
            
        bytes_final = self.frame.memory_usage(deep=True).sum()
        print(f"Initial Storage: {bytes_init / (1024**2):.3f} MB")
        print(f"Optimized Storage: {bytes_final / (1024**2):.3f} MB")

    def profile_distributions(self) -> str:
        print("\n=== Step 5: Statistical Distribution Profiling ===")
        # Drop columns explicitly dropped by threshold or non-numeric elements to keep this strict
        num_sub = self.frame.select_dtypes(include=[np.number])
        print(num_sub.describe())
        
        for feat in num_sub.columns:
            asymmetry = self.frame[feat].skew()
            self.skew_registry[feat] = asymmetry
            print(f"Feature: {feat:<20} | Fisher-Pearson Skewness: {asymmetry:.4f}")
            
        max_skew_feat = max(self.skew_registry, key=lambda k: abs(self.skew_registry[k]))
        print(f"\nCritical distributional asymmetry identified in: '{max_skew_feat}'")
        return max_skew_feat

    def evaluate_outliers(self, targets: list):
        print("\n=== Step 6: Non-Parametric Boundary Analysis ===")
        for target in targets:
            if target in self.frame.columns:
                q_25, q_75 = self.frame[target].quantile(0.25), self.frame[target].quantile(0.75)
                h_spread = q_75 - q_25
                floor = q_25 - (1.5 * h_spread)
                ceiling = q_75 + (1.5 * h_spread)
                
                anomalies = self.frame[(self.frame[target] < floor) | (self.frame[target] > ceiling)]
                print(f"Feature '{target}': Detected {len(anomalies)} structural anomalies outside fences.")

    def render_analytics_plots(self, structural_skew: str, grouping_var: str):
        print("\n=== Step 7: Generating Visual Assets ===")
        sns.set_style("ticks")
        numeric_keys = list(self.skew_registry.keys())
        
        # 1. Line Plot
        plt.figure(figsize=(8, 4))
        plt.plot(self.frame.index, self.frame[numeric_keys[0]], color='#2b5c8f', lw=1.2)
        plt.title(f"Sequential Volatility Index: {numeric_keys[0]}")
        plt.xlabel("Index Step")
        plt.ylabel("Magnitude")
        plt.tight_layout()
        plt.savefig("asset_line_sequence.png", dpi=200)
        plt.close()

        # 2. Bar Chart
        if grouping_var in self.frame.columns:
            plt.figure(figsize=(8, 4))
            self.frame.groupby(grouping_var)[numeric_keys[0]].mean().plot(kind='bar', color='#d97443')
            plt.title(f"Mean Baseline Distribution of {numeric_keys[0]} Across {grouping_var}")
            plt.ylabel("Mean Centroid Value")
            plt.tight_layout()
            plt.savefig("asset_bar_aggregation.png", dpi=200)
            plt.close()

        # 3. Histogram (Skewed variable)
        plt.figure(figsize=(8, 4))
        sns.histplot(self.frame[structural_skew], bins=20, kde=True, color='#9b2c2c')
        plt.title(f"Probability Mass Architecture: {structural_skew}")
        plt.tight_layout()
        plt.savefig("asset_density_histogram.png", dpi=200)
        plt.close()

        # 4. Scatter Plot
        if len(numeric_keys) > 1:
            plt.figure(figsize=(7, 5))
            sns.scatterplot(data=self.frame, x=numeric_keys[0], y=numeric_keys[1], color='#6b46c1', alpha=0.5)
            plt.title(f"Bivariate Trend Interface: {numeric_keys[0]} vs {numeric_keys[1]}")
            plt.tight_layout()
            plt.savefig("asset_bivariate_scatter.png", dpi=200)
            plt.close()

        # 5. Box Plot (Palette & Hue configuration updated to bypass warnings)
        if grouping_var in self.frame.columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(data=self.frame, x=grouping_var, y=numeric_keys[0], hue=grouping_var, legend=False, palette="Dark2")
            plt.title(f"Dispersion Profile: {numeric_keys[0]} Segmented by {grouping_var}")
            plt.tight_layout()
            plt.savefig("asset_dispersion_boxplot.png", dpi=200)
            plt.close()

    def process_advanced_analytics(self, grouping_var: str):
        print("\n=== Step 8: Comprehensive Association Matrix & Deep Analytics ===")
        num_sub = self.frame.select_dtypes(include=[np.number])
        
        # Heatmap
        p_matrix = num_sub.corr(method='pearson')
        plt.figure(figsize=(8, 6))
        sns.heatmap(p_matrix, annot=True, cmap="viridis", fmt=".3f")
        plt.title("Linear Association Matrix (Pearson)")
        plt.tight_layout()
        plt.savefig("asset_correlation_heatmap.png", dpi=200)
        plt.close()

        # Task A: Advanced Imputation
        sorted_asymmetry = sorted(self.skew_registry.items(), key=lambda x: abs(x[1]), reverse=True)
        top_2_skewed = [sorted_asymmetry[0][0], sorted_asymmetry[1][0]]
        
        print("\n[Task A] Central Tendency Discrepancy Matrix for Skewed Vectors:")
        for v in top_2_skewed:
            m_avg = self.frame[v].mean()
            m_mid = self.frame[v].median()
            print(f"Vector '{v}' -> Arithmetic Mean: {m_avg:.4f} | Resilient Median: {m_mid:.4f}")
            self.frame[v] = self.frame[v].fillna(m_mid)

        # Task B: Spearman Rank Correlation Comparison (Read-only bug corrected via explicit pandas modification)
        s_matrix = num_sub.corr(method='spearman')
        delta_matrix = (s_matrix - p_matrix).abs()
        
        for col in delta_matrix.columns:
            delta_matrix.loc[col, col] = 0.0
            
        print("\n--- Pearson Matrix ---")
        print(p_matrix)
        print("\n--- Spearman Matrix ---")
        print(s_matrix)
        
        print("\n[Task B] Top Non-Linear Discrepancy Interactions (|Spearman - Pearson|):")
        ranked_deltas = delta_matrix.unstack().drop_duplicates().sort_values(ascending=False)
        print(ranked_deltas.head(3))

        # Task C: Grouped Aggregation
        if grouping_var in self.frame.columns:
            print(f"\n[Task C] Segmented Partition Aggregates for '{grouping_var}':")
            aggregated_metrics = self.frame.groupby(grouping_var)[num_sub.columns[0]].agg(['mean', 'std', 'count'])
            print(aggregated_metrics)
            
            span_ratio = aggregated_metrics['mean'].max() / max(aggregated_metrics['mean'].min(), 1e-5)
            print(f"Structural Inter-Group Mean Variance Scale Factor: {span_ratio:.3f}")

    def export_clean_asset(self, destination: str = "cleaned_data.csv"):
        self.frame.to_csv(destination, index=False)
        print(f"\nProduction Ready Clean Dataset written to: {destination}")

if __name__ == "__main__":
    OBJECT_NUMERIC_COLUMN = "numeric_stored_as_object"  
    CATEGORICAL_STRING_COLUMN = "repetitive_string_col"  

    pipeline = AssetPipeline(data_path="raw_data.csv")
    pipeline.execute_ingestion()
    pipeline.evaluate_missingness()
    pipeline.scrub_redundancies()
    pipeline.optimize_types(misclassified_numeric=OBJECT_NUMERIC_COLUMN, high_frequency_string=CATEGORICAL_STRING_COLUMN)
    
    asym_vector = pipeline.profile_distributions()
    numeric_keys = list(pipeline.skew_registry.keys())
    
    if len(numeric_keys) >= 2:
        # Evaluate only existing, valid numeric keys up to two items
        pipeline.evaluate_outliers(targets=numeric_keys[:2])
        
    pipeline.render_analytics_plots(structural_skew=asym_vector, grouping_var=CATEGORICAL_STRING_COLUMN)
    pipeline.process_advanced_analytics(grouping_var=CATEGORICAL_STRING_COLUMN)
    pipeline.export_clean_asset()