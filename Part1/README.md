# Engineering Report: Core Data Ingestion & Exploratory Engineering Discovery

This log covers the implementation details of the exploratory preprocessing pipeline deployed to process the initial client payload into an optimized machine learning data matrix.

---

## 1. Data Cleaning Protocol & Missing Value Rationalization
*   **Missing Data Filter Threshold**: Columns showing a null rate **greater than 20%** are isolated and excluded from downstream analysis. Doing this prevents features that lack sufficient real information from corrupting model training patterns.
*   **Imputation Strategy Justification**: Missing cells in stable continuous columns (below the 20% null threshold) are filled using their **sample median** instead of the traditional arithmetic mean. The mean is highly vulnerable to out-of-distribution values and directional skew, which pulls its value away from the true center of data density. The median offers an ordinal center point that handles anomalies gracefully and avoids introducing mathematical bias into the dataset.

---

## 2. Redundancy Processing & Type Refactoring
*   **Duplicate Controls**: Row duplicates are systematically removed to prevent data leakage across validation boundaries and to stop repetitive observations from artificially warping performance metrics.
*   **Storage Configuration & Optimization**: Text attributes with repeating categories are downcast to the native `category` data type, and text-bound numbers are converted back to structural floats/ints. This significantly cuts operational memory overhead and increases calculation speeds.

---

## 3. Mathematical Asymmetry Profiling & Distribution Dynamics
Our skewness engine checks the Fisher-Pearson asymmetry indices across continuous attributes:
*   **Positive Skew (Right-Tail Distribution)**: The vast majority of observations sit low on the scale, but rare, large outliers extend a long right tail. Imputing using the mean here introduces upward bias because the mean gets pulled high by those tail values.
*   **Negative Skew (Left-Tail Distribution)**: The data shifts toward the high end, with a trailing tail extending down toward zero. The mean is pulled artificially lower in this case.
*   **Robust Imputation Execution**: Using the median protects our data-filling strategy against these tail influences, keeping center-point estimates stable across both types of distributions.

---

## 4. Anomaly Detection Using Non-Parametric Fences (IQR)
Outlying observations are identified using strict mathematical IQR fences:
$$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}$$
$$\text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$

*   **Handling Plan**: Rows flagged beyond these fences **are explicitly retained** in the data instead of being dropped. Truncating extreme data points risks erasing real-world variation that down-stream architectures need to see. These features will instead be scaled via robust transforms (like `RobustScaler`) or managed using targeted winsorization limits in Part 2.

---

## 5. Visual Asset Interpretations
*   **Asset Line Sequence**: Displays the continuous target variable sequential footprint to check for system volatility or temporal dependencies.
*   **Asset Bar Aggregation**: Compares categorical groups to highlight changes in mean target values.
*   **Asset Density Histogram**: Shows the actual distribution shape of the most asymmetrical variable, highlighting its tail structure.
*   **Asset Bivariate Scatter**: Plots two continuous variables against each other to check for linear paths or loose clouds.
*   **Asset Dispersion Boxplot**: Checks variation across categories; clear differences in median lines indicate strong grouping separation.

---

## 6. Association Logic: Pearson vs. Spearman Mechanics
### Confounding Drivers
High coefficients in the Pearson matrix show related movements but do not prove direct causality ($A \rightarrow B$). A strong correlation often stems from a hidden **confounding variable ($C$)** influencing both metrics:
*   *Alternative Explanation*: If "Device Temp" and "Failure Rate" rise together, a third factory factor—such as "Ambient Factory Humidity"—is often the actual driver behind both metrics.

### Monotonic vs. Linear Interactions
By comparing standard Pearson coefficients with rank-based Spearman coefficients, we can evaluate the shape of relationships:
*   **When $|Spearman| > |Pearson|$**: The relationship is non-linear but monotonic. The features trend together reliably, but not along a constant straight line.
*   **When $|Pearson| \geq |Spearman|$**: The relationship follows a standard straight-line path.
*   **Feature Selection Strategy**: Features showing strong non-linear monotonic trends ($|Spearman| > |Pearson|$) will be preserved for non-linear models (like Random Forests or Gradient Boosted Trees) that naturally handle ranked splits, rather than forcing them into linear structures.

---

## 7. Segmented Partition Aggregates
*   **Within-Group Variance Risk**: A category group showing a high within-group standard deviation indicates that the category label alone cannot provide stable predictions. High variance means the values are scattered, showing that additional features are needed to handle predictions for that specific group.
*   **Predictive Signal Strength**: The mean ratio (Highest Group Mean / Lowest Group Mean) serves as a proxy indicator for predictive strength. A ratio far from $1.0$ indicates that shifting across this categorical index changes the magnitude of the target response significantly, confirming that the variable holds valuable predictive signal.