---

# Engineering Log Part 2: Supervised Predictive Modeling & Validation Benchmarks

## 1. Multi-Target Label Definition & Design Strategy
*   **Regression Target ($y_{\text{reg}}$)**: Configured as the raw continuous feature `revenue`. This model captures granular changes in volume density.
*   **Classification Target ($y_{\text{clf}}$)**: Created by binarizing `revenue` at its statistical sample median value ($145.375$). Observations matching $y_{\text{reg}} > 145.375$ are encoded as $1$ (High Yield), and values below are labeled as $0$ (Low Yield). This balanced design ensures class weight distributions are even.

---

## 2. Leakage-Free Data Transformation Protocols
*   **Categorical Mapping Mechanics**: The `customer_segment` variable has clear ordinal relationships, which are preserved using ordered mapping parameters:
    $$\text{Discount} \rightarrow 0, \quad \text{Standard} \rightarrow 1, \quad \text{Premium} \rightarrow 2$$
*   **Nominal Multi-Collinearity Prevention**: Variables without natural ordering (like `repetitive_string_col`) are transformed using one-hot encoding. Dropping the first dummy column breaks structural multi-collinearity, allowing linear matrices to avoid the *dummy variable trap*. This prevents models from interpreting random textual names as continuous ordinal scales.
*   **Data Leakage Prevention Guardrail**: Feature scale transforms are computed **strictly on the training partition** ($X_{\text{train}}$) using `.fit_transform()`. The test matrix ($X_{\text{test}}$) is scaled using only those pre-computed training parameters via `.transform()`. Fitting transformations globally would leak future out-of-sample data points into the training process, leading to overfit evaluation metrics.

---

## 3. Continuous Target Regression Matrix & Regularization Comparisons
Continuous features map directly to estimated scale changes:
*   **Positive Scaling Coefficient**: A one-unit shift upward in a scaled feature corresponds to a proportional increase in predicted revenue equal to the value of that feature's coefficient.
*   **Negative Scaling Coefficient**: A one-unit shift upward in a scaled feature corresponds to a proportional decrease in predicted revenue equal to the value of that feature's coefficient.

### Model Performance Metrics

| Architectural Model Typology | Mean Squared Error (MSE) | R-Squared Metric ($R^2$) |
| :--- | :--- | :--- |
| **OLS Baseline Linear Regression** | 240451.0541 | 0.9592 |
| **Ridge Regularized Regression ($\alpha=1.0$)** | 258902.1147 | 0.9561 |

*   **Ridge Mechanics**: Ordinary Least Squares (OLS) tries to minimize residuals directly, which can result in highly volatile, inflated weights when features are highly correlated. Ridge regression stabilizes these parameters by introducing an $L_2$ shrinkage constraint scaled by hyperparameter $\alpha$:
    $$\text{Loss}_{\text{Ridge}} = \text{Loss}_{\text{OLS}} + \alpha \sum_{j=1}^{p} \beta_j^2$$
    This mathematical constraint shrinks coefficients toward zero, reducing variance and helping the model handle multicollinearity more effectively than plain OLS.

---

## 4. Binary Classification Metrics & Decision-Threshold Tuning
Classification performance is evaluated using standard structural matrices:

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

*   **Domain Priority Rule**: For this business system, **Recall** is prioritized over Precision. Missing out on a high-value customer (a False Negative) is costlier than accidentally flagging a standard customer as high-value (a False Positive). 
*   **Threshold Adjustment Logic**: To balance these costs, the decision threshold should be **lowered** to improve target capture rates. This expands the classification window, minimizing missed opportunities (False Negatives) at the cost of a minor increase in standard False Positives.
*   **AUC Interpretation**: The baseline Area Under the Curve (AUC) score measures the model's capacity to separate distinct categories. An AUC value of $1.0$ indicates perfect separation, while a score of $0.5$ represents random guessing.

### Decision Threshold Trade-Off Analysis

| Threshold Index | Precision Metric | Recall Metric | Max F1-Score |
| :--- | :--- | :--- | :--- |
| **0.30** | 1.0000 | 0.6667 | 0.8000 |
| **0.40** | 1.0000 | 0.6667 | 0.8000 |
| **0.50** | 1.0000 | 0.6667 | 0.8000 |
| **0.60** | 1.0000 | 0.3333 | 0.5000 |
| **0.70** | 1.0000 | 0.3333 | 0.5000 |

*   **F1 Optimization Index**: The F1 score reaches its maximum value ($0.8000$) across the threshold range of **0.30 to 0.50**, providing a stable window for final model configuration.

---

## 5. Regularization Experiments & Resampling Validations

### Complexity Configuration ($C$)

| Regularization Hyperparameter | Precision | Recall | ROC-AUC Metric |
| :--- | :--- | :--- | :--- |
| **Baseline Configuration ($C = 1.0$)** | 1.0000 | 0.6667 | 1.0000 |
| **Strong Regularization ($C = 0.01$)** | 1.0000 | 0.6667 | 1.0000 |

*   **Inverse Complexity Control ($C$)**: The hyperparameter $C$ acts as the inverse of regularization strength. Lowering its value shifts the optimization focus toward penalizing model complexity ($L_2$ weight reduction):
    $$\text{Objective} = C \times \text{Loss} + \text{Penalty}$$
    In small datasets with clear separations, reducing $C$ down to $0.01$ keeps metrics steady without dropping performance, showing that the model is robust against overfitting.

### Bootstrap Statistical Bounds Analysis
Using $n=500$ bootstrap resampling cycles with replacement, the model performance difference was calculated:
*   **Mean AUC Difference Metric**: $0.0000$
*   **95% Confidence Bounds Profile**: $[0.0000, 0.0000]$

*   **Statistical Evaluation**: Because the 95% bootstrap confidence interval includes zero ($0.0000$), the performance gap between the $C=1.0$ and $C=0.01$ setups is not statistically significant. This confirms that the classification boundary is highly stable and clean across different dataset subsets.