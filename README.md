# End-to-End Salary Prediction (Machine Learning Project)

## Description
This is an academic Machine Learning project demonstrating an end-to-end pipeline for predicting salaries based on multiple factors such as Years of Experience, Education Level, and Job Type. 

The project strictly meets advanced coursework requirements by dividing the task into **Regression** (predicting the exact continuous salary) and **Classification** (predicting whether a salary is above or below the median dataset threshold). It rigorously follows all standard phases of the ML pipeline: Data Generation & Preprocessing, Exploratory Data Analysis (EDA), Model Training, and Evaluation.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Setup & Execution](#project-setup--execution)
3. [Methodology & Pipeline](#methodology--pipeline)
4. [Models Implemented](#models-implemented)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results & Insights](#results--insights)

## Prerequisites
To run this project locally, ensure you have Python 3.8+ installed. The following libraries are required:
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`

You can install all dependencies via pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Project Setup & Execution
Run the main script from your terminal:
```bash
python salary_prediction_project.py
```
This script acts as the master pipeline. When executed, it will immediately:
1. Synthesize the dataframe
2. Automatically clean, encode, and scale the data
3. Train all 7 variants of the requested regression and classification models
4. Generate high-resolution visually rendered plots directly to this directory

## Methodology & Pipeline
1. **Data Loading (Synthetic Generation):** 
   A strict, reproducible random seed generating approximately 1,000 workers with defined bases salaries tied smoothly to their respective roles and education profiles alongside a linear growth for sequential years of experience.
2. **Exploratory Data Analysis (EDA):** 
   Visualizing raw correlations (Salary against Experience, grouped by Degree).
3. **Preprocessing:** 
   - **Imputation:** Using `SimpleImputer` (Median strategy) to gracefully handle purposely injected random missing values.
   - **Categorical Encoding:** Using Pandas dummy variables (`get_dummies`) to one-hot encode Job Types and Education Levels.
   - **Feature Scaling:** Using `StandardScaler` to ensure all numerical distributions have a Gaussian standard distribution (Mean = 0, Variance = 1). Essential for Support Vector Machine (SVM) coordinate distance mapping.
4. **Train-Test Splitting:** 80% Training parameter mapping, 20% unseen Testing validation.

## Models Implemented

### Regression (Continuous Target)
1. **Simple Linear Regression:** Restricted purely to 'Years of Experience' to demonstrate a baseline bivariate relationship calculation.
2. **Multiple Linear Regression:** Allowed to calculate the weight of all processed features simultaneously.
3. **Decision Tree Regressor:** Non-linear branching algorithm testing hierarchical splits.

### Classification (Binary Target: High vs Low Salary)
*(Salary > Median mapped as `1` / High)*
1. **Logistic Regression:** Probability-based solver utilizing the sigmoid function to map values.
2. **Decision Tree Classifier:** Gini-impurity based branching mechanism.
3. **Support Vector Machine (Linear Kernel):** Maximizing the margins between linearly separable points.
4. **Support Vector Machine (RBF Kernel):** Mapping into infinite dimensional space to trace difficult boundary lines.

## Evaluation Metrics
We rely on standard Scikit-Learn metrics to validate the mathematical validity of our models:
- **Regression:** Validated primarily on the **R² Score** (Coefficient of Determination), defining how much variance our features accurately map against target variance.
- **Classification:** Validated primarily on **Accuracy and F1-Scores**, mapping harmonic precision matrices. All top classification models also inherently generate **Confusion Matrix Heatmaps** to visualize True Positives and False Positives.

## Results & Insights

### 🏆 Best Model per Task
* **Best for Exact Salary (Regression): Multiple Linear Regression ($R^2$: 0.95)**
  Because salary structures inherently rely on additive linear bonuses (e.g. baseline + education bonus + fixed yearly raise), solving geometrically for multiple independent weights performs phenomenally well here. Simple Linear Regression ($R^2$: 0.65) was completely unaware of the Job Type features, leading to poorer performance.
  
* **Best for Salary Class (Classification): Logistic Regression / SVMs (Accuracy: 97%)**
  While Logistic Regression and both Support Vector Machines tied with exactly 97% accuracy, Logistic Regression is the functional winner for deployment because it is highly interpretable compared to SVM's black-box algorithm. The Decision Tree Classifier lagged slightly behind (89%) due to its rigid horizontal/vertical geometric cutting approach over smoothly correlated features.
