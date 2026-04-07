import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    r2_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING & PREPARATION
# ==========================================
print("========== 1. DATA LOADING ==========")
# Generate a synthetic dataset for demonstration (reproducible)
np.random.seed(42)
n_samples = 1000

# Generating Features
years_experience = np.random.uniform(0, 20, n_samples)
education_level = np.random.choice(['Bachelors', 'Masters', 'PhD'], n_samples, p=[0.6, 0.3, 0.1])
job_type = np.random.choice(['Developer', 'Data Scientist', 'Manager'], n_samples)

# Calculating Target Variable: Salary
base_salary = 40000
salary = base_salary + (years_experience * 3000)
salary += np.where(education_level == 'Masters', 15000, 0)
salary += np.where(education_level == 'PhD', 30000, 0)
salary += np.where(job_type == 'Data Scientist', 10000, 0)
salary += np.where(job_type == 'Manager', 20000, 0)
# Add some random noise to make it realistic
salary += np.random.normal(0, 5000, n_samples)

# Creating the DataFrame
df = pd.DataFrame({
    'YearsExperience': years_experience,
    'EducationLevel': education_level,
    'JobType': job_type,
    'Salary': salary
})

# Introduce some missing values specifically to demonstrate imputation (preprocessing)
missing_idx = np.random.choice(df.index, size=20, replace=False)
df.loc[missing_idx, 'YearsExperience'] = np.nan

print("Successfully loaded dataset. First 5 rows:")
print(df.head())
print("\nChecking for missing values:")
print(df.isnull().sum())


# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("\n========== 2. EXPLORATORY DATA ANALYSIS (EDA) ==========")
# We create a visualization to see how Experience affects Salary
plt.figure(figsize=(8,5))
sns.scatterplot(x='YearsExperience', y='Salary', hue='EducationLevel', data=df)
plt.title("Impact of Years of Experience and Education on Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.savefig('eda_salary_vs_experience.png') # Saving the plot as an image
plt.close()
print("-> Saved Scatter Plot as 'eda_salary_vs_experience.png'")


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================
print("\n========== 3. DATA PREPROCESSING ==========")

# A) Handling Missing Values
print("-> Handling Missing Values in 'YearsExperience' using Median.")
imputer = SimpleImputer(strategy='median')
df['YearsExperience'] = imputer.fit_transform(df[['YearsExperience']])

# B) Encoding Categorical Variables
# Machine learning models only understand numbers. We convert text categories into 1s and 0s.
print("-> Applying One-Hot Encoding to categorical columns ('EducationLevel', 'JobType').")
df_encoded = pd.get_dummies(df, columns=['EducationLevel', 'JobType'], drop_first=True)

# Define X (Features) and y (Target) for Regression
X_reg = df_encoded.drop('Salary', axis=1)
y_reg = df_encoded['Salary']

# C) Train-Test Split (Regression)
# 80% for training, 20% for testing
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# D) Feature Scaling
# Ensures all features have a mean of 0 and a variance of 1. Greatly affects SVM.
print("-> Applying Standardization to scale numerical features.")
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Convert Target into Binary for Classification tasks
# Example: If salary is above median, it's 'High' (1). Otherwise 'Low' (0).
median_salary = df['Salary'].median()
df_encoded['SalaryClass'] = (df_encoded['Salary'] > median_salary).astype(int)

# Re-split and re-scale for classification targets
y_clf = df_encoded['SalaryClass']
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_reg, y_clf, test_size=0.2, random_state=42)

X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)
print("-> Dataset is preprocessed and split into training/testing sets.")


# ==========================================
# 4. MODEL TRAINING & EVALUATION (REGRESSION)
# ==========================================
print("\n========== 4. MODEL TRAINING (REGRESSION) ==========")

# We need a strict 1-variable dataset for "Simple Linear Regression"
X_train_simple = X_train_reg[['YearsExperience']]
X_test_simple = X_test_reg[['YearsExperience']]

reg_results = {}

# 4.1 Simple Linear Regression (1 Feature)
print("\nTraining Simple Linear Regression...")
slr = LinearRegression()
slr.fit(X_train_simple, y_train_reg)
y_pred_slr = slr.predict(X_test_simple)
slr_r2 = r2_score(y_test_reg, y_pred_slr)
reg_results["Simple Linear Regression"] = slr_r2
print(f"-> R² Score: {slr_r2:.4f}")

# 4.2 Multiple Linear Regression (All Features)
print("\nTraining Multiple Linear Regression...")
mlr = LinearRegression()
mlr.fit(X_train_reg_scaled, y_train_reg)
y_pred_mlr = mlr.predict(X_test_reg_scaled)
mlr_r2 = r2_score(y_test_reg, y_pred_mlr)
reg_results["Multiple Linear Regression"] = mlr_r2
print(f"-> R² Score: {mlr_r2:.4f}")

# 4.3 Decision Tree Regressor
print("\nTraining Decision Tree Regressor...")
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train_reg_scaled, y_train_reg)
y_pred_dtr = dtr.predict(X_test_reg_scaled)
dtr_r2 = r2_score(y_test_reg, y_pred_dtr)
reg_results["Decision Tree Regressor"] = dtr_r2
print(f"-> R² Score: {dtr_r2:.4f}")


# ==========================================
# 5. MODEL TRAINING & EVALUATION (CLASSIFICATION)
# ==========================================
print("\n========== 5. MODEL TRAINING (CLASSIFICATION) ==========")
print(f"Target: Binary Salary (> ${median_salary:,.0f} is Class 1)")

classification_models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "SVM (Linear Kernel)": SVC(kernel='linear', random_state=42),
    "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

clf_results = {}

for name, model in classification_models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_clf_scaled, y_train_clf)
    y_pred_clf = model.predict(X_test_clf_scaled)
    
    # Calculate Evaluation Metrics
    acc = accuracy_score(y_test_clf, y_pred_clf)
    prec = precision_score(y_test_clf, y_pred_clf)
    rec = recall_score(y_test_clf, y_pred_clf)
    f1 = f1_score(y_test_clf, y_pred_clf)
    cm = confusion_matrix(y_test_clf, y_pred_clf)
    
    clf_results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
    
    print(f"-> Metrics: Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1-Score={f1:.4f}")
    print("-> Confusion Matrix:\n", cm)


# ==========================================
# 6. FINAL COMPARISON & VISUALIZATIONS
# ==========================================
print("\n========== 6. FINAL COMPARISON & VISUALIZATIONS ==========")

# --- 6.1 Regression Models (R² Score) ---
print("\n--- REGRESSION MODELS (R² Score) ---")
for model, score in reg_results.items():
    print(f"{model:<30}: {score:.4f}")

# Visualizing Regression Model Comparison
plt.figure(figsize=(10, 5))
sns.barplot(x=list(reg_results.values()), y=list(reg_results.keys()), palette='Blues_d')
plt.title('Regression Models Comparison (R² Score)')
plt.xlabel('R² Score')
plt.xlim(0, 1.05)
plt.tight_layout()
plt.savefig('regression_comparison.png')
plt.close()
print("-> Saved Plot: 'regression_comparison.png'")

# Visualizing Actual vs Predicted for Best Regression Model (Multiple Linear Regression)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_mlr, alpha=0.6, color='blue', edgecolor='k')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Salaries (Multiple Linear Regression)')
plt.xlabel('Actual Salary ($)')
plt.ylabel('Predicted Salary ($)')
plt.tight_layout()
plt.savefig('actual_vs_predicted_mlr.png')
plt.close()
print("-> Saved Plot: 'actual_vs_predicted_mlr.png'")

# --- 6.2 Classification Models (Accuracy & F1) ---
print("\n--- CLASSIFICATION MODELS (Accuracy & F1) ---")
clf_names = list(clf_results.keys())
accuracies = [metrics['Accuracy'] for metrics in clf_results.values()]

for model, metrics in clf_results.items():
    print(f"{model:<30}: Accuracy={metrics['Accuracy']:.4f} | F1-Score={metrics['F1-Score']:.4f}")

# Visualizing Classification Model Comparison
plt.figure(figsize=(10, 5))
sns.barplot(x=accuracies, y=clf_names, palette='Greens_d')
plt.title('Classification Models Comparison (Accuracy)')
plt.xlabel('Accuracy')
plt.xlim(0, 1.05)
plt.tight_layout()
plt.savefig('classification_comparison.png')
plt.close()
print("-> Saved Plot: 'classification_comparison.png'")

# Visualizing Confusion Matrices for the top Models
models_to_plot = ["Logistic Regression", "SVM (Linear Kernel)", "SVM (RBF Kernel)", "K-Nearest Neighbors"]
for model_name in models_to_plot:
    cm = confusion_matrix(y_test_clf, classification_models[model_name].predict(X_test_clf_scaled))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low Salary', 'High Salary'], 
                yticklabels=['Low Salary', 'High Salary'])
    plt.title(f'Confusion Matrix ({model_name})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Create a safe filename for the image
    filename = f'confusion_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()}.png'
    plt.savefig(filename)
    plt.close()
    print(f"-> Saved Plot: '{filename}'")

# --- 6.3 Combined All Models Comparison ---
print("\n--- COMBINED ALL MODELS COMPARISON ---")
all_model_names = list(reg_results.keys()) + list(clf_results.keys())
# Using R² for regression and Accuracy for classification
all_model_scores = list(reg_results.values()) + [metrics['Accuracy'] for metrics in clf_results.values()]

# We use blue for regression and green for classification to distinguish them
colors = ['skyblue'] * len(reg_results) + ['lightgreen'] * len(clf_results)

plt.figure(figsize=(12, 7))
# Reversing the order so Regression models appear at the top
bars = plt.barh(all_model_names[::-1], all_model_scores[::-1], color=colors[::-1], edgecolor='k')
plt.title('Complete End-to-End Model Comparison', fontsize=14, pad=15)
plt.xlabel('Performance Score (R² for Regression | Accuracy for Classification)', fontsize=12)
plt.xlim(0, 1.1)

# Overlaying exact scores directly onto the bars
for bar in bars:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.4f}', 
             va='center', ha='left', fontweight='bold')

# Creating a custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='skyblue', edgecolor='k', label='Regression Models (R² Score)'),
                   Patch(facecolor='lightgreen', edgecolor='k', label='Classification Models (Accuracy)')]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('all_models_combined_comparison.png')
plt.close()
print("-> Saved Plot: 'all_models_combined_comparison.png'")

print("\nEXECUTION COMPLETE. Check your project folder for the generated graphs!")
