import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score
)

def load_skills_data():
    with open('skills.json', 'r') as file:
        skills_data = json.load(file)
    return skills_data

def load_data():
    data = {
        "cgpa": [
            8.5, 7.0, 9.0, 6.5, 8.0, 7.5, 8.2, 6.8, 7.8, 8.1,
            7.3, 6.6, 7.9, 8.3, 6.9, 7.4, 8.4, 7.1, 8.6, 6.7,
            8.02  # New Data Point
        ],
        "final_technical_score": [
            91.67, 66.67, 100.0, 50.0, 83.33, 73.33, 78.33, 60.0, 70.0, 80.0,
            65.0, 53.33, 68.33, 81.67, 56.67, 71.67, 85.0, 63.33, 86.67, 61.67,
            71.73913043478261  # New Data Point
        ],
        "soft_skills_score": [
            21.0, 15.0, 24.0, 13.0, 20.0, 19.0, 22.0, 16.0, 18.5, 20.5,
            17.0, 14.0, 19.5, 23.0, 16.5, 18.0, 21.5, 17.5, 22.5, 16.8,
            22  # New Data Point
        ],
        "aptitude_test_score": [
            85, 70, 90, 60, 80, 75, 88, 65, 78, 82,
            72, 68, 74, 85, 69, 77, 83, 71, 86, 66,
            95.0  # New Data Point
        ],
        "internship_experience": [
            1, 0, 1, 0, 1, 1, 0, 0, 1, 1,
            0, 0, 1, 1, 0, 1, 1, 0, 1, 0,
            1  # New Data Point
        ],
        "domain_ml": [
            1, 0, 1, 0, 1, 1, 0, 0, 1, 1,
            0, 0, 1, 0, 0, 1, 1, 0, 1, 0,
            0  # New Data Point
        ],
        "domain_dev": [
            0, 1, 0, 1, 0, 0, 1, 1, 0, 0,
            1, 1, 0, 1, 1, 0, 0, 1, 0, 1,
            1  # New Data Point
        ],
        "domain_cloud": [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0  # New Data Point
        ],
        "placed": [
            1, 0, 1, 0, 1, 1, 0, 0, 1, 1,
            0, 0, 1, 1, 0, 1, 1, 0, 1, 0,
            1  # New Data Point
        ],
        "salary": [
            600000, 0, 750000, 0, 650000, 700000, 0, 0, 720000, 680000,
            0, 0, 690000, 710000, 0, 730000, 750000, 0, 760000, 0,
            800000  # New Data Point
        ]
    }

    # Enforce that each user has only one domain as 1
    # Priority: ML > Dev > Cloud
    for i in range(len(data['domain_ml'])):
        domains = [
            data['domain_ml'][i],
            data['domain_dev'][i],
            data['domain_cloud'][i]
        ]
        if sum(domains) > 1:
            # Assign 1 to the highest priority domain and 0 to others
            if domains[0] == 1:
                data['domain_dev'][i] = 0
                data['domain_cloud'][i] = 0
            elif domains[1] == 1:
                data['domain_ml'][i] = 0
                data['domain_cloud'][i] = 0
            elif domains[2] == 1:
                data['domain_ml'][i] = 0
                data['domain_dev'][i] = 0

    # Verify that each user has only one domain as 1
    for i in range(len(data['domain_ml'])):
        domains = [
            data['domain_ml'][i],
            data['domain_dev'][i],
            data['domain_cloud'][i]
        ]
        assert sum(domains) <= 1, f"User {i+1} has multiple domains as 1."

    return pd.DataFrame(data)

def visualize_feature_distribution():
    data = load_data()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='placed', y='final_technical_score', data=data)
    plt.title('Final Technical Score by Placement Status')
    plt.xlabel('Placement Status')
    plt.ylabel('Final Technical Score')
    plt.savefig('feature_distribution.png')  # Save the plot as an image
    plt.close()
    print("Feature distribution plot saved as 'feature_distribution.png'.")

def plot_feature_importances(clf_model):
    feature_names = ['CGPA', 'Final Technical Score', 'Soft Skills Score',
                     'Aptitude Test Score', 'Internship Experience',
                     'Domain ML', 'Domain Dev', 'Domain Cloud']
    coefficients = clf_model.coef_[0]
    feature_importance = pd.Series(np.abs(coefficients), index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title("Feature Importances (Logistic Regression Coefficients)")
    plt.ylabel("Absolute Coefficient Value")
    plt.tight_layout()
    plt.savefig('feature_importances.png')  # Save the plot as an image
    plt.close()
    print("Feature importances plot saved as 'feature_importances.png'.")

def train_logistic_regression(X_train, y_train):
    clf_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    clf_model.fit(X_train, y_train)
    joblib.dump(clf_model, 'placement_model.pkl')
    print("Logistic Regression model saved as 'placement_model.pkl'.")
    return clf_model

def train_random_forest_classifier(X_train, y_train):
    rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_clf.fit(X_train, y_train)
    joblib.dump(rf_clf, 'rf_placement_model.pkl')
    print("Random Forest Classifier saved as 'rf_placement_model.pkl'.")
    return rf_clf

def train_random_forest_regressor(X_train, y_train):
    reg_model = RandomForestRegressor(random_state=42)
    reg_model.fit(X_train, y_train)
    joblib.dump(reg_model, 'salary_model.pkl')
    print("Random Forest Regressor saved as 'salary_model.pkl'.")
    return reg_model

def evaluate_classification_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    roc_auc = roc_auc_score(y_test, proba)
    print(f"ROC-AUC Score: {roc_auc:.2f}")

def evaluate_regression_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print("Regression Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

def train_and_evaluate():
    data = load_data()
    X = data.drop(['placed', 'salary'], axis=1)
    y_class = data['placed']
    y_reg = data['salary']
    
    # Check class distribution
    print("Class Distribution:")
    print(y_class.value_counts())
    
    # Visualize feature distribution
    visualize_feature_distribution()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'scaler.pkl')
    print("StandardScaler saved as 'scaler.pkl'.")
    
    # Split dataset into training and testing sets for classification
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_scaled, y_class, test_size=0.25, random_state=42, stratify=y_class
    )
    
    # Train Logistic Regression model
    clf_model = train_logistic_regression(X_train_cls, y_train_cls)
    
    # Train Random Forest Classifier
    rf_clf = train_random_forest_classifier(X_train_cls, y_train_cls)
    
    # Evaluate Logistic Regression model
    print("\nEvaluating Logistic Regression Model:")
    evaluate_classification_model(clf_model, X_test_cls, y_test_cls)
    
    # Evaluate Random Forest Classifier
    print("\nEvaluating Random Forest Classifier:")
    evaluate_classification_model(rf_clf, X_test_cls, y_test_cls)
    
    # Plot and save feature importances for Logistic Regression
    plot_feature_importances(clf_model)
    
    # Perform Cross-Validation for Classification Models
    print("\nPerforming Cross-Validation for Logistic Regression:")
    cv_scores_lr = cross_val_score(clf_model, X_scaled, y_class, cv=5, scoring='accuracy')
    print(f"Logistic Regression Cross-Validation Scores: {cv_scores_lr}")
    print(f"Average CV Accuracy: {cv_scores_lr.mean():.2f}")
    
    print("\nPerforming Cross-Validation for Random Forest Classifier:")
    cv_scores_rf = cross_val_score(rf_clf, X_scaled, y_class, cv=5, scoring='accuracy')
    print(f"Random Forest Classifier Cross-Validation Scores: {cv_scores_rf}")
    print(f"Average CV Accuracy: {cv_scores_rf.mean():.2f}")
    
    # Split dataset into training and testing sets for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_scaled, y_reg, test_size=0.25, random_state=42
    )
    
    # Train Regression model
    reg_model = train_random_forest_regressor(X_train_reg, y_train_reg)
    
    # Evaluate Regression model
    print("\nEvaluating Regression Model:")
    evaluate_regression_model(reg_model, X_test_reg, y_test_reg)
    
    # Perform Cross-Validation for Regression Model
    print("\nPerforming Cross-Validation for Regression Model:")
    cv_scores_reg = cross_val_score(reg_model, X_scaled, y_reg, cv=5, scoring='neg_mean_squared_error')
    print(f"Regression Cross-Validation Scores (MSE): {cv_scores_reg}")
    print(f"Average CV MSE: {(-cv_scores_reg.mean()):.2f}")

if __name__ == "__main__":
    train_and_evaluate()