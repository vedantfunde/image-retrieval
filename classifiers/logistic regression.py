from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression classifier
logistic_regression = LogisticRegression(max_iter=1000, random_state=42)

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(logistic_regression, features_pca, labels, cv=kf)

# Display the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))
