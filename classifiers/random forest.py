from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest classifier with desired parameters
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(random_forest_classifier, features_pca, labels, cv=kf)

# Display the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))
