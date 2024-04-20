from sklearn.naive_bayes import GaussianNB

# Initialising the classifier(Gaussian Naive Bayes)
naive_bayes_classifier = GaussianNB()

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Cross validation
cv_scores = cross_val_score(naive_bayes_classifier, features_pca, labels, cv=kf)

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))
