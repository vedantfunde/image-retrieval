from sklearn.svm import SVC

# Loading the trained svm classifier
svm_classifier = SVC(kernel='linear')

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Cross Validation
cv_scores = cross_val_score(svm_classifier, features_pca, labels, cv=kf)

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))
