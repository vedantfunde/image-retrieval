from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold

param_grid = {'degree': [2, 3, 4, 5]}
svm_classifier_poly = SVC(kernel='poly')
grid_search = GridSearchCV(svm_classifier_poly, param_grid, cv=5, scoring='accuracy')
grid_search.fit(features_lda, labels)
best_degree = grid_search.best_params_['degree']
print("Best degree:", best_degree)
best_svm_classifier = grid_search.best_estimator_
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_svm_classifier, features_lda, labels, cv=kf)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))
predicted_labels = best_svm_classifier.predict(test_features_lda)
accuracy = np.mean(predicted_labels == test_labels)
print("Accuracy:", accuracy)
