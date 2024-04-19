from sklearn.model_selection import GridSearchCV

#consider any degrees of polynomial
param_grid = {'degree': [2, 3, 4, 5]} 

#initializing polynomial kernel
svm_classifier_poly = SVC(kernel='poly')

#initializing GridSearchCV with SVM classifier and parameter grid
grid_search = GridSearchCV(svm_classifier_poly, param_grid, cv=5, scoring='accuracy')

#fitting GridSearchCV to the training data
grid_search.fit(features_pca, labels)

#getting the best degree from the grid search results
best_degree = grid_search.best_params_['degree']
print("Best degree:", best_degree)

#getting the best SVM classifier with the best degree
best_svm_classifier = grid_search.best_estimator_

#making predictions on the test data using the best classifier
predicted_labels = best_svm_classifier.predict(test_features_pca)

#calculating accuracy
accuracy = np.mean(predicted_labels == test_labels)
#print("Accuracy:", accuracy)
