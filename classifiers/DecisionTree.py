from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier

decision_tree_classifier = DecisionTreeClassifier()

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
cv_scores = cross_val_score(decision_tree_classifier, features_pca, labels, cv=kf)


print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

test_labels = [test_batch_data[b'labels'][i] for i in range(num_test_images)]

# Predictions
predicted_labels = decision_tree_classifier.predict(test_features_pca)

accuracy = np.mean(predicted_labels == test_labels)
print("Accuracy:", accuracy)
