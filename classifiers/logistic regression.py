from sklearn.linear_model import LogisticRegression

#loading Logistic Regression classifier
logistic_regression = LogisticRegression(max_iter=10000, random_state=42)

#fitting the classifier to the training data
# logistic_regression.fit(extracted_features, labels)
# predicted_labels = logistic_regression.predict(test_extracted_features)

#making predictions on the test data
logistic_regression.fit(extracted_features, labels)
predicted_labels = logistic_regression.predict(test_extracted_features)

#calculating accuracy
accuracy = np.mean(predicted_labels == test_labels)
print("Accuracy:", accuracy)
