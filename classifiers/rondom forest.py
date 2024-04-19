from sklearn.ensemble import RandomForestClassifier

#initializing Random Forest classifier with desired parameters
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

#fitting the classifier to the training data
random_forest_classifier.fit(extracted_features, labels)

#making predictions on the test data
predicted_labels = random_forest_classifier.predict(test_extracted_features)

#calculating accuracy
accuracy = np.mean(predicted_labels == test_labels)
#print("Accuracy:", accuracy)
