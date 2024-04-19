from sklearn.naive_bayes import GaussianNB

#initialize classifier
naive_bayes_classifier = GaussianNB()

#fitting the classifier
naive_bayes_classifier.fit(features_pca, labels)

#make predictions
predicted_labels = naive_bayes_classifier.predict(test_features_pca)

#calculating accuracy
accuracy = np.mean(predicted_labels == test_labels)
print("Accuracy:", accuracy)
