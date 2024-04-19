from sklearn.neighbors import KNeighborsClassifier

#loading the trained KNN classifier
knn_classifier = KNeighborsClassifier()

#fitting the KNN classifier to the training data
# knn_classifier.fit(features_lda, labels)
knn_classifier.fit(features_pca, labels)
#making predictions
predicted_labels = knn_classifier.predict(test_features_pca)

#calculating accuracy
accuracy = np.mean(predicted_labels == test_labels)
#print("Accuracy:", accuracy)
