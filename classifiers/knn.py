from sklearn.neighbors import KNeighborsClassifier

#loading the trained KNN classifier, using library function
knn_classifier = KNeighborsClassifier()


# knn_classifier.fit(features_lda, labels)
knn_classifier.fit(features_pca, labels)
#making predictions
predicted_labels = knn_classifier.predict(test_features_pca)

accuracy = np.mean(predicted_labels == test_labels)
#print("Accuracy:", accuracy)
