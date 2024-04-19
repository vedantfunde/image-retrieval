from sklearn.svm import SVC

#loading the trained SVM classifier with RBF kernel
svm_classifier = SVC(kernel='linear')

#fitting the SVM classifier to the training data
svm_classifier.fit(features_pca, labels)

#making predictions on the test data using the trained classifier
predicted_labels = svm_classifier.predict(test_features_pca)

#calculating accuracy
accuracy = np.mean(predicted_labels == test_labels)
#print("Accuracy:", accuracy)
