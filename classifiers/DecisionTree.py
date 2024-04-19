from sklearn.tree import DecisionTreeClassifier
#loading the trained decision tree classifier
decision_tree_classifier = DecisionTreeClassifier()

#fitting the decision tree classifier to the training data
decision_tree_classifier.fit(features_pca, labels)

#loading the extracted features
test_extracted_features = np.load('test_extracted_features.npy')

#loading the labels
test_labels = [test_batch_data[b'labels'][i] for i in range(num_test_images)]

#making predictions using the trained classifier
predicted_labels = decision_tree_classifier.predict(test_features_pca)

#calculating accuracy
accuracy = np.mean(predicted_labels == test_labels)
#print("Accuracy:", accuracy)
