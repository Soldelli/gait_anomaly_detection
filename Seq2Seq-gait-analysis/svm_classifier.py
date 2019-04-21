"""
AUTHOR: Riccardo Bonetto
CONTACT: riccardo.bonetto@mailboxes.tu-dresden.de

NOTE:
	Standard SVM classifier, check scikitlearn documentation for insights. 
	
WARNING: 
	If this is the first time you run this bundle, this file MUST be run after running
	data_utils.

DEPENDENCIES: 
	Python >= 3.6
	sklearn
	numpy
	data_utils

"""

import os
import numpy as np 
from sklearn.externals import joblib
from sklearn import svm 

import data_utils as du 

def load_data():
	dataset = np.load(du.classifier_ds_fname)
	labels = np.load(du.classifier_l_fname)

	train_data = dataset[:int(len(dataset)*0.9)]
	train_labels = labels[:int(len(labels)*0.9)]

	test_data = dataset[int(len(dataset)*0.9):]
	test_labels = labels[int(len(labels)*0.9):]

	return (train_data, train_labels, test_data, test_labels)

def shape_data(dataset):
	shaped_dataset = np.reshape(dataset, 
		[dataset.shape[0], du.FEATURES*du.STEPS])
	return shaped_dataset

def get_classes_from_labels(labels):
	return np.argmax(labels, axis = 1)

def build_classifier():
	return svm.SVC()

def fit_svm(svm_object, train_data, train_labels):
	return svm_object.fit(train_data, train_labels)

def save_fit_model(fit_svm_object, dest_fname = './svm/svm.pkl'):
	if not os.path.exists('./svm/'):
		os.makedirs('./svm/')
	joblib.dump(fit_svm_object, dest_fname)

def load_fit_model(src_fname = './svm/svm.pkl'):
	return joblib.load(src_fname)

def evaluate_model(fit_svm_object, test_data, test_labels):
	predictions = fit_svm_object.predict(test_data)
	total = len(test_labels)
	correct_predictions = np.sum(
		[x==y for (x,y) in zip(test_labels, predictions)])
	metrics = {'accuracy': correct_predictions/total}
	return metrics

if __name__ == '__main__':
	train_data, train_labels, test_data, test_labels = load_data()
	print('Loaded Data')

	shaped_train_data = shape_data(train_data)
	shaped_test_data = shape_data(test_data)
	print('Reshaped Data')

	train_classes = get_classes_from_labels(train_labels)
	test_classes = get_classes_from_labels(test_labels)
	print('Got Classes from One Hot Vectors')

	clf = build_classifier()
	print('Fitting the SVC')
	fit_clf = fit_svm(clf, shaped_train_data, train_classes)

	save_fit_model(fit_svm_object = fit_clf)
	metrics = evaluate_model(fit_svm_object = fit_clf, 
		test_data = shaped_test_data, 
		test_labels = test_classes)
	print('Accuracy -> %f' % (metrics['accuracy']*100))


