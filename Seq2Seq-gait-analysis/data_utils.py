"""
AUTHOR: Riccardo Bonetto
CONTACT: riccardo.bonetto@mailboxes.tu-dresden.de

NOTE:
	This library the functions needed to load and parse the preprocessed
	multivariate timeseries representing the gait cycles.
	For the preprocessing phase, please refer to the paper:
	"Seq2Seq RNN based Gait Anomaly Detection".

WARNING: 
	If this is the first time you run this bundle, this file MUST be run first.

DEPENDENCIES: 
	Python >= 3.6
	numpy
"""

import numpy as np 

np.random.seed(seed=42)

BIG = False

"""
Just paths definitions
"""
original_filename = './data/final_matrix_big.csv' \
	if BIG else './data/final_matrix.csv'
np_data_fname = './data/dataset_big.npy' \
	if BIG else './data/dataset.npy'

classifier_normal_fname = \
	'./data/classification/final_matrix_class_normal.csv'
classifier_anomalous_fname = \
	'./data/classification/final_matrix_class_events.csv'

classifier_ds_fname = './data/classification/dataset.npy'
classifier_l_fname = './data/classification/labels.npy'

"""
Gait timeseries dimensions
"""
FEATURES = 9
STEPS = 200

"""
Load the data into RAM
"""
def load_original_data(fname = original_filename):
	# data are saved in a csv file as strings separated by a comma
	# split each line wrt ',', then take away '"' and '\n'
	# return a list of floats
	with open(fname, 'r') as f:
		df = [[float(i.strip('",\n')) for i in x.split(',')] for x in f]
	return df

"""
small matrix is in (n_steps*num_cycles) x n_feats format
big matrix is in n_feats x (n_steps*num_cycles) format
conditionally create a numpy array
reshape it to have n_steps*num_cycles sequences of
n_feats-dimensional samples
"""
def build_np_dataset(df, n_steps = STEPS, n_feats = FEATURES):
	array = np.transpose(np.array(df)) if BIG else np.array(df)
	array = np.reshape(array, [-1, n_steps, n_feats])
	return array

def concat_classification_datasets_with_labels(ds_n, ds_a, mix = True):
	class_ds = np.concatenate((ds_n, ds_a))
	class_labels = np.zeros([class_ds.shape[0],2], dtype = int)
	class_labels[:ds_n.shape[0],0] = 1
	class_labels[ds_n.shape[0]:,1] = 1
	if mix:
		idx = np.random.permutation(class_ds.shape[0])
		class_ds = class_ds[idx]
		class_labels = class_labels[idx]
	return (class_ds, class_labels)

"""
Save the processed data to dist, so that this has to be done only once
"""
def write_np_dataset_to_disk(dataset, dest_fname = np_data_fname):
	# nomen omen
	np.save(file = dest_fname, arr = dataset)
	print('numpy dataset saved to disk at %s' % dest_fname)


"""
Run this to get ready to train the models
"""
if __name__ == '__main__':
	df_normal = load_original_data(fname = classifier_normal_fname)
	df_events = load_original_data(fname = classifier_anomalous_fname)

	ds_normal = build_np_dataset(df_normal)
	ds_events = build_np_dataset(df_events)
	(class_ds, class_labels) = \
		concat_classification_datasets_with_labels(
			ds_n = ds_normal, 
			ds_a = ds_events)
	write_np_dataset_to_disk(dataset = class_ds, 
		dest_fname = classifier_ds_fname)
	write_np_dataset_to_disk(dataset = class_labels, 
		dest_fname = classifier_l_fname)

	df = load_original_data(fname = original_filename)
	write_np_dataset_to_disk(dataset = build_np_dataset(df), 
		dest_fname = np_data_fname)


