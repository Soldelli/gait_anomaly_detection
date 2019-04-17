"""
AUTHOR: Riccardo Bonetto
CONTACT: riccardo.bonetto@mailboxes.tu-dresden.de

NOTE:
	Here we implement the full model described in the paper
	"Seq2Seq RNN based Gait Anomaly Detection".
	Once the models have been trained, use this suite to perform actual
	anomaly detection on your own gaits.
	Please refer to the paper for further details.
	
WARNING: 
	If this is the first time you run this bundle, this file MUST be run after running
	conv_classifier.

DEPENDENCIES: 
	Python >= 3.6
	tensorflow nightly build (tf in the following)
	numpy
	data_utils
	bidirectional_autoencoder
	conv_classifier

FOR SPECIFIC TENSORFLOW CALLS, PLEASE REFER TO https://www.tensorflow.org
"""


import numpy as np
import tensorflow as tf

import data_utils as du
import bidirectional_autoencoder as bd 
import conv_classifier_eval as cv 

"""
Build the data iterators
"""
def build_iterators():
	_, _, test_data_iterator, test_target_iterator = cv.build_iterators()
	return (test_data_iterator, test_target_iterator)

"""
Create a tf Saver object
"""
def create_restore_object():
	return tf.train.Saver()

"""
Restore a trained model
"""
def restore_model(saver_object, session,
	checkpoint_fname = './checkpoint/classification/model'):
	saver_object.restore(session, checkpoint_fname)

"""
If you are testing the model, this function computes the accuracy on the test 
dataset
"""
def compute_accuracy(predictions, labels):
	total = len(predictions)
	test_classes = np.argmax(labels, 1) 
	pred_classes = predictions
	correct_predictions = np.sum(
		[x==y for (x,y) in zip(test_classes, pred_classes)])
	return correct_predictions/total


"""
Execute the model
"""
if __name__ == '__main__':
	classification_graph = tf.Graph()

	with classification_graph.as_default():
		data_handle = tf.placeholder(tf.string, shape = [], name = 'data_handle')
		target_handle = tf.placeholder(tf.string, shape = [], name = 'labels_handle')

		data_iterator = tf.data.Iterator.from_string_handle(data_handle, 
				output_types = tf.float32, 
				output_shapes = [None, du.STEPS, du.FEATURES])
		target_iterator = tf.data.Iterator.from_string_handle(target_handle, 
				output_types = tf.int32, 
				output_shapes = [None, 2])

		test_data_iterator, test_target_iterator = build_iterators()


		sequence = data_iterator.get_next()
		labels = target_iterator.get_next()

		keep_prob = tf.placeholder_with_default(1.0, shape=(),
			name = 'keep_prob_ph')


	
		fw_cell = bd.build_fw_cell(keep_prob = keep_prob)
		bw_cell = bd.build_bw_cell(keep_prob = keep_prob)

		output, state =  bd.build_rnn_autoencoder(
			fw_cell = fw_cell,
			bw_cell = bw_cell, 
			input_sequence = sequence, 
			sequence_lengths = du.STEPS)

		# concatenate output_fw and output_bw
		output = tf.concat(output, 2)
		# Here I obtain the predictions from the autoencoder
		predictions = bd.build_output_layer(source = output, 
			output_dim = 9)

		# Reshape the final state of the autoencoder so that
		# shape is: [batch_size, height, width, channels]
		batch_size = tf.placeholder_with_default(bd.batch_size, 
			shape=(), name = 'batch_size')
		s_full = cv.reshape_state(state, bd.num_units, batch_size) 
		#restore_rnn = tf.train.Saver()

		global_step = tf.Variable(0, trainable=False)
		conv_keep_prob = tf.placeholder(shape = (), dtype = tf.float32,
			name = 'kp_conv_ph')
		pooling_output = cv.build_conv_layers(s_full)
		relu = cv.add_relu_activation(pooling_output)
		flat_output = cv.flatten_layer(relu)
		do_output = cv.add_dropout(flat_output, conv_keep_prob)
		logits = cv.build_output_layer(do_output)
		saver = create_restore_object()

		# extract the predicted classes
		predictions = tf.argmax(logits, axis = 1)

	sess = tf.Session(graph = classification_graph)
	test_data_handle, test_target_handle = sess.run([
		test_data_iterator.string_handle(),
		test_target_iterator.string_handle()])

	sess.run([test_data_iterator.initializer, 
		test_target_iterator.initializer])

	restore_model(saver_object = saver, session = sess)

	p = sess.run(predictions, feed_dict = {data_handle: test_data_handle,
				target_handle: test_target_handle,
				batch_size: 572,
				conv_keep_prob: 1.0})

	labels = np.load(du.classifier_l_fname)

	test_labels = labels[int(len(labels)*0.9):]

	accuracy = compute_accuracy(predictions = p, labels = test_labels)
	print(accuracy*100)

