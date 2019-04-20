"""
AUTHOR: Riccardo Bonetto
CONTACT: riccardo.bonetto@mailboxes.tu-dresden.de

NOTE:
	This library implements the convolutional classifier described in the paper
	"Seq2Seq RNN based Gait Anomaly Detection".
	Please refer to the paper for further details.
	
WARNING: 
	If this is the first time you run this bundle, this file MUST be run after running
	bidirectional_autoencoder. Prepare to wait for a while while the training is performed.

DEPENDENCIES: 
	Python >= 3.6
	tensorflow nightly build (tf in the following)
	numpy
	data_utils
	bidirectional_autoencoder

FOR SPECIFIC TENSORFLOW CALLS, PLEASE REFER TO https://www.tensorflow.org
"""



import numpy as np 
import tensorflow as tf 

import bidirectional_autoencoder as bd 
import data_utils as du

# 5719 total samples 
# 5147 training samples

# 321 batches of size 16
# 10 epochs more or less
#STEPS = 3250
STEPS = 10000


#batch_size = bd.batch_size
starting_learning_rate = 1e-2
decay_rate = 5e-1
decay_steps = 1000


"""
Build the data iterators for training and testing.
For more information about data iterators and tf input
pipelines, please refer to the official tf documentation

(https://www.tensorflow.org/versions/r0.12/api_docs/python/io_ops/input_pipeline)

"""
def build_iterators():
	#split 90% train, 10% test
	dataset = np.load(du.classifier_ds_fname)
	labels = np.load(du.classifier_l_fname)

	train_data = dataset[:int(len(dataset)*0.9)]
	train_labels = labels[:int(len(labels)*0.9)]

	test_data = dataset[int(len(dataset)*0.9):]
	test_labels = labels[int(len(labels)*0.9):]


	train_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(train_data, 
		dtype = tf.float32))	
	test_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(test_data, 
		dtype = tf.float32))

	train_target = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, 
		dtype = tf.int32))	
	test_target = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, 
		dtype = tf.int32))

	train_dataset = train_dataset.batch(bd.batch_size)
	train_target = train_target.batch(bd.batch_size)

	# I batch the test just for convenience, otherwise I should
	# define the batch_size as a placeholder
	# and change it accordong the whether I'm in training or
	# testing
	test_dataset = test_dataset.batch(len(test_data))
	test_target = test_target.batch(len(test_data))


	train_data_iterator = train_dataset.make_initializable_iterator()
	train_target_iterator = train_target.make_initializable_iterator()

	test_data_iterator = test_dataset.make_initializable_iterator()
	test_target_iterator = test_target.make_initializable_iterator()

	return (train_data_iterator, 
		train_target_iterator, 
		test_data_iterator, 
		test_target_iterator)

"""
The final state of the autoencoder must be unpacked and reshaped.
Here we do this.
"""
def reshape_state(state, num_units, batch_size):
	log_dimension = np.log2(num_units)
	h = int(2**(np.floor(log_dimension/2)+1))
	w = int(2**np.floor(log_dimension/2))
	# Ugly as fuck unpacking of the output state
	# of the bidirectional LSTM blocks

	# After these operations s_[f|b]w_i_j has shape
	# (batch_size, num_units)
	s_fw, s_bw = tf.unstack(state)

	s_fw_1, s_fw_2 = tf.unstack(s_fw)
	s_fw_1_1, s_fw_1_2 = tf.unstack(s_fw_1)
	s_fw_2_1, s_fw_2_2 = tf.unstack(s_fw_2)

	s_bw_1, s_bw_2 = tf.unstack(s_bw)
	s_bw_1_1, s_bw_1_2 = tf.unstack(s_bw_1)
	s_bw_2_1, s_bw_2_2 = tf.unstack(s_bw_2)

	state_list = [s_fw_1_1,
		s_fw_1_2,
		s_fw_2_1,
		s_fw_2_2,
		s_bw_1_1,
		s_bw_1_2,
		s_bw_2_1,
		s_bw_2_2]

	# Now it is time to repack the tensors in a 
	# (batch_size, height, width, depth) shape
	# Namely, (16, 32, 16, 8)
	reshaped_state_list = [tf.reshape(x, [batch_size,h,w,1]) 
		for x in state_list]
	s_full =  tf.stack(reshaped_state_list,3)
	s_full = tf.squeeze(s_full, 4)
	#s_full = tf.reshape(s_full, [batch_size, h, w, 8])
	return s_full


"""
Here the convolutional architecture is defined.
The filters must be a tf variable. We initialize their weights by means
of truncated normal distribution with 0 mean and standard deviation std=5e-2.

We have one convolutional layer with [10, 6, 8, 16] filters, and a max pooling
operation with kernel size ksize = [1, 4, 4, 1], and strides = [1, 2, 2, 1]
"""
def build_conv_layers(in_tensor):
	filters = tf.get_variable('filter', 
		[10, 6, 8, 16], 
		initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), 
		dtype=tf.float32)

	x_1 = tf.nn.conv2d(input = in_tensor, 
		filter = filters,
		strides = [1, 2, 2, 1],
		padding = 'SAME',
		name = 'cv_1')

	pool_1 = tf.nn.max_pool(value = x_1,
		ksize = [1, 4, 4, 1],
		strides = [1, 2, 2, 1],
		padding = 'SAME',
		name = 'pool_1')
	return pool_1

"""
Add a RELU activation function
"""
def add_relu_activation(in_tensor):
	return tf.nn.relu(features = in_tensor, name = 'relu')


"""
Flatten the output
"""
def add_dropout(in_tensor, keep_prob):
	return tf.nn.dropout(x = in_tensor,
		keep_prob = keep_prob,
		name = 'conv_dropout')

def flatten_layer(in_tensor):
	# This layer flattens in_tensor so that it can be 
	# fed to the output layer.

	# in_tensor must have shape [batch_size, whatever]
	flat_output = tf.contrib.layers.flatten(in_tensor)
	return flat_output


"""
Run this to train the classifier. The trained model will be saved in
"./checkpoint/classification/model"
"""
def build_output_layer(in_tensor):

	logits = tf.layers.dense(inputs = in_tensor,
		units = 2,
		activation = None,
		name = 'logits')

	return logits

def build_loss(in_tensor, labels):
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels = labels,
		logits = in_tensor,
		name = 'xent'))
	return loss 

def training_step(loss, global_step):
	learning_rate = tf.train.exponential_decay(starting_learning_rate, 
		global_step = global_step,
		decay_steps = decay_steps, 
		decay_rate = decay_rate, 
		staircase=True)

	opt = tf.train.GradientDescentOptimizer(learning_rate)

	opt_op = opt.minimize(loss)
	return opt_op


"""
Run this to train the classifier. The trained model will be saved in
"./checkpoint/classification/model"
"""
if __name__ == '__main__':
	conv_train_graph = tf.Graph()
	data = np.load(du.np_data_fname)
	data = data[:int(len(data)*0.9)]

	with conv_train_graph.as_default(): 

		data_handle = tf.placeholder(tf.string, shape = [], name = 'data_handle')
		target_handle = tf.placeholder(tf.string, shape = [], name = 'labels_handle')

		data_iterator = tf.data.Iterator.from_string_handle(data_handle, 
				output_types = tf.float32, 
				output_shapes = [None, du.STEPS, du.FEATURES])
		target_iterator = tf.data.Iterator.from_string_handle(target_handle, 
				output_types = tf.int32, 
				output_shapes = [None, 2])

		train_data_iterator, train_target_iterator, \
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
		s_full = reshape_state(state, bd.num_units, batch_size) 
		restore_rnn = tf.train.Saver()

		conv_keep_prob = tf.placeholder(shape = (), dtype = tf.float32,
			name = 'kp_conv_ph')
		global_step = tf.Variable(0, trainable=False)
		pooling_output = build_conv_layers(s_full)
		relu = add_relu_activation(pooling_output)
		flat_output = flatten_layer(relu)
		do_output = add_dropout(flat_output, conv_keep_prob)
		logits = build_output_layer(do_output)
		loss = build_loss(logits, labels)
		update_step = training_step(loss, global_step)
		tf.summary.scalar('loss', loss)
		init = tf.global_variables_initializer()
		merged_summary = tf.summary.merge_all()
		train_saver = tf.train.Saver()

	sess = tf.Session(graph = conv_train_graph)
	sess.run(init)

	train_data_handle, train_target_handle = sess.run([
		train_data_iterator.string_handle(),
		train_target_iterator.string_handle()])

	test_data_handle, test_target_handle = sess.run([
		test_data_iterator.string_handle(),
		test_target_iterator.string_handle()])

	sess.run([train_data_iterator.initializer, 
		train_target_iterator.initializer, 
		test_data_iterator.initializer, 
		test_target_iterator.initializer])

	restore_rnn.restore(sess, './checkpoint/model')

	writer_1 = tf.summary.FileWriter('./checkpoint/classification/train/', conv_train_graph)
	writer_2 = tf.summary.FileWriter('./checkpoint/classification/eval/', conv_train_graph)

	for train_step in range(STEPS):
		if train_step%320!=0 or train_step==0:
			s, l, _ = sess.run([merged_summary, loss, update_step], 
				feed_dict = {data_handle: train_data_handle,
				target_handle: train_target_handle, conv_keep_prob: 0.5})
		else:
			sess.run(train_data_iterator.initializer)
			sess.run(train_target_iterator.initializer)
			s, l, _ = sess.run([merged_summary, loss, update_step], 
				feed_dict = {data_handle: train_data_handle,
				target_handle: train_target_handle, conv_keep_prob: 0.5})

		if train_step%100 == 0:
			writer_1.add_summary(s, train_step)
			print('Step -> %d\tLoss -> %f' % (train_step, l))
			sess.run([test_data_iterator.initializer, 
				test_target_iterator.initializer])
			s, eval_loss = sess.run([merged_summary, loss],
				feed_dict = {data_handle: test_data_handle,
				target_handle: test_target_handle,
				batch_size: 572,
				conv_keep_prob: 1.0})
			writer_2.add_summary(s, train_step)
			print('Eval Loss -> %f' % eval_loss)
		if train_step%500==0:
			train_saver.save(sess, 
					'./checkpoint/classification/model',
	    			global_step = train_step+1)
	train_saver.save(sess, 
		'./checkpoint/classification/model')
	writer_1.close()
	writer_2.close()




