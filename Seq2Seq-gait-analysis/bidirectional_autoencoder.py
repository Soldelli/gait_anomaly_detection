"""
AUTHOR: Riccardo Bonetto
CONTACT: riccardo.bonetto@mailboxes.tu-dresden.de

NOTE:
	This library implements the sequence to sequence model described in the paper
	"Seq2Seq RNN based Gait Anomaly Detection".
	Please refer to the paper for further details.
	
WARNING: 
	If this is the first time you run this bundle, this file MUST be run after running
	data_utils. Prepare to wait for a while while the training is performed.

DEPENDENCIES: 
	Python >= 3.6
	tensorflow nightly build (tf in the following)
	numpy
	data_utils

FOR SPECIFIC TENSORFLOW CALLS, PLEASE REFER TO https://www.tensorflow.org
"""
import numpy as np 
import tensorflow as tf
import data_utils as du 

"""
Parameters deinition

Change the batch_size according to the available memory, the bigger, the better
"""
batch_size = 16
num_layers = 2
num_units = 512
max_gradient_norm = 5
starting_learning_rate = 1e-2
decay_rate = 5e-1
decay_steps = 1000
STEPS = 6000
TOT_TRAIN_ITER = 10

"""
Build the data iterators for training and testing.
For more information about data iterators and tf input
pipelines, please refer to the official tf documentation

(https://www.tensorflow.org/versions/r0.12/api_docs/python/io_ops/input_pipeline)

"""
def build_training_iterator(data_filename = du.np_data_fname):
	data = np.load(data_filename)
	data = data[:int(len(data)*0.9)]
	print(len(data))
	dataset = tf.data.Dataset.from_tensor_slices(tf.cast(data, 
		dtype = tf.float32))	
	dataset = dataset.batch(batch_size)
	batched_iterator = dataset.make_initializable_iterator()
	return batched_iterator
def build_test_iterator(data_filename = du.np_data_fname):
	data = np.load(data_filename)
	data = data[int(len(data)*0.9):]
	dataset = tf.data.Dataset.from_tensor_slices(tf.cast(data, 
		dtype = tf.float32))
	dataset = dataset.batch(len(data))
	test_iterator = dataset.make_initializable_iterator()
	return test_iterator

"""
We use a bidirectional deep LSTM encoder, hence we define a forward (fw) block
and a backward (bw) block (in tf they are called cell).
Both cells are wrapped in a dropout wrapper to enhance the training
and to prevent overfitting.

PARAMETERS:
	num_layers: the number of layers of each cell
	num_cells_per_layer: the number of LSTM units per layer
	keep_prob: the keep probability applied to the dropout operation
"""	
def build_fw_cell(num_layers = num_layers, 
	num_cells_per_layer = [num_units for _ in range(num_layers)],
	keep_prob = 0.8):
	cell = [tf.contrib.rnn.LSTMBlockCell(
		size, use_peephole = True) for size in num_cells_per_layer]

	cell = [tf.contrib.rnn.DropoutWrapper(
			x,
			input_keep_prob = keep_prob) for x in cell]
	cell = tf.contrib.rnn.MultiRNNCell(cell)
	return cell 
def build_bw_cell(num_layers = num_layers, 
	num_cells_per_layer = [num_units for _ in range(num_layers)],
	keep_prob = 0.8):
	cell = [tf.contrib.rnn.LSTMBlockCell(
		size, use_peephole = True) for size in num_cells_per_layer]

	cell = [tf.contrib.rnn.DropoutWrapper(
			x,
			input_keep_prob = keep_prob) for x in cell]
	cell = tf.contrib.rnn.MultiRNNCell(cell)
	return cell 

"""
We wrap the fw and bw cells into a bidirectional_dynamic_rnn block.
This takes care of the BPTT process.

PARAMETERS:
	fw_cell: the fw block wrapped into a MultiRNNCell wrapper
	bw_cell: the bw block wrapped into a MultiRNNCell wrapper
	input_sequnce: the data to be processed (in batches). This is provided by an iterator
	sequence_lengths: a vector containing the length of each input sequence in a batch
"""
def build_rnn_autoencoder(fw_cell, bw_cell, input_sequence, sequence_lengths):
	output, state = tf.nn.bidirectional_dynamic_rnn(
		cell_fw = fw_cell,
		cell_bw = bw_cell,
		inputs = input_sequence,
		sequence_length = None,
		dtype = tf.float32)
	return (output, state)

"""
We build a shallow, linear decoder.

PARAMETERS:
	source: the output of the encoder
	output_dim: the dimension of each sample of the input 
				multivariate timeseries (here is 9)
"""
def build_output_layer(source, output_dim = du.FEATURES):
	return tf.layers.dense(inputs = source,
		units = output_dim)

"""
We define a MSE loss.

PARAMETERS:
		targets: the 1x9 sample that has to be replicated
		predictions: the 1x9 sample that has been decoded
"""
def training_loss(targets, predictions):
	loss = tf.losses.mean_squared_error(
		labels = targets,
		predictions = predictions)
	return loss

"""
We perform the training step by first applying the exponential decay to the learning rate,
we then compute the gradients, clip them if they exceed the maximum threshold (to prevent the
exploding gradient problem), and, eventually, we apply them to update the weights and biases.
"""
def training_step(loss, global_step):
	params = tf.trainable_variables()
	learning_rate = tf.train.exponential_decay(starting_learning_rate, 
		global_step = global_step,
		decay_steps = decay_steps, 
		decay_rate = decay_rate, 
		staircase=True)
	gradients = tf.gradients(loss, params)
	clipped_gradients, _ = tf.clip_by_global_norm(
		gradients, max_gradient_norm)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	update_step = optimizer.apply_gradients(
		zip(clipped_gradients, params),
		global_step = global_step)
	return (update_step, learning_rate)

"""
The __main__ block must be executed only when the Seq2Seq model has to be trained.

After training, the model is saved as "./checkpoint/model".
"""
if __name__ == '__main__':
	train_graph = tf.Graph()
	with train_graph.as_default(): 
		global_step = tf.Variable(0, trainable=False)
		keep_prob = tf.placeholder_with_default(0.8, shape=(),
			name = 'keep_prob_ph')
		handle = tf.placeholder(tf.string, shape = [], name = 'input_handle')
		iterator = tf.data.Iterator.from_string_handle(handle, 
			output_types = tf.float32, 
			output_shapes = [None, du.STEPS, du.FEATURES])
		sequence = iterator.get_next()
		training_iterator = build_training_iterator()
		test_iterator = build_test_iterator()
		fw_cell = build_fw_cell(keep_prob = keep_prob)
		bw_cell = build_bw_cell(keep_prob = keep_prob)
		output, state =  build_rnn_autoencoder(
			fw_cell = fw_cell,
			bw_cell = bw_cell, 
			input_sequence = sequence, 
			sequence_lengths = du.STEPS)
		output = tf.concat(output, 2)
		predictions = build_output_layer(source = output, 
			output_dim = 9)
		loss = training_loss(targets = sequence, 
			predictions = predictions)
		update_step, learning_rate = training_step(loss = loss, 
			global_step = global_step)
		tf.summary.scalar('loss', loss)
		init = tf.global_variables_initializer()
		merged_summary = tf.summary.merge_all()
		train_saver = tf.train.Saver()
	train_sess = tf.Session(graph = train_graph)
	train_sess.run(init)
	training_handle = train_sess.run(training_iterator.string_handle())
	test_handle = train_sess.run(test_iterator.string_handle())
	train_sess.run(training_iterator.initializer)
	writer_1 = tf.summary.FileWriter('./checkpoint/train/', train_graph)
	writer_2 = tf.summary.FileWriter('./checkpoint/eval/', train_graph)
	epoch = 0
	
	"""
	Main train/eval loop
	"""
	for step in range(STEPS):
		try:
			summary, l, u_s, l_r, o, p = train_sess.run(
				[merged_summary, loss, update_step, learning_rate, output, predictions],
				feed_dict = {handle: training_handle})
		except tf.errors.OutOfRangeError:
			train_sess.run(training_iterator.initializer)
			summary, l, u_s, l_r, o, p = train_sess.run(
				[merged_summary, loss, update_step, learning_rate, output, predictions],
				feed_dict = {handle: training_handle})
			epoch+=1
		if step%50 == 0:
			writer_1.add_summary(summary, step)
			print('Epoch -> %d\tStep -> %d,\tloss -> %f' % (epoch, step, l))
			train_sess.run(test_iterator.initializer)
			summary, val_loss = train_sess.run([merged_summary, loss], 
				feed_dict = {keep_prob: 1.0, handle: test_handle})
			writer_2.add_summary(summary, step)
			print('EVAL LOSS -> %f' % val_loss)
		if step%500 == 0:
			train_saver.save(train_sess, 
					'./checkpoint/model',
	    			global_step = step+1)
	train_saver.save(train_sess, 
		'./checkpoint/model')
	writer_1.close()
	writer_2.close()


