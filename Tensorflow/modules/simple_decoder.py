import tensorflow as tf
import numpy as np
from os.path import isfile


def sinusoid_from_phase(phases, n_T, omega):
    """
    This function produces an array where each row corresponds to a sinusoidal signal with a given phase and
    angular frequency omega. The columns represent the time sampling from the interval [0,1].
    :param phases: Vector with the phases to be used
    :param n_T: Number of elements in the partition of the interval [0,1]
    :param omega: Angular frequency
    :return: np.array with shape (len(phases),n_T)
    """
    time_range = np.linspace(0, 1, n_T)
    sinusoids = np.sin(np.subtract.outer(phases, -(omega * time_range)))
    return time_range, sinusoids


class simple_decoder():
    """
    Simple decoder neural network that is capable of taking a phase value and producing a sinusoidal signals after
    training.
    """
    def __init__(self, learning_rate, data_dimensions):
        tf.reset_default_graph()
        self.data_dimensions = data_dimensions
        self.intermediate_dim = data_dimensions//3
        self.learning_rate = learning_rate
        # Build the graph
        self.decoders_dictionary = self.build()


    def define_input(self):
        """
        Defines the input tensor
        :return: Tensor z
        """
        with tf.variable_scope("Input"):
            z = tf.placeholder(tf.float32, shape=[None, 2])
        return z


    def define_decoder(self, z, number):
        """
        Defines the subgraph of the decoder
        :param z: Code tensor
        :return: Reconstruction tensor x_r and a placeholder tensor for the true signal
        """
        with tf.variable_scope("decoder"+'_'+'Human'):
            # Hidden layers
            h_d = tf.layers.dense(z, self.intermediate_dim, activation=tf.nn.relu, name="h_dec"+'_'+'Human')
            h_d2 = tf.layers.dense(h_d, self.intermediate_dim, activation=tf.nn.relu, name="h_dec2"+'_'+'Human')
            # Output of the network
            x_r = tf.layers.dense(h_d2, self.data_dimensions, activation=None, name="x_r"+'_'+'Human')
            # Placeholder for the true signal
            real_signal = tf.placeholder(tf.float32, shape = [None, self.data_dimensions])
            return x_r, real_signal

    def define_loss(self, x_r, real_signal):
        """
        Defines the reconstruction loss
        :param x_r: Output tensor of the network
        :param real_signal: Real signal tensor
        :return:
        """
        with tf.name_scope('loss_calculation'):
            loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.losses.mean_squared_error(
                        labels=x_r,
                        predictions=real_signal,
                        reduction=tf.losses.Reduction.NONE),
                    axis=1)
                , name='loss')
        return loss

    def build(self):
        """
        This method builds the computation graph for the decoder network
        :return: dictionary with all the necessary tensors and classes needed for training and evaluation
        """
        # Define the network tensors
        z = self.define_input()
        x_r, real_signal = self.define_decoder(z, 0)
        loss = self.define_loss(x_r, real_signal)
        # Tensorboard summaries
        tf.summary.scalar('Loss', loss)
        summary_op = tf.summary.merge_all()
        # Saver
        saver = tf.train.Saver()
        # Optimizer
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        decoders_dictionary = {'z': z,
                          'x_r': x_r,
                          'real_signal': real_signal,
                          'loss': loss,
                          'summary_op': summary_op,
                          'saver': saver,
                          'train_step': train_step}
        return decoders_dictionary

    def sample_phases(self, num_samples:int):
        """
        This function samples uniformly a certain number of phases from [0,2pi]. It then embeds the phase into the
        unit circle in R^2.
        :param num_samples: Number of phases to be sampled uniformly
        :return: Returnes the sampled phases and the embedded samples
        """
        u_samples = np.random.uniform(-1.0,1.0, size = (num_samples))
        phases = u_samples*np.pi
        scaled = np.expand_dims(phases,1)
        x_proj = np.cos(scaled)
        y_proj = np.sin(scaled)
        samples = np.concatenate([x_proj, y_proj], axis = -1)
        return phases, samples



    def train(self, num_samples, epochs, batch_size, log_dir_tensorboard, weights_folder):
        """
        Trains the network with random samples and compares the reconstructions to the true sinusoidal signals
        :param num_samples: number of random phase samples used for the training
        :param epochs: number of training epochs
        :param batch_size: batch size used
        :param log_dir_tensorboard: path to the tensorboard directory
        :param weights_folder: path to the weights directory
        :return: phases and signals used for training
        """
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(log_dir_tensorboard, graph=sess.graph)
            sess.run(tf.global_variables_initializer())
            # Restore the decoder weights
            if isfile(weights_folder):
                print("Restoring saved parameters")
                self.decoders_dictionary['saver'].restore(sess, weights_folder)
            else:
                print("Initializing parameters")
                sess.run(tf.global_variables_initializer())
            # DATA
            phases, samples = self.sample_phases(num_samples)
            _, signals = sinusoid_from_phase(phases, self.data_dimensions, omega = np.pi*2)
            dataset = tf.data.Dataset.from_tensor_slices((samples, signals)).shuffle(num_samples).batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            # Loss per batch
            batch_loss = np.zeros(num_samples//batch_size)
            for epoch in range(epochs):
                sess.run(iterator.initializer)  # Initialize the data iterator
                for batch in range(num_samples // batch_size):
                    data_batch = sess.run(next_element)
                    feed_dict = {self.decoders_dictionary['z']: data_batch[0],
                                 self.decoders_dictionary['real_signal']: data_batch[1]}
                    # Training
                    _, batch_loss[batch], summary = \
                        sess.run([self.decoders_dictionary['train_step'],
                                  self.decoders_dictionary['loss'],
                                  self.decoders_dictionary['summary_op']
                                  ], feed_dict=feed_dict)
                    summary_writer.add_summary(summary, epoch)
                print("Epoch {} | Loss: {:.2E} ".format(epoch, np.mean(batch_loss)))
            self.decoders_dictionary['saver'].save(sess, weights_folder)
        return phases, signals


    def train_phases(self, phases, epochs, log_dir_tensorboard, weights_folder):
        """
        This method trains the network with a specific selection of phases
        :param phases: vector with the phases for training
        :param epochs: number of training epochs
        :param log_dir_tensorboard: path to the tensorboard directory
        :param weights_folder: path to the weights folder
        :return: None
        """
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(log_dir_tensorboard, graph=sess.graph)
            sess.run(tf.global_variables_initializer())
            # Restore the decoder weights
            if isfile(weights_folder):
                print("Restoring saved parameters")
                self.decoders_dictionary['saver'].restore(sess, weights_folder)
            else:
                print("Initializing parameters")
                sess.run(tf.global_variables_initializer())
            x_proj = np.cos(np.expand_dims(phases,1))
            y_proj = np.sin(np.expand_dims(phases,1))
            samples = np.concatenate([x_proj, y_proj], axis=-1)
            _,signals = sinusoid_from_phase(phases, self.data_dimensions, omega = np.pi*2)
            for epoch in range(epochs):
                feed_dict = {self.decoders_dictionary['z']: samples, self.decoders_dictionary['real_signal']: signals}
                # Training
                _, calc_loss, summary = \
                    sess.run([self.decoders_dictionary['train_step'],
                              self.decoders_dictionary['loss'],
                              self.decoders_dictionary['summary_op']
                              ], feed_dict=feed_dict)
                summary_writer.add_summary(summary, epoch)
                print("Epoch {} | Loss: {:.2E} ".format(epoch, calc_loss))
            self.decoders_dictionary['saver'].save(sess, weights_folder)



    def decode_phase_samples(self, num_samples:int, weights_folder:str):
        """
        This function takes a certain number of phase samples and produces decoded signals
        :param num_samples: number of phases sampled
        :param weights_folder: folder where the weights are saved
        :return: the function gives as an output the phases, the embedding of the phases in R^2 and the decoded signals
        """
        with tf.Session() as sess:
            self.decoders_dictionary['saver'].restore(sess, weights_folder)
            phases, samples = self.sample_phases(num_samples)
            decoded = sess.run([self.decoders_dictionary['x_r']],
                                      feed_dict={self.decoders_dictionary['z']: samples})
        return phases, samples, decoded[0]

    def decode_phases(self, phases, weights_folder:str):
        """
        This function takes a vector of phases and produces with the network decoded signals
        :param phases: vector with the phases to be used
        :param weights_folder: folder where the network weights are saved
        :return: decoded signals
        """
        with tf.Session() as sess:
            # Project the phase to the unit circle
            x_proj = np.cos(phases)
            y_proj = np.sin(phases)
            samples = np.concatenate([x_proj, y_proj], axis=-1)
            # Restore the weights
            self.decoders_dictionary['saver'].restore(sess, weights_folder)
            # Obtain the sinusoid signals
            decoded = sess.run([self.decoders_dictionary['x_r']],
                                            feed_dict={self.decoders_dictionary['z']: samples})
        return decoded[0]

