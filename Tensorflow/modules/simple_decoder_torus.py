import tensorflow as tf
import numpy as np
from os.path import isfile
from itertools import product


def sinusoid_image_phase_combination(phases1, phases2, n_T, omega_values):
    """
    This function produces an array where each row corresponds to a sinusoidal signal with a given phase and
    angular frequency omega. The columns represent the time sampling from the interval [0,1].
    :param phases: Vector with the phases to be used
    :param n_T: Number of elements in the partition of the interval [0,1]
    :param omega: Angular frequency
    :return: np.array with shape (len(phases),n_T)
    """

    # Sampling from phase and space
    space_linspace = np.linspace(0, 1, n_T)
    # Create all possible combinations of phi_1, phi_2
    phase_combinations = np.array(list(product(phases1, phases2)))
    sinusoid_images = np.zeros((n_T, n_T, len(phase_combinations)))

    # Create spatial mesh
    spatial_mesh = np.meshgrid(space_linspace, space_linspace)

    # Generate signals for each combination
    for num_mesh, mesh_dimension in enumerate(spatial_mesh):
        # Omega*dimension
        mesh_expanded_dim = omega_values[num_mesh] * mesh_dimension[:, :, np.newaxis]
        repeated_volume = np.repeat(mesh_expanded_dim, repeats=len(phase_combinations), axis=2)
        # sine(Omega*dimension+phase)
        sinusoid_images += np.sin(np.add(repeated_volume, phase_combinations[:, num_mesh]))
    sinusoid_images = np.swapaxes(sinusoid_images, 2, 0)
    return phase_combinations, sinusoid_images

def sinusoid_image_phase(phases1, phases2, n_T, omega_values):
    """
    This function produces an array where each row corresponds to a sinusoidal signal with a given phase and
    angular frequency omega. The columns represent the time sampling from the interval [0,1].
    :param phases: Vector with the phases to be used
    :param n_T: Number of elements in the partition of the interval [0,1]
    :param omega: Angular frequency
    :return: np.array with shape (len(phases),n_T)
    """

    # Sampling from phase and space
    space_linspace = np.linspace(0, 1, n_T)
    # Create all possible combinations of phi_1, phi_2
    phases1 = np.expand_dims(phases1, 1)
    phases2 = np.expand_dims(phases2, 1)
    phases = np.concatenate((phases1, phases2), axis=1)
    sinusoid_images = np.zeros((n_T, n_T, len(phases)))

    # Create spatial mesh
    spatial_mesh = np.meshgrid(space_linspace, space_linspace)

    # Generate signals for each combination
    for num_mesh, mesh_dimension in enumerate(spatial_mesh):
        # Omega*dimension
        mesh_expanded_dim = omega_values[num_mesh] * mesh_dimension[:, :, np.newaxis]
        repeated_volume = np.repeat(mesh_expanded_dim, repeats=len(phases), axis=2)
        # sine(Omega*dimension+phase)
        sinusoid_images += np.sin(np.add(repeated_volume, phases[:, num_mesh]))
    sinusoid_images = np.swapaxes(sinusoid_images, 2, 0)
    return phases, sinusoid_images


class simple_decoder():
    """
    Simple decoder neural network that is capable of taking a phase value and producing a sinusoidal signals after
    training.
    """
    def __init__(self, learning_rate, data_shape):
        tf.reset_default_graph()
        self.data_shape = data_shape
        self.intermediate_dim = np.product(data_shape)//3
        self.learning_rate = learning_rate
        # Build the graph
        self.decoders_dictionary = self.build()


    def define_input(self):
        """
        Defines the input tensor
        :return: Tensor z
        """
        with tf.variable_scope("Input"):
            z = tf.placeholder(tf.float32, shape=[None, 4])
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
            x_flat = tf.layers.dense(h_d2, np.product(self.data_shape), activation=None, name="x_r" + '_' + 'Human')
            x_r = tf.reshape(x_flat, [-1, self.data_shape[0], self.data_shape[1]])
            # Placeholder for the true signal
            real_signal = tf.placeholder(tf.float32, shape = [None, self.data_shape[0], self.data_shape[1]])
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
        This function samples uniformly a certain number of phases from [0,2pi]^2. It then embeds the phase into the
        thorus in R^4.
        :param num_samples: Number of phases to be sampled uniformly
        :return: Returnes the sampled phases and the embedded samples
        """
        u_samples1 = np.random.uniform(-1.0, 1.0, size = (num_samples))
        u_samples2 = np.random.uniform(-1.0, 1.0, size = (num_samples))
        phases1 = u_samples1*np.pi
        phases2 = u_samples2*np.pi
        torus = np.zeros((num_samples, 4))
        torus[:, 0] = 0.5 * np.cos(phases1)
        torus[:, 1] = 0.5 * np.sin(phases1)
        torus[:, 2] = 0.5 * np.cos(phases2)
        torus[:, 3] = 0.5 * np.sin(phases2)
        return phases1, phases2, torus

    def generate_torus_combinations(self, phases1, phases2):
        phase_combinations = np.array(list(product(phases1, phases2)))
        torus = np.zeros((len(phase_combinations), 4))
        torus[:, 0] = 0.5*np.cos(phase_combinations[:, 0])
        torus[:, 1] = 0.5*np.sin(phase_combinations[:, 0])
        torus[:, 2] = 0.5 * np.cos(phase_combinations[:, 1])
        torus[:, 3] = 0.5 * np.sin(phase_combinations[:, 1])
        return torus

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
            # FIx this
            phases1, phases2, samples = self.sample_phases(num_samples)
            phase_combinations, sinusoid_images = sinusoid_image_phase(phases1, phases2,
                                                   self.data_shape[0],
                                                   omega_values=[np.pi * 2, np.pi * 4])
            dataset = tf.data.Dataset.from_tensor_slices((samples, sinusoid_images)).shuffle(num_samples).batch(batch_size)
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
        return phases1, phases2, sinusoid_images


    def train_phases(self, phases1, phases2, epochs, log_dir_tensorboard, weights_folder):
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
            #Fix this
                sess.run(tf.global_variables_initializer())
            samples = self.generate_torus_combinations(phases1, phases2)
            _, sinusoid_images = sinusoid_image_phase_combination(phases1, phases2, self.data_shape[0], omega_values =[np.pi * 2, np.pi*4])
            for epoch in range(epochs):
                feed_dict = {self.decoders_dictionary['z']: samples, self.decoders_dictionary['real_signal']: sinusoid_images}
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
            phases1, phases2, samples = self.sample_phases(num_samples)
            decoded = sess.run([self.decoders_dictionary['x_r']],
                                      feed_dict={self.decoders_dictionary['z']: samples})
        return phases1, phases2, samples, decoded[0]

    def decode_phases(self, phases1, phases2, weights_folder:str):
        """
        This function takes a vector of phases and produces with the network decoded signals
        :param phases: vector with the phases to be used
        :param weights_folder: folder where the network weights are saved
        :return: decoded signals
        """
        with tf.Session() as sess:
            # Project the phase to the torus
            scaled1 = np.expand_dims(phases1, 1)
            scaled2 = np.expand_dims(phases2, 1)
            torus1 = 0.5 * np.cos(scaled1)
            torus2 = 0.5 * np.sin(scaled1)
            torus3 = 0.5 * np.cos(scaled2)
            torus4 = 0.5 * np.sin(scaled2)
            samples = np.concatenate([torus1, torus2, torus3, torus4], axis=-1)
            # Restore the weights
            self.decoders_dictionary['saver'].restore(sess, weights_folder)
            # Obtain the sinusoid signals
            decoded = sess.run([self.decoders_dictionary['x_r']],
                                            feed_dict={self.decoders_dictionary['z']: samples})
        return decoded[0]

