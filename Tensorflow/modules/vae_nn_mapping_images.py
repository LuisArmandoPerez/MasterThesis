import tensorflow as tf
import numpy as np
from os.path import isfile


class vae_nn_mapping():
    def __init__(self, shape, learning_rate, order):
        tf.reset_default_graph()
        # Initialize main values of the VAE
        self.shape = shape
        self.intermediate_dim = np.prod(self.shape) // 3
        self.learning_rate = learning_rate
        self.order = order
        self.names = ['Human', 'Normal']
        # Build the graph of the VAE
        print("Mapping from {} to {}".format(self.names[order[0]], self.names[order[1]]))
        self.decoders_dictionary = self.build_vae()

    def define_input(self):
        with tf.variable_scope("Input"):
            if self.order[0] == 0:
                z = tf.placeholder(tf.float32, shape=[None, 2])
            else:
                z = tf.placeholder(tf.float32, shape=[None, 4])
        return z

    def define_embedding(self, z, number):
        if number == 0:
            torus1 = 0.5 * tf.cos(np.pi * tf.expand_dims(z[:, 0], 1))
            torus2 = 0.5 * tf.sin(np.pi * tf.expand_dims(z[:, 0], 1))
            torus3 = 0.5 * tf.cos(np.pi * tf.expand_dims(z[:, 1], 1))
            torus4 = 0.5 * tf.sin(np.pi * tf.expand_dims(z[:, 1], 1))
            embedding = tf.concat((torus1, torus2, torus3, torus4), axis=-1)
        else:
            embedding = z
        return embedding

    def mapping(self, z):
        if self.order[1] == 0:
            l_map_h = tf.layers.dense(z, 2, activation=None, name='l_map_h')
            l_map = tf.mod(l_map_h, 2) - 1
        else:
            l_map = tf.layers.dense(z, 4, activation=None, name='l_map')
        return l_map

    def define_first_decoder(self, z_embedding, number):
        """
        Defines the subgraph of the decoder
        :param z_embedding: Code tensor
        :return: Reconstruction tensor x_r
        """
        with tf.variable_scope("decoder" + '_' + self.names[number]):
            h_d = tf.layers.dense(z_embedding, self.intermediate_dim, activation=tf.nn.relu,
                                  name="h_dec" + '_' + self.names[number], trainable=False)
            h_d2 = tf.layers.dense(h_d, self.intermediate_dim, activation=tf.nn.relu,
                                   name="h_dec2" + '_' + self.names[number], trainable=False)
            x_r_flat = tf.layers.dense(h_d2, np.prod(self.shape), activation=None,
                                       name="x_r" + '_' + self.names[number], trainable=False)
            x_r = tf.reshape(x_r_flat, [-1, self.shape[0], self.shape[1]], name='reshaped_output')
            return x_r

    def define_second_decoder(self, map_embedding, number):
        """
        Defines the subgraph of the decoder
        :param z: Code tensor
        :return: Reconstruction tensor x_r
        """
        with tf.variable_scope("decoder" + '_' + self.names[number]):
            h_d = tf.layers.dense(map_embedding, self.intermediate_dim, activation=tf.nn.relu,
                                  name="h_dec" + '_' + self.names[number], trainable=False)
            h_d2 = tf.layers.dense(h_d, self.intermediate_dim, activation=tf.nn.relu,
                                   name="h_dec2" + '_' + self.names[number], trainable=False)
            x_r_flat = tf.layers.dense(h_d2, np.prod(self.shape), activation=None,
                                       name="x_r" + '_' + self.names[number], trainable=False)
            x_r = tf.reshape(x_r_flat, [-1, self.shape[0], self.shape[1]], name='reshaped_output')
            return x_r

    def define_loss(self, x_1, x_2):
        with tf.name_scope('loss_calculation'):
            loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.losses.mean_squared_error(
                        labels=x_1,
                        predictions=x_2,
                        reduction=tf.losses.Reduction.NONE),
                    axis=1)
                , name='loss')
        return loss

    def build_vae(self):
        """
        This method produces the complete graph for the VAE.
        Defines the corresponding summaries for relevant scalars
        :return: dictionary with the relevant tensors, summary, save
        """
        # Tensors
        z = self.define_input()
        z_embedding = self.define_embedding(z, self.order[0])
        l_map = self.mapping(z)
        map_embedding = self.define_embedding(l_map, self.order[1])
        x_1 = self.define_first_decoder(z_embedding, self.order[0])
        x_2 = self.define_second_decoder(map_embedding, self.order[1])
        # Loss functions
        loss = self.define_loss(x_1, x_2)
        # Summary of scalars
        # Summary of scalars
        tf.summary.scalar('Loss', loss)
        tf.summary.histogram('l_map_hist', l_map)
        summary_op = tf.summary.merge_all()
        # Saver
        saver = tf.train.Saver()
        # Optimizer
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        decoders_dictionary = {'z': z,
                               'l_map': l_map,
                               'z_embedding': z_embedding,
                               'map_embedding': map_embedding,
                               'x_1': x_1,
                               'x_2': x_2,
                               'loss': loss,
                               'summary_op': summary_op,
                               'saver': saver,
                               'decoder_saver': saver,
                               'train_step': train_step}
        return decoders_dictionary

    def sample_latent_space(self, num_samples: int):
        if self.order[0] == 0:
            samples = np.random.uniform(-1.0, 1.0, (num_samples, 2))
        else:
            samples = np.random.normal(0.0, 1.0, (num_samples, 4))
        return samples

    def train(self, num_samples, epochs, log_dir_tensorboard, weights_folder):
        """
        Method for training the VAE network
        :param train_data: train data fed to the network
        :param epochs: number of epochs per training
        :param log_dir_tensorboard: folder directory to save tensorboard logs
        :param weights_folder: folder directory to save the weights
        :return:
        """
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(log_dir_tensorboard, graph=sess.graph)
            # if isfile(weights_folder):
            sess.run(tf.global_variables_initializer())
            print("Restoring saved parameters")
            self.decoders_dictionary['saver'].restore(sess, weights_folder)
            # else:
            #    print("Initializing parameters")
            #    sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                feed_dict = {self.decoders_dictionary['z']: self.sample_latent_space(num_samples)}
                # Training
                _, calc_loss, summary = \
                    sess.run([self.decoders_dictionary['train_step'],
                              self.decoders_dictionary['loss'],
                              self.decoders_dictionary['summary_op']
                              ], feed_dict=feed_dict)
                summary_writer.add_summary(summary, epoch)
                print("Epoch {} | Loss: {:.2E} ".format(epoch, calc_loss))
            self.decoders_dictionary['saver'].save(sess, weights_folder)

    def get_decoder_variables_names(self):
        decoder_variable_names = []
        decoder1_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope='decoder_' + self.names[self.order[0]])
        for variable in decoder1_variables:
            decoder_variable_names.append(variable.name)

        decoder2_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope='decoder_' + self.names[self.order[1]])
        for variable in decoder2_variables:
            decoder_variable_names.append(variable.name)

        return decoder_variable_names

    def assign_values(self, dictionary, saving_folder):
        """
        Assigns the weights to the variables in the current graph. The weights are obtained from the dictionary. The
        network is then saved for restoring it later.
        :param dictionary: Dictionary with the variable names and weights
        :param saving_folder: Path to the saved address
        :return:
        """
        print("Loading the weight of the variables ...")
        variables_names = self.get_decoder_variables_names()
        # List of the weight assignment operations
        assign_opers = []
        for name in variables_names:
            print("Loading the variable {}...\n".format(name))
            target_variable = [v for v in tf.global_variables() if v.name == name]
            assign_opers.append(target_variable[0].assign(dictionary[name]))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Run each of the weight assignment operations
            for assign_oper in assign_opers:
                sess.run(assign_oper)
            self.decoders_dictionary['saver'].save(sess, saving_folder)
        print("Finished loading variables.")

    def map_latent_space(self, num_samples, weights_folder):
        with tf.Session() as sess:
            self.decoders_dictionary['saver'].restore(sess, weights_folder)
            samples = self.sample_latent_space(num_samples)
            mapped_samples = sess.run([self.decoders_dictionary['l_map']],
                                      feed_dict={self.decoders_dictionary['z']: samples})
        return samples, mapped_samples[0]

    def map_latent_space_embedded(self, num_samples, weights_folder):
        with tf.Session() as sess:
            self.decoders_dictionary['saver'].restore(sess, weights_folder)
            samples = self.sample_latent_space(num_samples)
            mapped_samples = sess.run([self.decoders_dictionary['map_embedding']],
                                      feed_dict={self.decoders_dictionary['z']: samples})
        return samples, mapped_samples[0]

    def decode_latent_space_samples(self, num_samples, weights_folder):
        with tf.Session() as sess:
            self.decoders_dictionary['saver'].restore(sess, weights_folder)
            samples = self.sample_latent_space(num_samples)
            decoded_1, decoded_2 = sess.run([self.decoders_dictionary['x_1'], self.decoders_dictionary['x_2']],
                                            feed_dict={self.decoders_dictionary['z']: samples})
        return samples, decoded_1, decoded_2

    def decode_codes(self, codes, weights_folder):
        with tf.Session() as sess:
            self.decoders_dictionary['saver'].restore(sess, weights_folder)
            decoded_1, decoded_2 = sess.run([self.decoders_dictionary['x_1'], self.decoders_dictionary['x_2']],
                                            feed_dict={self.decoders_dictionary['z']: codes})
        return decoded_1, decoded_2

    def evaluate_latent_space_samples(self, num_samples, weights_folder):
        with tf.Session() as sess:
            self.decoders_dictionary['saver'].restore(sess, weights_folder)
            samples = self.sample_latent_space(num_samples)
            loss = sess.run([self.decoders_dictionary['loss']], feed_dict={self.decoders_dictionary['z']: samples})
        return loss


    def autoencode(self, data, weight_folder):
        """
        Takes data and produces similar reconstructed data
        :param data: array of data
        :param weight_folder: location of the weights for the network
        :return: np.array with reconstructions of data
        """
        with tf.Session() as sess:
            self.decoders_dictionary['saver'].restore(sess, weight_folder)
            reconstructed1 = sess.run([self.decoders_dictionary['x_r_1']],
                                      feed_dict={self.decoders_dictionary['x']: data})
            reconstructed2 = sess.run([self.decoders_dictionary['x_r_2']],
                                      feed_dict={self.decoders_dictionary['x']: data})
        return reconstructed1[0], reconstructed2[0]

    def map(self, data, weight_folder):
        with tf.Session() as sess:
            self.decoders_dictionary['saver'].restore(sess, weight_folder)
            mapped = sess.run([self.decoders_dictionary['l_map']],
                              feed_dict={self.decoders_dictionary['x']: data})
        return mapped[0]
