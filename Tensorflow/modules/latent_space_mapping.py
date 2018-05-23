import tensorflow as tf
import numpy as np


class vae_nn_mapping():
    def __init__(self, learning_rate, order: tuple, data_dimensions, non_linear_map=False):
        tf.reset_default_graph()
        assert len(order) == 2, "order needs to be a tuple of size 2"
        self.order = order
        self.data_dimensions = data_dimensions
        self.intermediate_dim = data_dimensions // 3
        self.names = ['HeatKernel', 'Normal', 'Human']
        self.learning_rate = learning_rate
        # Extra options
        self.non_linear_map = non_linear_map

        # Build the graph
        print("Mapping from {} to {}".format(self.names[order[0]], self.names[order[1]]))
        self.decoders_dictionary = self.build()

    def define_input(self):
        with tf.variable_scope("Input"):
            z = tf.placeholder(tf.float32, shape=[None, 2])
        return z

    def mapping(self, z):
        if self.non_linear_map:
            h_map = tf.layers.dense(z, self.intermediate_dim, activation=tf.nn.relu, name='h_map')
            l_map = tf.layers.dense(h_map, 2, activation=None, name='l_map')
        else:
            l_map = tf.layers.dense(z, 2, activation=None, name='l_map')
        return l_map

    def define_decoder(self, z, number):
        """
        Defines the subgraph of the decoder
        :param number:
        :param z: Code tensor
        :return: Reconstruction tensor x_r
        """
        with tf.variable_scope("decoder" + '_' + self.names[number]):
            h_d = tf.layers.dense(z, self.intermediate_dim, activation=tf.nn.relu,
                                  name="h_dec" + '_' + self.names[number], trainable=False)
            h_d2 = tf.layers.dense(h_d, self.intermediate_dim, activation=tf.nn.relu,
                                   name="h_dec2" + '_' + self.names[number], trainable=False)
            x_r = tf.layers.dense(h_d2, self.data_dimensions, activation=None, name="x_r" + '_' + self.names[number],
                                  trainable=False)
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

    def build(self):
        z = self.define_input()
        l_map = self.mapping(z)
        x_1 = self.define_decoder(z, self.order[0])
        x_2 = self.define_decoder(l_map, self.order[1])
        loss = self.define_loss(x_1, x_2)
        # Tensorboard summaries
        tf.summary.scalar('Loss', loss)
        tf.summary.histogram('l_map', l_map)
        summary_op = tf.summary.merge_all()
        # Saver
        saver = tf.train.Saver()
        # Optimizer
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        decoders_dictionary = {'z': z,
                               'l_map': l_map,
                               'x_1': x_1,
                               'x_2': x_2,
                               'loss': loss,
                               'summary_op': summary_op,
                               'saver': saver,
                               'train_step': train_step}
        return decoders_dictionary

    def sample_latent_space(self, num_samples: int):
        # Heat Kernel latent space
        if self.order[0] == 0:
            u_samples = np.random.uniform(-1.0, 1.0, size=(num_samples, 1))
            x_proj = np.cos(np.pi * u_samples)
            y_proj = np.sin(np.pi * u_samples)
            samples = np.concatenate([x_proj, y_proj], axis=-1)
        # Normal latent space
        elif self.order[0] == 1:
            samples = np.random.normal(size=(num_samples, 2))
        # Human latent space
        elif self.order[0] == 2:
            u_samples = np.random.uniform(-1.0, 1.0, size=(num_samples, 1))
            x_proj = np.cos(np.pi * u_samples)
            y_proj = np.sin(np.pi * u_samples)
            samples = np.concatenate([x_proj, y_proj], axis=-1)
        else:
            samples = None
        return samples

    def train(self, num_samples, epochs, log_dir_tensorboard, weights_folder):
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(log_dir_tensorboard, graph=sess.graph)
            sess.run(tf.global_variables_initializer())
            # Restore the decoder weights
            print("Restoring saved parameters")
            self.decoders_dictionary['saver'].restore(sess, weights_folder)
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

    def train_codes(self, codes, epochs, log_dir_tensorboard, weights_folder):
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(log_dir_tensorboard, graph=sess.graph)
            sess.run(tf.global_variables_initializer())
            # Restore the decoder weights
            print("Restoring saved parameters")
            self.decoders_dictionary['saver'].restore(sess, weights_folder)
            for epoch in range(epochs):
                feed_dict = {self.decoders_dictionary['z']: codes}
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
