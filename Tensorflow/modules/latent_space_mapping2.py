import tensorflow as tf
import numpy as np
import os


class vae_nn_mapping():
    def __init__(self, learning_rate, order: tuple, data_dimensions):
        tf.reset_default_graph()
        assert len(order) == 2, "order needs to be a tuple of size 2"
        self.order = order
        self.data_dimensions = data_dimensions
        self.intermediate_dim = data_dimensions // 3
        self.names = ['HeatKernel', 'Normal', 'Human']
        self.learning_rate = learning_rate

        # Build the graph
        print("Mapping from {} to {}".format(self.names[order[0]], self.names[order[1]]))
        self.decoders_dictionary = self.build()

    def define_input(self):
        with tf.variable_scope("Input"):
            if self.order[0] == 0 or self.order[0] == 2:
                z = tf.placeholder(tf.float32, shape=[None, 1])
            else:
                z = tf.placeholder(tf.float32, shape=[None, 2])
        return z

    def define_embedding(self, z, number):
        if number == 0 or number == 2:
            x_proj = tf.cos(np.pi * z)
            y_proj = tf.sin(np.pi * z)
            embedding = tf.concat([x_proj, y_proj], axis=-1)
        else:
            embedding = z
        return embedding

    def mapping(self, z):
        with tf.name_scope("mapping"):
            # From H->HK or HK->H
            if (self.order[1] == 0 or self.order[1] == 2) and (self.order[0] == 0 or self.order[0] == 2):
                bias = tf.Variable(tf.random_uniform([1], -1.0, 1.0, tf.float32), name='bias')
                l_map = tf.mod(-z + bias, 2) - 1
            # From N->H/HK
            elif self.order[0] == 1 and (self.order[1] == 0 or self.order[1] == 2):
                angle = tf.expand_dims(tf.atan2(z[:, 1], z[:, 0], name='angle')/np.pi, axis = -1)
                bias = tf.Variable(tf.random_uniform([1], -1.0, 1.0, tf.float32), name='bias')
                if self.order[1] == 0:
                    l_map = -angle+bias
                else:
                    l_map = angle+bias
            # From H/HK->N
            else:
                x_proj = tf.cos(np.pi * z)
                y_proj = tf.sin(np.pi * z)
                embedding = tf.concat([x_proj, y_proj], axis=-1)
                l_map = tf.layers.dense(embedding, 2, activation=None, name='l_map')
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
        z_embedding = self.define_embedding(z, self.order[0])
        l_map = self.mapping(z)
        map_embedding = self.define_embedding(l_map, self.order[1])
        x_1 = self.define_decoder(z_embedding, self.order[0])
        x_2 = self.define_decoder(map_embedding, self.order[1])
        loss = self.define_loss(x_1, x_2)
        # Tensorboard summaries
        tf.summary.scalar('Loss', loss)
        tf.summary.histogram('L_map', l_map)
        # tf.summary.histogram('bias', tf.get_default_graph().get_tensor_by_name('bias_l_map'))
        # Summary of the mapping variables
        if (self.order[1] == 0 or self.order[1] == 2) and (self.order[0] == 0 or self.order[0] == 2):
            tf.summary.histogram('map_bias', tf.get_default_graph().get_tensor_by_name('mapping/bias:0'))
        summary_op = tf.summary.merge_all()
        # Saver
        saver = tf.train.Saver()
        # Optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_step = optimizer.minimize(loss)
        gradient = optimizer.compute_gradients(loss, aggregation_method=tf.AggregationMethod.ADD_N)
        decoders_dictionary = {'z': z,
                               'l_map': l_map,
                               'z_embedding': z_embedding,
                               'map_embedding': map_embedding,
                               'x_1': x_1,
                               'x_2': x_2,
                               'loss': loss,
                               'summary_op': summary_op,
                               'saver': saver,
                               'train_step': train_step,
                               'gradient': gradient}
        return decoders_dictionary

    def sample_latent_space(self, num_samples: int):
        # Heat Kernel latent space
        if self.order[0] == 0:
            samples = np.random.uniform(-1.0, 1.0, size=(num_samples, 1))
        # Normal latent space
        elif self.order[0] == 1:
            samples = np.random.normal(size=(num_samples, 2))
        # Human latent space
        elif self.order[0] == 2:
            samples = np.random.uniform(-1.0, 1.0, size=(num_samples, 1))
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
                _, calc_loss, summary, gradient = \
                    sess.run([self.decoders_dictionary['train_step'],
                              self.decoders_dictionary['loss'],
                              self.decoders_dictionary['summary_op'],
                              self.decoders_dictionary['gradient']
                              ], feed_dict=feed_dict)
                summary_writer.add_summary(summary, epoch)
                print("Epoch {} | Loss: {:.2E} | Gradient: {}".format(epoch, calc_loss, gradient[0][0]))
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

    def map_codes(self, codes, weights_folder):
        with tf.Session() as sess:
            self.decoders_dictionary['saver'].restore(sess, weights_folder)
            mapped_codes = sess.run([self.decoders_dictionary['l_map']],
                                    feed_dict={self.decoders_dictionary['z']: codes})
        return mapped_codes[0]

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
