import tensorflow as tf
import numpy as np
from os.path import isfile


class vae_nn_mapping():
    def __init__(self, input_dim, learning_rate, log_t, order):
        tf.reset_default_graph()
        # Initialize main values of the VAE

        self.input_dim = input_dim
        self.intermediate_dim = self.input_dim // 3
        if order[0]==0:
            self.latent_dim = 1
        else:
            self.latent_dim = 2
        self.learning_rate = learning_rate
        self.log_t = log_t
        self.order = order
        self.names = ['HeatKernel', 'Normal']
        # Build the graph of the VAE
        self.vae_dictionary = self.build_vae()

    def define_encoder(self, number):
        """
        Defines the graph section for the encoder of the VAE
        :return: x, mu_z and log_var_z tensors
        """
        with tf.variable_scope("encoder" + '_' + self.names[number]):
            x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name="input_data")
            h = tf.layers.dense(x, self.intermediate_dim, activation=tf.nn.relu,
                                name="h_enc" + '_' + self.names[number], trainable=False)
            h2 = tf.layers.dense(h, self.intermediate_dim, activation=tf.nn.relu,
                                 name="h_enc2" + '_' + self.names[number], trainable=False)
            mu_z = tf.layers.dense(h2, self.latent_dim, activation=None, name="mu_z" + '_' + self.names[number], trainable=False)
            if self.order[0]==0:
                x_projection = tf.cos(tf.scalar_mul(np.pi, mu_z))
                y_projection = tf.sin(tf.scalar_mul(np.pi, mu_z))
                mu_z = tf.concat([x_projection, y_projection], axis=-1, name='z' + '_' + self.names[number])

            log_var_z = tf.layers.dense(h2, self.latent_dim, activation=None,
                                        name="log_var_z" + '_' + self.names[number], trainable=False)
        return x, mu_z, log_var_z

    def define_sampling(self, log_var_z, mu_z, number):
        """
        Defines the subgraph of the latent space sampling
        :param number:
        :param log_var_z: log of the encoding distribution's variance
        :param mu_z: mean of the encoding distribution's variance
        :return: z tensor
        """
        if number == 1:
            with tf.variable_scope("sampling" + '_' + self.names[number]):
                epsilon = tf.random_normal(tf.shape(log_var_z), name='epsilon' + '_' + self.names[number])
                sigma_z = tf.exp(0.5 * log_var_z, name='sigma_z' + '_' + self.names[number])
                z = tf.add(mu_z, tf.multiply(sigma_z, epsilon), name='z' + '_' + self.names[number])
                return z
        elif number == 0:
            with tf.variable_scope("sampling" + '_' + self.names[number]):
                epsilon = tf.random_normal(tf.shape(log_var_z), name='epsilon' + '_' + self.names[number])
                z = tf.add(np.float32(-1), tf.mod(mu_z + 1 + tf.scalar_mul(np.exp(self.log_t), epsilon), 2))
                x_projection = tf.cos(tf.scalar_mul(np.pi, z))
                y_projection = tf.sin(tf.scalar_mul(np.pi, z))
                z = tf.concat([x_projection, y_projection], axis=-1, name='z' + '_' + self.names[number])
                return z

    def define_mapping(self, mu_z):
        #h_map = tf.layers.dense(mu_z, self.intermediate_dim, activation=tf.nn.relu, name = 'h_map')
        #l_map = tf.layers.dense(h_map, 2, activation=None, name="l_map")
        l_map = tf.layers.dense(mu_z, 2, activation=None, name="l_map")
        return l_map

    def define_first_decoder(self, z, number):
        """
        Defines the subgraph of the decoder
        :param z: Code tensor
        :return: Reconstruction tensor x_r
        """
        with tf.variable_scope("decoder"+'_'+self.names[number]):
            h_d = tf.layers.dense(z, self.intermediate_dim, activation=tf.nn.relu, name="h_dec"+'_'+self.names[number], trainable=False)
            h_d2 = tf.layers.dense(h_d, self.intermediate_dim, activation=tf.nn.relu, name="h_dec2"+'_'+self.names[number], trainable=False)
            x_r = tf.layers.dense(h_d2, self.input_dim, activation=None, name="x_r"+'_'+self.names[number], trainable=False)
            return x_r
    def define_second_decoder(self, l_map, number):
        """
        Defines the subgraph of the decoder
        :param z: Code tensor
        :return: Reconstruction tensor x_r
        """
        with tf.variable_scope("decoder"+'_'+self.names[number]):
            h_d = tf.layers.dense(l_map, self.intermediate_dim, activation=tf.nn.relu, name="h_dec"+'_'+self.names[number], trainable=False)
            h_d2 = tf.layers.dense(h_d, self.intermediate_dim, activation=tf.nn.relu, name="h_dec2"+'_'+self.names[number], trainable=False)
            x_r = tf.layers.dense(h_d2, self.input_dim, activation=None, name="x_r"+'_'+self.names[number], trainable=False)
            return x_r


    def define_loss(self, x_r_1, x_r_2):
        with tf.name_scope('loss_calculation'):
            loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.losses.mean_squared_error(
                        labels=x_r_1,
                        predictions=x_r_2,
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
        x, mu_z, log_var_z = self.define_encoder(self.order[0])
        z = self.define_sampling(log_var_z, mu_z, self.order[0])
        l_map = self.define_mapping(mu_z)
        x_r_1 = self.define_first_decoder(mu_z, self.order[0])
        x_r_2 = self.define_second_decoder(l_map, self.order[1])
        # Loss functions
        loss = self.define_loss(x_r_1, x_r_2)
        # Summary of scalars
        tf.summary.scalar('Loss', loss)
        tf.summary.histogram('mapping', l_map)
        tf.summary.histogram('z_mean_hist', mu_z)
        tf.summary.histogram('log_z_sigma', log_var_z)
        summary_op = tf.summary.merge_all()
        # Saver
        saver = tf.train.Saver()
        # Optimizer
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        vae_dictionary = {'x': x,
                          'mu_z': mu_z,
                          'log_var_z': log_var_z,
                          'z': z,
                          'x_r_1': x_r_1,
                          'x_r_2': x_r_2,
                          'l_map': l_map,
                          'loss': loss,
                          'summary_op': summary_op,
                          'saver': saver,
                          'train_step': train_step}
        return vae_dictionary

    def train_vae(self, train_data, epochs, log_dir_tensorboard, weights_folder):
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
            #if isfile(weights_folder):
            sess.run(tf.global_variables_initializer())
            print("Restoring saved parameters")
            self.vae_dictionary['saver'].restore(sess, weights_folder)
            #else:
            #    print("Initializing parameters")
            #    sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                feed_dict = {self.vae_dictionary['x']: train_data}
                # Training
                _, calc_loss, summary = \
                    sess.run([self.vae_dictionary['train_step'],
                              self.vae_dictionary['loss'],
                              self.vae_dictionary['summary_op']
                              ], feed_dict=feed_dict)
                summary_writer.add_summary(summary, epoch)
                print("Epoch {} | Loss: {:.2E} ".format(epoch, calc_loss))
            self.vae_dictionary['saver'].save(sess, weights_folder)

    def get_variables_names(self):
        encoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_'+ self.names[self.order[0]])
        decoder_first_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,  scope='decoder_'+ self.names[self.order[0]])
        decoder_second_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,  scope='decoder_'+ self.names[self.order[1]])
        names = []
        for encoder_variable in encoder_variables:
            names.append(encoder_variable.name)
        for decoder_first_variable in decoder_first_variables:
            names.append(decoder_first_variable.name)
        for decoder_second_variable in decoder_second_variables:
            names.append(decoder_second_variable.name)
        return names

    def assign_values(self, dictionary, saving_folder):
        names = self.get_variables_names()
        assign_opers = []
        for name in names:
            print("Loading the variable {}...\n".format(name))
            target_variable = [v for v in tf.global_variables() if v.name == name]
            assign_opers.append(target_variable[0].assign(dictionary[name]))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for assign_oper in assign_opers:
                sess.run(assign_oper)
            self.vae_dictionary['saver'].save(sess, saving_folder)
            


    # OUTPUT GENERATING METHODS
    def encode(self, data, weight_folder):
        """
        Takes data an returns the mean of the encoding distribution for that
        data
        :param data: input data to encode
        :param weight_folder: location of the weights for the network
        :return: np.array with codes
        """
        with tf.Session() as sess:
            self.vae_dictionary['saver'].restore(sess, weight_folder)
            code = sess.run([self.vae_dictionary['mu_z']],
                            feed_dict={self.vae_dictionary['x']: data})[0]
            #if self.order[0] == 0:
            #    code_x = np.cos(np.pi * code)
            #    code_y = np.sin(np.pi * code)
            #    code = np.concatenate([code_x, code_y], axis=-1)
        return code

    def decode_1(self, code, weight_folder):
        """
        Reconstructs data from a code in latent space
        :param code: array of code in latent space
        :param weight_folder: location of the weights for the network
        :return: np.array with reconstructions of data
        """
        with tf.Session() as sess:
            self.vae_dictionary['saver'].restore(sess, weight_folder)
            reconstructed = sess.run([self.vae_dictionary['x_r_1']],
                                     feed_dict={self.vae_dictionary['mu_z']: code})
        return reconstructed[0]

    def decode_2(self, code, weight_folder):
        """
        Reconstructs data from a code in latent space
        :param code: array of code in latent space
        :param weight_folder: location of the weights for the network
        :return: np.array with reconstructions of data
        """
        with tf.Session() as sess:
            self.vae_dictionary['saver'].restore(sess, weight_folder)
            reconstructed = sess.run([self.vae_dictionary['x_r_2']],
                                       feed_dict={self.vae_dictionary['l_map']: code})
        return reconstructed[0]

    def autoencode(self, data, weight_folder):
        """
        Takes data and produces similar reconstructed data
        :param data: array of data
        :param weight_folder: location of the weights for the network
        :return: np.array with reconstructions of data
        """
        with tf.Session() as sess:
            self.vae_dictionary['saver'].restore(sess, weight_folder)
            reconstructed1 = sess.run([self.vae_dictionary['x_r_1']],
                                     feed_dict={self.vae_dictionary['x']: data})
            reconstructed2 = sess.run([self.vae_dictionary['x_r_2']],
                                     feed_dict={self.vae_dictionary['x']: data})
        return reconstructed1[0], reconstructed2[0]

    def map(self, data, weight_folder):
        with tf.Session() as sess:
            self.vae_dictionary['saver'].restore(sess, weight_folder)
            mapped = sess.run([self.vae_dictionary['l_map']],
                              feed_dict={self.vae_dictionary['x']: data})
        return mapped[0]



