import tensorflow as tf
import numpy as np
from os.path import isfile



class vae_nn():
    def __init__(self, input_dim, latent_dim, learning_rate):
        tf.reset_default_graph()
        # Initialize main values of the VAE
        self.input_dim = input_dim
        self.intermediate_dim = self.input_dim // 3
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        # Build the graph of the VAE
        self.vae_dictionary = self.build_vae()

    def define_encoder(self):
        """
        Defines the graph section for the encoder of the VAE
        :return: x, mu_z and log_var_z tensors
        """
        with tf.variable_scope("encoder"):
            x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name="input_data")
            h = tf.layers.dense(x, self.intermediate_dim, activation=tf.nn.relu, name="h_enc")
            mu_z = tf.layers.dense(h, self.latent_dim, activation=None, name="mu_z")
            log_var_z = tf.layers.dense(h, self.latent_dim, activation=None, name="log_var_z")
        return x, mu_z, log_var_z

    def define_sampling(self, log_var_z, mu_z):
        """
        Defines the subgraph of the latent space sampling
        :param log_var_z: log of the encoding distribution's variance
        :param mu_z: mean of the encoding distribution's variance
        :return: z tensor
        """
        with tf.variable_scope("sampling"):
            epsilon = tf.random_normal(tf.shape(log_var_z), name='epsilon')
            sigma_z = tf.exp(0.5 * log_var_z)
            z = tf.add(mu_z, tf.multiply(sigma_z, epsilon), name='z')
            return z

    def define_decoder(self, z):
        """
        Defines the subgraph of the decoder
        :param z: Code tensor
        :return: Reconstruction tensor x_r
        """
        with tf.variable_scope("decoder"):
            h_d = tf.layers.dense(z, self.intermediate_dim, activation=tf.nn.relu, name="h_dec")
            x_r = tf.layers.dense(h_d, self.input_dim, activation=None, name="x_r")
            return x_r

    def define_KL_divergence(self, mu_z, log_var_z):
        """
        Loss for the KL divergence w.r.t. normal distributions as encoder
        and decoder
        :param mu_z: tensor for the encoder mean
        :param log_var_z: tensor for the log_var of the encoder
        :return: Tensor with the KL_divergence
        """
        with tf.name_scope(name='loss_KL'):
            sigma_z = tf.exp(0.5 * log_var_z, name='sigma_z')
            KLD = tf.reduce_mean(-0.5 * (1 + log_var_z - tf.pow(mu_z, 2) - sigma_z), name='reduce_KL')
        return KLD

    def define_MSE(self, x, x_r):
        """
        Loss for the reconstruction error between x and x_r
        :param x: tensor of the input data
        :param x_r: tensor of the reconstructed data
        :return: tensor with the mean squared error loss
        """
        with tf.name_scope('loss_MSE'):
            MSE = tf.reduce_mean(
                tf.reduce_sum(
                    tf.losses.mean_squared_error(
                        labels=x,
                        predictions=x_r,
                        reduction=tf.losses.Reduction.NONE),
                    axis=1)
                , name='reduce_MSE')
        return MSE

    def define_loss(self, MSE, KLD):
        """
        Defines the total loss
        :param MSE: tensor for the MSE loss
        :param KLD: tensor for the KLD loss
        :return: tensor of the total loss
        """
        with tf.name_scope('loss_calculation'):
            loss = tf.add(MSE, KLD, name='loss')
        return loss

    def build_vae(self):
        """
        This method produces the complete graph for the VAE.
        Defines the corresponding summaries for relevant scalars
        :return: dictionary with the relevant tensors, summary, save
        """
        # Tensors
        x, mu_z, log_var_z = self.define_encoder()
        z = self.define_sampling(log_var_z, mu_z)
        x_r = self.define_decoder(z)
        # Loss functions
        MSE = self.define_MSE(x, x_r)
        KL = self.define_KL_divergence(mu_z, log_var_z)
        loss = self.define_loss(MSE, KL)
        # Summary of scalars
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('MSE', tf.reduce_mean(MSE))
        tf.summary.scalar('KL-Divergence', KL)
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
                          'x_r': x_r,
                          'MSE': MSE,
                          'KL': KL,
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
            summary_writer = tf.summary.FileWriter(log_dir_tensorboard, graph = sess.graph)
            if isfile(weights_folder):
                print("Restoring saved parameters")
                self.vae_dictionary['saver'].restore(sess, weights_folder)
            else:
                print("Initializing parameters")
                sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                feed_dict = {self.vae_dictionary['x']: train_data}
                # Training
                _, calc_loss, calc_KL, calc_MSE, summary = \
                sess.run([self.vae_dictionary['train_step'],
                          self.vae_dictionary['loss'],
                          self.vae_dictionary['KL'],
                          self.vae_dictionary['MSE'],
                          self.vae_dictionary['summary_op']
                          ], feed_dict = feed_dict)
                summary_writer.add_summary(summary, epoch)
                print("Epoch {} | Loss: {:.2E} | MSE: {:.2E} | KL: {:.2E}".format(epoch, calc_loss, np.mean(calc_MSE), np.mean(calc_KL)))
            self.vae_dictionary['saver'].save(sess, weights_folder)

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
                                  feed_dict={self.vae_dictionary['x']:data})
        return code[0]

    def decode(self, code, weight_folder):
        """
        Reconstructs data from a code in latent space
        :param code: array of code in latent space
        :param weight_folder: location of the weights for the network
        :return: np.array with reconstructions of data
        """
        with tf.Session() as sess:
            self.vae_dictionary['saver'].restore(sess, weight_folder)
            reconstructed = sess.run([self.vae_dictionary['x_r']],
                                  feed_dict = {self.vae_dictionary['z']:code})
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
            reconstructed = sess.run([self.vae_dictionary['x_r']],
                                     feed_dict={self.vae_dictionary['x']:data})
        return reconstructed[0]
