import tensorflow as tf
import numpy as np
from os.path import isfile


class vae_nn():
    def __init__(self, input_dim, learning_rate, mode = {'encoder':'Normal'}):
        tf.reset_default_graph()
        # Initialize main values of the VAE

        self.input_dim = input_dim
        self.intermediate_dim = self.input_dim // 3
        self.latent_dim = [2,1]
        self.learning_rate = learning_rate
        self.mode = mode
        self.names = ['normal','heatkernel']
        # Build the graph of the VAE
        self.vae_dictionary = self.build_vae()

    def define_encoder(self):
        """
        Defines the graph section for the encoder of the VAE
        :return: x, mu_z and log_var_z tensors
        """
        x = []
        h = []
        h2 = []
        mu_z = []
        log_var_z = []
        for num_name, name in enumerate(self.names):
            with tf.variable_scope("encoder"+"_"+name):
                x.append(tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name="input"+'_'+name))
                h.append(tf.layers.dense(x, self.intermediate_dim, activation=tf.nn.relu, name="h_enc"+'_'+name))
                h2.append(tf.layers.dense(h, self.intermediate_dim, activation=tf.nn.relu, name="h_enc2"+'_'+name))
                mu_z.append(tf.layers.dense(h2, self.latent_dim[num_name], activation=None, name="mu_z"+'_'+name))
                log_var_z.append(tf.layers.dense(h2, self.latent_dim[num_name], activation=None, name="log_var_z"+'_'+name))
        return x, mu_z, log_var_z



    def define_sampling(self, log_var_z, mu_z):
        """
        Defines the subgraph of the latent space sampling
        :param log_var_z: log of the encoding distribution's variance
        :param mu_z: mean of the encoding distribution's variance
        :return: z tensor
        """
        epsilon = []
        sigma_z = []
        z = []
        for num_name, name in enumerate(self.names):
            with tf.variable_scope("sampling"+"_"+name):
                epsilon.append(tf.random_normal(tf.shape(log_var_z[num_name]), name='epsilon'+'_'+name))
                sigma_z.append(tf.exp(0.5 * log_var_z[num_name], name='sigma_z'+'_'+name))
                if name == 'normal':
                    z.append(tf.add(mu_z[num_name], tf.multiply(sigma_z[num_name], epsilon[num_name]), name='z'+'_'+name))
                elif name == 'heatkernel':
                    z.append(tf. add(-1, tf.mod(mu_z + 1 + tf.scalar_mul(np.exp(self.mode['log_t']), epsilon), 2),name='z'+'_'+name))
        return z


    def define_decoder(self, z):
        """
        Defines the subgraph of the decoder
        :param z: Code tensor
        :return: Reconstruction tensor x_r
        """
        h_d = []
        h_d2 = []
        x_r = []
        for num_name, name in enumerate(self.names):
            with tf.variable_scope("decoder"+"_"+ name):
                if name == 'heatkernel':
                    with tf.variable_scope("projection"+'_'+name):
                        x_projection = tf.cos(tf.scalar_mul(np.pi, z[num_name]))
                        y_projection = tf.sin(tf.scalar_mul(np.pi, z[num_name]))
                        z[num_name] = tf.concat([x_projection, y_projection], axis = -1)
                h_d.append(tf.layers.dense(z[num_name], self.intermediate_dim, activation=tf.nn.relu, name="h_dec"+'_'+name))
                h_d2.append(tf.layers.dense(h_d[num_name], self.intermediate_dim, activation=tf.nn.relu, name="h_dec2"+'_'+name))
                x_r.append(tf.layers.dense(h_d2[num_name], self.input_dim, activation=None, name="x_r"+'_'+name))
        return x_r

    def define_KL_divergence(self, mu_z, z, log_var_z):
        """
        Loss for the KL divergence w.r.t. normal distributions as encoder
        and decoder
        :param mu_z: tensor for the encoder mean
        :param log_var_z: tensor for the log_var of the encoder
        :return: Tensor with the KL_divergence
        """
        KLD = []
        sigma_z = []
        for num_name, name in enumerate(self.names):
            if name == 'normal':
                with tf.name_scope(name='loss_KL'+'_'+name):
                    sigma_z.append(tf.exp(0.5 * log_var_z[num_name], name='sigma_z'+'_'+name))
                    KLD = tf.reduce_mean(-0.5 * (1 + log_var_z[num_name]- tf.pow(mu_z[num_name], 2) - sigma_z[num_name]), name='reduce_KL'+'_'+name)
            if name == 'heatkernel':
                with tf.name_scope(name='loss_KL'+'_'+name):
                    # Range for the sum of the Fourier expansion
                    sum_range = tf.range(1, self.mode['N'] + 1, dtype=np.float32, name='sum_range')
                    sum_range_t = tf.reshape(sum_range, [self.latent_dim[num_name], self.mode['N']])
                    # Fourier sum terms
                    position = np.pi* (mu_z[num_name] - z[num_name])  # Position
                    cosine = tf.cos(tf.matmul(position, sum_range_t), name='cosine')
                    # Temporal term
                    quadratic_term = tf.pow(np.pi*sum_range, 2, name='quadratic_term')
                    exponent = tf.scalar_mul(-tf.exp(self.mode['log_t']), quadratic_term)
                    exponential = tf.exp(exponent, name='exponential')
                    # Series sum from n = 1 to N
                    summand = tf.multiply(cosine, exponential, name='summand')
                    KLD.append(tf.reduce_mean(tf.log(tf.abs((1 / 2) + tf.reduce_sum(summand, axis=1))), name='reduce_KL'+'_'+name))
        return KLD



    def define_MSE(self, x, x_r):
        """
        Loss for the reconstruction error between x and x_r
        :param x: tensor of the input data
        :param x_r: tensor of the reconstructed data
        :return: tensor with the mean squared error loss
        """
        MSE = []
        for num_name, name in enumerate(self.names):
            with tf.name_scope('loss_MSE'+'_'+name):
                MSE.append(tf.reduce_mean(
                    tf.reduce_sum(
                        tf.losses.mean_squared_error(
                            labels=x[num_name],
                            predictions=x_r[num_name],
                            reduction=tf.losses.Reduction.NONE),
                        axis=1)
                    , name='reduce_MSE'+"_"+name))
        return MSE

    def define_loss(self, MSE, KLD):
        """
        Defines the total loss
        :param MSE: tensor for the MSE loss
        :param KLD: tensor for the KLD loss
        :return: tensor of the total loss
        """
        loss_partial = []
        for num_name, name in enumerate(self.names):
            with tf.name_scope('loss_calculation'+'_'+name):
                loss_partial.append(tf.add(MSE[num_name], KLD[num_name], name='loss'+'_'+name))
        loss = tf.add(loss_partial[0],loss_partial[1],name = 'loss')
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
        KL = self.define_KL_divergence(mu_z, z, log_var_z)
        loss = self.define_loss(MSE, KL)
        # Summary of scalars
        for num_name, name in enumerate(self.names):
            tf.summary.scalar('MSE'+'_'+name, MSE[num_name])
            tf.summary.scalar('KL-Divergence'+'_'+name, KL[num_name])
            tf.summary.histogram('z_mean_hist'+'_'+name, mu_z[num_name])
            tf.summary.histogram('log_z_sigma'+'_'+name, log_var_z[num_name])
        tf.summary.scalar('Loss', loss)
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
            summary_writer = tf.summary.FileWriter(log_dir_tensorboard, graph=sess.graph)
            if isfile(weights_folder):
                print("Restoring saved parameters")
                self.vae_dictionary['saver'].restore(sess, weights_folder)
            else:
                print("Initializing parameters")
                sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                feed_dict = {self.vae_dictionary['x'+'_'+'normal']: train_data,
                             self.vae_dictionary['x' + '_' + 'heatkernel']: train_data}
                # Training
                _, calc_loss, calc_KL_norm, calc_MSE_norm,calc_KL_HK, calc_MSE_HK, summary = \
                    sess.run([self.vae_dictionary['train_step'],
                              self.vae_dictionary['loss'],
                              self.vae_dictionary['KL'][0],
                              self.vae_dictionary['MSE'][0],
                              self.vae_dictionary['KL'][1],
                              self.vae_dictionary['MSE'][1],
                              self.vae_dictionary['summary_op']
                              ], feed_dict=feed_dict)
                summary_writer.add_summary(summary, epoch)
                print("Epoch {} | Loss: {:.2E} | MSE: {:.2E} | KL: {:.2E}".format(epoch, calc_loss, np.mean(calc_MSE_norm),
                                                                                  np.mean(calc_KL_norm),np.mean(calc_MSE_HK),
                                                                                  np.mean(calc_KL_HK)))
            self.vae_dictionary['saver'].save(sess, weights_folder, write_meta_graph=True)

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
            if self.mode['encoder'] == "HeatKernel" and self.mode['projection']:
                code_x = np.cos(np.pi*code)
                code_y = np.sin(np.pi*code)
                code = np.concatenate([code_x, code_y], axis = -1)
        return code

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
                                     feed_dict={self.vae_dictionary['z']: code})
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
                                     feed_dict={self.vae_dictionary['x']: data})
        return reconstructed[0]

    def extract_weights(self, weight_folder):
        with tf.Session() as sess:
            self.vae_dictionary['saver'].restore(sess, weight_folder)
            variables = tf.trainable_variables()
        return code

