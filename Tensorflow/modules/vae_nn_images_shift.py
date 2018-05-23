from os.path import isfile
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
import resources.image_utils


class vae_nn_images():
    def __init__(self, shape, latent_dim, learning_rate, mode={'encoder': 'Normal'}):
        tf.reset_default_graph()
        # Initialize main values of the VAE

        self.shape = shape
        self.intermediate_dim = np.prod(self.shape) // 3
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.mode = mode

        # Build the graph of the VAE
        self.vae_dictionary = self.build_vae()

    def define_encoder(self):
        """
        Defines the graph section for the encoder of the VAE
        :return: x, mu_z and log_var_z tensors
        """
        with tf.variable_scope("encoder"):
            x = tf.placeholder(dtype=tf.float32, shape=[None, self.shape[0], self.shape[1]], name="input_data")
            x_flat = tf.layers.Flatten()(x)
            # Hidden layers
            h = tf.layers.dense(x_flat, self.intermediate_dim, activation=tf.nn.relu, name="h_enc")
            h2 = tf.layers.dense(h, self.intermediate_dim, activation=tf.nn.relu, name="h2_enc")
            mu_z = tf.layers.dense(h2, self.latent_dim, activation=None, name="mu_z")
            log_var_z = tf.layers.dense(h2, self.latent_dim, activation=None, name="log_var_z")
        return x, mu_z, log_var_z

    def define_sampling(self, log_var_z, mu_z):
        """
        Defines the subgraph of the latent space sampling
        :param log_var_z: log of the encoding distribution's variance
        :param mu_z: mean of the encoding distribution's variance
        :return: z tensor
        """
        if self.mode['encoder'] == 'Normal':
            with tf.variable_scope("sampling" + '_' + self.mode['encoder']):
                epsilon = tf.random_normal(tf.shape(log_var_z), name='epsilon' + '_' + self.mode['encoder'])
                sigma_z = tf.exp(0.5 * log_var_z, name='sigma_z' + '_' + self.mode['encoder'])
                # Fixed sigma
                z = tf.add(mu_z, tf.multiply(sigma_z, epsilon), name='z' + '_' + self.mode['encoder'])
                # z = tf.add(mu_z, tf.scalar_mul(tf.exp(0.5 * -10), epsilon), name='z' + '_' + self.mode['encoder'])
                return z

    def define_decoder(self, z):
        """
        Defines the subgraph of the decoder
        :param z: Code tensor
        :return: Reconstruction tensor x_r
        """
        with tf.variable_scope("decoder" + '_' + self.mode['encoder']):
            h_d = tf.layers.dense(z, self.intermediate_dim, activation=tf.nn.relu,
                                  name="h_dec" + '_' + self.mode['encoder'])
            h_d2 = tf.layers.dense(h_d, self.intermediate_dim, activation=tf.nn.relu,
                                   name="h_dec2" + '_' + self.mode['encoder'])
            x_r_flat = tf.layers.dense(h_d2, np.prod(self.shape), activation=None,
                                       name="x_r" + '_' + self.mode['encoder'])
            x_r = tf.reshape(x_r_flat, [-1, self.shape[0], self.shape[1]], name='reshaped_output')
        # Identify the decoder variables to be saved
        decoder_variables = [v for v in tf.trainable_variables() if v.name.find('decoder') != -1]
        decoder_saver = tf.train.Saver(decoder_variables)
        return x_r, decoder_saver

    def define_KL_divergence(self, mu_z, log_var_z):
        """
        Loss for the KL divergence w.r.t. normal distributions as encoder
        and decoder
        :param mu_z: tensor for the encoder mean
        :param log_var_z: tensor for the log_var of the encoder
        :return: Tensor with the KL_divergence
        """
        if self.mode['encoder'] == 'Normal':
            print('Normal distribution chosen as encoder')
            with tf.name_scope(name='loss_KL'):
                sigma_z = tf.exp(log_var_z, name='sigma_z')
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
        x_r, decoder_saver = self.define_decoder(z)
        # Loss functions
        MSE = self.define_MSE(x, x_r)
        KL = self.define_KL_divergence(mu_z, log_var_z)
        loss = self.define_loss(MSE, KL)
        # Summary of scalars
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('MSE', MSE)
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
                          'decoder_saver': decoder_saver,
                          'train_step': train_step}
        return vae_dictionary

    def train_shift(self, image: np.ndarray, axis: int, batch_size: int, epochs: int, log_dir_tensorboard: str,
                    weights_folder: str, shuffle: bool = True):
        assert np.amax(image)<=1.0, "The maximum pixel value is {}. Normalize it to value 1".format(np.amax(image))
        # Define the dataset class and iterator
        generator = resources.image_utils.create_circular_shift_generator(image, axis, batch_size, shuffle)
        dataset = tf.data.Dataset.from_generator(generator, (tf.uint8))
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        batches_per_epoch = image.shape[axis]//batch_size
        batch_MSE = np.zeros(batches_per_epoch)
        batch_KL = np.zeros(batches_per_epoch)
        batch_loss = np.zeros(batches_per_epoch)
        # Start the tensorflow session
        with tf.Session() as sess:
            # Summary writer for tensorboard
            summary_writer = tf.summary.FileWriter(log_dir_tensorboard, graph=sess.graph)

            if isfile(weights_folder):
                print("Restoring saved parameters")
                self.vae_dictionary['saver'].restore(sess, weights_folder)
            else:
                print("Initializing parameters")
                sess.run(tf.global_variables_initializer())
            # TRAINING
            for epoch in range(epochs):
                sess.run(iterator.initializer)  # Initialize the data iterator

                # BATCHES
                for batch in range(batches_per_epoch):
                    # For testing the batching
                    image_batch = sess.run(next_element)
                    feed_dict = {self.vae_dictionary['x']: image_batch}
                    # Training
                    _, batch_loss[batch], batch_KL[batch], batch_MSE[batch], summary = \
                        sess.run([self.vae_dictionary['train_step'],
                                  self.vae_dictionary['loss'],
                                  self.vae_dictionary['KL'],
                                  self.vae_dictionary['MSE'],
                                  self.vae_dictionary['summary_op']
                                  ], feed_dict=feed_dict)
                    summary_writer.add_summary(summary, epoch)
                print("Epoch {} | Loss: {:.2E} | MSE: {:.2E} | KL: {:.2E}".format(epoch, np.mean(batch_loss),
                                                                                  np.mean(batch_MSE),
                                                                                  np.mean(batch_KL)))
            self.vae_dictionary['saver'].save(sess, weights_folder)
            print(weights_folder)

            self.vae_dictionary['decoder_saver'].save(sess, weights_folder.replace('.ckpt', '_decoder.ckpt'))
    def train_shift_images(self, images: np.ndarray, batch_size:int , epochs: int, log_dir_tensorboard: str,
                    weights_folder: str):
        assert np.amax(images) <= 1.0, "The maximum pixel value is {}. Normalize it to value 1".format(np.amax(images))
        # Define the dataset class and iterator
        dataset = tf.data.Dataset.from_tensor_slices(images).batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        batches_per_epoch = images.shape[1]//batch_size
        batch_MSE = np.zeros(batches_per_epoch)
        batch_KL = np.zeros(batches_per_epoch)
        batch_loss = np.zeros(batches_per_epoch)
        # Start the tensorflow session
        with tf.Session() as sess:
            # Summary writer for tensorboard
            summary_writer = tf.summary.FileWriter(log_dir_tensorboard, graph=sess.graph)

            if isfile(weights_folder):
                print("Restoring saved parameters")
                self.vae_dictionary['saver'].restore(sess, weights_folder)
            else:
                print("Initializing parameters")
                sess.run(tf.global_variables_initializer())
            # TRAINING
            for epoch in range(epochs):
                sess.run(iterator.initializer)  # Initialize the data iterator

                # BATCHES
                for batch in range(batches_per_epoch):
                    # For testing the batching
                    image_batch = sess.run(next_element)
                    feed_dict = {self.vae_dictionary['x']: image_batch}
                    # Training
                    _, batch_loss[batch], batch_KL[batch], batch_MSE[batch], summary = \
                        sess.run([self.vae_dictionary['train_step'],
                                  self.vae_dictionary['loss'],
                                  self.vae_dictionary['KL'],
                                  self.vae_dictionary['MSE'],
                                  self.vae_dictionary['summary_op']
                                  ], feed_dict=feed_dict)
                    summary_writer.add_summary(summary, epoch)
                print("Epoch {} | Loss: {:.2E} | MSE: {:.2E} | KL: {:.2E}".format(epoch, np.mean(batch_loss),
                                                                                  np.mean(batch_MSE),
                                                                                  np.mean(batch_KL)))
            self.vae_dictionary['saver'].save(sess, weights_folder)
            print(weights_folder)

            self.vae_dictionary['decoder_saver'].save(sess, weights_folder.replace('.ckpt', '_decoder.ckpt'))

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
        return code

    def encode_sigma(self, data, weight_folder):
        """
        Takes data an returns the mean of the encoding distribution for that
        data
        :param data: input data to encode
        :param weight_folder: location of the weights for the network
        :return: np.array with codes
        """
        with tf.Session() as sess:
            self.vae_dictionary['saver'].restore(sess, weight_folder)
            log_var_sigma = sess.run([self.vae_dictionary['log_var_z']],
                                     feed_dict={self.vae_dictionary['x']: data})[0]
            sigma = np.exp(0.5 * log_var_sigma)
        return sigma

    def encode_log_var(self, data, weight_folder):
        """
        Takes data an returns the mean of the encoding distribution for that
        data
        :param data: input data to encode
        :param weight_folder: location of the weights for the network
        :return: np.array with codes
        """
        with tf.Session() as sess:
            self.vae_dictionary['saver'].restore(sess, weight_folder)
            log_var_sigma = sess.run([self.vae_dictionary['log_var_z']],
                                     feed_dict={self.vae_dictionary['x']: data})[0]
        return log_var_sigma

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
