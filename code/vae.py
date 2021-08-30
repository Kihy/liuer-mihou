import sys
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow import keras
matplotlib.use('Agg')


class Sampling(Layer):
    """
    The reparameterization trick that samples from standard normal and reparameterize
    it into any normal distribution with mean and standard deviation.

    Note when training the variables are mean and log(var), thus to get std it is
    multiplied by exp(log(var)*0.5)
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def kullback_leibler_loss(z_mean, z_log_var, beta, out_stream=sys.stdout):
    """
    calculates kl divergence between standard normal and normal with z_mean
    and z_log_var as mean and log var.

    Note whether to use sum or mean is debatable, but both accepted. The difference is
    that mean calculates the average difference of the latent variables, and sum
    calculates total difference between latent variables.

    Args:
        z_mean (tensor): mean value of z distribution.
        z_log_var (tensor): log var of z distribution.
        beta (float): hyperparameter to weight kl divergence
    Returns:
        the kl divergence between the two distributions

    """
    kl_loss = - 0.5 * beta * \
        tf.math.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    # return xent_loss + kl_loss
    # tf.print("Average KL Div across Batch: ", tf.math.reduce_mean(kl_loss), output_stream=out_stream)
    # tf.summary.scalar('mean kl loss', data=tf.math.reduce_mean(kl_loss),step=1)

    return kl_loss


class Encoder(Model):
    """
    the encoder model used in variational autoencoder. this is purposely made
    to be a model rather than layer so we can use it directly for encoding. the
    encoder currently only has 1 intermediate layer and 1 output layer.

    Args:
        latent_dim (integer): latent dimension of the encoder output. Defaults to 2.
        intermediate_dim (integer): Description of parameter `intermediate_dim`. Defaults to 20.
        name (string): name of the model. Defaults to "encoder".
        **kwargs (type): any parameter that is used for model.

    Attributes:
        dense_proj (layer): projection layer from input to intermediate.
        dense_mean (layer): the mean values of the latent layer in the latent space.
        dense_log_var (layer): the log of variance of the latent layers in latent space.
        sampling (layer): the sampling layer used to sample data points.

    """

    def __init__(self, latent_dim=2, intermediate_dim=20, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        # regularizer=None
        self.dense_proj = Dense(intermediate_dim, activation='relu')
        self.dense_mean = Dense(latent_dim, activation='relu')
        self.dense_log_var = Dense(latent_dim, activation='relu')
        self.sampling = Sampling()

    def call(self, inputs):
        """runs the encoder model.

        Args:
            inputs (tensor): inputs to the network.

        Returns:
            type: the mean, log variance and z.

        """

        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(Model):
    """
    the decoder model used in variational autoencoder. this is purposely made
    to be a model rather than layer so we can use it directly for encoding. the
    encoder currently only has 1 intermediate layer and 1 output layer.

    Args:
        original_dim (integer): original dimensions of the input data.
        intermediate_dim (integer): intermediate layer dimension. Defaults to 20.
        name (string): name of this model. Defaults to "decoder".
        **kwargs (type): any parameter that is used for model.

    Attributes:
        dense_proj (type): dense layer that projects latent dim to intermeidate.
        dense_output (type): dense layer that maps intermediate to output.

    """

    def __init__(self, original_dim, intermediate_dim=20, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = Dense(intermediate_dim, activation='relu')
        self.dense_output = Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        """
        runs the decoder model

        Args:
            inputs (tensor): inputs of the decoder network.

        Returns:
            type: the output of this model which is the reformed data.

        """
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(Model):
    """
    the variational autoencoder model consisting of encoder and decoder

    Args:
        original_dim (integer): original dimension of input.
        latent_dim (integer): dimension of latent space. Defaults to 2.
        intermediate_dim (integer): dimension of intermediate layer. Defaults to 20.
        name (string): name of model. Defaults to "vae".
        **kwargs (type): other arguments.

    Attributes:
        encoder (model): encoder model.
        decoder (model): decoder model.
        original_dim

    """

    def __init__(self, original_dim, latent_dim=2, intermediate_dim=20, name="vae",** kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(
            latent_dim=latent_dim, intermediate_dim=intermediate_dim, name="encoder")
        self.decoder = Decoder(
            original_dim, intermediate_dim=intermediate_dim, name="decoder")

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                    keras.losses.mean_squared_error(data, reconstruction)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
