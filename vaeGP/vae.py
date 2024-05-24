#####
# It bulds the VAE class and further defines a custom vae function to use the vae model
#####
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras as keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def normalize(data):
  return [np.min(data),np.ptp(data)], (data - np.min(data))/(np.ptp(data))

class CustomVAE(Model):
    """
    Custom VAE model with custom loss function.
    
    """
    def __init__(self, encoder, decoder, image_size, **kwargs):
        super(CustomVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        if type(image_size) is int:
          self.image_size = image_size
        elif type(image_size) is tuple:
          self.image_size = image_size[0]*image_size[1]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def train_step(self, data):
        if isinstance(data, tuple):
            x, y = data
        else:
            x = data
            y = None

        # Check if labels are provided
        if y is None:
            raise ValueError("Digit labels are required for custom training.")


        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstructed = self.decoder(z)

            # Reconstruction loss
            reconstruction_loss = binary_crossentropy(K.flatten(x), K.flatten(reconstructed))
            reconstruction_loss *= self.image_size

            # KL divergence
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = -0.5*K.sum(kl_loss, axis=-1)

            # Custom loss
            batch_size = tf.shape(x)[0]
            y_true = tf.cast(y[:batch_size], dtype='float32')
            custom_loss = K.mean(K.square(z[:,0] - y_true), axis=-1)*10

            # Total loss
            total_loss = K.mean(reconstruction_loss + kl_loss + custom_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': total_loss, 'reconstruction_loss': reconstruction_loss, 'kl_loss': kl_loss, 'custom_loss': custom_loss}

    def plot_latent_space(self, n=30, figsize=15):
      # Display a grid of n*n digits (default 30*30)
      grid_x = np.linspace(-4, 4, n)
      grid_y = np.linspace(-4, 4, n)

      fig, ax = plt.subplots(n,n,figsize=(figsize, figsize), constrained_layout=True)
      for i, yi in enumerate(grid_y):
          for j, xi in enumerate(grid_x):
              z_sample = np.array([[xi, yi]])
              x_decoded = self.decoder.predict(z_sample, verbose=0)
              digit = x_decoded[0]
              ax[i,j].plot(digit)
              ax[i,j].set_xticks([])  # Hide x-axis ticks
              ax[i,j].set_yticks([])  # Hide y-axis ticks

      plt.show()

    def plot_2Dlatent_space(vae, n=30, figsize=15):
      # Display a grid of n*n digits (default 30*30)
      grid_x = np.linspace(-4, 4, n)
      grid_y = np.linspace(-4, 4, n)

      fig, ax = plt.subplots(n,n,figsize=(figsize, figsize), constrained_layout=True)
      for i, yi in enumerate(grid_y):
          for j, xi in enumerate(grid_x):
              z_sample = np.array([[xi, yi]])
              x_decoded = vae.decoder.predict(z_sample, verbose=0)
              digit = x_decoded[0]
              ax[i,j].imshow(digit)
              ax[i,j].set_xticks([])  # Hide x-axis ticks
              ax[i,j].set_yticks([])  # Hide y-axis ticks

      plt.show()


class Encoder:
  def __init__(self, input_shape, latent_dim=2, intermediate_dim=None):
    self.in_sh = input_shape
    self.latent_dim = latent_dim
    self.intermediate_dim = intermediate_dim

    if type(input_shape) is int:
      self.encoder = self.set_1D_encoder()
    elif (type(input_shape) is tuple) and (len(input_shape) == 2):#2D case
      self.encoder = self.set_2D_encoder()


  def set_1D_encoder(self, input_shape = None, latent_dim=None, intermediate_dim=128):
    if input_shape is None:
      input_shape = self.in_sh
    if latent_dim is None:
      latent_dim = self.latent_dim
    if self.intermediate_dim is not None:
      intermediate_dim = self.intermediate_dim

    encoder_inputs = keras.Input(shape=(input_shape,))
    x = layers.Dense(intermediate_dim, activation="relu")(encoder_inputs)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = layers.Lambda(self.sampling)([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder

  def set_2D_encoder(self, input_shape = None, latent_dim=None, intermediate_dim=[64,128,256]):
    if input_shape is None:
      input_shape = self.in_sh
    if latent_dim is None:
      latent_dim = self.latent_dim
    if self.intermediate_dim is not None:
      intermediate_dim = self.intermediate_dim

    encoder_inputs = keras.Input(shape=(input_shape[0], input_shape[1], 1))
    x = layers.Conv2D(intermediate_dim[0], 3, activation='relu', strides=2, padding='same')(encoder_inputs)
    x = layers.Conv2D(intermediate_dim[1], 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(intermediate_dim[2], activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    z = layers.Lambda(self.sampling)([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

  @staticmethod
  def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class Decoder:
  def __init__(self, output_shape, latent_dim=2, intermediate_dim=None):
    self.output_shape = output_shape
    self.latent_dim = latent_dim
    self.intermediate_dim = intermediate_dim

    #1D case
    if type(output_shape) is int:
      self.decoder = self.set_1D_decoder()
    #2D case
    elif (type(output_shape) is tuple) and (len(output_shape) == 2):
      self.decoder = self.set_2D_decoder()

  def set_1D_decoder(self, output_shape = None, latent_dim=None, intermediate_dim=128):
    if output_shape is None:
      output_shape = self.output_shape
    if latent_dim is None:
      latent_dim = self.latent_dim
    if self.intermediate_dim is not None:
      intermediate_dim = self.intermediate_dim

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
    decoder_outputs = layers.Dense(output_shape, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder

  def set_2D_decoder(self, output_shape = None, latent_dim=None, intermediate_dim=[64,128,256]):
    if output_shape is None:
      output_shape = self.output_shape
    if latent_dim is None:
      latent_dim = self.latent_dim
    if self.intermediate_dim is not None:
      intermediate_dim = self.intermediate_dim

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(intermediate_dim[-1], activation='relu')(latent_inputs)
    x = layers.Dense(output_shape[0]//4 * output_shape[1]//4 * intermediate_dim[-2], activation='relu')(x)
    x = layers.Reshape((output_shape[0]//4, output_shape[1]//4, intermediate_dim[-2]))(x)
    x = layers.Conv2DTranspose(intermediate_dim[-2], 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(intermediate_dim[-3], 3, activation='relu', strides=2, padding='same')(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


#@title LVAE for orchestration
from sklearn.preprocessing import MinMaxScaler

def VAE(dataset,
        batch_size=12,
        latent_dim=2,
        n_epoch=200,
        verbose=False,
        test = None,
        lvae_range = (0,1)):
  x_dataset, y_dataset = dataset
  norm_coef, norm_data = normalize(x_dataset)
  """
  
  Returns:
  vae: trained VAE model
  norm_coef: normalization coefficients
  (z_mean, z_std): latent space mean and std
  scaler: scaler object
  
  
  """

  if lvae_range is not None:
    scaler = MinMaxScaler(feature_range=lvae_range)# only scales the compostion
    y_dataset = scaler.fit_transform(y_dataset[:,np.newaxis]).squeeze()

  in_sh = norm_data.shape[1:]
  if len(in_sh) == 1:
    in_sh = in_sh[0]
  _encoder = Encoder(in_sh, latent_dim=latent_dim).encoder
  _decoder = Decoder(in_sh, latent_dim=latent_dim).decoder
  vae = CustomVAE(_encoder, _decoder, image_size=in_sh)
  vae.compile(optimizer='rmsprop')

  train_dataset = tf.data.Dataset.from_tensor_slices((norm_data, y_dataset))
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

  if verbose:
    vae.fit(train_dataset, epochs=n_epoch, verbose=2)
  else:
    vae.fit(train_dataset, epochs=n_epoch, verbose=0)

  if type(test) !=type(None):
    test_norm = (test - norm_coef[0])/norm_coef[1]
    full_mean, full_std, _ = vae.encoder.predict(test_norm)
    return vae, norm_coef, (full_mean, full_std), scaler#------------------> added scaler here

    
  else:
    z_mean, z_std, _ = vae.encoder.predict(norm_data)
    return vae, norm_coef, (z_mean, z_std), scaler#------------------> added scaler here



