""" Disclaimer: This method is based on the technique described in 'Variational 
Autoencoder with a scale Hyperprior'. Thus, the network is the same. """

""" This script requires TFC v2 ('pip install tensorflow-compression==2.*') and pywavelets ('pip install PyWavelets') """
import numpy as np
import pywt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import tensorflow_compression as tfc
import argparse
import glob
import sys


def read_png(filename):
  """Loads an image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=3)


def check_image_size(image, patchsize):
  """ Checks that the image has 3 channels and is large enough for processing."""
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def center_crop(image, width=None, height=None):
    """ Crops the center of an image to the specified dimensions."""
    assert width or height, 'At least one of width or height must be specified!'
    # use specified value, or fall back to the other one
    width = width or height
    height = height or width
    # determine/calculate relevant values
    old_height, old_width = image.shape[:2]
    c_x, c_y = old_width // 2, old_height // 2
    dx, dy = width // 2, height // 2
    # perform the crop
    image_cr = image[c_y-dy:c_y+dy, c_x-dx:c_x+dx]
    return tf.cast(image_cr, tf.keras.mixed_precision.global_policy().compute_dtype)


def dataset_processing(init_dataset,patch_size):
  data_set = []
  for img in init_dataset:
    image = read_png(img)
    check = check_image_size(image, patch_size)
    if check == True:
      data_set.append(img)
  return data_set


def waveletds_split(proc_dataset):
  """ Splits the RGB image into the 3 colour channels & applies 2D-DWT """  
  ds_A = []
  ds_H = []
  ds_V = []
  ds_D = []
  original_ds = []
  for img in proc_dataset:
    image1 = read_png(img)
    image = center_crop(image1, 504)
    datum = image.numpy()
    data0 = datum[:,:,0]/255.               # Normalises before the DWT
    coeffs0 = pywt.dwt2(data0, 'bior4.4')
    cA0, (cH0, cV0, cD0) = coeffs0 
    data1 = datum[:,:,1]/255.
    coeffs1 = pywt.dwt2(data1, 'bior4.4')
    cA1, (cH1, cV1, cD1) = coeffs1 
    data2 = datum[:,:,2]/255.
    coeffs2 = pywt.dwt2(data2, 'bior4.4')
    cA2, (cH2, cV2, cD2) = coeffs2 

    # Stacks the images by colour channel
    ds_A1 = tf.stack([cA0, cA1, cA2], axis=2)
    ds_A.append(ds_A1)
    ds_H1 = tf.stack([cH0, cH1, cH2], axis=2)
    ds_H.append(ds_H1)
    ds_V1 = tf.stack([cV0, cV1, cV2], axis=2)
    ds_V.append(ds_V1)
    ds_D1 = tf.stack([cD0, cD1, cD2], axis=2)
    ds_D.append(ds_D1)

  return ds_A, ds_H, ds_V, ds_D      


def get_custom_dataset(split, train_path):
  """Creates a dataset from custom images."""
  with tf.device("/cpu:0"):
    files_init = glob.glob(train_path)
    if not files_init:
      raise RuntimeError(f"No training images found with glob "
                         f"'{train_path}'.")
      
    files1 = dataset_processing(files_init, 504)
    
    wvlt_dsA, wvlt_dsH, wvlt_dsV, wvlt_dsD = waveletds_split(files1)

    
    wvlt_dsA1 = tf.data.Dataset.from_tensor_slices(wvlt_dsA)
    wvlt_dsA1 = wvlt_dsA1.shuffle(len(wvlt_dsA), reshuffle_each_iteration=True)   # The shuffling & repeating is only during training
    if split == "train":
      wvlt_dsA1 = wvlt_dsA1.repeat()
    wvlt_dsA1 = wvlt_dsA1.batch(8, drop_remainder=True)    # Default batch size is 8

    wvlt_dsH1 = tf.data.Dataset.from_tensor_slices(wvlt_dsH)
    wvlt_dsH1 = wvlt_dsH1.shuffle(len(wvlt_dsH), reshuffle_each_iteration=True)
    if split == "train":
      wvlt_dsH1 = wvlt_dsH1.repeat()
    wvlt_dsH1 = wvlt_dsH1.batch(8, drop_remainder=True)

    wvlt_dsV1 = tf.data.Dataset.from_tensor_slices(wvlt_dsV)
    wvlt_dsV1 = wvlt_dsV1.shuffle(len(wvlt_dsV), reshuffle_each_iteration=True)
    if split == "train":
      wvlt_dsV1 = wvlt_dsV1.repeat()
    wvlt_dsV1 = wvlt_dsV1.batch(8, drop_remainder=True)

    wvlt_dsD1 = tf.data.Dataset.from_tensor_slices(wvlt_dsD)
    wvlt_dsD1 = wvlt_dsD1.shuffle(len(wvlt_dsD), reshuffle_each_iteration=True)
    if split == "train":
      wvlt_dsD1 = wvlt_dsD1.repeat()
    wvlt_dsD1 = wvlt_dsD1.batch(8, drop_remainder=True)

  return wvlt_dsA1, wvlt_dsH1, wvlt_dsV1, wvlt_dsD1


class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  def __init__(self, num_filters):
    super().__init__(name="analysis")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_0")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_2")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=None))

class SynthesisTransform(tf.keras.Sequential):
  """The synthesis transform."""

  def __init__(self, num_filters):
    super().__init__(name="synthesis")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_2", inverse=True)))
    self.add(tfc.SignalConv2D(
        3, (5, 5), name="layer_3", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=None))


class HyperAnalysisTransform(tf.keras.Sequential):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters):
    super().__init__(name="hyper_analysis")
    self.add(tfc.SignalConv2D(
        num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))

class HyperSynthesisTransform(tf.keras.Sequential):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters):
    super().__init__(name="hyper_synthesis")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=None))

class DetectionModel(tf.keras.Model):
  """Main model."""

  def __init__(self, lmbda, num_filters, num_scales, scale_min, scale_max):
    super().__init__()
    self.lmbda = lmbda
    self.num_scales = num_scales
    offset = tf.math.log(scale_min)
    factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
        num_scales - 1.)
    self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
    self.analysis_transform = AnalysisTransform(num_filters)
    self.synthesis_transform = SynthesisTransform(num_filters)
    self.hyper_analysis_transform = HyperAnalysisTransform(num_filters)
    self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters)
    self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
    self.build((None, None, None, 3))

  def call(self, x, training):
    """Computes bit rate and distortion loss."""
    entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=False)
    side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=False)

    x = tf.cast(x, self.compute_dtype) 
    y = self.analysis_transform(x)
    z = self.hyper_analysis_transform(abs(y))
    z_hat, side_bits = side_entropy_model(z, training=training)
    indexes = self.hyper_synthesis_transform(z_hat)
    y_hat, bits = entropy_model(y, indexes, training=training)
    x_hat = self.synthesis_transform(y_hat)

    # Total number of bits divided by total number of pixels.
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
    bpp = (tf.reduce_sum(bits) + tf.reduce_sum(side_bits)) / num_pixels
    
    # Mean squared error used for loss.
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    mse = tf.cast(mse, bpp.dtype)
    
    # The three other quality metrics.
    psnr = tf.image.psnr(x, x_hat, max_val=1.)
    psnr = tf.cast(psnr, bpp.dtype)
    ssim = tf.image.ssim(x, x_hat, max_val=1.)
    ssim = tf.cast(ssim, bpp.dtype)
    ms_ssim = tf.image.ssim_multiscale(x, x_hat, max_val=1.)
    ms_ssim = tf.cast(ms_ssim, bpp.dtype)

    # The rate-distortion loss.
    loss = bpp + self.lmbda * mse
    if training:
      return loss, bpp, mse, psnr, ssim, ms_ssim
    else:
      return loss, bpp, mse, psnr, ssim, ms_ssim, x_hat
    
  def train_step(self, x):
    with tf.GradientTape() as tape:
      loss, bpp, mse, psnr, ssim, ms_ssim = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    self.psnr.update_state(psnr)
    self.ssim.update_state(ssim)
    self.ms_ssim.update_state(ms_ssim)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.psnr, self.ssim, self.ms_ssim]}

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.mse = tf.keras.metrics.Mean(name="mse")
    self.psnr = tf.keras.metrics.Mean(name="psnr")
    self.ssim = tf.keras.metrics.Mean(name="ssim")
    self.ms_ssim = tf.keras.metrics.Mean(name="ms_ssim")

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    # After training, fix range coding tables.
    self.entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=True)
    self.side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=True)    
    return retval
    
    
# Creates the 4 sub-datasets
traindataseta, traindataseth, traindatasetv, traindatasetd = get_custom_dataset("train", "Path to dataset")

"""Instantiates and trains the model 4 times - once for each sub-dataset."""

# Model for the approximation detail.
modela = DetectionModel(
    lmbda = 1000.0, num_filters = 192, num_scales = 64, scale_min = 0.11,
    scale_max = 256.0)
modela.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

modela.fit(
    traindataseta.prefetch(8),
    epochs=250,
    steps_per_epoch=200, 
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
    ],
    verbose= 0,
)
modela.save_weights("Path to weights a")


# Model for the horizontal detail.
modelh = DetectionModel(
    lmbda = 1000.0, num_filters = 192, num_scales = 64, scale_min = 0.11,
    scale_max = 256.0)
modelh.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

modelh.fit(
    traindataseth.prefetch(8),
    epochs=250,
    steps_per_epoch=200,
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
    ],
    verbose= 0,
)
modelh.save_weights("Path to weights h")


# Model for the vertical detail.
modelv = DetectionModel(
    lmbda = 1000.0, num_filters = 192, num_scales = 64, scale_min = 0.11,
    scale_max = 256.0)
modelv.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

modelv.fit(
    traindatasetv.prefetch(8),
    epochs=250,
    steps_per_epoch=200,
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
    ],
    verbose= 0,
)
modelv.save_weights("Path to weights v")


# Model for the diagonal detail.
modeld = DetectionModel(
    lmbda = 1000.0, num_filters = 192, num_scales = 64, scale_min = 0.11,
    scale_max = 256.0)
modeld.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

modeld.fit(
    traindatasetd.prefetch(8),
    epochs=250,
    steps_per_epoch=200,
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
    ], 
    verbose= 0,
)
modeld.save_weights("Path to weights d")

""" Reminder: The model must be trained twice, once on a real and once on a synthetic dataset. """

