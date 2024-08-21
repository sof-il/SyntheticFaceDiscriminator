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
    image2 = image
    datum = image2.numpy()
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

    original_ds.append(image)   # In the testing we need the original images too.

  return ds_A, ds_H, ds_V, ds_D, original_ds      

def get_custom_dataset(split, train_path):
  """Creates a dataset from custom images."""
  with tf.device("/cpu:0"):
    files_init = glob.glob(train_path)
    if not files_init:
      raise RuntimeError(f"No training images found with glob "
                         f"'{train_path}'.")

    files1 = dataset_processing(files_init, 504)

    wvlt_dsA, wvlt_dsH, wvlt_dsV, wvlt_dsD, original_ds = waveletds_split(files1)

    wvlt_dsA1 = tf.data.Dataset.from_tensor_slices(wvlt_dsA)
    wvlt_dsA1 = wvlt_dsA1.batch(1, drop_remainder=True)      # Default batch size is 1
    wvlt_dsH1 = tf.data.Dataset.from_tensor_slices(wvlt_dsH)
    wvlt_dsH1 = wvlt_dsH1.batch(1, drop_remainder=True)
    wvlt_dsV1 = tf.data.Dataset.from_tensor_slices(wvlt_dsV)
    wvlt_dsV1 = wvlt_dsV1.batch(1, drop_remainder=True)
    wvlt_dsD1 = tf.data.Dataset.from_tensor_slices(wvlt_dsD)
    wvlt_dsD1 = wvlt_dsD1.batch(1, drop_remainder=True)

  return wvlt_dsA1, wvlt_dsH1, wvlt_dsV1, wvlt_dsD1, original_ds

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
  """Main model class."""

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
    """Computes rate and distortion losses."""
    entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=False)
    side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=False)

    x = tf.cast(x, self.compute_dtype)  # TODO(jonycgn): Why is this necessary?
    y = self.analysis_transform(x)
    z = self.hyper_analysis_transform(abs(y))
    z_hat, side_bits = side_entropy_model(z, training=training)
    indexes = self.hyper_synthesis_transform(z_hat)
    y_hat, bits = entropy_model(y, indexes, training=training)
    x_hat = self.synthesis_transform(y_hat)

    # Total number of bits divided by total number of pixels.
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
    bpp = (tf.reduce_sum(bits) + tf.reduce_sum(side_bits)) / num_pixels
    # Mean squared error across pixels.
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    mse = tf.cast(mse, bpp.dtype)

    psnr = tf.image.psnr(x, x_hat, max_val=255.)
    psnr = tf.cast(psnr, bpp.dtype)
    ssim = tf.image.ssim(x, x_hat, max_val=255.)
    ssim = tf.cast(ssim, bpp.dtype)
    ms_ssim = tf.image.ssim_multiscale(x, x_hat, max_val=255.)
    ms_ssim = tf.cast(ms_ssim, bpp.dtype)

    # The rate-distortion Lagrangian.
    loss = bpp + self.lmbda * mse
    if training:
      return loss, bpp, mse, psnr, ssim, ms_ssim
    else:
      return loss, bpp, mse, psnr, ssim, ms_ssim, x_hat, x

  def test_step(self, x):
    loss, bpp, mse, psnr, ssim, ms_ssim, x_hat, x = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    self.psnr.update_state(psnr)
    self.ssim.update_state(ssim)
    self.ms_ssim.update_state(ms_ssim)

    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.psnr, self.ssim, self.ms_ssim, x_hat, x]}

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


# Creates the 4 sub-datasets
test_dataseta_r2, test_dataseth_r2, test_datasetv_r2, test_datasetd_r2, original_ds_r2 = get_custom_dataset("test", "Path to dataset")

"""Instantiates and tests the model 8 times - 4 for the model trained on 
on real images and 4 for the model trained on synthetic ones."""

modela_r2 = DetectionModel(
    lmbda = 1000.0, num_filters = 192, num_scales = 64, scale_min = 0.11,
    scale_max = 256.0)
modela_r2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

dummy_a_r2 = np.random.rand(1,256,256,3)
modela_r2.predict(dummy_a_r2)
modela_r2.load_weights("Path to weights real a",by_name=True)
test_resultsa_r2 = modela_r2.predict(test_dataseta_r2)


modela_r22 = DetectionModel(
    lmbda = 1000.0, num_filters = 192, num_scales = 64, scale_min = 0.11,
    scale_max = 256.0)
modela_r22.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

dummy_a_r22 = np.random.rand(1,256,256,3)
modela_r22.predict(dummy_a_r22)
modela_r22.load_weights("Path to weights synthetic a",by_name=True)
test_resultsa_r22 = modela_r22.predict(test_dataseta_r2)


modelh_r2 = DetectionModel(
    lmbda = 1000.0, num_filters = 192, num_scales = 64, scale_min = 0.11,
    scale_max = 256.0)
modelh_r2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

dummy_h_r2 = np.random.rand(1,256,256,3)
modelh_r2.predict(dummy_h_r2)
modelh_r2.load_weights("Path to weights real h",by_name=True)
test_resultsh_r2 = modelh_r2.predict(test_dataseth_r2)


modelh_r22 = DetectionModel(
    lmbda = 1000.0, num_filters = 192, num_scales = 64, scale_min = 0.11,
    scale_max = 256.0)
modelh_r22.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

dummy_h_r22 = np.random.rand(1,256,256,3)
modelh_r22.predict(dummy_h_r22)
modelh_r22.load_weights("Path to weights synthetic h",by_name=True)
test_resultsh_r22 = modelh_r22.predict(test_dataseth_r2)


modelv_r2 = DetectionModel(
    lmbda = 1000.0, num_filters = 192, num_scales = 64, scale_min = 0.11,
    scale_max = 256.0)
modelv_r2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

dummy_v_r2 = np.random.rand(1,256,256,3)
modelv_r2.predict(dummy_v_r2)
modelv_r2.load_weights("Path to weights real v",by_name=True)
test_resultsv_r2 = modelv_r2.predict(test_datasetv_r2)


modelv_r22 = DetectionModel(
    lmbda = 1000.0, num_filters = 192, num_scales = 64, scale_min = 0.11,
    scale_max = 256.0)
modelv_r22.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

dummy_v_r22 = np.random.rand(1,256,256,3)
modelv_r22.predict(dummy_v_r22)
modelv_r22.load_weights("Path to weights synthetic v",by_name=True)
test_resultsv_r22 = modelv_r22.predict(test_datasetv_r2)


modeld_r2 = DetectionModel(
    lmbda = 1000.0, num_filters = 192, num_scales = 64, scale_min = 0.11,
    scale_max = 256.0)
modeld_r2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

dummy_d_r2 = np.random.rand(1,256,256,3)
modeld_r2.predict(dummy_d_r2)
modeld_r2.load_weights("Path to weights real d",by_name=True)
test_resultsd_r2 = modeld_r2.predict(test_datasetd_r2)


modeld_r22 = DetectionModel(
    lmbda = 1000.0, num_filters = 192, num_scales = 64, scale_min = 0.11,
    scale_max = 256.0)
modeld_r22.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

dummy_d_r22 = np.random.rand(1,256,256,3)
modeld_r22.predict(dummy_d_r22)
modeld_r22.load_weights("Path to weights synthetic d",by_name=True)
test_resultsd_r22 = modeld_r22.predict(test_datasetd_r2)


def waveletds_unite(imA, imH, imV, imD):
  """ Applies 2D-IDWT and reconstructs the RGB image."""
  rec_ds = []

  for i in range(len(imA)):
    imgA = imA[i]
    imgH = imH[i]
    imgV = imV[i]
    imgD = imD[i]

    dataAY = imgA[:,:,0]
    dataAU = imgA[:,:,1]
    dataAV = imgA[:,:,2]
    dataHY = imgH[:,:,0]
    dataHU = imgH[:,:,1]
    dataHV = imgH[:,:,2]
    dataVY = imgV[:,:,0]
    dataVU = imgV[:,:,1]
    dataVV = imgV[:,:,2]
    dataDY = imgD[:,:,0]
    dataDU = imgD[:,:,1]
    dataDV = imgD[:,:,2]

    coeffsY = dataAY, (dataHY, dataVY, dataDY)
    latentsY = pywt.idwt2(coeffsY, 'bior4.4')
    coeffsU = dataAU, (dataHU, dataVU, dataDU)
    latentsU = pywt.idwt2(coeffsU, 'bior4.4')
    coeffsV = dataAV, (dataHV, dataVV, dataDV)
    latentsV = pywt.idwt2(coeffsV, 'bior4.4')

    image = tf.stack([latentsY*255., latentsU*255., latentsV*255.], axis=2)       # Denotmalise after the DWT.
    rec_ds.append(image)

  return rec_ds


reconstructed_ds_r2 = waveletds_unite(test_resultsa_r2[6], test_resultsh_r2[6], test_resultsv_r2[6], test_resultsd_r2[6])


""" Applies saturation arithmetic to the 3 bounded metrics. """
# The images produced by the model trained on a real dataset.
sat_output_r2 = []

for i in range(len(reconstructed_ds_r2)):
  sat_out_r2 = reconstructed_ds_r2[i]
  sat_r2 = np.clip(sat_out_r2, a_min=0, a_max=255)
  sat_output_r2.append(sat_r2)

sat_psnr_r2 = []
sat_ssim_r2 = []
sat_msssim_r2 = []

for i in range(len(sat_output_r2)):
  c_psnr_r2 = tf.image.psnr(original_ds_r2[i], np.float32(sat_output_r2[i]), max_val=255.)
  c_psnr_r2 = tf.cast(c_psnr_r2, dtype=tf.float32)
  sat_psnr_r2.append(c_psnr_r2)
  c_ssim_r2 = tf.image.ssim(original_ds_r2[i], np.float32(sat_output_r2[i]), max_val=255.)
  c_ssim_r2 = tf.cast(c_ssim_r2, dtype=tf.float32)
  sat_ssim_r2.append(c_ssim_r2)
  c_msssim_r2 = tf.image.ssim_multiscale(original_ds_r2[i], np.float32(sat_output_r2[i]), max_val=255.)
  c_msssim_r2 = tf.cast(c_msssim_r2, dtype=tf.float32)
  sat_msssim_r2.append(c_msssim_r2)

# The images produced by the model trained on a synthetic dataset.
reconstructed_ds_r22 = waveletds_unite(test_resultsa_r22[6], test_resultsh_r22[6], test_resultsv_r22[6], test_resultsd_r22[6])

sat_output_r22 = []

for i in range(len(reconstructed_ds_r22)):
  sat_out_r22 = reconstructed_ds_r22[i]
  sat_r22 = np.clip(sat_out_r22, a_min=0, a_max=255)
  sat_output_r22.append(sat_r22)

sat_psnr_r22 = []
sat_ssim_r22 = []
sat_msssim_r22 = []

for i in range(len(sat_output_r22)):
  c_psnr_r22 = tf.image.psnr(original_ds_r2[i], np.float32(sat_output_r22[i]), max_val=255.)
  c_psnr_r22 = tf.cast(c_psnr_r22, dtype=tf.float32)
  sat_psnr_r22.append(c_psnr_r22)
  c_ssim_r22 = tf.image.ssim(original_ds_r2[i], np.float32(sat_output_r22[i]), max_val=255.)
  c_ssim_r22 = tf.cast(c_ssim_r22, dtype=tf.float32)
  sat_ssim_r22.append(c_ssim_r22)
  c_msssim_r22 = tf.image.ssim_multiscale(original_ds_r2[i], np.float32(sat_output_r22[i]), max_val=255.)
  c_msssim_r22 = tf.cast(c_msssim_r22, dtype=tf.float32)
  sat_msssim_r22.append(c_msssim_r22)
  
  
""" Stacks the features extracted for the detection. """
features = []
label_r = 0.0      # The labels are 0 for real and 1 for synthetic images.

for i in range (len(reconstructed_ds_r2)):
    """ The 27 features are stacked together. """
    features_r = np.zeros(55)
    features_r[0] = test_resultsa_r2[0][i]
    features_r[1] = test_resultsa_r2[1][i]
    features_r[2] = test_resultsa_r2[2][i]
    features_r[3] = test_resultsa_r2[3][i]
    features_r[4] = test_resultsa_r2[4][i]
    features_r[5] = test_resultsa_r2[5][i]
    features_r[6] = test_resultsh_r2[0][i]
    features_r[7] = test_resultsh_r2[1][i]
    features_r[8] = test_resultsh_r2[2][i]
    features_r[9] = test_resultsh_r2[3][i]
    features_r[10] = test_resultsh_r2[4][i]
    features_r[11] = test_resultsh_r2[5][i]
    features_r[12] = test_resultsv_r2[0][i]
    features_r[13] = test_resultsv_r2[1][i]
    features_r[14] = test_resultsv_r2[2][i]
    features_r[15] = test_resultsv_r2[3][i]
    features_r[16] = test_resultsv_r2[4][i]
    features_r[17] = test_resultsv_r2[5][i]
    features_r[18] = test_resultsd_r2[0][i]
    features_r[19] = test_resultsd_r2[1][i]
    features_r[20] = test_resultsd_r2[2][i]
    features_r[21] = test_resultsd_r2[3][i]
    features_r[22] = test_resultsd_r2[4][i]
    features_r[23] = test_resultsd_r2[5][i]
    features_r[24] = sat_psnr_r2[i]
    features_r[25] = sat_ssim_r2[i]
    features_r[26] = sat_msssim_r2[i]
    features_r[27] = test_resultsa_r22[0][i]
    features_r[28] = test_resultsa_r22[1][i]
    features_r[29] = test_resultsa_r22[2][i]
    features_r[30] = test_resultsa_r22[3][i]
    features_r[31] = test_resultsa_r22[4][i]
    features_r[32] = test_resultsa_r22[5][i]
    features_r[33] = test_resultsh_r22[0][i]
    features_r[34] = test_resultsh_r22[1][i]
    features_r[35] = test_resultsh_r22[2][i]
    features_r[36] = test_resultsh_r22[3][i]
    features_r[37] = test_resultsh_r22[4][i]
    features_r[38] = test_resultsh_r22[5][i]
    features_r[39] = test_resultsv_r22[0][i]
    features_r[40] = test_resultsv_r22[1][i]
    features_r[41] = test_resultsv_r22[2][i]
    features_r[42] = test_resultsv_r22[3][i]
    features_r[43] = test_resultsv_r22[4][i]
    features_r[44] = test_resultsv_r22[5][i]
    features_r[45] = test_resultsd_r22[0][i]
    features_r[46] = test_resultsd_r22[1][i]
    features_r[47] = test_resultsd_r22[2][i]
    features_r[48] = test_resultsd_r22[3][i]
    features_r[49] = test_resultsd_r22[4][i]
    features_r[50] = test_resultsd_r22[5][i]
    features_r[51] = sat_psnr_r22[i]
    features_r[52] = sat_ssim_r22[i]
    features_r[53] = sat_msssim_r22[i]
    features_r[54] = label_r
    features.append(features_r)

features_real = np.asarray(features)
np.savetxt("Path to the the .csv file with the learned features", features_real, delimiter=',')

""" Reminder: This process needs to be done twice, once for a real and once for a synthetic dataset.
The final .csv file that will be used for the detection, contains the extracted features from both models. """

