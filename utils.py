import gc
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy import special
from scipy import signal
import matplotlib.pyplot as plt
from glob import glob
from GenerateDataset import GenerateDataset
from tqdm import tqdm

basedir = '.'

# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle




class train_data():
    def __init__(self, filepath="%s/data/image_clean_pat.npy" % basedir):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


def load_data(filepath="%s/data/image_clean_pat.npy" % basedir):
    return train_data(filepath=filepath)


def normalize_sar(im):
    return ((np.log(im+1e-6)-m)/(M-m)).astype(np.float32)

def denormalize_sar(im):
    return np.exp((np.clip(np.squeeze(im),0,1))*(M-m)+m)





def load_train_data(filepath, patch_size, batch_size, stride_size, n_data_augmentation): #TODO: add control on training data: exit if does not exists
    datagen = GenerateDataset()
    imgs = datagen.generate_patches(src_dir=filepath, pat_size=patch_size, step=0,
                             stride=stride_size, bat_size=batch_size, data_aug_times=n_data_augmentation)
    return normalize_sar(imgs)





def load_sar_images(filelist):
    if not isinstance(filelist, list):
        im = np.load(filelist)
        return np.array(im).reshape(np.size(im, 0), np.size(im, 1), 1)
    data = []
    for file in filelist:
        im = np.load(file)
        data.append(np.array(im).reshape(np.size(im, 0), np.size(im, 1), 1))
    return data





def store_data_and_plot(im, threshold, filename):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype('float64')).convert('L')
    im.save(filename.replace('npy','png'))


def save_sar_images(denoised, noisy, imagename, save_dir, noisy_bool=True, groundtruth=None):
    choices = {'marais1':190.92, 'marais2': 168.49, 'saclay':470.92, 'lely':235.90, 'ramb':167.22,
           'risoul':306.94, 'limagne':178.43, 'saintgervais':560, 'Serreponcon': 450.0,
          'Sendai':600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
          'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'domancy': 560, 'Brazil': 103.0,
              'region1_HH': 1068.85, 'region1_HV':402.75, 'region2_HH': 1039.59, 'region2_HV': 391.70}

    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None: threshold = np.mean(noisy) + 3 * np.std(noisy)

    if groundtruth:
        groundtruthfilename = save_dir+"/groundtruth_"+imagename
        np.save(groundtruthfilename,groundtruth)
        store_data_and_plot(groundtruth, threshold, groundtruthfilename)

    denoisedfilename = save_dir + "/denoised_" + imagename
    np.save(denoisedfilename, denoised)
    store_data_and_plot(denoised, threshold, denoisedfilename)

    if noisy_bool:
        noisyfilename = save_dir + "/noisy_" + imagename
        np.save(noisyfilename, noisy)
        store_data_and_plot(noisy, threshold, noisyfilename)


def save_real_imag_images(real_part, imag_part, imagename, save_dir):
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'Brazil': 103.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None: threshold = np.mean(imag_part) + 3 * np.std(imag_part)

    realfilename = save_dir + "/denoised_real_" + imagename
    np.save(realfilename, real_part)
    store_data_and_plot(real_part, threshold, realfilename)

    imagfilename = save_dir + "/denoised_imag_" + imagename
    np.save(imagfilename, imag_part)
    store_data_and_plot(imag_part, threshold, imagfilename)

def save_real_imag_images_noisy(real_part, imag_part, imagename, save_dir):
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'Brazil': 103.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None: threshold = np.mean(np.abs(imag_part)) + 3 * np.std(np.abs(imag_part))

    realfilename = save_dir + "/noisy_real_" + imagename
    np.save(realfilename, real_part)
    store_data_and_plot(np.sqrt(2)*np.abs(real_part), threshold, realfilename)

    imagfilename = save_dir + "/noisy_imag_" + imagename
    np.save(imagfilename, imag_part)
    store_data_and_plot(np.sqrt(2)*np.abs(imag_part), threshold, imagfilename)

def save_residual(filepath, residual):
    residual_image = np.squeeze(residual);
    plt.imsave(filepath, residual_image)


def cal_psnr(Shat, S):
    # takes amplitudes in input
    # Shat: a SAR amplitude image
    # S:    a reference SAR image
    P = np.quantile(S, 0.99)
    res = 10 * np.log10((P ** 2) / np.mean(np.abs(Shat - S) ** 2))
    return res


def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))


def init_weights(m):
      if type(m) == torch.nn.Linear:
          torch.nn.init.xavier_normal_(m.weight,gain=2.0)
          print('[*] inizialized weights')


def evaluate(model, loader):
    outputs = [model.validation_step(batch) for batch in loader]
    outputs = torch.tensor(outputs).T
    loss, accuracy = torch.mean(outputs, dim=1)
    return {"loss": loss.item(), "accuracy": accuracy.item()}


def save_model(model,destination_folder):
    """
      save the ".pth" model in destination_folder
    """
    # Check whether the specified path exists or not
    isExist = os.path.exists(destination_folder)

    if not isExist:

      # Create a new directory because it does not exist
      os.makedirs(destination_folder)
      print("The new directory is created!")

      torch.save(model.state_dict(),destination_folder+"/model.pth")

    else:
      torch.save(model.state_dict(),destination_folder+"/model.pth")

def save_checkpoint(model,destination_folder,epoch_num,optimizer,loss):
    """
      save the ".pth" checkpoint in destination_folder
    """
    # Check whether the specified path exists or not
    isExist = os.path.exists(destination_folder)

    if not isExist: os.makedirs(destination_folder)

    torch.save({
            'epoch_num': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, destination_folder+"/checkpoint_"+str(epoch_num)+".pth")
    print("\n Checkpoint saved at :",destination_folder)


def threshold(im, name):
  """
  A function to make a good threshold for the different type of images
  """
  choices = {'marais1':190.92, 'marais2': 168.49, 'saclay':470.92, 'lely':235.90, 'ramb':167.22, 'risoul':306.94, 'limagne':178.43}
  threshold = 0
  for x in choices:
    if name == x:
        threshold = choices.get(x)
  if threshold ==0:
    threshold= np.mean(im)+3*np.std(im) 

  dim = np.clip(im,0,threshold)
  dim = dim/threshold*255
  return dim

M = 10.089038980848645
m = -1.429329123112601

def normalized_SAR(im, M = 10.089038980848645, m = -1.429329123112601):
  """
  A function to normalized the SAR image. Return in the log domain between 0 and 1. 
  M is the max in the log domain, m the min.
  """
  return (torch.log(im+1e-6)-m)/(M-m)

def inverse_normalized_SAR(im, M = 10.089038980848645, m = -1.429329123112601):
  """
  A function to inverse the normalization the SAR image. From a log-domain image between 0 and 1 
  return in the natural domain with the good amplitude. 
  """
  return torch.exp((M-m)*im+m) - 1e-6

def denoiser_256(im, model):
  """
  Take one numpy image of size 256*256 an apply the NN SAR denoiser to it. Return a numpy image desnoised of size 256*256
  """
  im = torch.tensor(im)
  im_log = normalized_SAR(im)
  im_log = im_log[None,:,:,None]
  im_denoised_log = model.forward(im_log, batch_size = 1)[0,0,:,:]
  im_denoised = inverse_normalized_SAR(im_denoised_log)
  return im_denoised.detach().numpy()

def generate_speckle(x,L):
  """
  Add speckle to a log-normalized image (torch.tensor)
  """
  s = torch.zeros(x.size())
  for k in range(L):
      gamma = (torch.abs(torch.complex(torch.normal(0,1,x.size()),torch.normal(0,1,x.size()))) ** 2)/2
      s = s+gamma
  s_amplitude = torch.sqrt(s/L)
  s_log = torch.log(s_amplitude)
  log_norm_s = s_log / (M-m)
  return x + log_norm_s

#usefull function
def sar_to_sar_with_ratio(im, super_im, model):
  """
  A function to compute the denoising of an image with the SARtoSAR NN by computing the ratio image
  """
  super_im_log = np.log(super_im + 1e-6)
  lbda = np.exp(np.mean(super_im_log))
  super_im_norm = super_im / lbda
  im_ratio = im / super_im_norm
  return denoiser_256(im_ratio.astype(np.float32), model) *  super_im_norm

def sarratio_to_sarratio(im, super_im, model_ratio):
  """
  Apply the SARratiotoSARratio model to the ratio im/super_im
  """
  im_ratio = im / super_im
  return denoiser_256(im_ratio.astype(np.float32), model_ratio) * super_im

def Lee_filter(im, L, size_window = 4):
  """
  Apply the Lee filter to the SAR image im, with a number of look L and a convolution of size size_window*size_window
  """
  ima_int = im
  # create the average window
  masque_loc = np.ones((size_window,size_window))/(size_window*size_window)

  # compute the mean image
  ima_int_mean = signal.convolve2d(ima_int, masque_loc, mode='same')

  # compute the variance image (var{I} = E{I^2} - E{I}^2)
  ima_int_square = np.multiply(ima_int,ima_int) # I^2
  ima_int_mean_square = signal.convolve2d(ima_int_square, masque_loc, mode='same')
  ima_variance = ima_int_mean_square - np.multiply(ima_int_mean,ima_int_mean) 

  # compute coefficient of variation
  ima_coeff_var = np.sqrt(ima_variance) / ima_int_mean

  # compute ks, by taking ima_coeff_var previously computed
  ks = 1 - 1/(np.sqrt(L)*ima_coeff_var**2)

  # force k to have values comprised in the range [0,1]
  ks= np.clip(ks,0,1)

  image_lee_filtered = ima_int_mean + ks * (ima_int - ima_int_mean)
  return image_lee_filtered

def gradn(im):
  """
  Compute the norm of the gradient of the image im
  """
  return np.sqrt((np.diff(im, axis = 0)[:,:-1])**2+(np.diff(im, axis = 1)[:-1,:])**2)

def estimate_noise(im):
  """
  A function to estimate the noise level of an image
  """
  H, W = im.shape
  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]
  sigma = np.sum(np.abs(signal.convolve2d(im, M)))
  sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))
  return sigma

def denoise_BM3D(im, BM3DDenoiser):
  """
  To denoise an image im with BM3D in the log space. The noise is gaussian additive in the log space
  """
  sigma255 = 255*estimate_noise(np.log(im+1e-6))
  denoiser = BM3DDenoiser(sigma255)
  im_denoisyBM3D = np.exp(denoiser.denoise(np.log(im+1e-6)))
  return im_denoisyBM3D

def general_denoised(ims):
  """
  Apply the denoiser model to the set of image ims (of size bigger than 256*256).
  """
  #le noyau plante -> calcul trop lourd pour colab

  ims = normalized_SAR(Im_ratio_norm[:,:,:,None].to(torch.float32))
  print(ims.size())
  p, n, m, _ = ims.size()
  ims_denoised = torch.zeros(ims.size())
  for i in tqdm(range(n // 256)):
    for j in range(m // 256):
      ims_denoised[:,i*256:(i+1)*256,j*256:(j+1)*256,:] = torch.permute(model.forward(ims[:,i*256:(i+1)*256,j*256:(j+1)*256,:], batch_size = p), (0,2,3,1)) 
      
  for i in range(n // 256):
    ims_denoised[:,i*256:(i+1)*256,-m % 256:,:] = torch.permute(model.forward(ims[:,i*256:(i+1)*256,-256:,:], batch_size = p), (0,2,3,1))[:,:,-m % 256:,:]

  for j in range(m // 256):
    ims_denoised[:,-n % 256:,j*256:(j+1)*256,:] = torch.permute(model.forward(ims[:,-256:,j*256:(j+1)*256,:], batch_size = p), (0,2,3,1))[:,:,-n % 256:,:]

  print(ims_denoised.size())
  ims_denoised = inverse_normalized_SAR(ims_denoised)
  return ims_denoised