import numpy as np
import matplotlib.pyplot as plt

import h5py

import torch
import pyroved as pv

from IPython.core.display import Video
import imageio

import gpax
import numpyro
gpax.utils.enable_x64()

# from google.colab import drive

import numpyro.distributions as dist
import jax.numpy as jnp

import os
from igor2 import binarywave as bw
import aespm.tools as at
import logging
import datetime

# Initialize logging
def init_logging(save_dir):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Logging to {save_dir}log_{current_time}.log")
    filename = f"{save_dir}log_{current_time}.log"
    logging.basicConfig(filename=filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')

def normalize(data):
  return [np.min(data),np.ptp(data)], (data - np.min(data))/(np.ptp(data))

def main():
    save_dir = "."
    init_logging(save_dir=save_dir)
    
    logging.info("Starting main function")
    