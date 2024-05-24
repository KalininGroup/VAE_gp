#@title GP functions

from ast import Pass
from gpax import ExactGP
from pyroved.models import iVAE
from typing import Callable, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import gpax

#----------- GP BO -------------

class Orchestrated_BO(ExactGP):
  def __init__(self,
               kernel='Matern',
               lengthscale_prior_dist = None,
               noise_prior_dist = None,
               mean_fn = None,
               mean_fn_prior = None):

     super().__init__(1, kernel=kernel,
                         lengthscale_prior_dist=lengthscale_prior_dist,
                         noise_prior_dist = noise_prior_dist,
                         mean_fn = mean_fn,
                         mean_fn_prior = mean_fn_prior)
     self.keys = gpax.utils.get_keys(1)
     self.X_test  = None
     self.X_train = None
     self.y_train = None

  def iteration(self, X_train, y_train, X_test):
    self.X_test  = X_test
    self.X_train = X_train
    self.y_train = y_train

    self.fit(self.keys[0], X_train, y_train)
    self.y_mean, self.y_sampled = self.predict(self.keys[1], X_test, noiseless=True)

  def visualize_res(self, vae_latents=[], **kwargs):
    if self.X_test is None:
      raise ValueError('You must perform iteration first')

    fig = plt.figure(figsize=(6, 4))
    plt.plot(self.y_mean, c='red', label='Prediction')
    plt.fill_between(self.X_test,
                      self.y_mean - self.y_sampled.squeeze().std(0),
                      self.y_mean + self.y_sampled.squeeze().std(0),
                      alpha=0.3, color='red')

    plt.scatter(self.X_train,
                self.y_train,
                c='red',
                label='Experiment',
                alpha=0.6)
    if len(vae_latents) > 0:
        plt.plot(self.X_test,
                 vae_latents, '--',
                 color='black',
                 label='Ground Truth',
                 alpha=0.5)

    plt.legend(loc='lower center', fontsize=8, ncol=3)
    plt.grid(True)
    plt.show()
    return fig


  def acquisition(self, gp_model, acq_func='EI',  maximize=True, visualize=False):
    """
    Future: Sending model here is not the best practice, ideally self.model should be used, but that seems to give error
    """
    if acq_func == 'EI':
      _acq = gpax.acquisition.EI(self.keys[1], gp_model, self.X_test, maximize=maximize, noiseless=True)
    elif acq_func == 'MU':
      _acq = gpax.acquisition.UE(self.keys[1], gp_model, self.X_test, maximize=maximize, noiseless=True)
    elif acq_func == 'UCB':
      _acq = gpax.acquisition.UCB(self.keys[1], gp_model, self.X_test, maximize=maximize, noiseless=True)
    else:
      raise ValueError('Unknown acquisition function')

    _acq = np.array(_acq)
    _acq[np.isnan(_acq)==True] = np.random.uniform(low=0.0, high=1.0, size=_acq[np.isnan(_acq)==True].shape)*1e-3
    self.acq = _acq

    maxacq_idx = _acq.argmax()
    maxacq_val = _acq.max()
    self.nextpt = self.X_test[maxacq_idx]

    fig = None
    if visualize:
      fig = self.vis_acq(nextpt=self.nextpt, maxacq_val=maxacq_val)

    return maxacq_idx, maxacq_val, fig



  def vis_acq(self, nextpt=None, maxacq_val=None):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(self.acq, c='r', label="Acquisition function")

    if (nextpt is not None) and (maxacq_val is not None):
      plt.scatter(x=nextpt, y=maxacq_val, color='k', marker = 'X', s=100, label="Next point")

    plt.legend(loc='lower center', fontsize=8)
    plt.title("Acquisition function")
    plt.grid()
    plt.show()
    return fig

  def get_statistics(self,):
    dict_res = {}

    dict_res['X_train']       = np.array(self.X_train).astype(int)
    dict_res['y_train']       = np.array(self.y_train).astype(float)

    dict_res['X_test']        = np.array(self.X_test).astype(int)

    dict_res['noise']         = self.get_samples(1)['noise'].squeeze().mean(0).astype(float)
    dict_res['k_length']      = self.get_samples(1)['k_length'].squeeze().mean(0).astype(float)

    dict_res['y_res_mean']    = np.array(self.y_mean).astype(float)
    dict_res['y_res_uncertainty'] = np.array(tuple(x.squeeze().std(0) for x in self.y_sampled)).astype(float)
    dict_res['next_point'] = np.array(self.nextpt).astype(int)

    return dict_res