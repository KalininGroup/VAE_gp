import matplotlib.pyplot as plt

def plot_ld(train, test = None, highlight_latest=True, return_fig=False):
  """
        Plot VAE latent distribution

        Parameters
        ----------
        train: dataset used to train VAE

        test:  full dataset, if available

        highlight_latest: emphasis point, correspondent to the last selected location

        return_fig: return fig object (required if we would like to make video)

        Returns
        -------
        None of plt.fig

  """
  fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
  fig.subplots_adjust(wspace=.3)

  if test is not None:
    ax.scatter(test[:,-2], test[:,-1], color='grey', s=3)
  ax.scatter(train[:,-2], train[:,-1], color='red', s=20, label = 'training')
  if highlight_latest:
    ax.scatter(train[-1,-2], train[-1,-1], color='red', s=60, marker='x', label = 'last_added_point')


  ax.legend()
  for _ in [ax,]:
    _.grid()
    _.set_xlabel('z1')
    _.set_ylabel('z2')
    _.tick_params('both', direction='in')
  plt.show()

  if return_fig:
    return fig