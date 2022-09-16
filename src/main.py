import sys
import subprocess
from datasets import BinarizedMNIST
import datasets
from vae import VAE

import torch
import random
import numpy as np



from experiments.vae_k_1_layers_2 import experiment as vae_k_1_layers_2
from experiments.vae_k_5_layers_2 import experiment as vae_k_5_layers_2
from experiments.vae_k_50_layers_2 import experiment as vae_k_50_layers_2

from experiments.vae_k_1_layers_1 import experiment as vae_k_1_layers_1
from experiments.vae_k_5_layers_1 import experiment as vae_k_5_layers_1
from experiments.vae_k_50_layers_1 import experiment as vae_k_50_layers_1

from experiments.iwae_k_5_layers_2 import experiment as iwae_k_5_layers_2
from experiments.iwae_k_50_layers_2 import experiment as iwae_k_50_layers_2

from experiments.iwae_k_5_layers_1 import experiment as iwae_k_5_layers_1
from experiments.iwae_k_50_layers_1 import experiment as iwae_k_50_layers_1
from experiments.vae_k_5_layers_1_other_seed import experiment as vae_k_5_layers_1_other_seed

import experiment, utils

def main():
    ''' TODO:
            - Do same experiments as authors:
                1. Density Estimation
                    1.1 MNIST: 
                        1.1.1 VAE: 
                            stochastic_layers = 1 & k = [1, 5, 50]
                            stochastic_layers = 2 & k = [1, 5, 50]
                        1.1.2 IWAE:
                            stochastic_layers = 1 & k = [1, 5, 50]
                            stochastic_layers = 2 & k = [1, 5, 50]
                    1.2 OMNIGLOT:
                        1.2.1 VAE: 
                            stochastic_layers = 1 & k = [1, 5, 50]
                            stochastic_layers = 2 & k = [1, 5, 50]
                        1.2.2 IWAE:
                            stochastic_layers = 1 & k = [1, 5, 50]
                            stochastic_layers = 2 & k = [1, 5, 50]
                2. Latent space representation:
                    2.1 Best VAE -> keep training it as IWAE(k=50)
                    2.2 Best IWAE(k=50) -> keep training it as VAE
            
            - Run our own experiments:
                1. Venia's simple experiment: proves IWAEs superiority over VAEs
                2. FashionMNIST
    '''

    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    exper = vae_k_50_layers_1
    _, _, model_bias = utils.setup_data(exper["data"])
    model, _ = utils.setup_model(exper["model"], model_bias)
    experiment.load(model, 'vae_k_50_1_layers_1_chkp.pt')
    xs = [utils.interpolate_X(model, exper, amount=10) for _ in range(10)]
    f, a = utils.plot_images(xs)
    import matplotlib.pyplot as plt
    plt.show()
    


if __name__ == "__main__":
    main()
