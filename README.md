# [IWAEs](report.pdf): Importance Weighted Autoencoders Implementation
The goal of this project is to re-implement the paper by Burda et. al ”Importance Weighted Autoencoders”.

The theory behind regular Variational Autoencoders (VAE) and Importance Weighted Autoencoders (IWAE) is pre-
sented in this research, which compares and contrasts them. This paper also presents the related work done in this
field, where we analyse the results of three significant and recently released papers in the matter. In regards of the im-
plementation, we explain the methodology used such as the evaluation on density estimation, the weight initialization,
binarization, NLL, and the results of fitting a one-dimensional synthetic dataset. We can reproduce the original paper
showing that we reach similar scores. 

We also, as in the original paper, find that IWAEs are better at utilizing their
network capabilities than VAEs by learning more expressive latent representations which often results in improved
log-likelihood measurements. 

Furthermore, IWAE reaches better results than VAE on all MNIST models, as measured
by NLL and which we can generalize to all models.