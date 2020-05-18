# Deep_Discriminative_Clustering
Latent Space Clustering.
When training the Autoencoder, jointly optimise reconstruction loss and latent loss, defined as maximising the cosine similarity between anchor points.

Anchor points are defined as the K_Nearest_Neighbours and in real world datasets, this can be based on a subset of features. This will make the latent representaions orient towards distinguishing this subset of features.

The idea orginates from the paper Deep Discriminative Latent Space for Clustering.(https://arxiv.org/pdf/1805.10795.pdf) But when clustering is not task-specific, I think the clustring phase in the paper was sligthly overdone, as they only serve to increase the separatebility in the latent space and could make clusteirng unstable given different number of clusters. Therefore, only the first part of the paper is implemented. 


Dataset: MNIST 
Macihine Learning Methods: Autoencoder


 * [model](./model)
   * [model.py](./model/model.py)
   * [custom_layer.py](./model/custom_layer.py)
 * [src](./src)
   * [load_data.py](./src/load_data.py)
   * [metrics.py](./src/metrics.py)
 * [demonstration.ipynb](./demonstration.ipynb)
