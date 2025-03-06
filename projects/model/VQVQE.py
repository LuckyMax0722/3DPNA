import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
#from utils.loss import lovasz_softmax
from projects.model.vqvae_modules import C_Encoder, C_Decoder
from projects.model.vqvae_modules import VectorQuantizer

class vqvae(nn.Module):
    def __init__(
        self, 
        num_classes,
        init_size,
        l_size,
        l_attention,
        vq_size,
        ) -> None:
        super(vqvae, self).__init__()
        
        self.encoder = C_Encoder(nclasses=num_classes, init_size=init_size, l_size=l_size, attention=l_attention)
        self.quant_conv = nn.Conv3d(num_classes, num_classes, kernel_size=1, stride=1)

        self.VQ = VectorQuantizer(num_embeddings = num_classes * vq_size, embedding_dim = num_classes)

        self.post_quant_conv = nn.Conv3d(num_classes, num_classes, kernel_size=1, stride=1)
        self.decoder = C_Decoder(nclasses=num_classes, init_size=init_size, l_size=l_size, attention=l_attention)

    
    def encode(self, x):
        latent, down4b, down3b, down2b, down1b = self.encoder(x)  
        '''
        latent: torch.Size([1, 20, 16, 16, 8])
        down4b: torch.Size([1, 512, 32, 32, 8])
        down3b: torch.Size([1, 256, 64, 64, 8])
        down2b: torch.Size([1, 128, 128, 128, 16])
        down1b torch.Size([1, 64, 256, 256, 32])
        '''

        latent =  self.quant_conv(latent)

        return latent

    def vq(self, latent):
        quantized_latent, vq_loss, quantized_latent_ind, latents_shape = self.VQ(latent)
        
        return quantized_latent, vq_loss, quantized_latent_ind, latents_shape

    def decode(self, quantized_latent):
        quantized_latent = self.post_quant_conv(quantized_latent)
        recons = self.decoder(quantized_latent) # torch.Size([1, 20, 256, 256, 32])
        
        return recons

    def forward(self, x):
        x = self.encode(x)

        quantized_latent, vq_loss, _, _ = self.vq(x)

        x1, x2, x3 = self.decode(quantized_latent) 

        return x1, x2, x3, vq_loss



if __name__ == '__main__':
    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/VQVQE.py
    v = vqvae(
        num_classes = 20,
        init_size = 32,
        l_size = '882',
        l_attention = True,

        vq_size = 50
    ).cuda()

    x = np.load('/u/home/caoh/datasets/SemanticKITTI/dataset/pred/MonoScene/00/000000.npy')


    x = torch.from_numpy(x).long().cuda().unsqueeze(0)

    
    v(x)

