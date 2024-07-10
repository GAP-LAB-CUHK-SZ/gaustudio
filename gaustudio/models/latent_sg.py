from gaustudio.models.vanilla_sg import VanillaPointCloud
from gaustudio import models

import torch

@models.register('latent_pcd')
class LatentPointCloud(VanillaPointCloud):
    default_conf = {
        'sh_degree': 0,        
        'attributes':  {
            "xyz": 3, 
            'opacity': 1,
            "f_dc": 16,
            "f_rest": 0,
            "scale": 3,
            "rot": 16
        },
        'activations':{
            "scale": "exp",
            "opacity": "sigmoid",
            "rot": "normalize"
        }
    }

    @property
    def get_features(self):
        features_dc = self._f_dc
        features_rest = self._f_rest
        return torch.cat((features_dc, features_rest), dim=1)
