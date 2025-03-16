import torch
import torch.nn as nn

from projects.model.lseg.models import LSegNet

class LSegModule(nn.Module):
    def __init__(
        self, 
        ckpt_path, 
        backbone,
        num_features,
        arch_option,
        block_depth,
        activation,
        **kwargs):

        super(LSegModule, self).__init__()

        self.base_size = 520
        self.crop_size = 480

        use_pretrained = True
        norm_mean= [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]

        labels = self.get_labels()

        self.net = LSegNet(
            labels=labels,
            backbone=backbone,
            features=num_features,
            crop_size=self.crop_size,
            arch_option=arch_option,
            block_depth=block_depth,
            activation=activation,
        )

        self.net.pretrained.model.patch_embed.img_size = (
            self.crop_size,
            self.crop_size,
        )

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            #self.net.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
            
            if not isinstance(checkpoint, dict):
                raise RuntimeError(
                    f'No state_dict found in checkpoint file {filename}')
            # get state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            metadata = getattr(state_dict, '_metadata', OrderedDict())
            print(metadata)

        if True:
            for param in self.parameters():
                param.requires_grad = False

        self.eval()

    def forward(self, x):
        return self.net(x)

    def get_labels(self):
        label_src = 'car,bicycle,motorcycle,truck,vehicle,person,bicyclist,motorcyclist,road,parking,sidewalk,other-ground,building,fence,vegetation,trunk,terrain,pole,traffic-sign,other'

        labels = []
        lines = label_src.split(',')
        for line in lines:
            label = line
            labels.append(label)
        
        return labels

