import numpy as np

from PIL import Image
import torchvision.transforms as transforms

from projects.model.vqvae_modules import VectorQuantizer
from projects.model.lseg import LSegModule

if __name__ == '__main__':
    img_path = '/u/home/caoh/projects/MA_Jiachen/lang-seg/inputs/SK.png'
    crop_size = 480
    padding = [0.0] * 3
    image = Image.open(img_path)
    image = np.array(image)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image = transform(image).unsqueeze(0)
    img = image[0].permute(1,2,0)
    img = img * 0.5 + 0.5



    l = LSegModule(
        ckpt_path='/u/home/caoh/projects/MA_Jiachen/3DPNA/ckpts/demo_e200.ckpt', 
        backbone='clip_vitl16_384',
        num_features=256,
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )



    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/rh_tpv_seg.py