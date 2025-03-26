import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pytorch_lightning as pl

from projects.model.lseg.models import LSegNet
import torchvision.transforms as transforms

# Vis
from PIL import Image
import matplotlib.patches as mpatches

from configs.config import CONF

def resize_image(img, h, w):
    return F.interpolate(img, (h, w), mode='bilinear', align_corners=True)

def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)

def pad_image(img, mean, std, crop_size):
    b,c,h,w = img.shape #.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad

def get_image(img_path):
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

    return image

def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0,n):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while (lab > 0):
                pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                i = i + 1
                lab >>= 3
    return pallete

def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    out_img.putpalette(new_palette)
    
    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            label = labels[index]
            cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0, new_palette[index * 3 + 2] / 255.0]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

class LSegModule(pl.LightningModule):
    def __init__(
        self, 
        backbone,
        num_features,
        arch_option,
        block_depth,
        activation,
        **kwargs):

        super(LSegModule, self).__init__()

        self.base_size = 520
        self.crop_size = 480
        self.scales = ([0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25])

        self.norm_mean= [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]

        self.labels = self.get_labels()

        self.net = LSegNet(
            labels=self.labels,
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

        if True:
            for param in self.parameters():
                param.requires_grad = False
        
        self.img_seg_dir = os.path.join(CONF.PATH.DATA_SEG, 'Lseg')


    def forward(self, x):
        return self.net(x)
    
    def model_infer(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        
    def model_inference(self, image, flip=True):
        output = self.model_infer(image)

        if flip:
            fimg = flip_image(image)
            foutput = self.model_infer(fimg)
            output += flip_image(foutput)

        return output

    def infer(self, image, vis=False):
        batch, _, h, w = image.size()  # cat 1: 1 ,360, 480

        assert(batch == 1)
        self.nclass = len(self.labels)
        stride_rate = 2.0/3.0
        stride = int(self.crop_size * stride_rate)

        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch,self.nclass,h,w).zero_()
        
        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            
            # resize image to current size
            cur_img = resize_image(image, height, width)

            if long_size <= self.crop_size:
                pad_img = pad_image(cur_img, self.norm_mean, self.norm_std, self.crop_size)
                outputs = self.model_inference(pad_img)
                outputs = crop_image(outputs, 0, height, 0, width)

            else:
                if short_size < self.crop_size:
                    # pad if needed
                    pad_img = pad_image(cur_img, self.norm_mean, self.norm_std, self.crop_size)
                else:
                    pad_img = cur_img
                
                _,_,ph,pw = pad_img.shape #.size()
                assert(ph >= height and pw >= width)

                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph-self.crop_size)/stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw-self.crop_size)/stride)) + 1

                with torch.cuda.device_of(image):
                    outputs = image.new().resize_(batch,self.nclass,ph,pw).zero_()
                    count_norm = image.new().resize_(batch,1,ph,pw).zero_()

                 # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + self.crop_size, ph)
                        w1 = min(w0 + self.crop_size, pw)
                        crop_img = crop_image(pad_img, h0, h1, w0, w1)

                        # pad if needed
                        pad_crop_img = pad_image(crop_img, self.norm_mean, self.norm_std, self.crop_size)

                        output = self.model_inference(pad_crop_img)
                        outputs[:,:,h0:h1,w0:w1] += crop_image(output,0, h1-h0, 0, w1-w0)
                        count_norm[:,:,h0:h1,w0:w1] += 1

                assert((count_norm==0).sum()==0)  
                outputs = outputs / count_norm
                outputs = outputs[:,:,:height,:width]

            score = resize_image(outputs, h, w)
            scores += score

        # print(score[0].size()) torch.Size([20, 376, 1241])

        if vis:
            self.vis(image, scores)
            return scores
        else:
            return scores
                
    def vis(self, image, outputs):
        predict = torch.max(outputs[0].unsqueeze(0), 1)[1].cpu().numpy()

        # show results
        new_palette = get_new_pallete(len(self.labels))
        mask, patches = get_new_mask_pallete(predict, new_palette, out_label_flag=True, labels=self.labels)
        img = image[0].permute(1,2,0).cpu()
        img = img * 0.5 + 0.5
        img = Image.fromarray(np.uint8(255*img)).convert("RGBA")
        seg = mask.convert("RGBA")
        out = Image.blend(img, seg, alpha=0.5)


        if not os.path.exists(CONF.PATH.DEMO):
            os.makedirs(CONF.PATH.DEMO)

        out.save(os.path.join(CONF.PATH.DEMO, "segmentation_result.png"))
        img.save(os.path.join(CONF.PATH.DEMO, "original_image.png"))
        seg.save(os.path.join(CONF.PATH.DEMO, "segmentation_mask.png"))

        print(f'vis images are svaed to {CONF.PATH.DEMO}')

    def get_labels(self):
        label_src = 'other,car,bicycle,motorcycle,truck,other-vehicle,person,bicyclist,motorcyclist,road,parking,sidewalk,other-ground,building,fence,vegetation,trunk,terrain,pole,traffic-sign'

        labels = []
        lines = label_src.split(',')
        for line in lines:
            label = line
            labels.append(label)
        
        return labels

    def predict_step(self, batch, batch_idx):
        img_seg_dir_seq = os.path.join(self.img_seg_dir, batch['sequence'][0])
        
        check_path(img_seg_dir_seq)

        scores = self.infer(batch['img_seg'])

        predict = torch.max(scores[0].unsqueeze(0), 1)[1].cpu().numpy()

        output_file = os.path.join(img_seg_dir_seq, batch['frame_id'][0])

        np.save(output_file, predict)




if __name__ == '__main__':
    img_path = '/u/home/caoh/projects/MA_Jiachen/lang-seg/inputs/SK.png'

    image = get_image(img_path).cuda()

    l = LSegModule.load_from_checkpoint(
        checkpoint_path='/u/home/caoh/projects/MA_Jiachen/3DPNA/ckpts/demo_e200.ckpt', 
        backbone='clip_vitl16_384',
        num_features=256,
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    ).eval().cuda()

    with torch.no_grad():
        scores = l.infer(image, vis=False)

    predict = torch.max(scores[0].unsqueeze(0), 1)[1].cpu().numpy()

    print(predict.shape)
    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/lseg/lseg_net.py