import os
import numpy as np
import glob
import torch

from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset

class SemanticKITTILsegDataset(Dataset):
    def __init__(
        self,
        data_root,
        ann_file,
        pred_model,
        split,
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        test_mode=False,

        img_config={
            'input_size': (384, 1280),
            'resize': (0., 0.),
            'rot': (0.0, 0.0 ),
            'flip': (0.0, 0.0 ),
            'flip': False,
            'crop_h': (0.0, 0.0),
            'resize_test': 0.00,
        },
        color_jitter=(0.4, 0.4, 0.4)
    ):
        self.splits = {
            "predict": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
        }
        self.split = split
        self.sequences = self.splits[split]

        self.data_root = data_root
        self.ann_file = ann_file
        self.pred_model = pred_model
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.test_mode = test_mode

        self.data_infos = self.load_annotations(self.ann_file)

        self.img_config = img_config

        color_jitter = False
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.normalize_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.normalize_img_seg = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],  std=[0.5, 0.5, 0.5]
                ),
            ]
        )

    def __len__(self):
        return len(self.data_infos)


    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        
        return input_dict
    
    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None

        return input_dict
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)

        while True:
            data = self.prepare_train_data(idx)
            return data
    
    def get_data_info(self, index):
        info = self.data_infos[index]
        '''
        sample info includes the following:
            "sequence": sequence,
            "frame_id": frame_id,
            "voxel_path": voxel_path,
            "voxel_path_2": voxel_path_2,
            "voxel_path_4": voxel_path_4,
            "voxel_path_8": voxel_path_8,
        '''

        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            sequence = info['sequence'],
            frame_id = info['frame_id'],
        )
        
        # load images
        input_dict['img'], input_dict['img_seg'] = self.get_images_info(index, key='img_2_path')

        return input_dict

    def get_images_info(self, index, key='img_2_path'):
        info = self.data_infos[index][key]
        
        return self.load_image(info)


    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.img_config['input_size']
        
        if self.split == 'train':
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.img_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.img_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.img_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.img_config['rot'])

        else:
            resize = float(fW) / float(W)
            resize += self.img_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale

            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.img_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0

        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        
        return img

    def load_image(self, img_filename, flip=None, scale=None):
        img = Image.open(img_filename).convert('RGB')

        # perform image-view augmentation
        post_rot = torch.eye(2)
        post_trans = torch.zeros(2)

        img_augs = self.sample_augmentation(H=img.height, W=img.width, flip=flip, scale=scale)

        resize, resize_dims, crop, flip, rotate = img_augs

        img, post_rot2, post_tran2 = self.img_transform(
            img, post_rot, post_trans, resize=resize, 
            resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate
        )

        if self.color_jitter and self.split == 'train':
            img = self.color_jitter(img)

        img_seg = img.copy()

        img = self.normalize_img(img)
        img_seg = self.normalize_img_seg(img_seg)
        return img, img_seg


    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:

            voxel_base_path = os.path.join(self.ann_file, sequence)
            img_base_path = os.path.join(self.data_root, "sequences", sequence)   

            id_base_path = os.path.join(self.data_root, "pred", self.pred_model, sequence, '*.npy')

            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]

                # image
                img_2_path = os.path.join(img_base_path, 'image_2', img_id + '.png')
                
                scans.append(
                    {   
                        "sequence": sequence,
                        "frame_id": img_id,
                        "img_2_path": img_2_path,
                    })
                
        return scans  # return to self.data_infos


if __name__ == '__main__':
    s = SemanticKITTILsegDataset(
        data_root='/u/home/caoh/datasets/SemanticKITTI/dataset',
        ann_file='/u/home/caoh/datasets/SemanticKITTI/dataset/labels',
        pred_model='CGFormer',
        split='predict',
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
    )

    print(s[0])
    #print(s[0]['gt_occ'])
    #print(s[0]['gt_occ_2'])