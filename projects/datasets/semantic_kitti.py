import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset

class SemanticKITTIDataset(Dataset):
    def __init__(
        self,
        data_root,
        ann_file,
        pred_model,
        split,
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        test_mode=False
    ):
        self.splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["08"],
            "test_submit": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }

        self.sequences = self.splits[split]

        self.data_root = data_root
        self.ann_file = ann_file
        self.pred_model = pred_model
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.test_mode = test_mode

        self.data_infos = self.load_annotations(self.ann_file)

    def __len__(self):
        return len(self.data_infos)

    def convert_to_tensor(self, input_dict):
        # Convert numpy arrays to tensors and set data types
        if input_dict['input_occ'] is not None:
            input_dict['input_occ'] = torch.from_numpy(input_dict['input_occ']).long()
        if input_dict['gt_occ'] is not None:
            input_dict['gt_occ'] = torch.from_numpy(input_dict['gt_occ']).long()
        if input_dict['gt_occ_2'] is not None:
            input_dict['gt_occ_2'] = torch.from_numpy(input_dict['gt_occ_2']).long()
        if input_dict['gt_occ_4'] is not None:
            input_dict['gt_occ_4'] = torch.from_numpy(input_dict['gt_occ_4']).long()
        if input_dict['gt_occ_8'] is not None:
            input_dict['gt_occ_8'] = torch.from_numpy(input_dict['gt_occ_8']).long()
        
        return input_dict

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
        
        return self.convert_to_tensor(input_dict)
    
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

        return self.convert_to_tensor(input_dict)
    
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

        # load input voxels
        input_dict['input_occ'] = self.get_input_info(index, key='occ_path')

        # gt_occ is None for test-set
        input_dict['gt_occ'] = self.get_ann_info(index, key='voxel_path')
        input_dict['gt_occ_2'] = self.get_ann_info(index, key='voxel_path_2')
        input_dict['gt_occ_4'] = self.get_ann_info(index, key='voxel_path_4')
        input_dict['gt_occ_8'] = self.get_ann_info(index, key='voxel_path_8')
        
        return input_dict

    def get_ann_info(self, index, key='voxel_path'):
        info = self.data_infos[index][key]
        return None if info is None else np.load(info)
    
    def get_input_info(self, index, key='occ_path'):
        info = self.data_infos[index][key]
        return None if info is None else np.load(info)

    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:

            voxel_base_path = os.path.join(self.ann_file, sequence)
                        

            id_base_path = os.path.join(self.data_root, "pred", self.pred_model, sequence, '*.npy')

            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]

                # gt
                voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')
                voxel_path_2 = os.path.join(voxel_base_path, img_id + '_1_2.npy')
                voxel_path_4 = os.path.join(voxel_base_path, img_id + '_1_4.npy')
                voxel_path_8 = os.path.join(voxel_base_path, img_id + '_1_8.npy')

                
                # for sweep demo or test submission
                if not os.path.exists(voxel_path):
                    voxel_path = None
                if not os.path.exists(voxel_path_2):
                    voxel_path_2 = None
                if not os.path.exists(voxel_path_4):
                    voxel_path_4 = None
                if not os.path.exists(voxel_path_8):
                    voxel_path_8 = None

                scans.append(
                    {   
                        "sequence": sequence,
                        "frame_id": img_id,
                        "occ_path": id_path,
                        "voxel_path": voxel_path,
                        "voxel_path_2": voxel_path_2,
                        "voxel_path_4": voxel_path_4,
                        "voxel_path_8": voxel_path_8,
                    })
                
        return scans  # return to self.data_infos


if __name__ == '__main__':
    s = SemanticKITTIDataset(
        data_root='/u/home/caoh/datasets/SemanticKITTI/dataset',
        ann_file='/u/home/caoh/datasets/SemanticKITTI/dataset/labels',
        pred_model='CGFormer',
        split='train',
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
    )

    print(s[0]['input_occ'])
    #print(s[0]['gt_occ'])
    #print(s[0]['gt_occ_2'])