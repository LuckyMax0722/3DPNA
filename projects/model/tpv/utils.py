import torch

import numpy as np

from functools import lru_cache

@lru_cache(4)
def voxel_coord(voxel_shape):
    x = np.arange(voxel_shape[0])
    y = np.arange(voxel_shape[1])
    z = np.arange(voxel_shape[2])
    Y, X, Z = np.meshgrid(x, y, z)
    voxel_coord = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)
    return voxel_coord

def make_query(grid_size):
    gs = grid_size[1:]
    coords = torch.from_numpy(voxel_coord(gs))
    coords = coords.reshape(-1, 3)
    query = torch.zeros(coords.shape, dtype=torch.float32)
    query[:,0] = 2*coords[:,0]/float(gs[0]-1) -1
    query[:,1] = 2*coords[:,1]/float(gs[1]-1) -1
    query[:,2] = 2*coords[:,2]/float(gs[2]-1) -1
    
    query = query.reshape(-1, 3)
    return coords.unsqueeze(0), query.unsqueeze(0)

def get_pred(model_output, separate_decoder=False):
    preds = model_output
    pred_prob = torch.softmax(preds, dim=2)  # torch.Size([1, 262144, 20])
    pred_mask = pred_prob.argmax(dim=2).float()  # torch.Size([1, 262144])
    return pred_prob, pred_mask

def point2voxel(grid_size, pred_prob, pred_mask, coords):
    output_prob = torch.zeros((pred_prob.shape[0], pred_prob.shape[2], grid_size[1], grid_size[2], grid_size[3]), device=pred_prob.device)
    output_mask = torch.zeros((pred_mask.shape[0], grid_size[1], grid_size[2], grid_size[3]), device=pred_mask.device)

    for i in range(pred_mask.shape[0]):
        output_mask[i, coords[i, :, 0], coords[i, :, 1], coords[i, :, 2]] = pred_mask[i]

    for i in range(pred_prob.shape[0]):
        output_prob[i, :, coords[i, :, 0], coords[i, :, 1], coords[i, :, 2]] = pred_prob[i].transpose(0, 1)

    return output_prob, output_mask

if __name__ == '__main__':
    grid_size = (1, 256, 256, 32)

    coords, query = make_query(grid_size)

    print(query.size())

    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/model/tpv/utils.py