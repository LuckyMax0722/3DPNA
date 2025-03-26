import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def main():
    label_src = 'other,car,bicycle,motorcycle,truck,other-vehicle,person,bicyclist,motorcyclist,road,parking,sidewalk,other-ground,building,fence,vegetation,trunk,terrain,pole,traffic-sign'
    labels = label_src.split(',')

    img = '/u/home/caoh/datasets/SemanticKITTI/dataset/sequences/00/image_2/000140.png'
    img = plt.imread(img)


    # seg
    segmentation = np.load("/u/home/caoh/datasets/SemanticKITTI/dataset/seg/Lseg/00/000140.npy")
    
    segmentation = segmentation[0]


    cmap = plt.get_cmap('tab20', 20)
    norm = BoundaryNorm(np.arange(-0.5, 20, 1), cmap.N)


    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title("ori img")


    im = axes[1].imshow(segmentation, cmap=cmap, norm=norm)
    axes[1].axis('off')
    axes[1].set_title("Lseg vis demo")

    cbar = fig.colorbar(im, ax=axes[1], ticks=np.arange(0, 20))
    cbar.ax.set_yticklabels(labels)
    cbar.set_label("class")

    output_image = "/u/home/caoh/projects/MA_Jiachen/3DPNA/demo/file_0_visualization.png"

    output_dir = os.path.dirname(output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()

if __name__ == '__main__':
    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/tools/img_seg_vis.py
    main()