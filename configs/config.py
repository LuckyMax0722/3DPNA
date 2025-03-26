import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# Path
CONF.PATH = EasyDict()
CONF.PATH.BASE = '/u/home/caoh/projects/MA_Jiachen/3DPNA'  # TODO: Change path to your SGN-dir

## data
CONF.PATH.DATA_ROOT = '/u/home/caoh/datasets/SemanticKITTI/dataset'
CONF.PATH.DATA_LABEL = os.path.join(CONF.PATH.DATA_ROOT, 'labels')
CONF.PATH.DATA_SEG = os.path.join(CONF.PATH.DATA_ROOT, 'seg')

## log
CONF.PATH.LOG_DIR = os.path.join(CONF.PATH.BASE, 'output_new')

## ckpt
CONF.PATH.CKPT_DIR = os.path.join(CONF.PATH.BASE, 'ckpts')
CONF.PATH.CKPT_RESNET = os.path.join(CONF.PATH.CKPT_DIR, 'resnet50-19c8e357.pth')
CONF.PATH.CKPT_LSEG = os.path.join(CONF.PATH.CKPT_DIR, 'demo_e200.ckpt')
CONF.PATH.CKPT_SWIN = '/u/home/caoh/projects/MA_Jiachen/CGFormer/ckpts/swin_tiny_patch4_window7_224.pth'

## config
CONF.PATH.CONFIG_DIR = os.path.join(CONF.PATH.BASE, 'configs')

# Demo
CONF.PATH.DEMO = os.path.join(CONF.PATH.BASE, 'demo')
#
CONF.semantic_kitti_class_frequencies = [
        5.41773033e09, 1.57835390e07, 1.25136000e05, 1.18809000e05, 6.46799000e05,
        8.21951000e05, 2.62978000e05, 2.83696000e05, 2.04750000e05, 6.16887030e07,
        4.50296100e06, 4.48836500e07, 2.26992300e06, 5.68402180e07, 1.57196520e07,
        1.58442623e08, 2.06162300e06, 3.69705220e07, 1.15198800e06, 3.34146000e05,
    ]
