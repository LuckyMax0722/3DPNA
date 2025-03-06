import os

CONFIG_DIR = '/u/home/caoh/projects/MA_Jiachen/3DPNA/configs/config.py'
baseline_model = 'CGFormer'


yaml_config = os.path.join(CONFIG_DIR, ('REF_' + baseline_model + '.yaml'))

print(yaml_config)