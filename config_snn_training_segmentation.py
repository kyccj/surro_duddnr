'''
    Configuration for SNN direct training

'''

# GPU setting
import os
os.environ['NCCL_P2P_DISABLE']='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#
from config import config
conf = config.flags

conf.task = 'segmentation'

conf.model = 'ResNet50_DeepLab_V3'
conf.dataset = 'ADE20K'


config.set()