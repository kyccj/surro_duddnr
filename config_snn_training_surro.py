'''
    Configuration for SNN direct training

'''

# GPU setting
import os
os.environ['NCCL_P2P_DISABLE']='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


#
from config import config
conf = config.flags

##### training setting #####
conf.debug_mode = True
# conf.verbose_snn_train = True
conf.save_best_model_only = True
conf.save_models_max_to_keep = 1
##########

##### inference mode setting #####
# conf.mode='inference'
#conf.batch_size=400
# conf.name_model_load='/home/dydwls6598/PycharmProjects/Surro/model_ckpt_test/predictiveness_asy_all_timestep_sim_1.0_1.0_accumulate_5epoch/VGG16_AP_CIFAR10/ep-310_bat-100_opt-ADAMW_lr-COS-1E-05 to 6E-03_wd-2E-02_sc_ra_cm_re_ts-4_nc-R-R_nr-s/'
##########

##### hyper-parameter setting #####
conf.optimizer = 'ADAMW'
conf.lr_schedule = 'COS'

conf.learning_rate_init = 1e-5
conf.learning_rate = 6e-3
conf.weight_decay_AdamW = 2e-2
##########

##### neural network type setting #####
# conf.nn_mode = 'SNN'
conf.nn_mode = 'ANN'

conf.pooling_vgg = 'avg'
##########

##### augmentation setting #####
conf.label_smoothing=0.1
conf.debug_lr = True
conf.lmb=1E-3
conf.regularizer=None
#conf.data_aug_mix='mixup'

conf.mix_off_iter = 500*200
conf.mix_alpha = 0.5

# data augmentation
conf.randaug_en = True
conf.randaug_mag = 0.9
conf.randaug_mag_std = 0.4
conf.randaug_n = 1
conf.randaug_rate = 0.5

conf.rand_erase_en = True
##########

#### surrogate function setting #####

# surrogate function shape
# conf.fire_surro_grad_func = 'boxcar'
conf.fire_surro_grad_func = 'boxcar_height_fix'
# conf.fire_surro_grad_func = 'triangle'
# conf.fire_surro_grad_func = 'triangle_height_fix'
# conf.fire_surro_grad_func = 'asy'
# conf.fire_surro_grad_func = 'asy_height_fix'


# adaptive surrogate gradients
# conf.adaptive_surrogate = True

if conf.adaptive_surrogate == True :
    # conf.sparsity_aware_gradient_consistency = True
    # conf.temporal_gradient_consistency = True
    conf.surro_grad_beth = 0.5
    conf.find_beta_low = 0.1
    conf.find_beta_high = 0.5
    conf.train_beta_candidate_number = 30
    conf.test_beta_candidate_number_0 = 100
    conf.test_beta_candidate_number_1 = 30
    conf.accumulate_iteration = 500*1  #iteration * epoch
else :
    conf.surro_grad_beth = 1.0
##########

##### CPNG setting #####
# conf.chi_limit = 0.2
# conf.find_beta_low = 1
# conf.find_beta_high = 10.0
##########
conf.debug_grad = True
conf.debug_surro_grad = True
# conf.plot_predictiveness_in_neurons = True
# conf.predictiveness_in_model = True
conf.gradient_sparsity_in_model = True
conf.gradient_sparsity_in_neuron = True

if conf.predictiveness_in_model :
    conf.debug_mode = True
    conf.debug_grad = True



##### model save setting #####
# conf.root_model_save = f'./model_ckpt_1/relu'
conf.root_model_save = f'./model_ckpt_1/{conf.fire_surro_grad_func}_beta={conf.surro_grad_beth}'
# conf.root_model_save = f'./model_ckpt_1/test'

##########
conf.exp_set_name = 'gradient_gsnr_0717'
# conf.exp_set_name = 'compare_boxcar_asy'
# conf.exp_set_name = 'compare_boxcar_asy_0415'
# conf.exp_set_name = '0417'
# conf.exp_set_name = '0421'
# conf.exp_set_name='surro_grad_new'
# conf.exp_set_name='CPNG'
# conf.exp_set_name='test'
# conf.exp_set_name='distribution'
# conf.exp_set_name='adaptive_boxcar'
# conf.exp_set_name='adaptive_asy'
# conf.exp_set_name='younguk_convergenece_rate'
# conf.exp_set_name='confirm_0401'
# conf.exp_set_name='predictiveness_0408'
# conf.exp_set_name='asy'
# conf.exp_set_name='younguk_convergence_rate'
# conf.exp_set_name = 'NeurIPS_2025_predictiveness_asy/vggsnn_please'
# conf.exp_set_name = 'test'
##### Loss setting #####
# conf.rmp_en = 'True'
# conf.rmp_k = 0.0005
# conf.im_en = 'True'
# conf.im_k = 0.001
##########

##### Model setting #####
###### VGG16
conf.SEL_model_dataset = 'V16_C10'
# conf.SEL_model_dataset = 'V16_C100'
# conf.SEL_model_dataset = 'V16_DVS'

###### VGG11
# conf.SEL_model_dataset = 'V11_DVS'

###### VGGSNN
# conf.SEL_model_dataset = 'VSNN_DVS'

###### ResNet19
# conf.SEL_model_dataset = 'R19_C10'
# conf.SEL_model_dataset = 'R19_C100'
# conf.SEL_model_dataset = 'R19_DVS'

###### MS ResNet19
# conf.SEL_model_dataset = 'MS_R19_C10'

###### ResNet20
# conf.SEL_model_dataset = 'R20_C10'
# conf.SEL_model_dataset = 'R20_C100'
# conf.SEL_model_dataset = 'R20_DVS'


##### Spikformer
# conf.SEL_model_dataset = 'Spik_C10'
# conf.SEL_model_dataset = 'Spik_C100'
# conf.SEL_model_dataset = 'Spik_Img'
# conf.SEL_model_dataset = 'Spik_DVS'
##########

if conf.SEL_model_dataset == 'V16_C10':
    conf.model='VGG16'
    conf.dataset = 'CIFAR10'
    if conf.im_en:
        conf.adaptive_dec_vth_scale = 0.8
        conf.reg_psp_SEL_const = 5e-6
        conf.reg_psp_SEL_BN_ratio_value = -1
        conf.reg_psp_SEL_BN_ratio_rate = 1e-4
    else:
        conf.adaptive_dec_vth_scale = 0.8
        conf.reg_psp_SEL_const = 5e-6
        conf.reg_psp_SEL_BN_ratio_value = -1
        conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'V16_C100':
    conf.model='VGG16'
    conf.dataset = 'CIFAR100'
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -0.8
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'V16_DVS':
    conf.model='VGG16'
    conf.dataset = 'CIFAR10_DVS'
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-5 # 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1 # -1
    conf.reg_psp_SEL_BN_ratio_rate = 1e-2 # 1e-3
elif conf.SEL_model_dataset == 'V11_DVS':
    conf.model = 'VGG11'
    conf.dataset = 'CIFAR10_DVS'
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-5  # 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1  # -1
    conf.reg_psp_SEL_BN_ratio_rate = 1e-2  # 1e-3
elif conf.SEL_model_dataset == 'VSNN_DVS':
    conf.model = 'VGGSNN'
    conf.dataset = 'CIFAR10_DVS'
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-5  # 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1  # -1
    conf.reg_psp_SEL_BN_ratio_rate = 1e-2  # 1e-3
elif conf.SEL_model_dataset == 'R19_C10':
    conf.model='ResNet19'
    conf.dataset = 'CIFAR10'
    conf.adaptive_dec_vth_scale = 0.8 # not fix
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1.5
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'MS_R19_C10':
    conf.model='ResNet19_MS'
    conf.dataset = 'CIFAR10'
    conf.adaptive_dec_vth_scale = 0.8 # not fix
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1.5
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'R19_C100':
    conf.model='ResNet19'
    conf.dataset = 'CIFAR100'
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 3e-3
    conf.reg_psp_SEL_BN_ratio_value = -0.4
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'R19_DVS':
    conf.model='ResNet19'
    conf.dataset = 'CIFAR10_DVS'
    conf.adaptive_dec_vth_scale = 0.8 #not fix
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1.5
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'R20_C10':
    conf.model='ResNet20'
    conf.dataset = 'CIFAR10'
    conf.adaptive_dec_vth_scale = 0.2
    conf.reg_psp_SEL_const = 3e-3
    conf.reg_psp_SEL_BN_ratio_value = -0.4
    conf.reg_psp_SEL_BN_ratio_rate = 1e-3
elif conf.SEL_model_dataset == 'R20_C100':
    conf.model='ResNet20'
    conf.dataset = 'CIFAR100'
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 3e-3
    conf.reg_psp_SEL_BN_ratio_value = -0.3
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'R20_DVS':
    conf.model='ResNet20'
    conf.dataset = 'CIFAR10_DVS'
    conf.adaptive_dec_vth_scale = 0.8 #not fix
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -1.5
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'MS34_ImageNet':
    conf.model = 'ResNet34_MS'
    conf.dataset = 'ImageNet'
    conf.adaptive_dec_vth_scale = 0.8  # not fix
    conf.reg_psp_SEL_const = 3e-3
    conf.reg_psp_SEL_BN_ratio_value = -1
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == '34_ImageNet':
    conf.model = 'ResNet34'
    conf.dataset = 'ImageNet'
    conf.adaptive_dec_vth_scale = 0.8  # not fix
    conf.reg_psp_SEL_const = 3e-3
    conf.reg_psp_SEL_BN_ratio_value = -1
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'Spik_C10':
    conf.model='Spikformer'
    conf.dataset = 'CIFAR10'
    conf.patch_size = 4
    conf.embed_dims = 384
    conf.num_heads = 12
    conf.depths = 4
    conf.sr_ratios = 8
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -0.8
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'Spik_C100':
    conf.model='Spikformer'
    conf.dataset = 'CIFAR100'
    conf.patch_size = 4
    conf.embed_dims = 384
    conf.num_heads = 12
    conf.depths = 4
    conf.sr_ratios = 8
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -0.8
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'Spik_Img':
    conf.model='Spikformer'
    conf.dataset = 'ImageNet'
    conf.patch_size = 16
    conf.embed_dims = 512
    conf.num_heads = 16
    conf.depths = 10
    conf.sr_ratios = 8
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -0.8
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4
elif conf.SEL_model_dataset == 'Spik_DVS':
    conf.model='Spikformer'
    conf.dataset = 'CIFAR10_DVS'
    conf.batch_size = 10
    conf.patch_size = 16
    conf.embed_dims = 256
    conf.num_heads = 16
    conf.depths = 2
    conf.sr_ratios = 8
    conf.adaptive_dec_vth_scale = 0.8
    conf.reg_psp_SEL_const = 5e-6
    conf.reg_psp_SEL_BN_ratio_value = -0.8
    conf.reg_psp_SEL_BN_ratio_rate = 1e-4

##### training parameter setting #####
if conf.dataset == 'CIFAR10':
    conf.batch_size = 100
    conf.train_epoch = 310
    conf.time_step = 4
elif conf.dataset == 'CIFAR10_DVS':
    conf.batch_size = 32
    conf.train_epoch = 310
    conf.time_step = 4
    # conf.mix_off_iter = 281 * 150
    # conf.accumulate_iteration = 281*5
elif conf.dataset == 'ImageNet':
    conf.batch_size = 90
    conf.train_epoch = 90
    conf.step_decay_epoch = 30
##########

##### neuron setting #####
conf.n_reset_type = 'reset_by_sub'
# conf.n_reset_type = 'reset_to_zero'

conf.n_init_vth = 1.0

conf.vth_rand_static = False
conf.vrest = 0.0

conf.leak_const_init = 0.9
##########

##### stdp setting #####
#conf.en_stdp_pathway = True
# conf.stdp_pathway_weight = 0.1
##########

config.set()
