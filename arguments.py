import argparse
import numpy as np
import chainer.functions as F
from consts import activation_func,dtypes,uplayer,downlayer,norm_layer,optim,unettype
import os
from datetime import datetime as dt
from chainerui.utils import save_args

def arguments(suffix):
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix 3D')
    parser.add_argument('--root', '-R', default='data', help='directory containing image files')
    parser.add_argument('--argfile', '-a', help="specify args file to read")
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--loaderjob', '-j', type=int, default=5, help='Number of parallel data loading processes')

    parser.add_argument('--snapinterval', '-si', type=int, default=-1,  help='take snapshot every this epoch')
    parser.add_argument('--nvis', type=int, default=1, help='number of images in visualisation after each epoch')

    parser.add_argument('--crop_width', '-cw', type=int, default=224, help='this value may have to be divisible by a large power of two (if you encounter errors)')
    parser.add_argument('--crop_height', '-ch', type=int, default=224, help='this value may have to be divisible by a large power of two (if you encounter errors)')
    parser.add_argument('--crop_depth', '-cd', type=int, default=48, help='this value may have to be divisible by a large power of two (if you encounter errors)')
    parser.add_argument('--size_reduction_factor', '-srf', type=int, default=1, help='this value may have to be divisible by a large power of two (if you encounter errors)')
    parser.add_argument('--clipA', '-ca', type=float, nargs=2, default=[-1024,0], help="lower and upper limit for pixel values of images in domain A")
    parser.add_argument('--clipB', '-cb', type=float, nargs=2, default=[-1024,0], help="lower and upper limit for pixel values of images in domain B")
    parser.add_argument('--class_num', '-cn', type=int, default=5, help='number of classes for pixelwise classification (only for images in domain B)')
    parser.add_argument('--class_weight', type=float, nargs="*", help='weight for each class for pixelwise classification (only for images in domain B)')
    parser.add_argument('--local_avg_subtraction', '-las', type=int, default=-1, help='kernel size of local average subtraction: should be an odd number')

    #
    parser.add_argument('--lambda_rec_l1', '-l1', type=float, default=0, help='weight for L1 reconstruction loss')
    parser.add_argument('--lambda_rec_l2', '-l2', type=float, default=0.0, help='weight for L2 reconstruction loss')
    parser.add_argument('--lambda_focal', '-lce', type=float, default=1.0, help='weight for focal reconstruction loss')
    parser.add_argument('--lambda_dis', '-ldis', type=float, default=0.0, help='weight for adversarial loss')
    parser.add_argument('--lambda_dice', '-ldice', type=float, default=0.0, help='weight for channel-wise weighted dice loss')
    parser.add_argument('--lambda_tv', '-ltv', type=float, default=0.0, help='weight for total variation')
    parser.add_argument('--tv_tau', '-tt', type=float, default=1e-3, help='smoothing parameter for total variation')
    parser.add_argument('--focal_gamma', type=int, default=2)

    parser.add_argument('--load_optimizer', '-mo', action='store_true', help='load optimizer parameters')
    parser.add_argument('--model_gen', '-m', default='')
    parser.add_argument('--model_dis', '-md', default='')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam',help='optimizer')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-8, help='weight decay for regularization')
    parser.add_argument('--lr_decay_strategy', '-lrs', choices=['exp','linear','none'], default='linear', help='strategy for learning rate decay')
    parser.add_argument('--vis_freq', '-vf', type=int, default=500, help='visualisation frequency in iteration')

    parser.add_argument('--dtype', '-dt', choices=dtypes.keys(), default='fp32',
                        help='floating point precision')
    parser.add_argument('--eqconv', '-eq', action='store_true',
                        help='Equalised Convolution')
    parser.add_argument('--spconv', '-sp', action='store_true',
                        help='Enable Separable Convolution')
    parser.add_argument('--senet', '-se', action='store_true',
                        help='Enable Squeeze-and-Excitation mechanism')

    # data augmentation
    parser.add_argument('--random_translate', '-rt', type=int, default=4, help='jitter input images by random translation in w-axis')
    parser.add_argument('--noise', '-n', type=float, default=0.03, help='strength of noise injection')
    parser.add_argument('--noise_z', '-nz', type=float, default=0,
                        help='strength of noise injection for the latent variable')

    # discriminator
    parser.add_argument('--dis_activation', '-da', default='lrelu', choices=activation_func.keys())
    parser.add_argument('--dis_ksize', '-dk', type=int, default=4,    # default 4
                        help='kernel size for patchGAN discriminator')
    parser.add_argument('--dis_chs', '-dc', type=int, nargs="*", default=None,
                        help='Number of channels in down layers in discriminator')
    parser.add_argument('--dis_basech', '-db', type=int, default=64,
                        help='the base number of channels in discriminator (doubled in each down-layer)')
    parser.add_argument('--dis_ndown', '-dl', type=int, default=3,
                        help='number of down layers in discriminator')
    parser.add_argument('--dis_down', '-dd', default='down', choices=downlayer,  ## default down
                        help='type of down layers in discriminator')
    parser.add_argument('--dis_sample', '-ds', default='none',          ## default down
                        help='type of first conv layer for patchGAN discriminator')
    parser.add_argument('--dis_jitter', type=float, default=0,
                        help='jitter for discriminator label for LSGAN')
    parser.add_argument('--dis_dropout', '-ddo', type=float, default=None, 
                        help='dropout ratio for discriminator')
    parser.add_argument('--dis_norm', '-dn', default='instance',
                        choices=norm_layer)
    parser.add_argument('--dis_reg_weighting', '-dw', type=float, default=0,
                        help='regularisation of weighted discriminator. Set 0 to disable weighting')
    parser.add_argument('--dis_wgan', action='store_true',help='WGAN-GP')

    # generator
    parser.add_argument('--gen_activation', '-ga', default='relu', choices=activation_func.keys())
    parser.add_argument('--gen_fc_activation', '-gfca', default='relu', choices=activation_func.keys())
    parser.add_argument('--gen_out_activation', '-go', default='softmax', choices=activation_func.keys())
    parser.add_argument('--gen_chs', '-gc', type=int, nargs="*", default=None,
                        help='Number of channels in down layers in generator')
    parser.add_argument('--gen_ndown', '-gl', type=int, default=4,
                        help='number of down layers in generator')
    parser.add_argument('--gen_basech', '-gb', type=int, default=64,
                        help='the base number of channels in generator (doubled in each down-layer)')
    parser.add_argument('--gen_fc', '-gfc', type=int, default=0,
                        help='number of fc layers before convolutional layers')
    parser.add_argument('--gen_nblock', '-gnb', type=int, default=9,
                        help='number of residual blocks in generators')
    parser.add_argument('--gen_ksize', '-gk', type=int, default=3,
                        help='kernel size for generator')
    parser.add_argument('--gen_sample', '-gs', default='none',
                        help='first and last conv layers for generator')
    parser.add_argument('--gen_down', '-gd', default='down', choices=downlayer,
                        help='down layers in generator')
    parser.add_argument('--gen_up', '-gu', default='resize', choices=uplayer,
                        help='up layers in generator')
    parser.add_argument('--gen_dropout', '-gdo', type=float, default=None, 
                        help='dropout ratio for generator')
    parser.add_argument('--gen_norm', '-gn', default='instance',
                        choices=norm_layer)
    parser.add_argument('--unet', '-u', default='concat', choices=unettype,
                        help='use u-net for generator')

    args = parser.parse_args()
    args.out = os.path.join(args.out, dt.now().strftime('%Y%m%d_%H%M_3d')+"_"+suffix)
    args.wgan=False

    if not args.gen_chs:
        args.gen_chs = [int(args.gen_basech) * (2**i) for i in range(args.gen_ndown)]
    if not args.dis_chs:
        args.dis_chs = [int(args.dis_basech) * (2**i) for i in range(args.dis_ndown)]
    if args.local_avg_subtraction % 2 ==0:
        print("local_avg_subtraction should be an ODD number!")
        args.local_avg_subtraction -= 1
    return(args)

