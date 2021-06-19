#!/usr/bin/env python
#############################
##
## Image converter by learned models
##
#############################

import argparse
import os
import glob
import json
import codecs
from datetime import datetime as dt
import time
import chainer.cuda
from chainer import serializers, Variable, dataset
import numpy as np
import net3d
import cv2
import random
import chainer.functions as F
from chainercv.utils import write_image
from chainercv.transforms import resize
from chainerui.utils import save_args
from arguments import arguments 
from consts import dtypes
from dataset3d import Dataset
import scipy.special as sps

def heatmap(heat,src):  ## heat [0,1], src [0,1] grey => [0,1] colour
#    h = cv2.normalize(heat[0], h, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    if len(src.shape)==2:
        src = src[:,:,np.newaxis]
    h = np.uint8(np.clip(heat,0,1)*255)
    h = cv2.resize(h, (src.shape[1],src.shape[0]))
    h = cv2.applyColorMap(np.uint8(h), cv2.COLORMAP_JET)
    h = 255*src + 0.5*h
    h = np.uint8(255 * h / h.max())
    return(h)

if __name__ == '__main__':
    args = arguments("cgan_out")
    args.suffix = "out"
    args.imgtype="npy"
    outdir = os.path.join(args.out, dt.now().strftime('out_%Y%m%d_%H%M'))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        print('use gpu {}'.format(args.gpu))

    ## load arguments from "arg" file used in training
    if args.argfile:
        with open(args.argfile, 'r') as f:
            larg = json.load(f)
            root=os.path.dirname(args.argfile)
            for x in ['out','local_avg_subtraction','clip_max',
              'dis_norm','dis_activation','dis_chs','dis_ksize','dis_sample','dis_down',
              'gen_norm','gen_activation','gen_out_activation','gen_nblock','gen_chs','gen_sample','gen_down','gen_up','gen_ksize','unet',
              'gen_fc','gen_fc_activation','spconv','eqconv','dtype','clipA','clipB','class_num','out_ch',
              'args.crop_depth', 'args.crop_height', 'args.crop_width']:
                if x in larg:
                    setattr(args, x, larg[x])
            if not args.model_gen:
                if larg["epoch"]:
                    args.model_gen=os.path.join(root,'gen_{}.npz'.format(larg["epoch"]))
                    
    chainer.config.dtype = dtypes[args.dtype]

    dataset = Dataset(args.root, phase=None, random_tr=0, args=args)

    print(args)

    iterator = chainer.iterators.MultiprocessIterator(dataset, args.nvis, n_processes=8, repeat=False, shuffle=False)
#    iterator = chainer.iterators.MultithreadIterator(dataset, args.batch_size, n_threads=3, repeat=False, shuffle=False)   ## best performance
#    iterator = chainer.iterators.SerialIterator(dataset, args.batch_size,repeat=False, shuffle=False)

    args.ch = dataset.A_ch
    print("Output channels {}".format(args.out_ch))

    ## load generator models
    if args.model_gen:
        gen = net3d.Generator(args)
        print('Loading {:s}..'.format(args.model_gen))
        serializers.load_npz(args.model_gen, gen)
        if args.gpu >= 0:
            gen.to_gpu()
        xp = gen.xp
    else:
        print("Specify a learned model.")
        exit()        

    ## start measuring timing
    start = time.time()
    # gaussian kernel
    gauss = xp.array([sps.comb(args.local_avg_subtraction - 1, i) for i in range(args.local_avg_subtraction)]).reshape((1,1,1,1,-1))
    gauss /= gauss.sum()

    cnt = 0
    salt = str(random.randint(1000, 999999))
    for batch in iterator:
        x_in, t_out = chainer.dataset.concat_examples(batch, device=args.gpu)
        if args.local_avg_subtraction>0: # make it a 3D vol and apply 3D conv
            x_in = x_in-F.convolution_3d(x_in[:,xp.newaxis,:,:,:],gauss,pad=(0,0,args.local_avg_subtraction//2))[:,0,:,:,:]
        else:
            x_in = Variable(x_in)
        #x_in = F.clip(x_in,-args.clip_max,args.clip_max)/args.clip_max # clipping and scaling
        with chainer.using_config('train', False), chainer.function.no_backprop_mode():
            x_out = gen(x_in)
        if args.gpu >= 0:
            x_in = xp.asnumpy(x_in.array)
            x_out = xp.asnumpy(x_out.array)
            t_out = xp.asnumpy(t_out)
            x_in = xp.asnumpy(x_in)
        else:
            x_in = x_in.array
            x_out = x_out.array
            x_in = x_in.array
#        print(x_in.shape,out.shape,t_out.shape)

        ## output images
        for i in range(len(x_out)):
            fname = dataset.dirs[cnt]
            path = os.path.join(outdir,fname) ## preserve directory structures
            os.makedirs(path, exist_ok=True)
            print("{}, shape, raw value: {} {} {}".format(fname,x_out[i].shape,np.min(x_out[i]),np.max(x_out[i]),x_out[i].shape))
            if args.class_num>0:
                new = np.argmax(x_out[i],axis=0)
                airvalue = 0
                print(new.shape)
            else:
                airvalue = None
                new = dataset.var2img(x_out[i],args.clipB)
            np.save(os.path.join(path,"{}.npy".format(fname)),new)
            for j in range(len(new)):
                print(path)
                fn = dataset.names[cnt][j + (len(dataset.names[cnt])-len(new))//2]
                ref_dicom = dataset.overwrite_dicom(new[j],fn,salt,airvalue=airvalue)
                ref_dicom.save_as(os.path.join(path,os.path.basename(fn)))
            cnt += 1
        ####
    elapsed_time = time.time() - start
    print ("{} volumes in {} sec".format(cnt,elapsed_time))
    iterator.finalize()


