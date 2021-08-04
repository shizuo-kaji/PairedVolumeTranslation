# -*- coding:utf-8 -*-
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
from skimage.transform import rescale
import nrrd

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
    outdir = args.out

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        print('use gpu {}'.format(args.gpu))

    ## load arguments from "arg" file used in training
    if args.argfile:
        with open(args.argfile, 'r', encoding="utf-8_sig") as f:
            larg = json.load(f)
            root=os.path.dirname(args.argfile)
            for x in ['out','local_avg_subtraction','clip_max',
              'dis_norm','dis_activation','dis_chs','dis_ksize','dis_sample','dis_down',
              'gen_norm','gen_activation','gen_out_activation','gen_nblock','gen_chs','gen_sample','gen_down','gen_up','gen_ksize','unet',
              'gen_fc','gen_fc_activation','spconv','eqconv','dtype','clipA','clipB','class_num','out_ch',
              'crop_depth', 'crop_height', 'crop_width','size_reduction_factor','plane']:
                if x in larg:
                    setattr(args, x, larg[x])
            if not args.model_gen:
                if larg["epoch"]:
                    args.model_gen=os.path.join(root,'gen_{}.npz'.format(larg["epoch"]))
    
    if args.slide_step <= 0:
        args.slide_step = args.crop_depth

    chainer.config.dtype = dtypes[args.dtype]

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    print(args)

    dataset = Dataset(args.root, phase=None, random_tr=0, args=args)

#    iterator = chainer.iterators.MultiprocessIterator(dataset, args.nvis, n_processes=8, repeat=False, shuffle=False)
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

    airvalue = 0 if args.class_num>0 else args.clipB[0]

    cnt = 0
    salt = str(random.randint(1000, 999999))
    for idx in range(len(dataset.dcms["A"])):
        dname = dataset.dirs[cnt]
        path = os.path.join(outdir,dname) ## preserve directory structures
        os.makedirs(path, exist_ok=True)
        org_shape = dataset.dcms["A"][idx].shape[1:]
        z_len = org_shape[0]
        z_max = z_len-args.crop_depth
        zl = (args.crop_depth - args.slide_step)//2
        print("{}, shape {}".format(path, org_shape))
        # for each volume
        out_vol=np.zeros( (args.out_ch,*org_shape) ) #(C,D,H,W)
        for z in range(0,z_len,args.slide_step):
            z = min(z,z_max)
            x_in = xp.asarray(dataset.get_example(idx,z_offset=z)[0])[np.newaxis,:]
            #print(z, z_len, z_max, x_in.shape)
            if args.local_avg_subtraction>0: # make it a 3D vol and apply 3D conv
                x_in = x_in-F.convolution_3d(x_in[:,xp.newaxis,:,:,:],gauss,pad=(0,0,args.local_avg_subtraction//2))[:,0,:,:,:]
            with chainer.using_config('train', False), chainer.function.no_backprop_mode():
                x_out = gen(x_in)
            if args.gpu >= 0:
                x_out = xp.asnumpy(x_out.array)[0]
            else:
                x_out = x_out.array[0]
            if args.class_num==0:
                x_out = dataset.var2img(x_out,args.clipB)
    #        print(x_in.shape,out.shape,t_out.shape)
            ## output images
            print("{}_{}, {}, min-max values: {} {}".format(dname,z,x_out.shape,np.min(x_out),np.max(x_out)))
            # pad airvalue outside the cropped box
            ph = (org_shape[1]-x_out.shape[2])//2
            pw = (org_shape[2]-x_out.shape[3])//2
            x_out = np.pad(x_out, ((0,0), (0,0), (ph,org_shape[1]-x_out.shape[2]-ph), (pw,org_shape[2]-x_out.shape[3]-pw)),'constant', constant_values=airvalue)
            if z==0:
                out_vol[:,:x_out.shape[1]] = x_out
            elif z==z_max:
                out_vol[:,(z+zl):] = x_out[:,zl:]
                break
            else:
                out_vol[:,(z+zl):(z+zl+args.slide_step)] = x_out[:,zl:(zl+args.slide_step)]
        # inverse transpose
        if args.plane == "sagittal":
            out_vol = out_vol.transpose((0,2,3,1))
        elif args.plane == "coronal":
            out_vol = out_vol.transpose((0,2,1,3))
        # highest score
        if args.class_num>0:
            prob_vol = out_vol
            out_vol = np.argmax(prob_vol,axis=0)
        # rescale back
        if args.size_reduction_factor != 1:
            out_vol = rescale(out_vol,args.size_reduction_factor,order=0, mode="reflect",preserve_range=True).astype(out_vol.dtype)
        # save volume
        print("{}, {}, min-max values: {} {}".format(dname,out_vol.shape,np.min(out_vol),np.max(out_vol)))
        if args.output_image_type == "dcm":
            for j in range(len(out_vol)):
                fn = dataset.names[cnt][j]
                ref_dicom = dataset.overwrite_dicom(out_vol[j],fn,salt,airvalue=airvalue)
                ref_dicom.save_as(os.path.join(path,os.path.basename(fn)))
        elif args.output_image_type == "nrrd":
            nrrd.write(os.path.join(path,"{}.nrrd".format(dname)), out_vol.astype(np.int8), index_order='C')
            nrrd.write(os.path.join(path,"{}_prob.nrrd".format(dname)), prob_vol.astype(np.float32), index_order='C')
        elif args.output_image_type == "npy":
            np.save(os.path.join(path,"{}.npy".format(dname)),out_vol)
        cnt += 1  # number of vols
        ####
    elapsed_time = time.time() - start
    print ("{} volumes in {} sec".format(cnt,elapsed_time))


