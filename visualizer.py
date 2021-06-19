#!/usr/bin/env python
# -*- coding: utf-8 -*-
# implementation of pix2pix
# By Shizuo Kaji

import os
import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F
from chainer.training import extensions
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import warnings
import scipy.special as sps
import cv2
import losses
from matplotlib import colors

def var2unit_img(var, base=-1.0, rng=2.0):
    img = var.data.get()
    img = (img - base) / rng  # [0, 1)
    img = img.transpose(0, 2, 3, 4, 1) # BDHWC
    return img

def heatmap(heat,src):  ## heat [0,1], src [0,1] grey => [0,1] colour
    if len(src.shape)==2:
        src = src[:,:,np.newaxis]
#    h = cv2.normalize(heat[0], h, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    h = np.uint8(np.clip(heat,0,1)*255)
    h = cv2.resize(h, (src.shape[1],src.shape[0]))
    h = cv2.applyColorMap(np.uint8(h), cv2.COLORMAP_JET)
    h = 255*src + 0.5*h
    h = np.uint8(255 * h / h.max())
    return(h)

class VisEvaluator(extensions.Evaluator):
    name = "myval"
    def __init__(self, *args, **kwargs):
        params = kwargs.pop('params')
        super(VisEvaluator, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.vis_out = params['vis_out']
        self.count = 0
        self.xp = self._targets['gen'].xp
        self.class_weight = 1.0
        # kernel for local average subtraction
        self.gauss = self.xp.array([sps.comb(self.args.local_avg_subtraction - 1, i) for i in range(self.args.local_avg_subtraction)]).reshape((1,1,1,1,-1))
        self.gauss /= self.gauss.sum()

    def evaluate(self):
        self.count += self.args.vis_freq
        if self.eval_hook:
            self.eval_hook(self)

        # visualisation 
        for k,dataset in enumerate(['test','train']):
            batch =  self._iterators[dataset].next()
            x_in, t_out = chainer.dataset.concat_examples(batch, self.device)
            org=Variable(x_in)
            t_out = Variable(t_out) # corresponding translated image (ground truth)
            if self.args.local_avg_subtraction>0: # make it a 3D vol and apply 3D conv
                x_in = org-F.convolution_3d(x_in[:,self.xp.newaxis,:,:,:],self.gauss,pad=(0,0,self.args.local_avg_subtraction//2))[:,0,:,:,:]
            else:
                x_in = Variable(x_in)

            with chainer.using_config('train', False), chainer.function.no_backprop_mode():
                x_out = self._targets['gen'](x_in)

            if dataset == 'test':  # for test dataset, compute some statistics
                fig = plt.figure(figsize=(12, 6 * len(x_out)))
                gs = gridspec.GridSpec(3*2* len(x_out), 4, wspace=0.1, hspace=0.1)
                loss_rec_L1 = F.mean_absolute_error(x_out, t_out)
                loss_rec_L2 = F.mean_squared_error(x_out, t_out)
                loss_rec_CE = losses.softmax_focalloss(x_out, t_out, gamma=self.args.focal_gamma, class_weight=self.class_weight)
                result = {"myval/loss_L1": loss_rec_L1, "myval/loss_L2": loss_rec_L2, "myval/loss_CE": loss_rec_CE}

            #heat = [heatmap(x_out[j],x_in[j]) for j in range(len(x_out))]
            #print(np.min(x_in),np.max(x_in),np.min(x_out),np.max(x_out),np.min(t_out),np.max(t_out))
            domain = ["A","truth","B"]
            for i, var in enumerate([x_in, t_out, x_out]):
                if i % 3 != 0 and self.args.class_num>0: # t_out, x_out
                    outs = var2unit_img(var,0,1) # softmax
                else:
                    outs = var2unit_img(var) # tanh
#                print(imgs.shape,np.min(imgs),np.max(imgs))
                imgs = []
                for im in outs:
                    imgs.extend([im[im.shape[0]//2], im[:,im.shape[1]//2], im[:,:,im.shape[2]//2]])
                for j,im in enumerate(imgs):
                    ax = fig.add_subplot(gs[j+k*len(imgs),i])
                    ax.set_title(dataset+"_"+domain[i], fontsize=8)
                    #print(imgs.shape, im.shape)
                    if(im.shape[2] == 3): ## RGB
                        ax.imshow(im, interpolation='none',vmin=0,vmax=1)
                    elif(im.shape[2] >= 4): ## categorical
                        cols = ['k','b','c','g','y','r','m','w']*5
                        cmap = colors.ListedColormap(cols)
                        im = np.argmax(im, axis=2)
                        norm = colors.BoundaryNorm(list(range(len(cols)+1)), cmap.N)
                        ax.imshow(im, interpolation='none', cmap=cmap, norm=norm)
                    else:
                        ax.imshow(im[:,:,-1], interpolation='none',cmap='gray',vmin=0,vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])

        gs.tight_layout(fig)
        plt.savefig(os.path.join(self.vis_out,'count{:0>7}.jpg'.format(self.count)), dpi=200)
        plt.close()
        return result

        # statistics
#         test_all_iter = self._iterators["test_all"]
#         test_all_iter.reset()
#         sum_fn,sum_tp,sum_fp = 0,0,0
#         loss_l2 = 0
#         b = 0
#         while not test_all_iter.is_new_epoch:
#             for batch in test_all_iter:
#                 x_in, t_out = chainer.dataset.concat_examples(batch, self.device)
#                 org=Variable(x_in)
#                 if self.args.local_avg_subtraction>0: # make it a 3D vol and apply 3D conv
#                     x_in = org-F.convolution_3d(x_in[:,self.xp.newaxis,:,:,:],self.gauss,pad=(0,0,self.args.local_avg_subtraction//2))[:,0,:,:,:]
#                 else:
#                     x_in = Variable(x_in)
#                 x_in = F.clip(x_in,-self.args.clip_max,self.args.clip_max)/self.args.clip_max # clipping and scaling
#                 t_out = Variable(t_out)
#                 with chainer.using_config('train', False), chainer.function.no_backprop_mode():
#                     x_out = self._targets['gen'](x_in)
#                 loss_l2 += len(batch)*F.mean_squared_error(x_out, F.cast(t_out,'float32'))
#                 ## convert to numpy
#                 x_in = postprocess((x_in+1)/2)
#                 t_out = postprocess(t_out).astype(np.float32)
#                 x_out = postprocess(x_out)
#                 # IOU
#                 for j in range(len(t_out)):
#                     tp,fp,fn= IOU(t_out[j],x_out[j],self.args.threshould,self.args.min_IOU,radius=7)
# #                    print(tp,fp,fn)
#                     sum_fn += fn
#                     sum_tp += tp
#                     sum_fp += fp
#                     b += 1
#                     #heat = heatmap(x_out[j],x_in[j])
#                     heat = np.stack([x_in[j],x_out[j]+x_in[j],t_out[j]+x_in[j]],axis=2)
#                     heat = np.uint8(255*(heat/np.max(heat)))
# #                    print(heat.shape,np.min(heat),np.max(heat))
#                     cv2.imwrite(os.path.join(self.vis_out,'test{:0>7}_{:0>3}.jpg'.format(self.count,b)),heat)
#                 # save volume
#                 #np.save(os.path.join(self.vis_out,'count_t{:0>4}.npy'.format(self.count)),t_out.data.get()[0])
#                 #np.save(os.path.join(self.vis_out,'count_out{:0>4}.npy'.format(self.count)),x_out.data.get()[0])
