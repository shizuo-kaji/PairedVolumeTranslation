from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable,cuda

import random
import numpy as np
from PIL import Image
import scipy.special as sps

from chainer import function
from chainer.utils import type_check
from losses import ImagePool, add_noise, sigmoid_focalloss
import losses

class pixupdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        params = kwargs.pop('params')
        super(pixupdater, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.xp = self.gen.xp
        if self.args.lambda_dis>0:
            self._buffer = ImagePool(50 * self.args.batch_size)

        self.class_weight = 1.0
        # kernel for local average subtraction
        self.gauss = self.xp.array([sps.comb(self.args.local_avg_subtraction - 1, i) for i in range(self.args.local_avg_subtraction)]).reshape((1,1,1,1,-1))
        self.gauss /= self.gauss.sum()

    def update_core(self):        
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        gen, dis = self.gen, self.dis

        ## image conversion
        batch = self.get_iterator('main').next()
        x_in, t_out = self.converter(batch, self.args.gpu)
        ### local average subtraction
        if self.args.local_avg_subtraction>0:
            x_in = x_in-F.convolution_3d(x_in[:,self.xp.newaxis,:,:,:],self.gauss,pad=(0,0,self.args.local_avg_subtraction//2))[:,0,:,:,:]
        else:
            x_in = Variable(x_in)
#        print(x_in.shape,t_out.shape)
        x_out = gen(add_noise(x_in, sigma=self.args.noise))
#        print(x_in.shape,x_out.shape,t_out.shape)
        if self.args.lambda_dis>0:
            x_in_out_copy = Variable(self._buffer.query(F.concat([x_in,x_out]).data))

        loss_gen=0
        # reconstruction error
        if self.args.lambda_rec_l1>0:
            loss_rec_l1 = F.mean_absolute_error(x_out,t_out)
            loss_gen = loss_gen + self.args.lambda_rec_l1*loss_rec_l1       
            chainer.report({'loss_L1': loss_rec_l1}, gen)
        if self.args.lambda_rec_l2>0:
            loss_rec_l2 = F.mean_squared_error(x_out, t_out)
            loss_gen = loss_gen + self.args.lambda_rec_l2*loss_rec_l2
            chainer.report({'loss_L2': loss_rec_l2}, gen)
        if self.args.lambda_focal>0:
            loss_focal = losses.softmax_focalloss(x_out, t_out, gamma=self.args.focal_gamma)
            loss_gen = loss_gen + self.args.lambda_focal * loss_focal
            chainer.report({'loss_focal': loss_focal}, gen)

        # total variation
        if self.args.lambda_tv > 0:
            loss_tv = losses.total_variation(x_out, self.args.tv_tau)
            loss_gen = loss_gen + self.args.lambda_tv * loss_tv
            chainer.report({'loss_tv': loss_tv}, gen)
 
        # Adversarial loss
        if self.args.lambda_dis>0:
            y_fake = dis(F.concat([x_in, x_out]))
            loss_adv = losses.loss_func_comp(y_fake,1.0,self.args.dis_jitter)
            chainer.report({'loss_adv': loss_adv}, gen)
            loss_gen = loss_gen + self.args.lambda_dis * loss_adv

        # update generator model
        gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update(loss=loss_gen)

        ## discriminator
        if self.args.lambda_dis>0:
            loss_real = losses.loss_func_comp(dis(F.concat([x_in, F.cast(t_out,'float32')])),1.0,self.args.dis_jitter)
            loss_fake = losses.loss_func_comp(dis(x_in_out_copy),0.0,self.args.dis_jitter)
            chainer.report({'loss_fake': loss_fake}, dis)
            chainer.report({'loss_real': loss_real}, dis)
            loss_dis = 0.5*(loss_fake + loss_real)
            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update(loss=loss_dis)
