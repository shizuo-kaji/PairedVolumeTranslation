import functools

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from consts import activation_func, norm_layer


class ResBlock(chainer.Chain):
    def __init__(self, ch, norm='instance', activation='relu', equalised=False, separable=False, skip_conv=False):
        super(ResBlock, self).__init__()
        w = chainer.initializers.HeNormal()
        bias = chainer.initializers.Zero()
        self.activation = activation_func[activation]
        nobias = True if 'batch' in norm or 'instance' in norm else False
        with self.init_scope():
            self.c0 = L.Convolution3D(ch, ch, 3, 1, 1, nobias=nobias)
            self.c1 = L.Convolution3D(ch, ch, 3, 1, 1, nobias=nobias)
            if skip_conv:  # skip connection
                self.cs = L.Convolution3D(ch, ch, 1, 1, 0)
            else:
                self.cs = F.identity
            self.norm0 = norm_layer[norm](ch)
            self.norm1 = norm_layer[norm](ch)

    def __call__(self, x):
        h = self.c0(x)
        h = self.norm0(h)
        h = self.activation(h)
        h = self.c1(h)
        h = self.norm1(h)
        return self.activation(h + self.cs(x))


class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, ksize=3, pad=1, norm='instance',
                 sample='down', activation='relu', dropout=False, equalised=False, separable=False, senet=False):
        super(CBR, self).__init__()
        self.activation = activation_func[activation]
        self.dropout = dropout
        self.sample = sample
        w = chainer.initializers.HeNormal()
        bias = chainer.initializers.Zero()
        nobias = True if 'batch' in norm or 'instance' in norm else False

        with self.init_scope():
            if sample == 'down':
                self.c = L.Convolution3D(ch0, ch1, ksize, 2, pad, initialW=w, nobias=nobias)
            elif sample == 'none-7':
                self.c = L.Convolution3D(ch0, ch1, (7,7,3), stride=1, pad=(3,3,1), initialW=w, nobias=nobias) 
            elif sample == 'deconv':
                self.c = L.Deconvolution3D(ch0, ch1, ksize, 2, pad,initialW=w, nobias=nobias)
            else: ## maxpool,avgpool,resize,unpool
                self.c = L.Convolution3D(ch0, ch1, ksize, 1, pad,initialW=w, nobias=nobias)
            self.norm = norm_layer[norm](ch1)
            if '_res' in sample:
                self.normr = norm_layer[norm](ch1)
                self.cr = L.Convolution3D(ch1, ch1, ksize, 1, pad,initialW=w, nobias=nobias)
                self.cskip = L.Convolution3D(ch0, ch1, 1, 1, 0,initialW=w, nobias=False)

    def __call__(self, x):
        if self.sample in ['maxpool_res','avgpool_res']:
            h = self.activation(self.norm(self.c(x)))
            h = self.normr(self.cr(h))
            if self.sample == 'maxpool_res':
                h = F.max_pooling_3d(h, 2, 2, 0)
                h = h + F.max_pooling_3d(self.cskip(x), 2, 2, 0)
            elif self.sample == 'avgpool_res':
                h = F.average_pooling_3d(h, 2, 2, 0)
                h = h + F.average_pooling_3d(self.cskip(x), 2, 2, 0)                
        elif 'unpool' in self.sample or 'resize' in self.sample:  # NO 3d resize!
            h0 = F.unpooling_3d(x, 2, 2, 0, cover_all=False)
            h = self.norm(self.c(h0))
            if self.sample == 'unpool_res':
                h = self.activation(h)
                h = self.cskip(h0) + self.normr(self.cr(h))
        else:
            if self.sample == 'maxpool':
                h = self.c(x)
                h = F.max_pooling_3d(h, 2, 2, 0)
            elif self.sample == 'avgpool':
                h = self.c(x)
                h = F.average_pooling_3d(h, 2, 2, 0)
            else:
                h = self.c(x)
            h = self.norm(h)
        if self.dropout:
            h = F.dropout(h, ratio=self.dropout)
        if self.activation is not None:
            h = self.activation(h)
        return h

class Generator(chainer.Chain):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.n_resblock = args.gen_nblock
        self.chs = args.gen_chs
        if hasattr(args,'unet'):
            self.unet = args.unet
        else:
            self.unet = 'none'
        if self.unet=='concat':
            up_chs = [2*self.chs[i] for i in range(len(self.chs))]
        elif self.unet in ['add','none']:
            up_chs = self.chs
        elif self.unet=='conv':
            up_chs = [self.chs[i]+4 for i in range(len(self.chs))]                
        if hasattr(args,'noise_z'):
            self.noise_z = args.noise_z
        else:
            self.noise_z = 0
        self.nfc = args.gen_fc
        with self.init_scope():
            self.c0 = CBR(None, self.chs[0], norm=args.gen_norm, sample=args.gen_sample, activation=args.gen_activation, equalised=args.eqconv, separable=args.spconv)
            for i in range(1,len(self.chs)):
                setattr(self, 'd' + str(i), CBR(self.chs[i-1], self.chs[i], ksize=args.gen_ksize, norm=args.gen_norm, sample=args.gen_down, activation=args.gen_activation, dropout=args.gen_dropout, equalised=args.eqconv, separable=args.spconv))
            if self.unet=='conv':
                for i in range(len(self.chs)):
                    setattr(self, 's' + str(i), CBR(self.chs[i], 4, ksize=3, norm=args.gen_norm, sample='none', activation='lrelu', dropout=args.gen_dropout, equalised=args.eqconv, separable=args.spconv))
            for i in range(self.n_resblock):
                setattr(self, 'r' + str(i), ResBlock(self.chs[-1], norm=args.gen_norm, activation=args.gen_activation, equalised=args.eqconv, separable=args.spconv))
            for i in range(1,len(self.chs)):
                setattr(self, 'ua' + str(i), CBR(up_chs[-i], self.chs[-i-1], ksize=args.gen_ksize, norm=args.gen_norm, sample=args.gen_up, activation=args.gen_activation, dropout=args.gen_dropout, equalised=args.eqconv, separable=args.spconv))
            setattr(self, 'ua'+str(len(self.chs)),CBR(up_chs[0], args.out_ch, norm='none', sample=args.gen_sample, activation=args.gen_out_activation, equalised=args.eqconv, separable=args.spconv))

    def __call__(self, x):
        h = x
        e = self.c0(h)
        if self.unet=='conv':
            h = [self.s0(e)]
        elif self.unet in ['concat','add']:
            h = [e]
        else:
            h=[0]
        # down-sampling
        for i in range(1,len(self.chs)):
            e = getattr(self, 'd' + str(i))(e)
            if self.unet=='conv':
                h.append(getattr(self, 's' + str(i))(e))
            elif self.unet in ['concat','add']:
                h.append(e)
            else:
                h.append(0)
#            print(e.data.shape)
        # residual
        for i in range(self.n_resblock):
            e = getattr(self, 'r' + str(i))(e)
            ## add noise
            if chainer.config.train and self.noise_z>0 and i == self.n_resblock//2:
                e.data += self.noise_z * e.xp.random.randn(*e.data.shape, dtype=e.dtype)
#        print(e.data.shape)
        # up-sampling
        for i in range(1,len(self.chs)+1):
            if self.unet in ['conv','concat']:
                e = getattr(self, 'ua' + str(i))(F.concat([e,h[-i]]))
            elif self.unet=='add':
                e = getattr(self, 'ua' + str(i))(e+h[-i])
            else:
                e = getattr(self, 'ua' + str(i))(e)
#            print(e.data.shape)
        return e

class Discriminator(chainer.Chain):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.n_down_layers = args.dis_ndown
        self.activation = args.dis_activation
        self.wgan = args.dis_wgan
        self.chs = args.dis_chs
        dis_out = 2 if args.dis_reg_weighting>0 else 1  ## weighted discriminator
        with self.init_scope():
            self.c0 = CBR(None, self.chs[0], ksize=args.dis_ksize, norm='none', 
                          sample=args.dis_sample, activation=args.dis_activation,dropout=args.dis_dropout, equalised=args.eqconv,senet=args.senet) #separable=args.spconv)
            for i in range(1, len(self.chs)):
                setattr(self, 'c' + str(i),
                        CBR(self.chs[i-1], self.chs[i], ksize=args.dis_ksize, norm=args.dis_norm,
                            sample=args.dis_down, activation=args.dis_activation, dropout=args.dis_dropout, equalised=args.eqconv, separable=args.spconv, senet=args.senet))
            self.csl = CBR(self.chs[-1], 2*self.chs[-1], ksize=args.dis_ksize, norm=args.dis_norm, sample='none', activation=args.dis_activation, dropout=args.dis_dropout, equalised=args.eqconv, separable=args.spconv, senet=args.senet)
            if self.wgan:
                self.fc1 = L.Linear(None, 1024)
                self.fc2 = L.Linear(None, 1)
            else:
                self.cl = CBR(2*self.chs[-1], dis_out, ksize=args.dis_ksize, norm='none', sample='none', activation='none', dropout=False, equalised=args.eqconv, separable=args.spconv, senet=args.senet)

    def __call__(self, x):
        h = self.c0(x)
        for i in range(1, len(self.chs)):
            h = getattr(self, 'c' + str(i))(h)
        h = self.csl(h)
        if self.wgan:
            h = F.average(h, axis=(2, 3))  # Global pooling
            h = activation_func[self.activation](self.fc1(h))
            h = self.fc2(h)
        else:
            h = self.cl(h)
        return h
