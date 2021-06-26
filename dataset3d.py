# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import os,glob,re
import random

from chainer.dataset import dataset_mixin
from chainercv.transforms import resize,random_flip,random_crop,rotate,center_crop,resize
from chainercv.utils import read_image,write_image
from skimage.transform import rescale
import PIL
import pydicom as dicom

class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, path, phase="train", random_tr=0, args=None):
        self.dcms = {'A': [], 'B': []}
        self.clip = {"A": args.clipA, "B": args.clipB}
        self.dtype=np.float32
        self.forceSpacing = -1
        self.names = []
        self.dirs = []
        self.random_tr = random_tr
        self.crop = (args.crop_depth, args.crop_height, args.crop_width)
        num = lambda val : int(re.sub("\\D", "", val+"0"))

        self.class_num=args.class_num

        if phase is not None:
            self.validation_mode = False
            phases = ["A","B"]
            target_dir = os.path.join(path, phase+"A")
        else:
            self.validation_mode = True
            phases = ["A"]
            target_dir = path

        print("Loading Dataset from: {}".format(path))
        dirlist = []
        for f in os.listdir(target_dir):
            if os.path.isdir(os.path.join(target_dir, f)):
                dirlist.append(os.path.join(target_dir,f))
        #print(dirlist)
        # each directory is considered to form a volume
        for dirname in sorted(dirlist):
            for ph in phases:
                if phase is not None:
                    dnph = dirname.replace(phase+"A",phase+ph)
                else:
                    dnph = dirname
                print(dnph)
                files = [os.path.join(dnph, fname) for fname in sorted(os.listdir(dnph), key=num) if fname.endswith(".dcm")]
                slices = []
                filenames = []
                for f in files:
                    ds = dicom.dcmread(f, force=True)
                    #ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
                    slices.append(ds)
                    filenames.append(f)
                s = sorted(range(len(slices)), key=lambda k: filenames[k])

                # if the current dir contains at least one slice
                if len(s)>0:
                    vollist = []
                    for i in s:
                        sl = slices[i].pixel_array+slices[i].RescaleIntercept
                        # resize
                        if self.forceSpacing>0:
                            scaling = float(slices[i].PixelSpacing[0])/self.forceSpacing
                            sl = rescale(sl,scaling,mode="reflect",preserve_range=True)
                        # one-hot encoding
                        if self.class_num>0 and ph=="B":  # Only images in domain B
                            sl = np.eye(self.class_num)[sl.astype(np.uint64)].transpose((2,0,1))
                        else:
                            sl = self.img2var(sl[np.newaxis,],self.clip[ph]) # scaling to [-1,1]
                        vollist.append(sl.astype(self.dtype))
                    volume = np.stack(vollist,axis=1)   # shape = (c,z,h,w)
                    if args.size_reduction_factor != 1:
                        scl = 1.0/args.size_reduction_factor
                        volume = rescale(volume,(1,scl,scl,scl),mode="reflect",preserve_range=True)
                    self.dcms[ph].append(volume)
                    if ph == "A":
                        self.names.append( [filenames[i] for i in s] )
                        self.dirs.append(os.path.basename(dirname))

        print("Volume size: ", self.dcms["A"][0].shape, "Cropped size: ",self.crop, "ClipA: ",self.clip["A"], "ClipB: ",self.clip["B"])
        print("#dir {}, #file {}, #slices {}".format(len(dirlist),len(self.dcms),sum([len(fd) for fd in self.names])))
        self.A_ch = self.dcms["A"][0].shape[0]
        if phase is not None:
            self.B_ch = self.dcms["B"][0].shape[0]
    
    def __len__(self):
        return len(self.dcms["A"])

    def var2img(self,var,clip):
        return(0.5*(1.0+var)*(clip[1]-clip[0])+clip[0])

    def img2var(self,img, clip): 
        return(2*(np.clip(img,clip[0],clip[0]+(clip[1]-clip[0]))-clip[0])/(clip[1]-clip[0])-1.0)        
    
    def get_example(self, i, z_offset=-1):
        imgs_in = self.dcms["A"][i]
        if not self.validation_mode:
            imgs_out = self.dcms["B"][i]
        #print(imgs_in.shape, np.min(imgs_in),np.max(imgs_in),np.min(imgs_out),np.max(imgs_out))

        slices = [0,0,0]
        if self.random_tr>0: # random crop/flip
            if random.choice([True, False]): # x-flip
                imgs_in = imgs_in[:, :, :, ::-1]
                imgs_out = imgs_out[:, :, :, ::-1]
            for i,C in enumerate(self.crop):
                offsets = random.randint( max(0,(imgs_in.shape[i+1]-C)//2-self.random_tr), min((imgs_in.shape[i+1]-C)//2+self.random_tr,imgs_in.shape[i+1]-C) )
                slices[i] = slice(offsets, offsets + C)
            # slide z-axis
            r = random.randint(0,imgs_in.shape[1]-self.crop[0])
            slices[0] = slice(r, r+self.crop[0])
        else: # centre crop
            for i,C in enumerate(self.crop):
                offsets = max(imgs_in.shape[i+1] - C,0) // 2
                slices[i] = slice(offsets, offsets + C)
            # slide z-axis
            if z_offset >= 0:
                slices[0] = slice(z_offset, z_offset+self.crop[0])
        #print(imgs_in.shape, x_offset, imgs_in[:,y_slice,x_slice].shape, imgs_out[:,y_slice,x_slice].shape)
        if self.validation_mode:
            return imgs_in[:,slices[0],slices[1],slices[2]], imgs_in[:,slices[0],slices[1],slices[2]]
        else:
            return imgs_in[:,slices[0],slices[1],slices[2]], imgs_out[:,slices[0],slices[1],slices[2]]

    ## TODO    
    def overwrite_dicom(self,new,fn,salt,airvalue=None):
        if airvalue is None:
            airvalue = self.clip_B[0]
        ref_dicom = dicom.read_file(fn, force=True)
        dt=ref_dicom.pixel_array.dtype
        img = np.full(ref_dicom.pixel_array.shape, airvalue, dtype=np.float32)
        ch,cw = img.shape
        h,w = new.shape
        img[np.newaxis,(ch-h)//2:(ch+h)//2,(cw-w)//2:(cw+w)//2] = new
        # if np.min(img - ref_dicom.RescaleIntercept)<0:
        #     ref_dicom.RescaleIntercept -= 1024
        #     img -= ref_dicom.RescaleIntercept
        ref_dicom.RescaleIntercept = 0
        ref_dicom.RescaleSlope = 1

        img = img.astype(dt)           
#        print("min {}, max {}, intercept {}\n".format(np.min(img),np.max(img),ref_dicom.RescaleIntercept))
#            print(img.shape, img.dtype)
        ref_dicom.PixelData = img.tostring()
        ## UID should be changed for dcm's under different dir
        #                uid=dicom.UID.generate_uid()
        #                uid = dicom.UID.UID(uid.name[:-len(args.suffix)]+args.suffix)
        uid = ref_dicom[0x8,0x18].value.split(".")
        uid[-2] = salt
        uidn = ".".join(uid)
        uid = ".".join(uid[:-1])

#            ref_dicom[0x2,0x3].value=uidn  # Media SOP Instance UID                
        ref_dicom[0x8,0x18].value=uidn  #(0008, 0018) SOP Instance UID              
        ref_dicom[0x20,0xd].value=uid  #(0020, 000d) Study Instance UID       
        ref_dicom[0x20,0xe].value=uid  #(0020, 000e) Series Instance UID
        ref_dicom[0x20,0x52].value=uid  # Frame of Reference UID
        return(ref_dicom)