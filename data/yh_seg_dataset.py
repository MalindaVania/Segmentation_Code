import os.path
import ipdb
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
import random_crop_yh


class yhSegDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        #self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        #self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.dir_A = opt.raw_CT_dir
        self.dir_B = opt.raw_MRI_dir
        self.dir_Seg = opt.raw_MRI_seg_dir
        
        self.A_paths = opt.imglist_CT
        self.B_paths = opt.imglist_MRI
        self.Seg_paths = opt.imglist_Label

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.Seg_size = len(self.Seg_paths)
        if not self.opt.isTrain:
            self.skipcrop = True
        else:
            self.skipcrop = False
        # self.transform = get_transform(opt)

        if self.skipcrop:
            osize = [opt.fineSize, opt.fineSize]
        else:
            osize = [opt.loadSize, opt.loadSize]
        transform_list = []
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        self.transforms_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Resize(osize, Image.NEAREST))
        self.transforms_seg_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(random_crop_yh.randomcrop_yh(opt.fineSize))
        self.transforms_crop = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)
        #ipdb.set_trace()

        transform_list = []
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5)))
        self.transforms_normalize = transforms.Compose(transform_list)


    def __getitem__(self, index):
        index_A = index % self.A_size
        #ipdb.set_trace()

        A_path = self.A_paths[index_A]
        Seg_path = self.Seg_paths[index_A]
        #Seg_path =self.dir_Seg[index_A]
        #Seg_path = A_path.replace(self.dir_A,self.dir_Seg)
        #Seg_path = Seg_path.replace('_rawimg','_organlabel')
        
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('L')
        #A_img.show()
        #ipdb.set_trace()
        Seg_img = Image.open(Seg_path).convert('L') #I
        B_img = Image.open(B_path).convert('L')
        #ipdb.set_trace()

        A_img = self.transforms_scale(A_img)
        B_img = self.transforms_scale(B_img)
        #ipdb.set_trace()
        Seg_img = self.transforms_seg_scale(Seg_img)
        #ipdb.set_trace()
        if not self.skipcrop:
            [A_img,Seg_img] = self.transforms_crop([A_img, Seg_img])
            [B_img] = self.transforms_crop([B_img])

        A_img = self.transforms_toTensor(A_img)
        B_img = self.transforms_toTensor(B_img)
        Seg_img = self.transforms_toTensor(Seg_img)
        #print("check label_seg_img value")
        #ipdb.set_trace()
        A_img = self.transforms_normalize(A_img)
        B_img = self.transforms_normalize(B_img)
        

        Seg_img[Seg_img == 255] = 1.0
        #Seg_img[Seg_img == 7] = 5
        #Seg_img[Seg_img == 14] = 6

        Seg_imgs = torch.Tensor(self.opt.output_nc_seg, self.opt.fineSize, self.opt.fineSize)
        #ipdb.set_trace()
        #Seg_imgs = Seg_imgs.float()
        Seg_imgs[0, :, :] = Seg_img == 0.0
        Seg_imgs[1, :, :] = Seg_img == 1.0
        #ipdb.set_trace()    
        
        #Seg_imgs[2, :, :] = Seg_img == 2
        #Seg_imgs[3, :, :] = Seg_img == 3
        #Seg_imgs[4, :, :] = Seg_img == 4
        #Seg_imgs[5, :, :] = Seg_img == 5
        #Seg_imgs[6, :, :] = Seg_img == 6

        return {'A': A_img, 'B': B_img, 'Seg': Seg_imgs, 'Seg_one': Seg_img,
                'A_paths': A_path, 'B_paths': B_path, 'Seg_paths':Seg_path}


    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
