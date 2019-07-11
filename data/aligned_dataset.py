import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import util.util as util


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.dir_C = os.path.join(opt.dataroot, 'unlabeled')

        #self.dir_AB = os.path.join(opt.dataroot, 'train')

        # image path
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.C_paths = sorted(make_dataset(self.dir_C))

        # transform
        self.transformPIL = transforms.ToPILImage()
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        AB_path = self.AB_paths[index]
        # and C is the unlabel hazy image
        C_path = self.C_paths[random.randint(0, int((len(self.AB_paths)-1)/6))]

        AB = Image.open(AB_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        ## resize the real image without label
        C = C.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)

        ori_w = AB.width
        ori_h = AB.height

        # transform to (0, 1) tensor
        AB = self.transform1(AB)
        C = self.transform1(C)



        ######### crop the training image into fineSize ########
        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        w = C.size(2)
        h = C.size(1)

        # transform to (-1, 1)
        A = self.transform2(A)
        B = self.transform2(B)
        C = self.transform2(C)

        # noise
        # if random.random()<0.5:
        #     noise = torch.randn(3, self.opt.fineSize, self.opt.fineSize) / 100
        #     #A = A + noise


        if self.opt.phase == 'test':
            AB = self.transform1(AB)
            AB = self.transform2(AB)
            A = AB[:, :, 0:int(ori_w/2)]
            B = AB[:, :, int(ori_w/2):ori_w]

        return {'A': A, 'B': B, 'C':C, 'C_paths': C_path,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
