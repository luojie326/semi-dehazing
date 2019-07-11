import numpy as n
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
from .losses import init_loss
import pdb
from ECLoss.ECLoss import DCLoss
from TVLoss.TVLossL1 import TVLossL1 

try:
    xrange          # Python2
except NameError:
    xrange = range  # Python 3

class ConditionalGAN(BaseModel):
    def name(self):
        return 'ConditionalGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batch_size, opt.input_nc,
                                   opt.fineSize, opt.fineSize)

        self.input_B = self.Tensor(opt.batch_size, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        self.input_C = self.Tensor(opt.batch_size, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        #Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
        use_parallel = not opt.gan_type == 'wgan-gp'
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual)
        if self.isTrain:
            use_sigmoid = opt.gan_type == 'gan'
            self.netD = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
                                                
            self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1
            
            # define loss functions
            self.discLoss, self.contentLoss, self.loss_L2, self.loss_ssim = init_loss(opt, self.Tensor)



        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_C = input['C']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_C.resize_(input_C.size()).copy_(input_C)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = self.input_A
        self.real_C = self.input_C


        # A, B is the supervised branch
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = self.input_B

        # C, D is the unsupervised branch
        self.fake_D = self.netG.forward(self.real_C)

    # no backprop gradients
    def test(self):
        self.real_A = self.input_A
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = self.input_B

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        self.loss_D = self.discLoss.get_loss(self.netD, self.real_A, self.fake_B, self.real_B)
        self.loss_D.backward(retain_graph=True)

    def backward_G(self, iter):
        self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.real_A, self.fake_B)
        self.loss_G_Content = 0
        # Second, G(A) = B
        if self.opt.lambda_vgg != 0:
            self.vgg_loss = self.contentLoss.get_loss(self.fake_B, self.real_B) * self.opt.lambda_vgg
            self.loss_G_Content += self.vgg_loss

        if self.opt.lambda_mse != 0:
            self.mse_loss = self.loss_L2.get_loss(self.fake_B, self.real_B) * self.opt.lambda_mse
            self.loss_G_Content += self.mse_loss

        if self.opt.lambda_ssim != 0:
            self.ssim_loss = self.loss_ssim.get_loss(self.fake_B, self.real_B) * self.opt.lambda_ssim
            self.loss_G_Content += self.ssim_loss


        update_ratio =  self.opt.update_ratio

        if iter%2000 == 1 and iter > 2000:
            self.opt.lambda_DC *= self.opt.unlabel_decay
            self.opt.lambda_TV *= self.opt.unlabel_decay
            print('unlabel loss decay {}, is DC:{}, TV:{}'.format(self.opt.unlabel_decay, self.opt.lambda_DC, self.opt.lambda_TV))

        if iter%update_ratio ==0:
            self.DC_loss_unsuper = self.opt.lambda_DC * DCLoss((self.fake_D+1)/2, self.opt)
            #self.BC_loss_unsuper = self.opt.lambda_DC * BCLoss((self.fake_D+1)/2)
            self.TV_loss_unsuper = self.opt.lambda_TV * TVLossL1(self.fake_D)

            self.DC_loss_super = self.opt.lambda_DC * DCLoss((self.fake_B+1)/2, self.opt)
            #self.BC_loss_super = self.opt.lambda_DC * BCLoss((self.fake_B+1)/2)
            self.TV_loss_super = self.opt.lambda_TV * TVLossL1(self.fake_B)

            self.loss_G = self.loss_G_GAN + self.loss_G_Content

            #self.loss_G = self.loss_G_GAN + self.loss_G_Content +  self.DC_loss_super  +self.DC_loss_unsuper + self.TV_loss_super  +self.TV_loss_unsuper - self.BC_loss_super  -self.BC_loss_unsuper 
            if self.opt.semi:
                self.loss_G += self.DC_loss_unsuper + self.TV_loss_unsuper #+ self.BC_loss_unsuper
            if self.opt.all_loss:
                self.loss_G += self.DC_loss_super + self.TV_loss_super #+ self.BC_loss_super
        else:
            self.loss_G = self.loss_G_GAN + self.loss_G_Content
    
        self.loss_G.backward()

    def optimize_parameters(self, iter):
        self.forward()

        for iter_d in xrange(self.criticUpdates):
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G(iter)
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_Content', self.loss_G_Content.item()),
                            ('DCLoss', self.DC_loss_unsuper.item()) 
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        real_C = util.tensor2im(self.real_C.data)
        fake_D = util.tensor2im(self.fake_D.data)
        return OrderedDict([('Hazed_Train', real_A), ('Sharp_Train', real_B), ('Restored_Train', fake_B),  ('Real_Ori', real_C), ('Real_Restored', fake_D)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
