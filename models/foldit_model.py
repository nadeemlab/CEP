#THIS IS THE FOLDIT MICCAI21 MODEL

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks



class FoldItModel(BaseModel):
    """
    This class implements the FoldIt model.
    """
    @staticmethod

    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser
        """
        parser.set_defaults(no_dropout=True) 

        if is_train:
            parser.add_argument('--lambda_adv', type=float, default=1.0, help='weight for adversarial portion')
            parser.add_argument('--lambda_T', type=float, default=10.0, help='weight for transitve loss')
            parser.add_argument('--lambda_GT', type = float, default = 10.0, help = 'weight for ground truth loss')

            parser.add_argument('--lambda_identity', type=float, default= 1.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')


        return parser



    def __init__(self, opt):
        """Initialize the FoldIt class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_A', 'G_B', 'G_AC', 'G_BC', 'D_A','D_B', 'D_AC', 'D_BC','T_AC', 'T_BC','GT','idt_AC','idt_BC',]#'G_big','D_big']


        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_AC', 'fake_AC']
        visual_names_B = ['real_B', 'fake_A', 'rec_BC', 'fake_BC']
        visual_names_C = ['real_C','idt_AC','idt_BC']



        self.visual_names = visual_names_A + visual_names_B +visual_names_C # combine visualizations for A and B and C

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:

            self.model_names = ['G_A', 'G_B', 'G_AC', 'G_BC', 'D_A','D_B', 'D_BC', 'D_AC']

        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B', 'G_AC', 'G_BC']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_AC = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_BC = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)




        if self.isTrain:  # define discriminators

            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)            

            self.netD_BC = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_AC = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)




        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)


            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.L1Loss = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(),self.netG_AC.parameters(),self.netG_BC.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(),self.netD_BC.parameters(),self.netD_AC.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # print(input['B_paths'],input['C_paths'])

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)      # G_A(A)
        self.rec_AC = self.netG_BC(self.fake_B)     # G_BC(G_A(A))
        self.fake_AC = self.netG_AC(self.real_A)    # G_AC(A)


        self.fake_A = self.netG_B(self.real_B)      # G_B(B)
        self.rec_BC = self.netG_AC(self.fake_A)     # G_AC(G_B(B))
        self.fake_BC = self.netG_BC(self.real_B)    # G_BC(B)

        self.idt_AC = self.netG_AC(self.real_C)     # G_AC(C)
        self.idt_BC = self.netG_BC(self.real_C)     # G_BC(C)



    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""

        # fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_A"""

        # fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)



    def backward_D_AC(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_AC = self.backward_D_basic(self.netD_AC, self.real_C, self.fake_AC.detach())


    def backward_D_BC(self):
        """Calculate GAN loss for discriminator D_BC"""
        self.loss_D_BC = self.backward_D_basic(self.netD_BC, self.real_C, self.fake_BC.detach())

    # def backward_D_big(self):
    #     """Calculate GAN loss for discriminator D_A"""

    #     #conditional loss
    #     real = torch.cat((self.real_B,self.real_C),1)
    #     syn = torch.cat((self.fake_B.detach(),self.fake_AC.detach()),1)
    #     self.loss_D_big = self.backward_D_basic(self.netD_big, real, syn)




    def backward_G(self):


        """Calculate the loss for generators G_A and G_B"""

        #get weights
        lambda_idt = self.opt.lambda_identity
        lambda_T = self.opt.lambda_T
        lambda_GT = self.opt.lambda_GT
        lambda_adv = self.opt.lambda_adv

        #______________________________________________________
        #Adversarial Losses
        #______________________________________________________

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)* lambda_adv

        # GAN loss D_AC(G_AC(A))
        self.loss_G_AC = self.criterionGAN(self.netD_AC(self.fake_AC),True)* lambda_adv

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) * lambda_adv

        # GAN loss D_BC(G_BC(B))
        self.loss_G_BC = self.criterionGAN(self.netD_BC(self.fake_BC),True)* lambda_adv


        #sum adversarial losses
        self.loss_G = self.loss_G_B + self.loss_G_BC + self.loss_G_A + self.loss_G_AC


        #______________________________________________________
        #Transitive Losses
        #______________________________________________________

        self.loss_T_BC = self.criterionCycle(self.fake_BC, self.rec_BC) * lambda_T

        self.loss_T_AC = self.criterionCycle(self.fake_AC, self.rec_AC) * lambda_T

        #add transitive loss
        self.loss_G += self.loss_T_BC + self.loss_T_AC


        #______________________________________________________
        #Ground Truth Losses
        #______________________________________________________

        self.loss_GT = self.L1Loss(self.fake_BC, self.real_C) * lambda_GT 

        #add ground truth loss
        self.loss_G += self.loss_GT

        #______________________________________________________
        #Identity Losses
        #______________________________________________________
        self.loss_idt_BC = self.criterionIdt(self.idt_BC, self.real_C) * lambda_idt

        self.loss_idt_AC = self.criterionIdt(self.idt_AC, self.real_C) * lambda_idt

        #add identity loss
        self.loss_G += self.loss_idt_BC + self.loss_idt_AC

        #apply backwards operation
        self.loss_G.backward()


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward


        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B,self.netD_BC, self.netD_AC], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights


        self.set_requires_grad([self.netD_A, self.netD_B,self.netD_BC, self.netD_AC], True)

        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_BC()
        self.backward_D_AC()

        # self.backward_D()      # calculate gradients for D_A
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights








