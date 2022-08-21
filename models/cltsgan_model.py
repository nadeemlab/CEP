#THIS IS THE ONE WITH THE SINGLE CYCLE DISCRIMINATOR!!!
import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import transforms


class CLTSGANModel(BaseModel):
    """
    This class implements the XDCycleGAN model, for learning a one-to-many image-to-image translation without paired data.

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_extend', type=float, default = 0.5, help='weight for adversarial loss on reconstructed image G_A(G_B(B))')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the XDCycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', "D_A", "noiseA", "noiseB_n","noiseB_t", 'idt_B', 'G_Brec', "D_Brec"]
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A',"rec_At"]
        visual_names_B = ['real_B', 'fake_A','fake_At', 'fake_A2', 'rec_B']

        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B','D_Brec']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, rev = True)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, addNoise = True)

        if self.isTrain:  # define discriminators
 
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_Brec = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)

            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.pool_size)  
            self.fake_B_pool = ImagePool(opt.pool_size)  
            self.rec_A_pool  = ImagePool(opt.pool_size)
            self.AtoB_pool = ImagePool(opt.pool_size)
            self.BtoA_pool = ImagePool(opt.pool_size)
            self.real_pool = ImagePool(opt.pool_size)


            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionVec = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B.parameters(),self.netD_A.parameters(), self.netD_Brec.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))


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
        self.image_paths = input['A_paths' if AtoB else 'B_paths']





    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""


        self.nreal_B = torch.rand(self.real_A.shape[0],1,128).to(self.real_A.device)
        self.nreal_A = torch.rand(self.real_A.shape[0],1,128).to(self.real_A.device)

        self.treal_B = torch.rand(1, 3+9, 64, 64).to(self.real_A.device)

        self.nreal_B2 = torch.rand(self.real_A.shape[0],1,128).to(self.real_A.device)
        self.treal_B2 = torch.rand(1, 3+9, 64, 64).to(self.real_A.device)


        self.fake_B, self.nfake_B, self.tfake_B = self.netG_A(self.real_A)  # G_A(A)

        self.rec_A   = self.netG_B(self.fake_B, self.nfake_B, self.tfake_B)   # G_B(G_A(A))
        self.rec_At  = self.netG_B(self.fake_B, self.nreal_B, self.treal_B)   # G_B(G_A(A))
        self.rec_At2 = self.netG_B(self.fake_B, self.nfake_B, self.treal_B)   # G_B(G_A(A))


        self.fake_A= self.netG_B(self.real_B, self.nreal_B, self.treal_B)  # G_B(B)
        self.fake_At = self.netG_B(self.real_B, self.nreal_B, self.treal_B2)  # G_B(B)

        self.fake_A2  = self.netG_B(self.real_B, self.nreal_B2, self.treal_B2)  # G_B(B)


        self.rec_B, self.nrec_B, self.trec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))



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

    def backward_D_Brec(self):
        """Calculate GAN loss for discriminator D_B"""

        self.loss_D_Brec = self.backward_D_basic(self.netD_Brec, self.rec_A.detach(), self.rec_At.detach())


    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""



        fake_A = self.fake_A_pool.query(self.fake_A.detach())
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)



    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)


    def backward_G(self):

        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_E = self.opt.lambda_extend

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A,n,t = self.netG_A(self.real_B)
            self.loss_idt_A  = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A,self.nfake_B, self.tfake_B )
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            self.loss_idt_B = 0

        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0


        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) #+ self.criterionGAN(self.netD_B(self.rec_A), True)


        rec_A = self.netG_B(self.fake_B.detach(), self.nfake_B, self.tfake_B)   # G_B(G_A(A))
        rec_At = self.netG_B(self.fake_B.detach(), self.nreal_B, self.treal_B)   # G_B(G_A(A))

        self.loss_G_Brec = self.criterionGAN(self.netD_Brec(rec_At), True)#/2
        self.loss_G_Brec += self.criterionGAN(self.netD_Brec(rec_A), False)#/2
        ##WWORKING prevv
        # self.loss_G_Brec = self.criterionGAN(self.netD_Brec(self.rec_At), True)/2
        # self.loss_G_Brec += self.criterionGAN(self.netD_Brec(self.rec_A), False)/2


        #Extended Cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss 
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B



        self.loss_noiseB_n  = self.criterionVec(self.nrec_B,self.nreal_B) * lambda_B ##/2
        self.loss_noiseB_t  = self.criterionVec(self.trec_B,self.treal_B) * lambda_B ##/2
        self.loss_noiseB = self.loss_noiseB_n + self.loss_noiseB_t 



        # self.loss_noise = self.loss_noise
        self.loss_noiseA = 0

        self.loss_noiseA = torch.clamp(.1 - torch.abs(self.criterionCycle(self.fake_At, self.fake_A)), min = 0) * 20  #had * 2 when workign


        # combined loss and calculate gradients#
        self.loss_G = (self.loss_G_A  + self.loss_G_B + self.loss_G_Brec) * 2  + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_noiseB + self.loss_noiseA
        self.loss_G.backward()



    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_B,self.netD_A,self.netD_Brec], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D and D_B
        self.set_requires_grad([self.netD_B,self.netD_A,self.netD_Brec], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.backward_D_Brec()      # calculate graidents for D_Brec

        self.optimizer_D.step()  # update D_A and D_B's weights
