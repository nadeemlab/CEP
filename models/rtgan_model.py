import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import random

#RT_CycleGAN Model
class RTGANModel(BaseModel):
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
            # parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            # parser.add_argument('--lambda_extend', type=float, default = 0.5, help='weight for adversarial loss on reconstructed image G_A(G_B(B))')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the TempCycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["temp_cycle_B","G_Bt","D", 'D_ref', 'G_ref','idtBt']

        

        # self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        self.visual_names = ['real_B0','real_B1', 'fake_B0', 'fake_B1', 'Bt1', 'Bt2','real_B2','idt_B1t','real_A0','real_A1']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['D_ref','D', "G_Btp",]
        else:  # during test time, only load Gs
            self.model_names = ['G_A', "G_Btp"]

        # # define networks (both Generators and discriminators)
        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, 80, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,)
        self.netG_A = networks.define_G(opt.output_nc, opt.input_nc, 80, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, addNoise = True)
        # self.netG_A.module.load_state_dict(torch.load("checkpoints/detcahrec_g0/latest_net_G_A.pth",map_location=str(self.device)))
        # self.netG_A.module.load_state_dict(torch.load("checkpoints/xdcyclegan_depth/latest_net_G_A.pth",map_location=str(self.device)))
        # self.netG_A.module.load_state_dict(torch.load("checkpoints/folds/latest_net_G_A.pth",map_location=str(self.device)))
        self.netG_A.module.load_state_dict(torch.load("checkpoints/CLTS-GAN/latest_net_G_B.pth",map_location=str(self.device)))


        self.netG_A.eval()

        self.netG_Btp = networks.define_G(opt.output_nc*3, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # define discriminators
            self.netD = networks.define_D(opt.output_nc, opt.ndf, '3D',
                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # # self.netD = networks.define_D(opt.output_nc*2, opt.ndf, opt.netD,
            # #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # self.netD_B = networks.define_D(opt.input_nc, opt.ndf, 'spec',
            #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_Atp = networks.define_D(opt.input_nc*2, opt.ndf, 'spec',
            #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_ref = networks.define_D(opt.output_nc*2, opt.ndf, 'spec',
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)

            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.pool_size)  
            self.fake_B_pool = ImagePool(opt.pool_size)  
            self.real_pool = ImagePool(opt.pool_size)
            self.Bt_pool = ImagePool(opt.pool_size)
            self.Bt_single_pool = ImagePool(opt.pool_size)
            self.Bt_ref = ImagePool(opt.pool_size)



            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netG_Atp.parameters(), self.netG_Btp.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_Btp.parameters(),), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters(),self.netD_ref.parameters()), lr=opt.lr/5, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.SGD(itertools.chain(self.netD_Btp.parameters(),self.netD_B.parameters()), lr=opt.lr, momentum = .9 ,nesterov = True)

            # self.optimizer_D = torch.optim.SGD(itertools.chain(self.netD_B.parameters(),self.netD.parameters(), self.netD_Atp.parameters(), self.netD_Btp.parameters()), lr=opt.lr, momentum = .9 ,nesterov = True)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A0 = input['A0' if AtoB else 'B0'].to(self.device)
        self.real_A1 = input['A1' if AtoB else 'B1'].to(self.device)
        self.real_A2 = input['A2' if AtoB else 'B2'].to(self.device)


        self.real_B0 = input['B0' if AtoB else 'A0'].to(self.device)
        self.real_B1 = input['B1' if AtoB else 'A1'].to(self.device)
        self.real_B2 = input['B2' if AtoB else 'A2'].to(self.device)

        # self.real_C0 = input['C0']


        self.image_paths = input['A0_paths' if AtoB else 'B1_paths']

    def forward(self):
        # """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.nreal_B0 = torch.rand(self.real_A0.shape[0],1,128).to(self.real_A0.device)
        self.treal_B0 = torch.rand(1, 3+9, 64, 64).to(self.real_A0.device)

        self.nreal_B1 = torch.rand(self.real_A0.shape[0],1,128).to(self.real_A0.device)
        self.treal_B1 = torch.rand(1, 3+9, 64, 64).to(self.real_A0.device)

        self.nreal_B2 = torch.rand(self.real_A0.shape[0],1,128).to(self.real_A0.device)
        self.treal_B2 = torch.rand(1, 3+9, 64, 64).to(self.real_A0.device)


        self.fake_B0,_ = self.netG_A(self.real_A0,self.nreal_B0,self.treal_B0)  # G_A(A)
        self.fake_B1,_ = self.netG_A(self.real_A1,self.nreal_B1,self.treal_B1)  # G_A(A)
        self.fake_B2,_ = self.netG_A(self.real_A2,self.nreal_B2,self.treal_B2)  # G_A(A)

        # print(self.netG_A(self.real_B0))
        # self.fake_B0 = self.netG_A(self.real_B0)  # G_A(A)
        # self.fake_B1 = self.netG_A(self.real_B1)  # G_A(A)
        # self.fake_B2 = self.netG_A(self.real_B2)  # G_A(A)



        self.Bt1 = self.netG_Btp(torch.cat((self.real_B0,self.real_B1,self.fake_B0.detach()),1))
        self.Bt2 = self.netG_Btp(torch.cat((self.real_B1,self.real_B2,self.Bt1),1))


        # self.rec_B1 = self.netG_Btp(torch.cat((self.real_B2,self.real_B1,self.Bt2),1))
        # self.rec_B0 = self.netG_Btp(torch.cat((self.real_B1,self.real_B0,self.rec_B1),1))



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

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_B = self.fake_B_pool.query(self.fake_B0)
        # fake_B = self.fake_B_pool.query(self.fake_B1)

        Bt = self.Bt_single_pool.query(random.choice([self.Bt1,self.Bt2]))
        # Bt = self.Bt_single_pool.query(self.Bt1)

        self.loss_D_B = self.backward_D_basic(self.netD_B, fake_B, Bt)

    # def backward_D(self):
    #     """Calculate GAN loss for discriminator D"""

    #     AB = torch.cat((self.real_A0,self.real_A1,self.Bt0,self.fake_B1),1)
    #     BA = torch.cat((self.real_A0,self.real_A1,self.fake_B0,self.Bt1),1)

    #     self.loss_D = self.backward_D_basic(self.netD, AB.detach(), BA.detach())
    def backward_D_ref(self):
        """Calculate GAN loss for discriminator D"""

        BA = random.choice([torch.cat((self.real_B2,self.Bt2),1),torch.cat((self.real_B1,self.Bt1),1)])
        # AB = random.choice([torch.cat((self.real_B0,self.fake_B0),1),torch.cat((self.real_B1,self.fake_B1),1)])
        AB = self.Bt_ref.query(random.choice([torch.cat((self.real_B0,self.fake_B0),1),torch.cat((self.real_B1,self.fake_B1),1)]))
        self.loss_D_ref = self.backward_D_basic(self.netD_ref, AB.detach(), BA.detach())



    def backward_D(self):
        """Calculate GAN loss for discriminator D"""

        real_pair = random.choice([torch.stack((self.real_A0,self.real_A1,self.real_A2),2),torch.stack((self.real_A2,self.real_A1,self.real_A0),2)])

        fake_pair = self.Bt_pool.query(random.choice([torch.stack((self.fake_B0, self.Bt1,self.Bt2),2),torch.stack((self.Bt2,self.Bt1, self.fake_B0),2)]))
        # fake_pair = self.Bt_pool.query(torch.stack((self.Bt2,self.Bt1, self.fake_B0),2))

        # fake_pair = self.Bt_pool.query(torch.cat((self.Bt0,self.fake_B1),1))


        self.loss_D = self.backward_D_basic(self.netD, real_pair, fake_pair.detach())


        # real_pair2 = random.choice([torch.stack((self.real_A0,self.real_A1,self.real_A2),2),torch.stack((self.real_A2,self.real_A1,self.real_A0),2)])
        real_pair2 = torch.stack((self.fake_B0,self.fake_B0,self.fake_B0),2)

        orderings = [torch.stack((self.real_A2,self.real_A0,self.real_A1),2)]
        orderings.append(torch.stack((self.real_A1,self.real_A0,self.real_A2),2))
        orderings.append(torch.stack((self.real_A0,self.real_A2,self.real_A1),2))
        orderings.append(torch.stack((self.real_A1,self.real_A0,self.real_A2),2))


        fake_pair2 = random.choice(orderings)
        # fake_pair2 = random.choice([torch.stack((self.fake_B2, self.fake_B1,self.fake_B0),2),torch.stack((self.fake_B2,self.fake_B1, self.fake_B0),2)])
        self.loss_D += self.backward_D_basic(self.netD, real_pair2.detach(), fake_pair2)


    def backward_G(self):

        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        # lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # lambda_E = self.opt.lambda_extend

        #compare paried input and output (make sure output image matches input)
        ref = random.choice([torch.cat((self.real_B2,self.Bt2),1),torch.cat((self.real_B1,self.Bt1),1)])
        self.loss_G_ref = self.criterionGAN(self.netD_ref(ref), True)

        # temporal consitency
        # AB = random.choice([torch.cat((self.Bt2,self.Bt1,self.fake_B0.detach()),1),torch.cat((self.fake_B0.detach(),self.Bt1,self.Bt2),1)])
        # AB_0 = torch.stack((self.Bt2,self.Bt1,self.fake_B0.detach()),2)
        # AB_1 = torch.stack((self.fake_B0.detach(),self.Bt1,self.Bt2),2)

        AB_0 = torch.stack((self.Bt2,self.Bt1,self.fake_B0.detach()),2)
        AB_1 = torch.stack((self.fake_B0.detach(),self.Bt1,self.Bt2),2)

        AB = random.choice([AB_0,AB_1])

        self.loss_G_Bt = self.criterionGAN(self.netD(AB), True)* .5

        #image quality
        # self.loss_G_B = self.criterionGAN(self.netD_B(random.choice([self.Bt2,self.Bt1])),True)
        # self.loss_G_B = 0

        temp_gan = self.loss_G_Bt + self.loss_G_ref 

        # #Extended Cycle loss
        # self.treal_B0 = torch.rand(1, 3+9, 64, 64).to(self.real_A0.device)

        # self.nreal_B1 = torch.rand(self.real_A0.shape[0],1,128).to(self.real_A0.device)
        # self.treal_B1 = torch.rand(1, 3+9, 64, 64).to(self.real_A0.device)

 
        #temporal additions here
        # self.loss_temp_cycle_B = (self.criterionCycle(self.fake_B0.detach(), self.rec_B0) + self.criterionCycle(self.fake_B1.detach(), self.rec_B1)) * lambda_B * temp_weight
        self.loss_temp_cycle_B = 0
        self.idt_B1t = self.netG_Btp(torch.cat((self.real_B1,self.real_B1,self.Bt1.detach()),1))

        self.loss_idtBt = self.criterionIdt(self.idt_B1t, self.Bt1)
        temp_idt = self.loss_idtBt 
        # self.loss_idt = 0 

        self.loss_G = self.loss_temp_cycle_B + temp_gan + temp_idt
        self.loss_G.backward()



    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        # self.set_requires_grad([self.netD_B,self.netD,self.netD_Atp,self.netD_Btp], False)  # Ds require no gradients when optimizing Gs
        # self.set_requires_grad([self.netD_Btp,self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD,self.netD_ref], False)  # Ds require no gradients when optimizing Gs

        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D and D_B
        # self.set_requires_grad([self.netD_B,self.netD,self.netD_Atp,self.netD_Btp], True)
        # self.set_requires_grad([self.netD_Btp,self.netD_B], True)
        self.set_requires_grad([self.netD,self.netD_ref], True)


        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        # self.backward_D_B()      # calculate graidents for D_B
        self.backward_D_ref()      # calculate graidents for D_B

        self.optimizer_D.step()  # update D_A and D_B's weights
