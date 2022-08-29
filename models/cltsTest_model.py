from .base_model import BaseModel
from . import networks
import torch

class cltsTestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')

        # parser.set_defaults(dataset_mode='unaligned')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        parser.add_argument('--freeze_texture', action='store_true', help='whether texture will be fixed')
        parser.add_argument('--freeze_color', action='store_true', help='whether the color will be fixed')
        parser.add_argument('--augment', action='store_true', help='set true if augmenting image with texture and color with new textures and colors')




        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        #self.visual_names = ['real_A', 'fake_B']
        self.visual_names = ['fake_B1','fake_B2','fake_B3','fake_B4','fake_B5',]
        # self.visual_names = ['fake_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G_A', 'G_B']  # only generator is needed.
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, rev = True)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, addNoise = True)
        self.freeze_texture = opt.freeze_texture
        self.freeze_color = opt.freeze_color
        self.augment = opt.augment

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG_A' , self.netG_A)  # store netG in self.
        setattr(self, 'netG_B' , self.netG_B)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real_A = input['A'].to(self.device)
        # self.real_B = inputs['B'].to(self.device)

        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""

        text = torch.rand(self.real_A.shape[0], 3+9, 64, 64).to(self.real_A.device)
        color = torch.rand(self.real_A.shape[0],1,128).to(self.real_A.device)

        if self.augment:
            a,color,text = self.netG_A(self.real_A)  # G(A)
        else:
            a = self.real_A

        t = []
        if self.freeze_texture:
            for i in range(5):
                t.append(text)
        else:
            for i in range(5):
                t.append(torch.rand(self.real_A.shape[0], 3+9, 64, 64).to(self.real_A.device))

        n = []
        if self.freeze_color:
            for i in range(5):
                n.append(color)
        else:
            for i in range(5):
                n.append(torch.rand(self.real_A.shape[0],1,128).to(self.real_A.device))


        self.fake_B1 = self.netG_B(a,n[0],t[0])
        self.fake_B2 = self.netG_B(a,n[1],t[1])
        self.fake_B3 = self.netG_B(a,n[2],t[2])
        self.fake_B4 = self.netG_B(a,n[3],t[3])
        self.fake_B5 = self.netG_B(a,n[4],t[4])
    def optimize_parameters(self):
        """No optimization for test model."""
        pass
