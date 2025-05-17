from .base_model import BaseModel
from . import networks
import torch



class TestRTCycleModel(BaseModel):
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
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

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
        # self.visual_names = ['fake_B','prev','img']
        self.visual_names = ['fake_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc*2, opt.output_nc*2, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, 80, opt.netG, opt.norm,
                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, addNoise = False)

        self.netG_A.module.load_state_dict(torch.load("checkpoints/folds/latest_net_G_A.pth",map_location=str(self.device)))
        self.netG_A.eval()
        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.


        self.prev = None
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']


    def forward(self):
        """Run forward pass."""
        # self.fake_B = self.netG(self.real_A)  # G(A)
        # self.nreal_B0 = torch.rand(self.real_A.shape[0],1,128).to(self.real_A.device)
        # self.treal_B0 = torch.rand(1, 3+9, 64, 64).to(self.real_A.device)
        # self.img,_m = self.netG_A(self.real_A,self.nreal_B0,self.treal_B0)

        if self.prev == None:

            # self.fake_B = self.netG(torch.cat((img,img),1))
            self.fake_B = self.real_A

        else:
            self.fake_B = self.netG(torch.cat((self.real_A,self.prev),1))
            _, self.fake_B = self.fake_B.chunk(2,1)
        self.prev = self.real_A
        # self.prevA = self.real_A

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
