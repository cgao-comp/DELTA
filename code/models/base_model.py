import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from .networks import tools

class BaseModel(ABC):

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.save_shared_dir = os.path.join(opt.shared_dir, opt.name)
        if opt.cuda_benchmark:
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.optimizers = []
        self.metric = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self, opt):
        if self.isTrain:
            self.schedulers = [tools.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            for name in self.model_names:
                net = getattr(self, 'net' + name)
                net = tools.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
                setattr(self, 'net' + name, net)
        else:
            self.eval()
        self.print_networks(opt.verbose)
        self.post_process()

    def cuda(self):
        assert torch.cuda.is_available()
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            net.to(self.gpu_ids[0])
            net = torch.nn.DataParallel(net, self.gpu_ids)

    def eval(self):
        self.isTrain = False
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        self.isTrain = True
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        with torch.no_grad():
            self.forward()

    def compute_visuals(self):
        pass

    def update_learning_rate(self, logger):
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        logger.info('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

    def load_networks_cv(self, folder_path):
        checkpoints = list(filter(lambda x: x.endswith('.pth'), os.listdir(folder_path)))
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = list(filter(lambda x: x.split('.')[0].endswith('net_' + name), checkpoints))
                print('load_filename is:', load_filename)
                if len(load_filename) == 1:
                    load_filename = load_filename[0]
                    load_path = os.path.join(folder_path, load_filename)
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=self.device)
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                    net.load_state_dict(state_dict)
                else:
                    continue

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1000000.0))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def post_process(self):
        pass
