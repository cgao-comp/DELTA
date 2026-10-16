import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import numpy as np

def find_dataset_using_name(dataset_name):
    dataset_filename = 'data.' + dataset_name + '_dataset'
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls
    if dataset is None:
        raise NotImplementedError('In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase.' % (dataset_filename, target_dataset_name))
    return dataset

def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataset(opt):
    data_loader = CustomDatasetDataLoader(opt)
    return data_loader

def create_dataset_with_args(opt, **kwargs):
    _kwargs = []
    for key in kwargs:
        value = kwargs[key]
        if not isinstance(value, (list, tuple)):
            value = [value]
        lens = len(value)
        _kwargs += list(map(lambda x: {}, range(lens))) if len(_kwargs) == 0 else []
        for i, v in enumerate(value):
            _kwargs[i][key] = v
    dataloaders = tuple(map(lambda x: CustomDatasetDataLoader(opt, **x), _kwargs))
    return dataloaders if len(dataloaders) > 1 else dataloaders[0]

class CustomDatasetDataLoader:

    def __init__(self, opt, **kwargs):
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt, **kwargs)
        print('dataset [%s] was created' % type(self.dataset).__name__)
        if self.dataset.manual_collate_fn:
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches, num_workers=int(opt.num_threads), drop_last=opt.drop_last, collate_fn=self.dataset.collate_fn)
        else:
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches, num_workers=int(opt.num_threads), drop_last=False)

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

def collate_fn(batch):
    print(batch)
    pass

def _init_fn(random_seed):
    np.random.seed(random_seed)
