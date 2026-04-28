import os

from Utils.device_utils import configure_runtime, resolve_device
from Utils.parameter import *

num_cpu_cores = max(1, 6)


def main(config):
    configure_runtime(config.device_backend, config.cuda_devices)
    config.device = resolve_device(config.device_backend)

    from torch.backends import cudnn
    from torch.utils.data import DataLoader

    from data_loader import Data_Item, FECGDataset
    from trainer import Trainer
    from Utils.utils import make_folder
    
    data_item = Data_Item()
    train_dataset = FECGDataset(data_item,train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_cpu_cores)   
    val_dataset = FECGDataset(data_item,train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=num_cpu_cores)    
    test_dataset = FECGDataset(data_item,train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=num_cpu_cores)

    
    
    # For fast training
    cudnn.benchmark = True
    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    if config.train:
        if config.model=='sagan':
            trainer = Trainer(train_dataloader, config)
        elif config.model == 'qgan':
            trainer = qgan_trainer(train_dataloader, config)
        trainer.train()
    else:
        tester = Tester(data_loader.loader(), config)
        tester.test()

if __name__ == '__main__':
    config = get_parameters()
    # print(config)
    main(config)