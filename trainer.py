import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,3"
import time
import torch
import datetime
import math
import signal
import sys

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from Utils.sagan_models import Generator, Discriminator
from Utils.utils import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from Utils.utils import (
    ALL_MODULE_SUFFIXES,
    CHECKPOINT_MARKER,
    DISCRIMINATOR_SUFFIXES,
    GENERATOR_SUFFIXES,
    RESUME_STATE_FILENAME,
    atomic_torch_save,
    capture_rng_state,
    find_latest_checkpoint,
    make_folder,
    previous_generation_path,
    restore_rng_state,
    torch_load,
    unwrap_module,
)
from Utils.device_utils import get_device_backend, resolve_device


device = None

class logcosh(nn.Module):
    def __init__(self):
        super().__init__()
        self._log2 = math.log(2.0)
        
    def forward(self, true, pred):
        # Numerically stable log-cosh:
        # log(cosh(x)) = |x| + softplus(-2|x|) - log(2)
        # This avoids overflow in cosh(x) for large |x|.
        x = pred - true
        ax = torch.abs(x)
        loss = ax + F.softplus(-2.0 * ax) - self._log2
        return torch.sum(loss)




class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal1(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0, 0.1)
        m.bias.data.zero_()
    elif isinstance(m, nn.InstanceNorm1d):
        # pass
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias, 0)
        
def weights_init_normal(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.InstanceNorm1d):
        pass
        # nn.init.constant_(m.weight,1)
        # nn.init.constant_(m.bias, 0)        
        
        


class Trainer(object):
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_AECG_lr = config.g_AECG_lr
        self.g_MECG_lr = config.g_MECG_lr
        self.g_FECG_lr = config.g_FECG_lr
        self.g_BIAS_lr = config.g_BIAS_lr

        self.d_AECG_lr = config.d_AECG_lr
        self.d_MECG_lr = config.d_MECG_lr
        self.d_FECG_lr = config.d_FECG_lr
        self.d_BIAS_lr = config.d_BIAS_lr
        
        
        self.decay_start_epoch = round(config.total_step / config.batch_size) - 1
        self.decay_start_epoch = 1
        
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model
        self.device = getattr(config, "device", resolve_device(getattr(config, "device_backend", get_device_backend())))
        global device
        device = self.device

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.resume_save_hours = getattr(config, 'resume_save_hours', 12.0)
        self.archive_discriminators = getattr(config, 'archive_discriminators', False)
        self.resume_state_path = os.path.join(self.model_save_path, RESUME_STATE_FILENAME)
        self.epoch_log_path = os.path.join(self.log_path, 'training_log.txt')
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        self.start_step = 0
        self.loss_c = float('inf')
        self.resume_suffixes = None
        self.resume_from_state = False
        self._stop_requested = False

        if getattr(config, 'resume', False):
            self._plan_resume()

        self.build_model()
        
        
        
        self.gamma_FECG_fake  = nn.Parameter(torch.zeros(1)).to(device)
        self.lambda_MECG_fake = nn.Parameter(torch.zeros(1)).to(device)
        self.beta_BIAS_fake   = nn.Parameter(torch.zeros(1)).to(device)
        
        self.gamma_FECG_reconstr  = nn.Parameter(torch.zeros(1)).to(device)
        self.lambda_MECG_reconstr = nn.Parameter(torch.zeros(1)).to(device)
        self.beta_BIAS_reconstr   = nn.Parameter(torch.zeros(1)).to(device)
        
        self.gamma_AECG_loss = nn.Parameter(torch.ones(1)*0.8).to(device)
        self.gamma_FECG_loss = nn.Parameter(torch.ones(1)*0.4).to(device)
        self.gamma_MECG_loss = nn.Parameter(torch.ones(1)*0.4).to(device)
        self.gamma_BIAS_loss = nn.Parameter(torch.ones(1)*0.2).to(device)
        

        if self.use_tensorboard:
            self.build_tensorboard()

        if self.resume_from_state:
            if not self.load_resume_state():
                print(f'No usable {RESUME_STATE_FILENAME}; falling back to the '
                      f'step-numbered archive (weights only)', flush=True)
                self.resume_from_state = False
                self._plan_archive_resume()

        if not self.resume_from_state and self.pretrained_model is not None:
            self.load_pretrained_model(self.resume_suffixes)
            self.start_step = self.pretrained_model + 1


    def _plan_resume(self):
        """Pick a resume source: the rolling state file, else an older checkpoint set."""
        if self.pretrained_model is not None:
            print(f'Resuming weights from checkpoint step {self.pretrained_model}')
            return

        if any(os.path.isfile(path) for path in self._resume_state_candidates()):
            self.resume_from_state = True
            return

        if not self._plan_archive_resume():
            raise FileNotFoundError(
                f'--resume set but no {RESUME_STATE_FILENAME} and no complete checkpoint '
                f'set found in {self.model_save_path}'
            )

    def _plan_archive_resume(self):
        """Point pretrained_model at the newest usable step-numbered set."""
        self.pretrained_model = find_latest_checkpoint(
            self.model_save_path, required_suffixes=ALL_MODULE_SUFFIXES)
        if self.pretrained_model is not None:
            print(f'Resuming weights from complete checkpoint set at step '
                  f'{self.pretrained_model}')
            return True

        self.pretrained_model = find_latest_checkpoint(
            self.model_save_path, required_suffixes=GENERATOR_SUFFIXES)
        if self.pretrained_model is None:
            return False

        self.resume_suffixes = list(GENERATOR_SUFFIXES)
        print(f'Resuming generators only from step {self.pretrained_model}; '
              f'discriminators start fresh')
        return True

    def _resume_state_candidates(self):
        return [self.resume_state_path, previous_generation_path(self.resume_state_path)]

    def train(self):
        start = self.start_step

        #  Train Discriminators
        self.model_train()
        # Start time
        start_time = time.time()
        resume_save_interval = self.resume_save_hours * 3600.0
        last_resume_save = time.time()
        show_progress = sys.stderr.isatty()
        self._install_stop_handler()
        MECG_factor = 5.0
        FECG_factor = 5.0
        BIAS_factor = 0.5
        last_step = start - 1
        last_saved_step = None
        for step in range(start, self.total_step):
            last_step = step
            tbar = tqdm(self.data_loader, desc='epoch'+str(step), disable=not show_progress)
            for AECG_signals, FECG_signals, MECG_signals,BIAS_signals in tbar: 
                # print(AECG_signals.shape)
                # MECG_signals =  AECG_signals - FECG_signals   
                # BIAS_signals = tensor2var(torch.randn(AECG_signals.size(0), AECG_signals.size(1), AECG_signals.size(2))*torch.max(AECG_signals)*0.1).expand_as(AECG_signals)
                
                
                # print('A',AECG_signals,'\n')
                # print('\n')
                # print('F',FECG_signals,'\n')
                # print('\n')
                # print('M',MECG_signals,'\n')
                # print('\n')
                # print('B',BIAS_signals,'\n')
                # print('\n')
                # BIAS_signals = AECG_signals - FECG_signals - MECG_signals
                valid = torch.ones((AECG_signals.shape[0],1,128),dtype=torch.float32).to(device)
                fake = torch.zeros((AECG_signals.shape[0],1,128),dtype=torch.float32).to(device)
                # valid = Variable(torch.Tensor((AECG_signals.shape[0],1,1)).fill_(1.0).float(), requires_grad=False).to(device)
                # fake =  Variable(torch.Tensor((AECG_signals.shape[0],1,1)).fill_(0.0).float(), requires_grad=False).to(device)
                
                
                AECG_signals = AECG_signals.to(device, dtype=torch.float32)
                FECG_signals = FECG_signals.to(device, dtype=torch.float32)
                MECG_signals = MECG_signals.to(device, dtype=torch.float32)
                BIAS_signals = BIAS_signals.to(device, dtype=torch.float32)
                # plt.plot(AECG_signals[0].t().cpu().numpy(),'r')    
                # plt.show()
                # plt.plot(FECG_signals[0].t().cpu().numpy(),'g')
                # plt.show()
                # plt.plot(MECG_signals[0][0].detach().cpu().numpy(),'b')
                # plt.title('MECG')
                # plt.show()
                


                self.optimizer_G_zero_grad()
                #AECG to MECG
                #1 generator loss
                # fake_MECG_signals = self.G_AECG2MECG(AECG_signals)
                # fake_AECG_signals = self.G_MECG2AECG(MECG_signals)
                
                # reco_AECG_signals = self.G_MECG2AECG(fake_MECG_signals)
                # reco_MECG_signals = self.G_AECG2MECG(fake_AECG_signals)
                
                # d_AECG_signals = self.D_AECG2MECG(AECG_signals)
                # d_AECG_loss_real = self.loss_discriminator(d_AECG_signals,valid)    
                # d_fake_AECG_signals = self.D_AECG2MECG(fake_AECG_signals)
                # d_AECG_loss_fake = self.loss_discriminator(d_fake_AECG_signals,fake)  
                # d_AECG_loss = (d_AECG_loss_real + d_AECG_loss_fake)*0.5 
                
                # d_MECG_signals = self.D_MECG2AECG(MECG_signals)
                # d_MECG_loss_real = self.loss_discriminator(d_MECG_signals,valid)
                # d_fake_MECG_signals = self.D_MECG2AECG(fake_MECG_signals)
                # d_MECG_loss_fake = self.loss_discriminator(d_fake_MECG_signals,fake) 
                # d_MECG_loss = (d_MECG_loss_real + d_MECG_loss_fake)*0.5 
                
                # d_AECG2MECG_loss = (d_AECG_loss + d_MECG_loss)*0.5
                
                
                
                
                same_MECG_signals = self.G_AECG2MECG(AECG_signals)
                loss_generator_MECG = self.loss_generator(same_MECG_signals,MECG_signals.float())*1             
                same_AECG_signals = self.G_MECG2AECG(MECG_signals)   
                loss_generator_AECG = self.loss_generator(same_AECG_signals,AECG_signals.float())*1
                
                
                #2 forwardGAN loss
                fake_MECG_signals = self.G_AECG2MECG(AECG_signals)
                pred_fake_MECG_signals = self.D_AECG2MECG(fake_MECG_signals)
                loss_forwardGAN_AECG2MECG = self.loss_forwardGAN(pred_fake_MECG_signals,valid)
                
                fake_AECG_signals_from_MECG = self.G_MECG2AECG(MECG_signals)
                pred_fake_AECG_signals_from_MECG = self.D_MECG2AECG(fake_AECG_signals_from_MECG)
                loss_forwardGAN_MECG2AECG = self.loss_forwardGAN(pred_fake_AECG_signals_from_MECG,valid)
                
                #3 cycleGAN loss
                reconstr_AECG_signals_from_MECG = self.G_MECG2AECG(fake_MECG_signals)
                loss_cycleGAN_AECG2MECG2AECG = self.loss_cycleGAN(reconstr_AECG_signals_from_MECG,AECG_signals.float())*0.04
                
                reconstr_MECG_signals = self.G_AECG2MECG(fake_AECG_signals_from_MECG)
                loss_cycleGAN_MECG2AECG2MECG = self.loss_cycleGAN(reconstr_MECG_signals,MECG_signals.float())*0.04
                
                loss_G_total_AECG2MECG = loss_generator_MECG + loss_generator_AECG + loss_forwardGAN_AECG2MECG + loss_forwardGAN_MECG2AECG + loss_cycleGAN_AECG2MECG2AECG + loss_cycleGAN_MECG2AECG2MECG
                loss_G_total_AECG2MECG.backward(retain_graph=True)
                
                
                
                
                #AECG to FECG
                #1 generator loss
                same_FECG_signals = self.G_AECG2FECG(AECG_signals)
                loss_generator_FECG = self.loss_generator(same_FECG_signals,FECG_signals.float())*4               
                same_AECG_signals = self.G_FECG2AECG(FECG_signals)   
                loss_generator_AECG = self.loss_generator(same_AECG_signals,AECG_signals.float())*4
                
                
                #2 forwardGAN loss
                fake_FECG_signals = self.G_AECG2FECG(AECG_signals)
                pred_fake_FECG_signals = self.D_AECG2FECG(fake_FECG_signals)
                loss_forwardGAN_AECG2FECG = self.loss_forwardGAN(pred_fake_FECG_signals,valid)
                
                fake_AECG_signals_from_FECG = self.G_FECG2AECG(FECG_signals)
                pred_fake_AECG_signals_from_FECG = self.D_FECG2AECG(fake_AECG_signals_from_FECG)
                loss_forwardGAN_FECG2AECG = self.loss_forwardGAN(pred_fake_AECG_signals_from_FECG,valid)
                
                #3 cycleGAN loss
                reconstr_AECG_signals_from_FECG = self.G_FECG2AECG(fake_FECG_signals)
                loss_cycleGAN_AECG2FECG2AECG = self.loss_cycleGAN(reconstr_AECG_signals_from_FECG,AECG_signals.float())*0.04
                
                reconstr_FECG_signals = self.G_AECG2FECG(fake_AECG_signals_from_FECG)
                loss_cycleGAN_FECG2AECG2FECG = self.loss_cycleGAN(reconstr_FECG_signals,FECG_signals.float())*0.04
                
                loss_G_total_AECG2FECG = loss_generator_FECG + loss_generator_AECG + loss_forwardGAN_AECG2FECG + loss_forwardGAN_FECG2AECG + loss_cycleGAN_AECG2FECG2AECG + loss_cycleGAN_FECG2AECG2FECG
                loss_G_total_AECG2FECG.backward(retain_graph=True)

                
                #AECG to BIAS
                #1 generator loss
                same_BIAS_signals = self.G_AECG2BIAS(AECG_signals)
                loss_generator_BIAS = self.loss_generator(same_BIAS_signals,BIAS_signals.float())*1               
                same_AECG_signals = self.G_BIAS2AECG(BIAS_signals)   
                loss_generator_AECG = self.loss_generator(same_AECG_signals,AECG_signals.float())*1
                
                
                #2 forwardGAN loss
                fake_BIAS_signals = self.G_AECG2BIAS(AECG_signals)
                pred_fake_BIAS_signals = self.D_AECG2BIAS(fake_BIAS_signals)
                loss_forwardGAN_AECG2BIAS = self.loss_forwardGAN(pred_fake_BIAS_signals,valid)
                
                fake_AECG_signals_from_BIAS = self.G_BIAS2AECG(BIAS_signals)
                pred_fake_AECG_signals_from_BIAS = self.D_BIAS2AECG(fake_AECG_signals_from_BIAS)
                loss_forwardGAN_BIASAECG = self.loss_forwardGAN(pred_fake_AECG_signals_from_BIAS,valid)
                
                #3 cycleGAN loss
                reconstr_AECG_signals_from_BIAS = self.G_BIAS2AECG(fake_BIAS_signals)
                loss_cycleGAN_AECG2BIAS2AECG = self.loss_cycleGAN(reconstr_AECG_signals_from_BIAS,AECG_signals.float())*0.04
                
                reconstr_BIAS_signals = self.G_AECG2BIAS(fake_AECG_signals_from_BIAS)
                loss_cycleGAN_BIAS2AECG2BIAS = self.loss_cycleGAN(reconstr_BIAS_signals,BIAS_signals.float())*0.04
                
                loss_G_total_AECG2BIAS = loss_generator_BIAS + loss_generator_AECG + loss_forwardGAN_AECG2BIAS + loss_forwardGAN_BIASAECG + loss_cycleGAN_AECG2BIAS2AECG + loss_cycleGAN_BIAS2AECG2BIAS
                loss_G_total_AECG2BIAS.backward(retain_graph=True) 
                

                
                
                
                
                #D loss      
                self.optimizer_D_zero_grad()  
                #AECG to MECG
                pred_MECG_signals = self.D_AECG2MECG(AECG_signals)
                loss_D_real_forwardGAN_AECG2MECG = self.loss_forwardGAN(pred_MECG_signals,valid)
                pred_fake_MECG_signals = self.D_AECG2MECG(fake_AECG_signals_from_MECG)
                loss_D_fake_forwardGAN_AECG2MECG = self.loss_forwardGAN(pred_fake_MECG_signals, fake)               
                loss_D_forwardGAN_AECG2MECG= (loss_D_real_forwardGAN_AECG2MECG +  loss_D_fake_forwardGAN_AECG2MECG)*0.5
                loss_D_forwardGAN_AECG2MECG.backward(retain_graph=True)
                
                pred_AECG_signals = self.D_MECG2AECG(MECG_signals)
                loss_D_real_forwardGAN_MECG2AECG = self.loss_forwardGAN(pred_AECG_signals,valid)
                pred_fake_AECG_signals_from_MECG = self.D_MECG2AECG(fake_MECG_signals)
                loss_D_fake_forwardGAN_MECG2AECG = self.loss_forwardGAN(pred_fake_AECG_signals_from_MECG, fake)               
                loss_D_forwardGAN_MECG2AECG= (loss_D_real_forwardGAN_MECG2AECG +  loss_D_fake_forwardGAN_MECG2AECG)*0.5

                
                loss_D_AECG2MECG = (loss_D_forwardGAN_AECG2MECG + loss_D_forwardGAN_MECG2AECG)*0.5
                loss_D_AECG2MECG.backward(retain_graph=True)
                

                #AECG to FECG
                pred_FECG_signals = self.D_AECG2FECG(AECG_signals)
                loss_D_real_forwardGAN_AECG2FECG = self.loss_forwardGAN(pred_FECG_signals,valid)
                pred_fake_FECG_signals = self.D_AECG2FECG(fake_AECG_signals_from_FECG)
                loss_D_fake_forwardGAN_AECG2FECG = self.loss_forwardGAN(pred_fake_FECG_signals, fake)               
                loss_D_forwardGAN_AECG2FECG= (loss_D_real_forwardGAN_AECG2FECG +  loss_D_fake_forwardGAN_AECG2FECG)*0.5
                loss_D_forwardGAN_AECG2FECG.backward(retain_graph=True)
                
                pred_AECG_signals = self.D_FECG2AECG(FECG_signals)
                loss_D_real_forwardGAN_FECG2AECG = self.loss_forwardGAN(pred_AECG_signals,valid)
                pred_fake_AECG_signals_from_FECG = self.D_FECG2AECG(fake_FECG_signals)
                loss_D_fake_forwardGAN_FECG2AECG = self.loss_forwardGAN(pred_fake_AECG_signals_from_FECG, fake)               
                loss_D_forwardGAN_FECG2AECG= (loss_D_real_forwardGAN_FECG2AECG +  loss_D_fake_forwardGAN_FECG2AECG)*0.5

                
                loss_D_AECG2FECG = (loss_D_forwardGAN_AECG2FECG + loss_D_forwardGAN_FECG2AECG)*0.5
                loss_D_AECG2FECG.backward(retain_graph=True)
                
                
                #AECG to BIAS
                pred_BIAS_signals = self.D_AECG2BIAS(AECG_signals)
                loss_D_real_forwardGAN_AECG2BIAS = self.loss_forwardGAN(pred_BIAS_signals,valid)
                pred_fake_BIAS_signals = self.D_AECG2BIAS(fake_AECG_signals_from_BIAS)
                loss_D_fake_forwardGAN_AECG2BIAS = self.loss_forwardGAN(pred_fake_BIAS_signals, fake)               
                loss_D_forwardGAN_AECG2BIAS= (loss_D_real_forwardGAN_AECG2BIAS +  loss_D_fake_forwardGAN_AECG2BIAS)*0.5
                loss_D_forwardGAN_AECG2BIAS.backward(retain_graph=True)
                
                pred_AECG_signals = self.D_BIAS2AECG(BIAS_signals)
                loss_D_real_forwardGAN_BIAS_AECG = self.loss_forwardGAN(pred_AECG_signals,valid)
                pred_fake_AECG_signals_from_BIAS = self.D_BIAS2AECG(fake_BIAS_signals)
                loss_D_fake_forwardGAN_BIAS2AECG = self.loss_forwardGAN(pred_fake_AECG_signals_from_BIAS, fake)               
                loss_D_forwardGAN_BIAS2AECG= (loss_D_real_forwardGAN_BIAS_AECG +  loss_D_fake_forwardGAN_BIAS2AECG)*0.5

                
                loss_D_AECG2BIAS = (loss_D_forwardGAN_AECG2BIAS + loss_D_forwardGAN_BIAS2AECG)*0.5
                loss_D_AECG2BIAS.backward(retain_graph=True)
                
                self.clip_gradients(max_norm=5.0)
                
                self.optimizer_G_step()  
                self.optimizer_D_step() 
                
            self.optimizer_G_lr_step()
            self.optimizer_D_lr_step()
                


            tbar.close()
            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], D_FECG_loss: {:.6f}, ".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                              self.total_step , 
                              loss_generator_FECG.item() ), flush=True)

                # print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], G_FECG_loss: {:.6f}, G_FECG_lr: {:.6f}, D_FECG_lr: {:.6f}, "
                #       " G_AECG2FECG_ccar03_ave_gamma: {:.4f}, G_AECG2FECG_ccar02_ave_gamma: {:.4f}".
                #       format(elapsed, step + 1, self.total_step, (step + 1),
                #               self.total_step , 
                #               loss_generator_FECG.item(),self.G_AECG2FECG_exp_lr_scheduler.get_last_lr()[0], self.D_AECG2FECG_exp_lr_scheduler.get_last_lr()[0],
                #               self.G_AECG2FECG.module.ccar03.gamma.mean().item(), self.G_AECG2FECG.module.ccar02.gamma.mean().item() ))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_FECG_signals = self.G_AECG2FECG(AECG_signals)
                self.sample_images(epoch=step, batch_i=step, MECG=denorm(AECG_signals.cpu().detach().numpy()), FECG_reconstr=denorm(fake_FECG_signals.cpu().detach().numpy()),FECG=denorm(FECG_signals.cpu().detach().numpy()), sample_path=self.sample_path)


        
            archived = self.loss_c > loss_generator_FECG.item()
            if archived:
                self.loss_c = loss_generator_FECG.item()
                self.save_archive(step + 1)

            now = time.time()
            resume_saved = resume_save_interval > 0 and (now - last_resume_save) >= resume_save_interval
            if resume_saved:
                self.save_resume_state(step)
                last_resume_save = now
                last_saved_step = step

            self._append_epoch_log(step, loss_generator_FECG.item(),
                                   str(datetime.timedelta(seconds=time.time() - start_time)),
                                   archived, resume_saved)

            if self._stop_requested:
                if last_saved_step != step:
                    self.save_resume_state(step)
                print(f'Stopped after epoch {step}; relaunch with --resume True to continue',
                      flush=True)
                return

        if last_step >= start and last_saved_step != last_step:
            self.save_resume_state(last_step)

    def _install_stop_handler(self):
        """Turn SIGINT/SIGTERM into a graceful stop at the next epoch boundary.

        A shared lab machine means someone else can kill this run at any time; on
        the first signal we finish the epoch and write resume state. A second
        signal restores the default handler so an impatient kill still works.
        """
        def handler(signum, frame):
            if self._stop_requested:
                signal.signal(signum, signal.SIG_DFL)
                return
            self._stop_requested = True
            print(f'\nSignal {signum} received; finishing this epoch, then saving resume state.',
                  flush=True)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def _append_epoch_log(self, step, loss_value, elapsed, archived, resume_saved):
        """One tail-able line per epoch, so progress is readable after reconnecting."""
        saved = ','.join(tag for tag, wrote in (('archive', archived), ('resume', resume_saved)) if wrote)
        line = (f'{datetime.datetime.now().isoformat(timespec="seconds")}\t'
                f'epoch={step + 1}/{self.total_step}\t'
                f'FECG_loss={loss_value:.6f}\tbest={self.loss_c:.6f}\t'
                f'g_lr={self.G_AECG2FECG_exp_lr_scheduler.get_last_lr()[0]:.6g}\t'
                f'saved={saved or "none"}\telapsed={elapsed}\n')
        try:
            with open(self.epoch_log_path, 'a') as handle:
                handle.write(line)
        except OSError as exc:
            print(f'Could not append to {self.epoch_log_path}: {exc}', flush=True)

    def _module_map(self):
        return {suffix: getattr(self, suffix) for suffix in ALL_MODULE_SUFFIXES}

    def save_archive(self, step):
        """Write the step-numbered archive that the evaluation sweeps read.

        Generators only by default: nothing downstream loads a discriminator, and
        the discriminators are 16MB each against 299KB for a generator. The full
        set including discriminators lives in resume.pth.
        """
        suffixes = [s for s in GENERATOR_SUFFIXES if s != CHECKPOINT_MARKER]
        if self.archive_discriminators:
            suffixes.extend(DISCRIMINATOR_SUFFIXES)
        suffixes.append(CHECKPOINT_MARKER)

        modules = self._module_map()
        for suffix in suffixes:
            atomic_torch_save(
                unwrap_module(modules[suffix]).state_dict(),
                os.path.join(self.model_save_path, f'{step}_{suffix}.pth'),
            )

    def save_resume_state(self, step):
        """Overwrite resume.pth with everything needed to continue training."""
        payload = {
            'epoch_completed': step,
            'loss_c': self.loss_c,
            'models': {
                suffix: unwrap_module(module).state_dict()
                for suffix, module in self._module_map().items()
            },
            'optimizers': {
                suffix: getattr(self, f'{suffix}_optimizer').state_dict()
                for suffix in ALL_MODULE_SUFFIXES
            },
            'schedulers': {
                suffix: getattr(self, f'{suffix}_exp_lr_scheduler').state_dict()
                for suffix in ALL_MODULE_SUFFIXES
            },
            'rng': capture_rng_state(),
        }
        try:
            atomic_torch_save(payload, self.resume_state_path, keep_previous=True)
        except OSError as exc:
            print(f'Could not write {self.resume_state_path}: {exc}', flush=True)
            return
        print(f'Saved resume state after epoch {step} -> {self.resume_state_path}', flush=True)

    def load_resume_state(self):
        """Restore full training state, preferring resume.pth over its predecessor."""
        for path in self._resume_state_candidates():
            if not os.path.isfile(path):
                continue
            try:
                # map_location='cpu' keeps the saved RNG tensors on CPU where they
                # belong; load_state_dict moves weights and Adam moments onto the
                # live params.
                payload = torch_load(path, map_location='cpu')
                self._apply_resume_state(payload)
            except Exception as exc:
                print(f'Could not resume from {path} ({exc})', flush=True)
                continue
            print(f'Resumed from {path}: starting at epoch {self.start_step}, '
                  f'best FECG loss so far {self.loss_c:.6f}', flush=True)
            return True
        return False

    def _apply_resume_state(self, payload):
        modules = self._module_map()
        for suffix, state in payload['models'].items():
            if suffix in modules:
                unwrap_module(modules[suffix]).load_state_dict(state)

        for suffix, state in payload.get('optimizers', {}).items():
            optimizer = getattr(self, f'{suffix}_optimizer', None)
            if optimizer is not None:
                optimizer.load_state_dict(state)

        for suffix, state in payload.get('schedulers', {}).items():
            scheduler = getattr(self, f'{suffix}_exp_lr_scheduler', None)
            if scheduler is None:
                continue
            try:
                scheduler.load_state_dict(state)
            except Exception as exc:
                print(f'Could not restore {suffix} scheduler ({exc}); '
                      f'falling back to its epoch counter', flush=True)
                scheduler.last_epoch = state.get('last_epoch', 0)

        restore_rng_state(payload.get('rng'))

        self.loss_c = payload.get('loss_c', float('inf'))
        self.start_step = payload['epoch_completed'] + 1

    def build_model(self):
        
        #first, create generator 
        
        #AECG to MECG
        self.G_AECG2MECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.G_MECG2AECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)     
        self.D_AECG2MECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        self.D_MECG2AECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        
        if torch.cuda.device_count() > 1:
                self.G_AECG2MECG = nn.DataParallel(self.G_AECG2MECG)
                self.G_MECG2AECG = nn.DataParallel(self.G_MECG2AECG)
                self.D_AECG2MECG = nn.DataParallel(self.D_AECG2MECG)
                self.D_MECG2AECG = nn.DataParallel(self.D_MECG2AECG)

        
        #AECG to FECG
        self.G_AECG2FECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.G_FECG2AECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.D_AECG2FECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        self.D_FECG2AECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        
        if torch.cuda.device_count() > 1:
                self.G_AECG2FECG = nn.DataParallel(self.G_AECG2FECG)
                self.G_FECG2AECG = nn.DataParallel(self.G_FECG2AECG)
                self.D_AECG2FECG = nn.DataParallel(self.D_AECG2FECG)
                self.D_FECG2AECG = nn.DataParallel(self.D_FECG2AECG)

        #AECG to BIAS
        self.G_AECG2BIAS = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.G_BIAS2AECG = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.D_AECG2BIAS = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        self.D_BIAS2AECG = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(device)
        
        if torch.cuda.device_count() > 1:
                self.G_AECG2BIAS = nn.DataParallel(self.G_AECG2BIAS)
                self.G_BIAS2AECG = nn.DataParallel(self.G_BIAS2AECG)
                self.D_AECG2BIAS = nn.DataParallel(self.D_AECG2BIAS)
                self.D_BIAS2AECG = nn.DataParallel(self.D_BIAS2AECG)

        #second, initialize weights 
        # self.G_AECG2MECG.apply(weights_init_normal)
        # self.G_MECG2AECG.apply(weights_init_normal)    
        # self.D_AECG2MECG.apply(weights_init_normal)
        # self.D_MECG2AECG.apply(weights_init_normal)

        # self.G_AECG2FECG.apply(weights_init_normal)
        # self.G_FECG2AECG.apply(weights_init_normal)
        # self.D_AECG2FECG.apply(weights_init_normal)
        # self.D_FECG2AECG.apply(weights_init_normal)

        # self.G_AECG2BIAS.apply(weights_init_normal)
        # self.G_BIAS2AECG.apply(weights_init_normal)
        # self.D_AECG2BIAS.apply(weights_init_normal)
        # self.D_BIAS2AECG.apply(weights_init_normal)
        
        #third, loss definition
        # self.loss_generator = torch.nn.L1Loss()
        # self.loss_forwardGAN = torch.nn.MSELoss() 
        # self.loss_cycleGAN = torch.nn.L1Loss()
        
        self.loss_generator = logcosh()
        self.loss_forwardGAN = torch.nn.L1Loss() 
        self.loss_cycleGAN = logcosh()
        
        
        # self.loss_discriminator = torch.nn.L1Loss()

        #fourth, optimizer definition
        self.G_AECG2MECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G_AECG2MECG.parameters()), self.g_AECG_lr, [self.beta1, self.beta2])        
        self.G_MECG2AECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G_MECG2AECG.parameters()), self.g_MECG_lr, [self.beta1, self.beta2])
        self.D_AECG2MECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_AECG2MECG.parameters()), self.d_AECG_lr, [self.beta1, self.beta2])
        self.D_MECG2AECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_MECG2AECG.parameters()), self.d_MECG_lr, [self.beta1, self.beta2])
  
        self.G_AECG2FECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G_AECG2FECG.parameters()), self.g_AECG_lr, [self.beta1, self.beta2])        
        self.G_FECG2AECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G_FECG2AECG.parameters()), self.g_FECG_lr, [self.beta1, self.beta2])
        self.D_AECG2FECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_AECG2FECG.parameters()), self.d_AECG_lr, [self.beta1, self.beta2])
        self.D_FECG2AECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_FECG2AECG.parameters()), self.d_FECG_lr, [self.beta1, self.beta2])
    
        self.G_AECG2BIAS_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G_AECG2BIAS.parameters()), self.g_AECG_lr, [self.beta1, self.beta2])        
        self.G_BIAS2AECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G_BIAS2AECG.parameters()), self.g_BIAS_lr, [self.beta1, self.beta2])
        self.D_AECG2BIAS_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_AECG2BIAS.parameters()), self.d_AECG_lr, [self.beta1, self.beta2])
        self.D_BIAS2AECG_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_BIAS2AECG.parameters()), self.d_BIAS_lr, [self.beta1, self.beta2])

        # print(self.decay_start_epoch)
        #fifth, lr_scheduler definition
 
        self.G_AECG2MECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_AECG2MECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.G_MECG2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_MECG2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_AECG2MECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_AECG2MECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_MECG2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_MECG2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        
        self.G_AECG2FECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_AECG2FECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.G_FECG2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_FECG2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_AECG2FECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_AECG2FECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_FECG2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_FECG2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        
        self.G_AECG2BIAS_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_AECG2BIAS_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.G_BIAS2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_BIAS2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_AECG2BIAS_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_AECG2BIAS_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        self.D_BIAS2AECG_exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_BIAS2AECG_optimizer, lr_lambda=LambdaLR(self.total_step, 0, self.decay_start_epoch).step)
        
    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def _load_module_checkpoint(self, module, suffix, required=True):
        path = os.path.join(self.model_save_path, f'{self.pretrained_model}_{suffix}.pth')
        if not os.path.isfile(path):
            if required:
                raise FileNotFoundError(f'Checkpoint file not found: {path}')
            return False
        state_dict = torch_load(path, map_location='cpu')
        unwrap_module(module).load_state_dict(state_dict)
        return True

    def load_pretrained_model(self, suffixes=None):
        modules = self._module_map()
        missing = []
        for suffix in (suffixes or ALL_MODULE_SUFFIXES):
            # Discriminators are optional: the archive stopped carrying them once
            # they moved into resume.pth, and training can re-learn them.
            required = suffix not in DISCRIMINATOR_SUFFIXES
            if not self._load_module_checkpoint(modules[suffix], suffix, required=required):
                missing.append(suffix)
        if missing:
            print(f'No discriminator weights at step {self.pretrained_model} '
                  f'({", ".join(missing)}); those start from fresh initialisation')
        print(f'Loaded checkpoint weights from step {self.pretrained_model} '
              f'({self.model_save_path}); continuing at step {self.pretrained_model + 1}')

    def optimizer_G_zero_grad(self):
        self.G_AECG2MECG_optimizer.zero_grad()
        self.G_MECG2AECG_optimizer.zero_grad()
        self.G_AECG2FECG_optimizer.zero_grad()
        self.G_FECG2AECG_optimizer.zero_grad()
        self.G_AECG2BIAS_optimizer.zero_grad()
        self.G_BIAS2AECG_optimizer.zero_grad()
        
    def optimizer_D_zero_grad(self):    
        self.D_AECG2MECG_optimizer.zero_grad()
        self.D_MECG2AECG_optimizer.zero_grad()
        self.D_AECG2FECG_optimizer.zero_grad()
        self.D_FECG2AECG_optimizer.zero_grad()
        self.D_AECG2BIAS_optimizer.zero_grad()
        self.D_BIAS2AECG_optimizer.zero_grad()
        
    def optimizer_G_step(self):
        self.G_AECG2MECG_optimizer.step()
        self.G_MECG2AECG_optimizer.step()
        self.G_AECG2FECG_optimizer.step()
        self.G_FECG2AECG_optimizer.step()
        self.G_AECG2BIAS_optimizer.step()
        self.G_BIAS2AECG_optimizer.step()
        
    def optimizer_D_step(self):    
        self.D_AECG2MECG_optimizer.step()
        self.D_MECG2AECG_optimizer.step()
        self.D_AECG2FECG_optimizer.step()
        self.D_FECG2AECG_optimizer.step()
        self.D_AECG2BIAS_optimizer.step()
        self.D_BIAS2AECG_optimizer.step()

    def clip_gradients(self, max_norm=5.0):
        modules = [
            self.G_AECG2MECG, self.G_MECG2AECG,
            self.G_AECG2FECG, self.G_FECG2AECG,
            self.G_AECG2BIAS, self.G_BIAS2AECG,
            self.D_AECG2MECG, self.D_MECG2AECG,
            self.D_AECG2FECG, self.D_FECG2AECG,
            self.D_AECG2BIAS, self.D_BIAS2AECG,
        ]
        for module in modules:
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)
        
        
        

    def optimizer_G_lr_step(self):     
        self.G_AECG2MECG_exp_lr_scheduler.step()
        self.G_MECG2AECG_exp_lr_scheduler.step()
        self.G_AECG2FECG_exp_lr_scheduler.step()
        self.G_FECG2AECG_exp_lr_scheduler.step()
        self.G_AECG2BIAS_exp_lr_scheduler.step()
        self.G_BIAS2AECG_exp_lr_scheduler.step()

    def optimizer_D_lr_step(self):
        self.D_AECG2MECG_exp_lr_scheduler.step()
        self.D_MECG2AECG_exp_lr_scheduler.step() 
        self.D_AECG2FECG_exp_lr_scheduler.step()
        self.D_FECG2AECG_exp_lr_scheduler.step()
        self.D_AECG2BIAS_exp_lr_scheduler.step()
        self.D_BIAS2AECG_exp_lr_scheduler.step()


      
        
    def model_train(self):
        self.G_AECG2MECG.train()
        self.G_MECG2AECG.train()    
        self.D_AECG2MECG.train()
        self.D_MECG2AECG.train()
        
        self.G_AECG2FECG.train()
        self.G_FECG2AECG.train()
        self.D_AECG2FECG.train()
        self.D_FECG2AECG.train()
    
        self.G_AECG2BIAS.train()
        self.G_BIAS2AECG.train()
        self.D_AECG2BIAS.train()
        self.D_BIAS2AECG.train()
        

            

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
        
        
    def sample_images(self, epoch, batch_i, MECG, FECG_reconstr, FECG, sample_path):
        r, c = 1, 3
        gen_imgs = [MECG, FECG_reconstr,FECG]
        titles = ['MECG', 'FECG_reconstr','FECG']
        

        fig, axs = plt.subplots(r, c,figsize=(15, 5))
        cnt = 0
        for i in range(r):
            for j in range(c):
                for bias in range(1):
                    tt = gen_imgs[cnt][bias,:]
                    axs[j].plot(tt[0])
                axs[j].set_title(titles[j])
                cnt += 1
        fig.savefig("%s/%d_%d.png" % (sample_path, epoch,batch_i),dpi=500,bbox_inches = 'tight')
        plt.close()
        
        
        
        
        
  