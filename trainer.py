import wandb
import torch
import os
import time
from tqdm import tqdm

import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from evaluation import make_evaluation_on_L_p_grid, find_threshold, make_threshold_plot


class Trainer:
    def __init__(self, net, optim, criterion, use_amp):
        self.net = net
        self.optim = optim
        self.criterion = criterion
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def step(self, syndrome, logical, train=True):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            self.optim.zero_grad()
            logits_logical = self.net(syndrome)
            loss = self.criterion(logits_logical, logical)
            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            return loss, logits_logical


class LRFinder:
    def __init__(self, trainer, loader):
        self.trainer = trainer
        self.loader = loader
        self.history = {'lr': [], 'loss': []}

    def find_lr(self, batches_n, b_size, start_lr=1e-5, end_lr=1, step_mode='exp'):

        if step_mode == 'exp':
            func_scheduler = lambda e: start_lr * np.exp(e / batches_n * np.log(end_lr/start_lr))
        elif step_mode == 'linear':
            func_scheduler = lambda e: start_lr * e / batches_n * (end_lr/start_lr)
        else:
            raise NotImplementedError
        scheduler = LambdaLR(func_scheduler)

        for b in range(batches_n):
            logical_b, syndrome_b = self.loader.sample(b_size)
            loss, prediction_logical = self.trainer.step(syndrome_b, logical_b, train=True)



def start_train(args, loader, trainer, scheduler):
    batches_n = args.default.b_n
    epoch_n = args.default.e_n
    b_size = args.b_size
    test_size = args.default.test_size

    save_rate = args.default.save_rate
    name = args.name
    save_dir = wandb.run.dir

    torch.backends.cudnn.benchmark = True
    for e in range(epoch_n):
        running_loss = 0.
        epoch_start_time = time.time()
        for b in range(batches_n):
            logical_b, syndrome_b = loader.sample(b_size)
            loss, prediction_logical = trainer.step(syndrome_b, logical_b, train=True)

            running_loss += loss.item()

            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(running_loss)
        elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            pass
        else:
            scheduler.step()
        epoch_duration = time.time() - epoch_start_time
        with torch.no_grad():
            loader.batch_generator.noise_model.qmc_flag = False
            logical_b, syndrome_b = loader.sample(test_size)
            loss, logits_logical = trainer.step(syndrome_b, logical_b, train=False)

            metrics = trainer.criterion.get_metrics(logits=logits_logical,
                                                    labels=logical_b)
            metrics['loss'] = running_loss / batches_n
            metrics['lr'] = scheduler.optimizer.param_groups[0]['lr']
            metrics['epoch_time'] = epoch_duration
            wandb.log(metrics)

            if (e % save_rate == 0) or (e == (epoch_n - 1)):
                save_name = name + '_' + str(e) + '.pth'
                save_path = os.path.join(save_dir, save_name)

                torch.save({
                    'epoch': e,
                    'model_state_dict': trainer.net.state_dict(),
                    'optimizer_state_dict': trainer.optim.state_dict(),
                    'loss': running_loss,
                    'metrics': metrics}, save_path)

    if args.evaluation.current:
        print('Evaluation on 1e6 samples starts...')
        with torch.no_grad():
            trainer.net.eval()
            acc, std, acc_array, std_array = loader.make_evaluation(trainer.net)
        metrics['final_acc'] = acc
    else:
        acc, std, acc_array, std_array = 0, 0, 0, 0
        wandb.log(metrics)
        save_path = os.path.join(save_dir, 'after_training.pth')
        torch.save({
            'epoch': -1,
            'model_state_dict': trainer.net.state_dict(),
            'metrics': metrics,
            'final_acc_array': acc_array,
            'final_std_array': std_array
        }, save_path)

    if args.evaluation.grid:
        print('Evaluation on grid 1e5 samples starts...')
        with torch.no_grad():
            L_grid, p_grid = args.evaluation.L_grid, args.evaluation.p_grid
            grid_results = make_evaluation_on_L_p_grid(loader, trainer.net.eval(), L_grid, p_grid)
        metrics['acc_grid'] = grid_results

        popt, pcov, xdata = find_threshold(grid_results[0], np.array(p_grid), np.array(L_grid))
        save_fig = os.path.join(save_dir, 'threshold_fig.png')
        make_threshold_plot(L_grid, p_grid, popt, grid_results, ax=None, save_path=save_fig)
        metrics['p_threshold'] = popt


    wandb.log(metrics)
    save_path = os.path.join(save_dir, 'final.pth')
    torch.save({
        'epoch': -1,
        'model_state_dict': trainer.net.state_dict(),
        'metrics': metrics,
        'final_acc_array': acc_array,
        'final_std_array': std_array
    }, save_path)

    return 0





