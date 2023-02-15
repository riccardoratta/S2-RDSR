import torch
import torch.nn as nn

from typing import List, Callable, Union

from torch.utils.data import DataLoader

from torchmetrics.metric import Metric

from torch.utils.tensorboard.writer import SummaryWriter

def _cuda(dataLoader):
    for batch in dataLoader:
        batch[0][0] = batch[0][0].cuda()
        batch[0][1] = batch[0][1].cuda()
        batch[1] = batch[1].cuda()
        yield batch

def model_step(
    model: nn.Module,
    dataLoader: DataLoader,
    loss_fn: Callable,
    optimizer,
    scheduler=None,
    gradient_clipping=None):
    
    model.train()

    m = len(dataLoader)
        
    n = 0; s_loss = 0
    for (x1, x2), targets in _cuda(dataLoader):

        print(f'\rBatch: {n:03d}/{m:03d}', end='')

        outputs = model(x1, x2)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        if gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        n += 1
        s_loss += loss.item()
                
    if scheduler:
        scheduler.step()

    print('\r')
            
    return s_loss / n

def model_eval(
    model: nn.Module,
    dataLoader: DataLoader,
    eval_fns: Union[Callable, List[Callable]]):
    
    model.eval()
    
    if not isinstance(eval_fns, list):
        eval_fns = [eval_fns]
        
    for i, eval_fn in enumerate(eval_fns):
        if isinstance(eval_fn, Metric):
            eval_fns[i] = eval_fn.cuda()
    
    n = 0; stats = [None] * len(eval_fns)
    with torch.no_grad():
        for (x1, x2), targets in _cuda(dataLoader):
        
            outputs = model(x1, x2)
            
            for i, eval_fn in enumerate(eval_fns):
                stat = eval_fn(outputs, targets)
                if stats[i] is not None:
                    stats[i] += stat
                else:
                    stats[i] = stat.detach().clone()

            n += 1
                                    
    for i, _ in enumerate(stats):
        stats[i] /= n 
    
    for i, _ in enumerate(stats):
        stats[i] = stats[i].cpu()
    
    if len(stats) == 1:
        return stats[0]
    return stats

def at_epoch(
    model,
    epochs,
    dataLoader,
    eval_dataLoader,
    loss_fn,
    eval_fn,
    optimizer,
    scheduler=None,
    gradient_clipping=None):
    '''
    Train a model. Epochs could be the number of epochs to train, or a tuple with `start` and `end`
    if the model has already been trained and you wanted to restart it.
    
    The method should be considered as in itererator and for every step it yield the epoch number
    (starts from 1), and a tuple with the step and eval loss (i.e., training loss and validation 
    loss).    
    '''
    
    if isinstance(epochs, int):
        epochs = [epochs]
            
    for epoch in range(*epochs):
        step_v = model_step(
            model,
            dataLoader,
            loss_fn,
            optimizer,
            scheduler,
            gradient_clipping)
        
        eval_v = model_eval(
            model,
            eval_dataLoader,
            eval_fn)
        
        yield epoch+1, (step_v, eval_v)
        
class Logger(SummaryWriter):
    '''
    Wrapper class for `SummaryWriter`.
    '''
    def __init__(self, name):
        super().__init__(f'runs/{name}')
        
    def add(self, epoch, step_v, eval_v):
        self.add_scalar('training/loss'  , step_v, epoch)
        self.add_scalar('validation/loss', eval_v, epoch)
