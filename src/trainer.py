
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils import save_metrics_to_txt, plot_training_curves


def evaluate(model, data_loader, criterion, device):
    model.eval()
    epoch_total_loss = 0

    with torch.no_grad():
        for enc_inputs, dec_inputs, dec_outputs in data_loader:
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            epoch_total_loss += loss.item()
    
    avg_loss = epoch_total_loss / len(data_loader)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.best_valid_ppl = float('inf')
        self.early_stopped = False
        self.stopped_epoch = None
        self.best_epoch = 0
    
    def check(self, valid_ppl, epoch):
        improved = False
        if valid_ppl < (self.best_valid_ppl - self.min_delta):
            self.best_valid_ppl = valid_ppl
            self.best_epoch = epoch
            self.patience_counter = 0
            improved = True
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stopped = True
                self.stopped_epoch = epoch
                print(f"\n{'='*50}")
                print(f"Early stopping triggered!")
                print(f"No improvement for {self.patience} consecutive epochs.")
                print(f"Best model was at epoch {self.best_epoch} with validation PPL: {self.best_valid_ppl:.4f}")
                print(f"{'='*50}\n")
        
        return improved


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    train_total_loss = 0

    for i, (enc_inputs, dec_inputs, dec_outputs) in enumerate(loader):
        try:
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            train_total_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f'Batch: {i + 1:04d}/{len(loader)}, loss = {loss.item():.6f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f'| WARNING: CUDA out of memory on batch {i+1}. Skipping batch.')
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            else:
                raise e
    
    avg_loss = train_total_loss / len(loader)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def save_checkpoint(model, optimizer, epoch, valid_ppl, args, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_ppl': valid_ppl,
        'args': vars(args)
    }, save_path)


def train(model, train_loader, valid_loader, criterion, optimizer, args, device, exp_dir):
    print("开始训练...")
    
    train_losses = []
    train_ppls = []
    valid_losses = []
    valid_ppls = []
    
    early_stopping = None
    if args.early_stop:
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        print(f"Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")
    
    best_valid_ppl = float('inf')
    best_epoch = 0
    metrics_path = os.path.join(exp_dir, 'metrics.txt')
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        avg_train_loss, train_ppl = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_ppl = evaluate(model, valid_loader, criterion, device)
        
        train_losses.append(avg_train_loss)
        train_ppls.append(train_ppl)
        valid_losses.append(valid_loss)
        valid_ppls.append(valid_ppl)
        
        improved = False
        if valid_ppl < (best_valid_ppl - args.min_delta):
            best_valid_ppl = valid_ppl
            best_epoch = epoch + 1
            best_model_path = os.path.join(exp_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch + 1, valid_ppl, args, best_model_path)
            print(f'*** New best model saved! (Epoch {best_epoch}, Valid PPL: {best_valid_ppl:.4f}) ***')
            improved = True
        
        if early_stopping is not None:
            early_stopping.check(valid_ppl, epoch + 1)
            if early_stopping.early_stopped:
                best_epoch = early_stopping.best_epoch
                best_valid_ppl = early_stopping.best_valid_ppl
        
        save_metrics_to_txt(
            train_losses, train_ppls, valid_losses, valid_ppls, 
            metrics_path, args, best_epoch, best_valid_ppl, device, 
            verbose=False, 
            early_stopped=early_stopping.early_stopped if early_stopping else False,
            stopped_epoch=early_stopping.stopped_epoch if early_stopping else None
        )
        
        epoch_time = time.time() - epoch_start_time
        print("-" * 50)
        print(f'Epoch: {epoch + 1:04d} 结束 | 耗时: {epoch_time:.2f}s')
        print(f'\t(Train)  Loss: {avg_train_loss:.4f} | PPL: {train_ppl:7.4f}')
        print(f'\t(Valid)  Loss: {valid_loss:.4f} | PPL: {valid_ppl:7.4f}')
        if early_stopping is not None:
            print(f'\tEarly Stop: {early_stopping.patience_counter}/{early_stopping.patience} (no improvement)')
        print("-" * 50)
        
        if early_stopping is not None and early_stopping.early_stopped:
            break
    
    print("\n" + "=" * 50)
    if early_stopping is not None and early_stopping.early_stopped:
        print("Training stopped early! Saving results...")
    else:
        print("Training completed! Saving results...")
    print("=" * 50)
    
    curve_path = os.path.join(exp_dir, 'training_curves.png')
    plot_training_curves(train_losses, train_ppls, valid_losses, valid_ppls, curve_path)
    
    save_metrics_to_txt(
        train_losses, train_ppls, valid_losses, valid_ppls, 
        metrics_path, args, best_epoch, best_valid_ppl, device, 
        verbose=True,
        early_stopped=early_stopping.early_stopped if early_stopping else False,
        stopped_epoch=early_stopping.stopped_epoch if early_stopping else None
    )
    
    print(f"\nAll results saved to: {exp_dir}")
    print(f"  - Best model: best_model.pth (Epoch {best_epoch}, Valid PPL: {best_valid_ppl:.4f})")
    if early_stopping is not None and early_stopping.early_stopped:
        print(f"  - Training stopped early at epoch {early_stopping.stopped_epoch}")
    print(f"  - Training curves: training_curves.png")
    print(f"  - Metrics: metrics.txt (updated in real-time during training)")
    print("=" * 50)
    
    return train_losses, train_ppls, valid_losses, valid_ppls, best_epoch, best_valid_ppl

