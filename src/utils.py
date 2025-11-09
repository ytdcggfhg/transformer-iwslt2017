"""工具函数模块"""
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(device_str=None):
    if device_str is not None:
        device = torch.device(device_str)
        if 'cuda' in device_str and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    return device


def get_ablation_info(args):
    ablation_info = []
    if args.no_residual:
        ablation_info.append("no_residual")
    if args.no_positional_encoding:
        ablation_info.append("no_positional_encoding")
    if args.single_head:
        ablation_info.append("single_head")
    ablation_str = f" [Ablation: {', '.join(ablation_info)}]" if ablation_info else ""
    return ablation_str


def create_experiment_dir(results_dir, args):
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr_str = str(args.lr).replace('.', '_')
    dr_str = str(args.dropout).replace('.', '_')
    param_str = f"ep{args.epochs}_bs{args.batch_size}_dm{args.d_model}_df{args.d_ff}_dk{args.d_k}_nl{args.n_layers}_nh{args.n_heads}_lr{lr_str}_dr{dr_str}"
    
    ablation_suffix = ""
    if args.no_residual:
        ablation_suffix += "_noRes"
    if args.no_positional_encoding:
        ablation_suffix += "_noPos"
    if args.single_head:
        ablation_suffix += "_singleHead"
    
    exp_name = f"{timestamp}_{param_str}{ablation_suffix}"
    exp_dir = os.path.join(results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Results will be saved to: {exp_dir}")
    return exp_dir


def plot_training_curves(train_losses, train_ppls, valid_losses, valid_ppls, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, train_ppls, 'b-', label='Train PPL', linewidth=2)
    axes[1].plot(epochs, valid_ppls, 'r-', label='Validation PPL', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Perplexity (PPL)', fontsize=12)
    axes[1].set_title('Training and Validation Perplexity', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def save_metrics_to_txt(train_losses, train_ppls, valid_losses, valid_ppls, save_path, args, 
                        best_epoch, best_valid_ppl, device, verbose=False, 
                        early_stopped=False, stopped_epoch=None):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Transformer Training Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Hyperparameters:\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  d_model: {args.d_model}\n")
        f.write(f"  d_ff: {args.d_ff}\n")
        f.write(f"  d_k: {args.d_k}\n")
        f.write(f"  n_layers: {args.n_layers}\n")
        f.write(f"  n_heads: {args.n_heads}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Train Samples: {args.train_samples}\n")
        f.write(f"  Random Seed: {args.seed}\n")
        f.write(f"  Early Stop: {'Enabled' if args.early_stop else 'Disabled'}\n")
        f.write(f"  Ablation Studies:\n")
        f.write(f"    No Residual: {'Yes' if args.no_residual else 'No'}\n")
        f.write(f"    No Positional Encoding: {'Yes' if args.no_positional_encoding else 'No'}\n")
        f.write(f"    Single Head: {'Yes' if args.single_head else 'No'}\n")
        if args.early_stop:
            f.write(f"  Patience: {args.patience}\n")
            f.write(f"  Min Delta: {args.min_delta}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        if best_epoch > 0:
            f.write(f"Best Model: Epoch {best_epoch}, Validation PPL: {best_valid_ppl:.4f}\n")
            if early_stopped and stopped_epoch is not None:
                f.write(f"Training stopped early at epoch {stopped_epoch} (no improvement for {args.patience} epochs)\n")
            f.write("\n")
        else:
            f.write("Best Model: (Training in progress...)\n\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Epoch-by-Epoch Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<15} {'Train PPL':<15} {'Valid Loss':<15} {'Valid PPL':<15}\n")
        f.write("-" * 80 + "\n")
        
        for epoch in range(len(train_losses)):
            f.write(f"{epoch+1:<8} {train_losses[epoch]:<15.4f} {train_ppls[epoch]:<15.4f} "
                   f"{valid_losses[epoch]:<15.4f} {valid_ppls[epoch]:<15.4f}\n")
        
        f.write("-" * 80 + "\n")
    
    if verbose:
        print(f"Metrics saved to: {save_path}")


def initialize_metrics_file(metrics_path, args, device):
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Transformer Training Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Hyperparameters:\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  d_model: {args.d_model}\n")
        f.write(f"  d_ff: {args.d_ff}\n")
        f.write(f"  d_k: {args.d_k}\n")
        f.write(f"  n_layers: {args.n_layers}\n")
        f.write(f"  n_heads: {args.n_heads}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Train Samples: {args.train_samples}\n")
        f.write(f"  Random Seed: {args.seed}\n")
        f.write(f"  Ablation Studies:\n")
        f.write(f"    No Residual: {'Yes' if args.no_residual else 'No'}\n")
        f.write(f"    No Positional Encoding: {'Yes' if args.no_positional_encoding else 'No'}\n")
        f.write(f"    Single Head: {'Yes' if args.single_head else 'No'}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        f.write("Best Model: (Training in progress...)\n\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Epoch-by-Epoch Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<15} {'Train PPL':<15} {'Valid Loss':<15} {'Valid PPL':<15}\n")
        f.write("-" * 80 + "\n")
    
    print(f"Metrics file initialized at: {metrics_path}")

