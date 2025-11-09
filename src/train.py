"""主训练脚本"""
import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import parse_args
from utils import set_seed, setup_device, get_ablation_info, create_experiment_dir, initialize_metrics_file
from data import (
    load_tokenizers, load_dataset_iwslt, build_vocab, 
    create_collate_fn, create_dataloaders, UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX
)
from model import Transformer
from trainer import train
from inference import test_translation


def main():
    args = parse_args()
    
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    device = setup_device(args.device)
    
    ablation_str = get_ablation_info(args)
    print(f"Hyperparameters: epochs={args.epochs}, batch_size={args.batch_size}, d_model={args.d_model}, "
          f"d_ff={args.d_ff}, d_k={args.d_k}, n_layers={args.n_layers}, n_heads={args.n_heads}, "
          f"lr={args.lr}, dropout={args.dropout}{ablation_str}")
    
    spacy_en, spacy_de = load_tokenizers()
    train_data, valid_data, test_data = load_dataset_iwslt(args.train_samples)
    vocab_src, vocab_tgt = build_vocab(train_data, spacy_en, spacy_de)
    collate_fn = create_collate_fn(vocab_src, vocab_tgt, spacy_en, spacy_de, device)
    loader, valid_loader = create_dataloaders(train_data, valid_data, args.batch_size, collate_fn)
    
    src_vocab_size = len(vocab_src)
    tgt_vocab_size = len(vocab_tgt)
    d_model = args.d_model
    d_ff = args.d_ff
    d_k = d_v = args.d_k
    n_layers = args.n_layers
    n_heads = args.n_heads
    
    print("正在初始化模型...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_k=d_k,
        d_v=d_v,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=args.dropout,
        device=device,
        no_residual=args.no_residual,
        no_positional_encoding=args.no_positional_encoding
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    
    results_dir = "results"
    exp_dir = create_experiment_dir(results_dir, args)
    metrics_path = os.path.join(exp_dir, 'metrics.txt')
    initialize_metrics_file(metrics_path, args, device)
    
    train_losses, train_ppls, valid_losses, valid_ppls, best_epoch, best_valid_ppl = train(
        model, loader, valid_loader, criterion, optimizer, args, device, exp_dir
    )
    
    test_translation(model, test_data, vocab_src, vocab_tgt, spacy_en, device)


if __name__ == '__main__':
    main()
