
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Transformer for IWSLT2017 English-German Translation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--d_model', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--d_ff', type=int, default=2048, help='FeedForward dimension')
    parser.add_argument('--d_k', type=int, default=64, help='Dimension of K(=Q), V')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of Encoder/Decoder layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads in Multi-Head Attention')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--train_samples', type=int, default=40000, help='Number of training samples')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., "cuda", "cuda:0", "cuda:1", "cpu"). If not specified, auto-detect.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs with no improvement after which training will be stopped (only used if --early_stop is enabled)')
    parser.add_argument('--min_delta', type=float, default=0.0, help='Minimum change in the monitored quantity to qualify as an improvement (only used if --early_stop is enabled)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_residual', action='store_true', help='Disable residual connections (ablation study)')
    parser.add_argument('--no_positional_encoding', action='store_true', help='Disable positional encoding (ablation study)')
    parser.add_argument('--single_head', action='store_true', help='Use single-head attention instead of multi-head (ablation study)')
    
    args = parser.parse_args()
    
    if args.single_head:
        args.n_heads = 1
        print("Single-head attention enabled: n_heads set to 1")
    
    return args

