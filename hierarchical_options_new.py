import argparse
import os
import time
import torch


def get_hierarchical_options(args=None):
    parser = argparse.ArgumentParser(
        description="Improved Hierarchical Attention Model Training for MCLP problem")

    # ===== Data Configuration =====
    parser.add_argument('--problem', default='MCLP', choices=['PM', 'PC', 'MCLP'],
                        help="The problem to solve")
    parser.add_argument('--n_users', type=int, default=1312,
                        help='Number of users')
    parser.add_argument('--n_facilities', type=int, default=100,
                        help='Number of facilities')
    parser.add_argument('--p', type=int, default=30,
                        help='Number of facilities to select')
    parser.add_argument('--r', type=float, default=None,
                        help='Coverage radius')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--epoch_size', type=int, default=128000,
                        help='Number of instances per epoch')
    parser.add_argument('--val_size', type=int, default=2000,
                        help='Number of instances for validation')
    parser.add_argument('--val_dataset', type=str, default=None,
                        help='Dataset file for validation')
    parser.add_argument('--train_dataset', type=str, default=None,
                        help='Dataset file for training (optional)')
    parser.add_argument('--data_distribution', type=str, default='normal',
                        help='Data distribution type')

    # ===== Model Architecture =====
    parser.add_argument('--model', default='hierarchical_v2',
                        help="Model type: hierarchical or hierarchical_v2")
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of embeddings')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of hidden layers')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of encoder layers')
    parser.add_argument('--n_decode_layers', type=int, default=2,
                        help='Number of decoder layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Tanh clipping')
    parser.add_argument('--normalization', default='batch',
                        help='Normalization type: batch or layer')
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Use checkpoint for encoder to save memory')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink size for attention')

    # ===== Hierarchical Model Specific =====
    parser.add_argument('--max_facilities_per_step', type=int, default=5,
                        help='Maximum facilities to select per step')
    parser.add_argument('--option_loss_weight', type=float, default=0.5,
                        help='Weight for option (upper-level) loss')
    parser.add_argument('--use_kv_cache', action='store_true', default=True,
                        help='Use KV caching for faster decoding')
    parser.add_argument('--use_critic', action='store_true',
                        help='Use Actor-Critic architecture instead of REINFORCE')

    # ===== Training Configuration =====
    parser.add_argument('--lr_model', type=float, default=1e-3,
                        help='Learning rate for main model')
    parser.add_argument('--lr_option', type=float, default=5e-4,
                        help='Learning rate for option/strategy network')
    parser.add_argument('--lr_critic', type=float, default=1e-3,
                        help='Learning rate for critic network')
    parser.add_argument('--lr_decay', type=float, default=0.99,
                        help='Learning rate decay per epoch')
    parser.add_argument('--lr_schedule', type=str, default='exponential',
                        choices=['exponential', 'cosine', 'linear', 'constant'],
                        help='Learning rate schedule type')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate model without training')
    parser.add_argument('--n_epochs', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--eval_batch_size', type=int, default=1000,
                        help='Batch size for evaluation')
    parser.add_argument('--normalize_advantage', action='store_true', default=True,
                        help='Normalize advantages for stable training')

    # ===== Exploration and Temperature =====
    parser.add_argument('--temp_start', type=float, default=2.0,
                        help='Starting temperature for exploration')
    parser.add_argument('--temp_end', type=float, default=0.5,
                        help='Final temperature')
    parser.add_argument('--temp_decay', type=float, default=0.95,
                        help='Temperature decay per epoch')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy regularization coefficient')

    # ===== Baseline Configuration =====
    parser.add_argument('--baseline', default='rollout',
                        choices=['rollout', 'critic', 'exponential', 'actor_critic'],
                        help="Baseline type")
    parser.add_argument('--bl_alpha', type=float, default=0.3,
                        help='Significance threshold for updating rollout baseline (higher = more updates)')
    parser.add_argument('--bl_warmup_epochs', type=int, default=5,
                        help='Number of epochs to warmup baseline')
    parser.add_argument('--bl_update_interval', type=int, default=5,
                        help='Maximum epochs between baseline updates')
    parser.add_argument('--exp_beta', type=float, default=0.9,
                        help='Exponential moving average baseline decay')
    parser.add_argument('--option_exp_beta', type=float, default=0.9,
                        help='Option exponential baseline decay')
    parser.add_argument('--adaptive_baseline', action='store_true', default=True,
                        help='Use adaptive baseline with dynamic parameters')
    parser.add_argument('--force_baseline_update', type=int, default=10,
                        help='Force baseline update after this many epochs without improvement')

    # ===== Curriculum Learning =====
    parser.add_argument('--curriculum', action='store_true', default=True,
                        help='Use curriculum learning (start with easier problems)')
    parser.add_argument('--curriculum_rate', type=float, default=0.6,
                        help='Fraction of training to reach full difficulty')
    parser.add_argument('--curriculum_type', type=str, default='sigmoid',
                        choices=['linear', 'sigmoid', 'exponential'],
                        help='Type of curriculum progression')

    # ===== Actor-Critic Specific =====
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='Coefficient for value loss in Actor-Critic')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for rewards')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='Lambda for Generalized Advantage Estimation')

    # ===== PPO Configuration (Advanced) =====
    parser.add_argument('--use_ppo', action='store_true',
                        help='Use PPO instead of vanilla policy gradient')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                        help='Number of PPO update epochs')
    parser.add_argument('--ppo_clip_epsilon', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--ppo_batch_size', type=int, default=256,
                        help='Mini-batch size for PPO updates')

    # ===== Logging and Checkpointing =====
    parser.add_argument('--log_step', type=int, default=50,
                        help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs',
                        help='Directory to write TensorBoard information')
    parser.add_argument('--run_name', default='run',
                        help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs',
                        help='Directory to write output models')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (for resuming)')
    parser.add_argument('--checkpoint_epochs', type=int, default=5,
                        help='Save checkpoint every n epochs')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to load model parameters')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true',
                        help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true',
                        help='Disable progress bar')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    # ===== Advanced Options =====
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (requires apex)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='Pin memory for faster GPU transfer')

    opts = parser.parse_args(args)

    # ===== Post-processing Options =====

    # Set device
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Set default run name with more information
    if opts.run_name == 'run':
        model_suffix = "_v2" if "v2" in opts.model else ""
        baseline_suffix = f"_{opts.baseline[:3]}"  # First 3 chars of baseline
        opts.run_name = "hier{}{}_n{}_p{}_{}".format(
            model_suffix,
            baseline_suffix,
            opts.n_facilities,
            opts.p,
            time.strftime("%Y%m%dT%H%M%S")
        )

    # Create save directory
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.n_users),
        opts.run_name
    )

    # Validate and adjust parameters
    assert opts.epoch_size % opts.batch_size == 0, \
        "Epoch size must be integer multiple of batch size!"

    # Adjust warmup epochs based on baseline type
    if opts.baseline == 'rollout' and opts.bl_warmup_epochs < 1:
        opts.bl_warmup_epochs = 5
        print(f"Setting bl_warmup_epochs to {opts.bl_warmup_epochs} for rollout baseline")

    # Adjust learning rates for Actor-Critic
    if opts.use_critic or opts.baseline == 'actor_critic':
        if opts.lr_critic == opts.lr_model:
            opts.lr_critic = opts.lr_model * 2  # Critic often benefits from higher LR
            print(f"Adjusting critic learning rate to {opts.lr_critic}")

    # Set curriculum parameters
    if opts.curriculum:
        opts.min_p = max(3, opts.p // 3)  # Start with at least 3 facilities
        print(f"Curriculum learning enabled: starting with p={opts.min_p}, target p={opts.p}")

    # Print configuration summary if verbose
    if opts.verbose:
        print("\n" + "=" * 50)
        print("Configuration Summary:")
        print("=" * 50)
        print(f"Model: {opts.model}")
        print(f"Problem: {opts.problem} (n_users={opts.n_users}, n_facilities={opts.n_facilities}, p={opts.p})")
        print(f"Training: {opts.n_epochs} epochs, batch_size={opts.batch_size}")
        print(f"Baseline: {opts.baseline}")
        print(f"Learning rates: model={opts.lr_model}, option={opts.lr_option}, critic={opts.lr_critic}")
        print(f"Device: {opts.device}")
        print(f"Save directory: {opts.save_dir}")
        print("=" * 50 + "\n")

    return opts


def get_improved_options():
    """Get options with improved default settings for better training"""
    args = [
        # Model improvements
        '--model', 'hierarchical_v2',
        '--use_kv_cache',  # Enable KV caching
        '--use_critic',  # Use Actor-Critic

        # Better baseline
        '--baseline', 'rollout',  # Start with rollout
        '--bl_alpha', '0.3',  # More aggressive updates
        '--bl_warmup_epochs', '5',  # Proper warmup
        '--adaptive_baseline',  # Adaptive parameters
        '--force_baseline_update', '10',  # Force update if stuck

        # Training improvements
        '--normalize_advantage',  # Normalize advantages
        '--entropy_coef', '0.01',  # Entropy regularization
        '--curriculum',  # Curriculum learning
        '--curriculum_type', 'sigmoid',  # Smooth progression

        # Learning rates
        '--lr_model', '1e-3',
        '--lr_option', '5e-4',
        '--lr_critic', '2e-3',
        '--lr_decay', '0.99',

        # Temperature
        '--temp_start', '2.0',
        '--temp_end', '0.5',
        '--temp_decay', '0.95',

        # Other improvements
        '--max_grad_norm', '1.0',
        '--checkpoint_epochs', '5',
        '--verbose',
    ]
    return get_hierarchical_options(args)


def get_ppo_options():
    """Get options for PPO training (advanced)"""
    args = [
        # Base improvements
        '--model', 'hierarchical_v2',
        '--use_kv_cache',
        '--use_critic',

        # PPO specific
        '--use_ppo',
        '--ppo_epochs', '4',
        '--ppo_clip_epsilon', '0.2',
        '--ppo_batch_size', '256',

        # Actor-Critic settings
        '--value_loss_coef', '0.5',
        '--gamma', '0.99',
        '--gae_lambda', '0.95',

        # Other settings
        '--baseline', 'actor_critic',
        '--normalize_advantage',
        '--entropy_coef', '0.01',
        '--curriculum',
        '--verbose',
    ]
    return get_hierarchical_options(args)


if __name__ == "__main__":
    # Test configuration
    opts = get_hierarchical_options()
    print(f"Configuration loaded successfully")
    print(f"Save directory: {opts.save_dir}")
    print(f"Device: {opts.device}")