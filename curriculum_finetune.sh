#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --job-name=FT
#SBATCH --output=out/FT/%A_%a.out
#SBATCH --error=out/FT/%A_%a.err
#SBATCH --mem=64G
#SBATCH --gpus-per-node=a100:2
#SBATCH --partition=gpu
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL
#SBATCH --array=1-1

# Curriculum fine-tuning job array with hyperparameter sweeps (mirrors curriculum2m.sh)
cd $SLURM_SUBMIT_DIR

# Ensure required output directories exist
mkdir -p out/CUR
mkdir -p results/curriculum
mkdir -p configs

# Set up distributed training environment
export MASTER_PORT=$((10000 + ((SLURM_JOB_ID % 30000) + (SLURM_ARRAY_TASK_ID % 1000))))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export PYTHONUNBUFFERED=1

# Hyperparameters to sweep
learning_rates=(1e-4 5e-4)
weight_decays=(0 0.1)


# Calculate indices for hyperparameter sweep
num_lr=${#learning_rates[@]}
num_wd=${#weight_decays[@]}

lr_idx=$((SLURM_ARRAY_TASK_ID / num_wd % num_lr))
wd_idx=$((SLURM_ARRAY_TASK_ID % num_wd))

lr=${learning_rates[$lr_idx]}
wd=${weight_decays[$wd_idx]}

TEMP_CONFIG="configs/curriculum_temp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
# Pretrained model configuration
# Set these variables based on your pretrained model
###PRETRAINED_PATH="results/FT/multi_curriculum_model_XSLRR_moduli262144_1048576_transexp_cp1s50000u5000t50000dconsta0.01,0.99-0.0,1.0_p2s25000u0t0dcosa0.0,1.0-0.0,1.0_n1024_h8_d4_na2048_nc2048_ne1_vs2048_sl641_cb3_T75000_B512_lr0.0003_wd0.1_dI1_I1.pt"
PRETRAIN_M=None  # Largest modulus from pretrained model (e.g., 262144 from moduli65536_262144)

# Exclude AC paths configuration
# Add paths to AC files from pretrained models that should be excluded from test set generation
# This prevents data leakage by ensuring test sets don't use AC values from previous training runs
# EXCLUDE_AC_PATHS=(
#     # Uncomment and modify these paths based on your pretrained models
#     # Examples:
#     # "results/pretrain/trainac_XSLRR_*.npz"
#     # "results/curriculum/trainac_XSLRR_*.npz" 
#     # "results/FT/trainac_XSLRR_*.npz"
#     # "results/pretrain/training_ac_values_*.pt"
# )
# Convert EXCLUDE_AC_PATHS array to Python list format
if [ ${#EXCLUDE_AC_PATHS[@]} -eq 0 ]; then
    EXCLUDE_AC_PATHS_PY="[]"
else
    EXCLUDE_AC_PATHS_PY="["
    for i in "${!EXCLUDE_AC_PATHS[@]}"; do
        if [ $i -gt 0 ]; then
            EXCLUDE_AC_PATHS_PY="$EXCLUDE_AC_PATHS_PY, "
        fi
        EXCLUDE_AC_PATHS_PY="$EXCLUDE_AC_PATHS_PY\"${EXCLUDE_AC_PATHS[$i]}\""
    done
    EXCLUDE_AC_PATHS_PY="$EXCLUDE_AC_PATHS_PY]"
fi

# Check if yaml module is available
if ! python3 -c "import yaml" 2>/dev/null; then
    echo "WARNING: PyYAML module not available. Installing..."
    pip install PyYAML
    if ! python3 -c "import yaml" 2>/dev/null; then
        echo "ERROR: Failed to install PyYAML. Please install manually: pip install PyYAML"
        exit 1
    fi
fi

# Generate YAML configuration (same structure, but single modulus 2^20 and vocab/base=1024)
python3 -c "
import yaml
import os
import sys



config = {
    'model': {
        'n_layer': 4,
        'n_head': 8,
        'n_embd': 1024,
    },
    'data': {
        'type': 'XSLRR',
        'vocab_size': 1024,
        'base': 1024,
        'seq_len': 513,
        'control_bits': 3,
        'n_a': 1024,
        'n_c': 1024,
        'n_test_a': 128,
        'n_test_c': 64,
        'n_example': 1,
        'digits': 1,
    },
    'curriculum': {
        'moduli': [65536, 262144],
        'bits_to_keep': [8, 9],
        'sampler_update_interval': 100,
        'phases': [
            {
                'name': 'Phase 1',
                'phase_steps': 4000,
                'transition_steps': 4000,
                'warmup_steps': 500,
                'lr_decay': 'constant',
                'start_weights': [0.001, 0.999],
                'end_weights': [0,1],
                'transition': 'exp',
            },
            {
                'name': 'Phase 2',
                'phase_steps': 1000,
                'transition_steps': 0,
                'warmup_steps': 0,
                'lr_decay': 'cosine',
                'start_weights': [0,1],
                'end_weights': [0,1],
                'transition': 'exp',
            },
        ],
    },
    'training': {
        'batch_size': 256,
        'grad_acc_steps': 1,
        'eval_interval': 200,
        'lr_trgt': """$lr""",
        'lr_min': 1e-7,
        'weight_decay': """$wd""",
        'beta1': 0.9,
        'beta2': 0.999,
    },
    'seeds': {
        'main_seed': 1,
        'data_seed': 2,
    },
    'output': {
        'results_dir': 'results/FT',
        'save_checkpoints': True,
        'checkpoint_interval': 1000,
        'save_params': True,
        'save_correctness': True,
    },
    'pretrained': {
        'pretrained_path': '$PRETRAINED_PATH',
        'pretrain_m': $PRETRAIN_M,
    },
    'wandb': {
        'use_wandb': False,
        'project': 'prng_curriculum',
        'entity': None,
        'name': None,
        'tags': None,
        'notes': None,
    },
}

# Handle exclude_ac_paths properly
exclude_ac_paths = $EXCLUDE_AC_PATHS_PY
if exclude_ac_paths:
    config['exclude_ac_paths'] = exclude_ac_paths

# Ensure configs directory exists
import os
os.makedirs('configs', exist_ok=True)

with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f'Generated config file: $TEMP_CONFIG')
"

echo "Job $SLURM_ARRAY_TASK_ID configuration:"
echo "Generated config file: $TEMP_CONFIG"
echo "Parameters: lr=$lr, wd=$wd"
echo "Using pretrained: $PRETRAINED_PATH"
echo "Excluding AC paths: ${EXCLUDE_AC_PATHS[*]}"

# Check if config file was created successfully
if [ -f "$TEMP_CONFIG" ]; then
    echo "Config file created successfully: $TEMP_CONFIG"
    echo "Config file size: $(wc -l < "$TEMP_CONFIG") lines"
else
    echo "ERROR: Config file was not created: $TEMP_CONFIG"
    exit 1
fi

srun python curriculum_lowmem.py --config "$TEMP_CONFIG"
rm -f "$TEMP_CONFIG"


