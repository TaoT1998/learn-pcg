## Transformer PRNG Training

This repository is for training transformers on PRNG tasks:

- `prng_train_lowmem.py`: single‑modulus training.
- `curriculum_lowmem.py`: multi‑modulus curriculum training/fine‑tuning.


---

### Environment


Required:
- torch
- numpy
- pandas
- psutil
- sympy

Optional (features/scripts):
- wandb  [experiment tracking]
- PyYAML (yaml)  [required for `curriculum_lowmem.py --config`]

Install (example):

```bash
pip install torch numpy pandas psutil sympy
# Optional
pip install pyyaml matplotlib seaborn scikit-learn wandb
```

---

## Single‑modulus training: prng_train_lowmem.py


### Example run

```bash


python prng_train_lowmem.py \
  --eval_interval 2000 \
  --num_workers 8 \
  --m 262144 \
  --type XSLRR \
  --n_layer 4 \
  --n_head 8 \
  --n_embd 1024 \
  --seq_len 513 \
  --batch_size 512 \
  --grad_acc_steps 1 \
  --lr_trgt 0.0005 \
  --weight_decay 0.1 \
  --data_seed 1 \
  --main_seed 1 \
  --n_a 1024 \
  --n_c 1024 \
  --n_example 1 \
  --n_test_a 128 \
  --n_test_c 64 \
  --num_steps 400000 \
  --warm_steps 5000 \
  --save_correctness \
  --save_params \
  --save_checkpoints \
  --digits 1 \
  --base 512 \
  --bits_to_keep 9 \
  --control_bits 3
```






---

## Curriculum training/fine‑tuning: curriculum_lowmem.py

Trains across multiple moduli with weighted‑sampler curriculum, using a YAML config. 

### YAML‑driven run

Create a YAML config (see example below), then:

```bash
python curriculum_lowmem.py --config configs/curriculum.yaml
```



### Example YAML

```yaml
model:
  n_layer: 4
  n_head: 8
  n_embd: 1024

data:
  type: XSLRR
  vocab_size: 1024
  base: 1024
  seq_len: 513
  control_bits: 3
  n_a: 1024
  n_c: 1024
  n_test_a: 128
  n_test_c: 64
  n_example: 1
  digits: 1

curriculum:
  moduli: [65536, 262144]
  bits_to_keep: [8, 9]
  sampler_update_interval: 100
  phases:
    - name: Phase 1
      phase_steps: 4000
      transition_steps: 4000
      warmup_steps: 500
      lr_decay: constant
      start_weights: [0.001, 0.999]
      end_weights: [0, 1]
      transition: exp
    - name: Phase 2
      phase_steps: 1000
      transition_steps: 0
      warmup_steps: 0
      lr_decay: cosine
      start_weights: [0, 1]
      end_weights: [0, 1]
      transition: exp

training:
  batch_size: 256
  grad_acc_steps: 1
  eval_interval: 200
  lr_trgt: 0.0005
  lr_min: 1e-7
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.999

seeds:
  main_seed: 1
  data_seed: 2

output:
  results_dir: results/FT
  save_checkpoints: true
  checkpoint_interval: 1000
  save_params: true
  save_correctness: true

pretrained:
  pretrained_path: null   # or path/to/checkpoint.pt
  pretrain_m: null        # largest modulus in the pretrained model (optional)

wandb:
  use_wandb: false
  project: prng_curriculum
  entity: null
  name: null
  tags: null
  notes: null

# Optionally exclude AC values used in prior training to avoid leakage
exclude_ac_paths: []
```

### SLURM curriculum fine‑tune (from curriculum_finetune.sh)

`curriculum_finetune.sh` programmatically builds a YAML (with a small LR/WD sweep) and runs:

```bash
srun python curriculum_lowmem.py --config <TEMP_CONFIG_YAML>
```
---

## Key arguments

- **Data/PRNG**: `--type` (e.g., LCG, TLCG, RS, RR, XSHRR, XSHRS, XSLRR), `--m` (single‑modulus training), `--control_bits`, `--bits_to_keep`, `--base`, `--vocab_size`, `--seq_len`, `--digits`.
- **Dataset sizes**: `--n_a`, `--n_c`, `--n_test_a`, `--n_test_c`, `--n_example`.
- **Model**: `--n_layer`, `--n_head`, `--n_embd`, `--no_rope`.
- **Training**: `--batch_size`, `--grad_acc_steps`, `--lr_trgt`, `--lr_min`, `--weight_decay`, `--beta1`, `--beta2`, `--num_steps` (single‑modulus) or derived from curriculum phases (curriculum), `--eval_interval`.
- **Curriculum**: YAML `curriculum.moduli`, `curriculum.bits_to_keep` (optional), `curriculum.phases[*]` with `phase_steps`, `transition_steps`, `warmup_steps`, `lr_decay`, `start_weights`, `end_weights`, `transition`; `--sampler_update_interval` (defaults to `eval_interval`).
- **Saving**: `--results_dir`, `--save_checkpoints`, `--checkpoint_interval`, `--save_params`, `--save_correctness`.
- **Exclusions**: `--exclude_ac_paths` to avoid leaking AC values from previous trainings.
- **W&B**: provided through shared `utils.wandb_utils.add_wandb_args` (disabled unless set).




