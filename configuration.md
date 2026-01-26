## Training 
### Model Training Config
To train model you need to setup config under `config` directory

config/stage1.yaml:
```yaml
# Stage 1: PointNet Backbone Training Configuration
data_dir: "/path/to/your/dataset"
val_split: 0.1

sampling: "random"  # Options: fps, random, original_size
num_points: 8192

grid_size: 7

epochs: 50
min_epochs: 50
batch_size: 12
lr: 0.001
workers: 4

accelerator: "auto"
device: "auto"

stn_3d: true
stn_nd: true
head_size: 256
stn_head_norm: false
pooling: "max"  # Options: max, avg

count_penalty_type: "kl_to_counts"  # Options: softmax, ratio, kl_to_counts
count_penalty_weight: 3.5
count_penalty_beta: 1.0
count_penalty_tau: 1.0

model_save_path: "checkpoints/stage_1"
model_save_name: "pointnet_backbone"

fast_dev_run: false
```

config/stage2.yaml:
```yaml
# Stage 2: Disentangler Training Configuration
pointnet_ckpt: "/path/to/backbone/checkpoint"

data_dir: "data"
val_split: 0.1

sampling: "original_size"  # Options: fps, random, original_size
num_samples: 8192

grid_size: 7

epochs: 30
batch_size: 8
num_workers: 2
lr: 0.001
accumulate_grad_batches: 4

prototype_update_freq: 2
initial_topk: 40
final_topk: 5
num_channels: 256

early_stopping_patience: 15

output_dir: "/path/where/output/should/be"
log_name: "/path/to/store/logs"

viz_channels: 6
viz_topk: 5
```

## Multi-Stage Training
### Stage 1: Backbone Training Full Argument List

```bash
python3 train_stage_1_backbone.py --data_dir data --epochs 90 \
  --batch_size 16 --workers 2 --sampling random --num_samples 8192 \
  --stn_3d --stn_nd --pooling max --grid_size 7 --head_size 256 \
  --count_penalty_type kl_to_counts --count_penalty_weight 3.5 \
  --model_save_path checkpoints/stage_1 --model_save_name pointnet_backbone
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `None` | Path to stage config, instead of parser |
| `--data_dir` | `data` | Root directory with training data |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `12` | Training batch size |
| `--lr` | `0.001` | Learning rate |
| `--val_split` | `0.1` | Validation split ratio |
| `--workers` | `4` | Number of data loading workers |
| `--sampling` | `random` | Sampling method: `fps`, `random`, `original_size` |
| `--num_samples` | `8192` | Number of points to sample |
| `--grid_size` | `7` | Voxel grid size for aggregation |
| `--stn_3d` | `false` (recommend to true) | Enable 3D Spatial Transformer Network |
| `--stn_nd` | `false` (recommend to true) | Enable feature-space STN |
| `--head_size` | `256` | Classification head hidden size |
| `--pooling` | `max` | Pooling type: `max`, `avg` |
| `--count_penalty_type` | `kl_to_counts` | Regularization: `softmax`, `ratio`, `kl_to_counts` |
| `--count_penalty_weight` | `3.5` | Weight for count penalty (disabled if None) |
| `--model_save_path` | `checkpoints` | Directory to save checkpoints |
| `--model_save_name` | `model` | Checkpoint filename |
| `--accelerator` | `auto` | Training accelerator |
| `--device` | `auto` | Device specification |
| `--fast_dev_run` | `false` | Quick debug run |

### Stage 2: Disentangler Training Full Argument List

```bash
python3 train_stage_2_disentangler.py \
  --pointnet_ckpt checkpoints/stage_1/pointnet_backbone.ckpt \
  --data_dir data --epochs 30 --val_split 0.1 \
  --sampling random --num_samples 8192 --grid_size 7 \
  --batch_size 5 --lr 0.0001 --initial_topk 40 --final_topk 5 \
  --num_channels 256 --output_dir checkpoints/stage_2 \
  --log_name disentangler_logs --viz_channels 6 --viz_topk 5
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `None` | Path to stage config, instead of parser |
| `--pointnet_ckpt` | *required* | Path to Stage 1 checkpoint |
| `--data_dir` | *required* | Root directory with training data |
| `--output_dir` | *required* | Output directory for checkpoints |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `8` | Training batch size |
| `--lr` | `0.001` | Learning rate |
| `--val_split` | `0.1` | Validation split ratio |
| `--num_workers` | `2` | Number of data loading workers |
| `--sampling` | `random` | Sampling method: `fps`, `random`, `original_size` |
| `--num_samples` | `8192` | Number of points to sample |
| `--grid_size` | `7` | Voxel grid size (should match Stage 1) |
| `--num_channels` | `256` | Number of feature channels |
| `--initial_topk` | `40` | Initial prototypes per channel |
| `--final_topk` | `5` | Final prototypes per channel |
| `--prototype_update_freq` | `2` | Epochs between prototype updates |
| `--early_stopping_patience` | `15` | Early stopping patience |
| `--accumulate_grad_batches` | `4` | Gradient accumulation steps |
| `--log_name` | `disentangler_logs` | TensorBoard log name |
| `--viz_channels` | `6` | Channels to visualize |
| `--viz_topk` | `5` | Top-k prototypes to visualize |


## Explaining
### Model Explaining Full Argument List

```bash
python3 explain.py --ply_path path/to/input.ply --pointnet_ckpt checkpoints/stage_2/pointnet_disentangler_compensated.pt
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--ply_path` | *required* | Path to input PLY file to explain |
| `--pointnet_ckpt` | `checkpoints/stage_2/pointnet_disentangler_compensated.pt` | Path to trained model checkpoint |
| `--data_dir` | `data` | Directory with reference samples for prototypes |
| `--output_path` | `explanations` | Output directory for results |
| `--grid_size` | `7` | Voxel grid size (must match training) |
| `--num_channels` | `256` | Number of feature channels (must match training) |
| `--head_size` | `256` | Classification head size (must match training) |
| `--stn_3d` | `false` (recommend to true) | Enable 3D STN (must match training) |
| `--stn_nd` | `false` (recommend to true) | Enable feature-space STN (must match training) |
| `--sampling` | `random` | Point sampling: `fps`, `random`, `original_size` |
| `--num_samples` | `8192` | Number of points to sample |
| `--num_prototypes` | `5` | Number of prototype examples per channel |
| `--save_viz` | `true` | Save point cloud visualizations |
| `--no_viz` | `false` | Disable visualization output |
| `--config` | `None` | Path to YAML config file |

#### Model Explaining Config

Example `config/explain.yaml`:
```yaml
pointnet_ckpt: "checkpoints/stage_2/pointnet_disentangler_compensated.pt"
data_dir: "data"
output_path: "explanations"
grid_size: 10
num_channels: 256
head_size: 256
stn_3d: true
stn_nd: true
num_prototypes: 5
save_viz: true
```

### Batch Explanations Full Argument List

```bash
python3 run_explanations.py \
  --data_root data/test \
  --pointnet_ckpt checkpoints/stage_2/pointnet_disentangler_compensated.pt \
  --data_dir data/train \
  --explanation_root explanations \
  --max_per_dir 5 \
  --save_viz
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `data/test` | Root folder containing subdirectories with PLY files |
| `--pointnet_ckpt` | *required* | Path to trained model checkpoint |
| `--data_dir` | `data/train` | Directory with reference samples for prototypes |
| `--explanation_root` | *required* | Output directory for all explanations |
| `--config` | `None` | Path to YAML config file (passed to explain.py) |
| `--grid_size` | `None` | Voxel grid size (must match training) |
| `--num_channels` | `None` | Number of feature channels (must match training) |
| `--head_size` | `None` | Classification head size (must match training) |
| `--stn_3d` | `false` | Enable 3D STN (must match training) |
| `--stn_nd` | `false` | Enable feature-space STN (must match training) |
| `--sampling` | `None` | Point sampling: `fps`, `random`, `original_size` |
| `--num_samples` | `None` | Number of points to sample |
| `--num_prototypes` | `5` | Number of prototypes per channel |
| `--save_viz` | `true` | Save point cloud visualizations |
| `--no_viz` | `false` | Disable visualization output |
| `--script` | `explain.py` | Explanation script to run |
| `--python_exe` | `python` | Python executable path |
| `--max_per_dir` | `5` | Max files to process per subdirectory |
| `--merge_out` | `merged_inference_stats.json` | Output filename for merged stats |
| `--dry_run` | `false` | Print files without processing |
