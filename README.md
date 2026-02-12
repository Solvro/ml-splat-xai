# <div align="center">XSPLAIN: XAI-enabling Splat-based Prototype Learning for Attribute-aware INterpretability</div>

<p align="center">
  <a href="https://arxiv.org/abs/2602.10239"><img src="https://img.shields.io/badge/arXiv-2602.10239-b31b1b.svg" alt="arXiv"></a>
  <a href="https://dominik-galus.github.io/XSPLAIN/"><img src="https://img.shields.io/badge/🌐-Project%20Page-blue" alt="Project Page"></a>
</p>

<p align="center">
  <em>Demo (loops):</em><br>
  <img src="resources/microphone.gif" alt="XSPLAIN demo" width="920">
</p>

<p align="center">
  <em>Teaser:</em><br>
  <img src="resources/teaser.png" alt="Teaser" width="920">
</p>

## ABSTRACT
3D Gaussian Splatting (3DGS) has rapidly become a standard for high-fidelity 3D reconstruction, yet its adoption in multiple critical domains is hindered by the lack of interpretability of the generation models as well as classification of the Splats. While explainability methods exist for other 3D representations, like point clouds, they typically rely on ambiguous saliency maps that fail to capture the volumetric coherence of Gaussian primitives. We introduce XSPLAIN, the first ante-hoc, prototype-based interpretability framework designed specifically for 3DGS classification. Our approach leverages a voxel-aggregated PointNet backbone and a novel, invertible orthogonal transformation that disentangles feature channels for interpretability while strictly preserving the original decision boundaries. Explanations are grounded in representative training examples, enabling intuitive "this looks like that" reasoning without any degradation in classification performance. A rigorous user study (N=51) demonstrates a decisive preference for our approach: participants selected XSPLAIN, explanations 48.4% of the time as the best, significantly outperforming baselines $(p < 0.001)$, showing that XSPLAIN, provides transparency and user trust.

<p align="center">
  <em>XSPLAIN Architecture:</em><br>
  <img src="resources/XSPLAIN-arch.png" alt="XSPLAIN" width="920">
</p>

### Environment Setup

Model had been tested on python version >= 3.11

Create the virtual environment
```bash
python3 -m venv .venv
```
Then install dependencies from requirements
```bash
pip install -r requirements.txt
```

### Datasets
We used datasets:
- [ShapeSplat](https://arxiv.org/abs/2408.10906)
- [3DRealCar](https://xiaobiaodu.github.io/3drealcar/)
- [Toys generated from TRELLIS](https://drive.google.com/drive/folders/19crk-rGoSAPTU8x58ELSNsqq6R9eTNEu?usp=sharing)

We assumed the structure of the dataset:
```
.
├── data
│   ├── test
│   │   ├── 10
│   │     └── sample_001.ply
│   └── train
│   │     └── sample_001.ply
```

### Model Training
To train model you need to setup config under `config` directory. See [Model Training Config](configuration.md/#model-training-config) for specific configuration file.

To train model and disentangler you can use directly `train_xsplain.py`:
```bash
python3 train_xsplain.py
```

Or use two scripts separately:

#### Stage 1: Backbone Training

```bash
python3 train_stage_1_backbone.py --data_dir data --epochs 90 \
  --batch_size 16 --workers 2 --sampling random --num_samples 8192 \
  --stn_3d --stn_nd --pooling max --grid_size 7 --head_size 256 \
  --count_penalty_type kl_to_counts --count_penalty_weight 3.5 \
  --model_save_path checkpoints/stage_1 --model_save_name pointnet_backbone
```

#### Stage 2: Disentangler Training

```bash
python3 train_stage_2_disentangler.py \
  --pointnet_ckpt checkpoints/stage_1/pointnet_backbone.ckpt \
  --data_dir data --epochs 30 --val_split 0.1 \
  --sampling random --num_samples 8192 --grid_size 7 \
  --batch_size 5 --lr 0.0001 --initial_topk 40 --final_topk 5 \
  --num_channels 256 --output_dir checkpoints/stage_2 \
  --log_name disentangler_logs --viz_channels 6 --viz_topk 5
```

See [Multi-Stage Training](configuration.md/#multi-stage-training) for detailed explanation for each argument.

### Explaining
To generate explanations for a trained XSPLAIN model, use `explain.py`:

```bash
python3 explain.py --ply_path path/to/input.ply --pointnet_ckpt checkpoints/stage_2/pointnet_disentangler_compensated.pt
```

See [Model Explaining Full Argument List](configuration.md/#explaining-full-argument-list) for detailed explanation for each argument.

#### Using config file:

```bash
python3 explain.py --ply_path path/to/input.ply --config config/explain.yaml
```

See [Model Explaining Config](configuration.md/#model-explaining-config) for specific configuration file..

### Batch Explanations

To generate explanations for multiple PLY files at once, use `run_explanations.py`:

```bash
python3 run_explanations.py \
  --data_root data/test \
  --pointnet_ckpt checkpoints/stage_2/pointnet_disentangler_compensated.pt \
  --data_dir data/train \
  --explanation_root explanations \
  --max_per_dir 5 \
  --save_viz
```

This script processes all PLY files in subdirectories of `--data_root` and merges results into a single JSON file.

#### Using config file:

You can pass a config file that will be forwarded to `explain.py`:

```bash
python3 run_explanations.py \
  --data_root data/test \
  --explanation_root explanations \
  --pointnet_ckpt checkpoints/stage_2/pointnet_disentangler_compensated.pt \
  --config config/explain.yaml
```

See [Batch Explanations Full Argument List](configuration.md/#batch-explanations-full-argument-list) for detailed explanation for each argument.

#### Output structure:

```
explanations/
├── sample_001/
│   ├── inference_stats.json
│   └── channel_XXXX/
│       ├── prototypes_full_3d.png
│       └── *.ply
├── sample_002/
│   └── ...
└── merged_inference_stats.json
```


## User Study Results

We collected full responses from $N = 51$ respondents. The study group consisted of 6 women and 45 men. Regarding Machine Learning expertise, the majority were intermediate $(N=26)$ or advanced $(N=14)$ practitioners, with the remainder being beginners $(N=9)$ or having no experience $(N=2)$. Experience with 3D modelling was more limited, with most participants identifying as beginners $(N=22)$ or having no experience $(N=21)$, and only 8 intermediate users. Familiarity with Explainable AI (XAI) was mixed: 23 participants had no prior experience, while the rest were intermediate $(N=16)$, beginners $(N=11)$, or advanced $(N=1)$ users. Each respondent evaluated 3 separate items. In each item, the respondent saw 3 explanation methods, denoted Method A, Method B, and Method C. Two information were collected per item.

- **Best method selection:** a single categorical choice among $\{A,B,C\}$.

- **Confidence rating:** a four level confidence rating, mapped to integers ("Fairly confident correct" to 4,
"Somewhat confident correct" to 3, "Somewhat confident incorrect" to 2, "Fairly confident incorrect" maps to 1) that the selected explanation supports a correct understanding.

### Key Findings
The study focused on two main metrics: **User Preference** (which explanation is the most helpful?) and **Perceived Confidence** (does the explanation convince the user that the prediction is correct?).

* **Decisive Preference:** Participants selected **XSPLAIN** explanations as the "best" choice **49.4%** of the time, significantly outperforming SHAP (32.7%) and LIME (18.0%) with statistical significance ($p < 0.001$).
* **Increased Trust:** Our method achieved the highest share of "Confident Correct" ratings (**46.5%**), indicating that prototype-based explanations effectively foster user trust in correct model predictions.

### Quantitative Comparison

| Metric Category | Response Label | LIME | SHAP | **XSPLAIN (Ours)** |
| :--- | :--- | :---: | :---: | :---: |
| **Preference** | *Chosen as Best Explanation* | 17.95% | 32.69% | **49.36%** |
| **Confidence** | *High Confidence in Prediction* | 22.58% | 30.97% | **46.45%** |
| **Confidence** | *Perceived as Incorrect/Unsure* | 41.18% | 32.35% | **26.47%** |

> **Analysis:** While baseline methods often produce ambiguous saliency maps, users consistently found XSPLAIN's "this looks like that" reasoning grounded in volumetric prototypes to be more transparent and trustworthy.

## Citation
If you find our work useful, please consider citing:

```bibtex
@misc{galus2026xsplainxaienablingsplatbasedprototype,
      title={XSPLAIN: XAI-enabling Splat-based Prototype Learning for Attribute-aware INterpretability}, 
      author={Dominik Galus and Julia Farganus and Tymoteusz Zapala and Mikołaj Czachorowski and Piotr Borycki and Przemysław Spurek and Piotr Syga},
      year={2026},
      eprint={2602.10239},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.10239}, 
}
```
