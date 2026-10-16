# DELTA-IJCAI2026

The code related to the paper below: Shenjie Jiang, Xiangfeng Liu, Xuecheng Wu, and Xianghua Li. 2026. DELTA: Disentangled Hierarchical Interaction and Adaptive Adjustment for Emotion and Intent Understanding in Multimodal Conversations. In Proceedings of the 35th International Joint Conference on Artificial Intelligence (IJCAI '26).


# Run

## Project Structure

The project is organized as follows:

* **`train.py`** — Main training script for DELTA, handling data loading, optimization, validation, checkpoint saving, and testing.
* **`models/`**
  * **`base_model.py`** — Base training and evaluation utilities used by all models.
  * **`our/our_model.py`** — Main DELTA implementation, including disentangled feature decomposition, adaptive interaction, and multimodal classification.
  * **`our/htbi_block.py`** — Hierarchical Transformer-Based Interaction block for emotion and intent streams.
  * **`networks/`** — Backbone encoders and classifiers such as LSTM, TextCNN, and fully connected classifiers.
  * **`config/`** — Dataset configuration files for feature and label paths.
* **`opts/get_opts.py`** — Command-line options and experiment configuration.
* **`utils/`** — Logging, checkpoint, and result recording utilities.



## Datasets

- **MCEIU (Chinese)**
- **MCEIU (English)**

You should put them in the `./dataset` folder, or modify the dataset path on your own in `config/config.json`. The dataset loader expects pre-extracted multimodal features stored in `.h5` files. Emotion and intent labels are loaded from `.npy` files under the configured target directory.

## Download Links
To access the [MCEIU](https://github.com/MC-EIU/MC-EIU) and [MINE](https://github.com/yan9qu/CVPR25-MINE) datasets, please reach out to the original authors and complete the required User Agreement.

## Installation
1. **Clone the repository**:
```bash
git clone https://github.com/Username0729/DELTA.git
cd DELTA
```
2. **Create a virtual environment** (recommended):
```bash
python3 -m venv delta_env
source delta_env/bin/activate
```
3. **Install dependencies**:
```bash
pip install -r requirements.txt
```


## Execution

Before running, make sure the dataset paths in `./data/config/*.json` are correctly set for your machine.

### Training

Run the main training script with the following command:

```bash
python train.py --gpu_ids=1 --model=our --niter=50 --niter_decay=100 --print_freq=100 --batch_size=32 --lr=2e-4 --weight_decay=1e-5
```

By default, checkpoints are saved in `./checkpoints/<experiment_name>/`, and logs are written to `./logs/<experiment_name>/`.

### Testing

After training, evaluate the model using the corresponding checkpoint and the same dataset configuration used during training.




# Project Structure

```
DELTA/
├── train.py                  # Main training and evaluation script
├── models/
│   ├── base_model.py         # Base model utilities
│   ├── our/
│   │   ├── our_model.py      # DELTA model implementation
│   │   └── htbi_block.py     # Hierarchical interaction block
│   ├── networks/             # Backbone encoders and classifiers
│   └── utils/                # Model-related config helpers
├── data/
│   └── config/               # Dataset path configuration files
│   
├── opts/
│   └── get_opts.py           # Command-line argument parsing
├── utils/                    # Logging and result recording utilities
└── README.md
```


# Acknowledgment

This framework is built upon and improves the [MEI-Pro](https://github.com/ZhengL00/MEI-Pro) model. We would like to express our special gratitude to the authors of MEI-Pro for their foundational work, and sincerely thank the broader community, especially [MC-EIU](https://github.com/MC-EIU/MC-EIU), for their open-source contributions.


# Contact

If you have any questions about the project, please contact 
If you have any questions, please feel free to contact the authors: **Shenjie Jiang**( [jiangshenjie@mail.nwpu.edu.cn](jiangshenjie@mail.nwpu.edu.cn))


# Reference

If you make advantage of DELTA in your research, please cite the following paper:

```bibtex
@inproceedings{jiang2026delta,
  title     = {{DELTA}: Disentangled Hierarchical Interaction and Adaptive Adjustment for Emotion and Intent Understanding in Multimodal Conversations},
  author    = {Jiang, Shenjie and Liu, Xiangfeng and Wu, Xuecheng and Li, Xianghua},
  booktitle = {Proceedings of the 35th International Joint Conference on Artificial Intelligence (IJCAI '26)},
  year      = {2026}
}
```
