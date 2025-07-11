works with python 3.10

change wandb login credential to your own:
in train_shadow and train_attack

requirements changed, update wandb and numpy to 1.x version


GPU usage: 
pip uninstall torch torchvision torchaudio
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
sudo apt install nvcc
check gpu driver (nvidia-smi)


Run python train_target.py (or use own target model)
set target_model_path in config.yaml (nach target training, model lays in ./ckpt)

Run python train_shadow.py
set attack_dset_path (nach shadow training(changes with shadow model number), ./attack)

Run python train_attack.py
set attack_model_path (nach attack model training, ./attack)


# MIA_ML

### Implementation of Shokri et al(2016) [Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/pdf/1610.05820.pdf)

Modifications were made on shadow models' training methodology in order to prevent overfitting

1. Added weight decay factor
2. Implemented early stopping
3. Loads & saves best model based on evaluation metrics
4. Creates member vs non-member attack dataset based on shadow testset

### How to run

1. (Optional) Customize train / inference configurations in config.yaml

2. (Optional) `python train_target.py`: Train the victim model which is the target of the extraction.

3. `python train_shadow.py`: Corresponds to Diagram 1-1 ~ Diagram 2-2 illustrated below.

4. `python train_attack.py`: Corresponds to Diagram 2-3 ~ Diagram 3 illustrated below.

5. `python inference_attack.py`: Corresponds to Diagram 4 illustrated below.

### Result

- Replicated the paper's configuration on [config.yaml](./config.yaml)
- ROC Curve is plotting `TPR / FPR` according to MIA classification thresholds

| MIA Attack Metrics | Accuracy | Precision | Recall | F1 Score |
| :----------------: | :------: | :-------: | :----: | :------: |
|      CIFAR10       |  0.7761  |  0.7593   | 0.8071 |  0.7825  |
|      CIFAR100      |  0.9746  |  0.9627   | 0.9875 |  0.9749  |

|             MIA ROC Curve CIFAR10              |              MIA ROC Curve CIFAR100              |
| :--------------------------------------------: | :----------------------------------------------: |
| ![roc_curve CIFAR10](./assets/roc_cifar10.png) | ![roc_curve CIFAR100](./assets/roc_cifar100.png) |

### Paper's Methodology in Diagrams

![Page2](./assets/README/Page2.jpg)

![Page3](./assets/README/Page3.jpg)

![Page4](./assets/README/Page4.jpg)

![Page5](./assets/README/Page5.jpg)

![Page6](./assets/README/Page6.jpg)

![Page7](./assets/README/Page7.jpg)

![Page8](./assets/README/Page8.jpg)

![Page9](./assets/README/Page9.jpg)

![Page10](./assets/README/Page10.jpg)
