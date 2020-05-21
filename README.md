# Ensemble Model with Batch Spectral Regularization and Data Blending for Cross-Domain Few-Shot Learning with Unlabeled Data

Code release for Track 2 in Cross-Domain Few-Shot Learning (CD-FSL) Challenge .

## Enviroment

Python 2.7.16

Pytorch 1.0.0

## Steps

1. Prepare the source dataset miniImageNet and four target datasets CropDisease, EuroSAT, ISIC and ChestX.

2. Modify the paths of the datasets in `configs.py` according to the real paths.

3. Train base models on miniImageNet (pre-train)

    • *Train single model*
    
    ```bash
    python ./train_bsr.py --model ResNet10 --train_aug
    ```
    
    • *Generate projection matrices and train ensemble model*
    
    ```bash
    python ./create_Pmatrix.py
    python ./train_Pbsr.py --model ResNet10 --train_aug
    ```
4. Fine-tune and test for the 5-shot task in CropDisease as an example (change `n_shot` parameter to 20, 50 for 20-shot and 50-shot evaluations and change `dtarget` parameter to EuroSAT, ISIC, ChestX for the other target domains)

    • *Test the BSDB and BSDB+LP methods for single model*
    
    ```bash
     python ./finetune_lp_bsdb.py --model ResNet10 --train_aug --use_saved --dtarget CropDisease --n_shot 5
    ```
    
    The `use_saved` flag is used to test with our saved models. You can close it to test with the reproduced models.
    
    Example output:
    
        BSDB: 600 Test Acc = 93.48% +- 0.42%
        BSDB+LP: 600 Test Acc = 95.31% +- 0.37%
    
    • *Test the BSDB and BSDB+LP methods for ensemble model*
    
    ```bash
     python ./finetune_P_lp_bsdb.py --model ResNet10 --train_aug --use_saved --dtarget CropDisease --n_shot 5
    ```
    
    Example output:
    
        BSDB (Ensemble): 600 Test Acc = 94.05% +- 0.41%
        BSDB+LP (Ensemble): 600 Test Acc = 95.93% +- 0.37%
