# Sequential wafer map inspection via feedback loop with reinforcement learning

This repository contains the official implementation of Sequential wafer map inspection via feedback loop with reinforcement learning [paper](https://www.sciencedirect.com/science/article/pii/S0957417425006189) by Aleksandr Dekhovich, Oleg Soloviev & Michel Verhaegen.

![Sequential wafer map inspection](https://github.com/adekhovich/sequential_wafer_inspection/blob/main/docs/Figure1.svg)

## Abstract

Wafer map defect recognition is a vital part of the semiconductor manufacturing process that requires a high level of precision. Measurement tools in such manufacturing systems can scan only a small region (patch) of the map at a time. However, this can be resource-intensive and lead to unnecessary additional costs if the full wafer map is measured. Instead, selective sparse measurements of the image save a considerable amount of resources (e.g. scanning time). Therefore, in this work, we propose a feedback loop approach for wafer map defect recognition. The algorithm aims to find sequentially the most informative regions in the image based on previously acquired ones and make a prediction of a defect type by having only these partial observations without scanning the full wafer map. To achieve our goal, we introduce a reinforcement learning-based measurement acquisition process and a recurrent neural network-based classifier that takes the sequence of these measurements as input. Additionally, we employ an ensemble technique to increase the accuracy of the prediction. As a result, we reduce the need for scanned patches by 38%, having higher accuracy than the conventional convolutional neural network-based approach on a publicly available WM-811k dataset.


## Dataset

The dataset for this research is provided by [Wu et al.](10.1109/TSM.2014.2364237) and can be downloaded [here](http://mirlab.org/dataSet/public/). 

## Installation

* Clone this github repository using:
```
git clone https://github.com/adekhovich/sequential_wafer_inspection.git
cd sequential_wafer_inspection
```

* Install requirements using:
```
pip install -r requirements.txt
```

## Train the model

Run the code with the following command to train the model from scratch:
```
python3 src/main.py --train_classifier --train_confidnet --train_patch_selector

Possible arguments:
    --data_name                     Name of the dataset (default: wm811k)
    --data_path                     Path to the dataset (default: ./data/WM811K_labeled.pkl)
    --input_channels                Number of image channels (default: 1)
    --patch_size                    Patch size (default: 8)
    --seed                          Random seed (default: 0)
    
    # Classifier related:
    --classifier_name               Name of the classification model (default: gru)
    --classifier_hidden_dim         Size of the hidden state (default: 256)
    --classifier_num_layers         Number of RNN cells (default: 2)
    --classifier_num_epochs         Number of training epochs for the classifier (default: 240)
    --classifier_lr                 Learning rate for the classifier (default: 1e-3)
    --classifier_wd                 Weight decay for the classifier (default: 1e-5)
    --classifier_optimizer_name     Optimizer to train the classifier (default: Adam)
    --classifier_batch_size         Batch size to train the classifier (default: 128) 
    --classifier_num_heads          Number of RNN heads for which the loss is computed (default: 8) 
    --train_classifier              Use this flag to train the classifier
    --supcon                        Use this flag to include the SupCon loss term. Only for ResNet
    
    # Confidnet related:   
    --confidnet_name                Name of the confidnet model (default: gru)
    --confidnet_hidden_dim          Size of the hidden state(default: 128)
    --confidnet_num_layers          Number of confident RNN cells (default: 2)
    --confidnet_num_epochs          Number of training epochs for the confident (default: 300)
    --confidnet_lr                  Learning rate for the confident (default: 1e-3)
    --confidnet_wd                  Weight decay for the confident (default: 1e-5)
    --confidnet_optimizer_name      Optimizer to train the confident (default: Adam)
    --confidnet_batch_size          Batch size to train the confident (default: 128)
    --train_confidnet               Use this flag to train the confidnet
    
    # Patch selector related:
    --patch_selector_num_iters      Number of training epochs for the patch_selector (default: 100)
    --patch_selector_lr             Learning rate for the patch_selector (default: 1e-3)
    --patch_selector_wd             Weight decay for the patch_selector (default: 1e-5)
    --patch_selector_batch_size     Batch size to train the patch_selector (default: 16)  
    --gamma                         Discounting factor (default: 0.99)
    --train_patch_selector          Use this flag to train the patch_selector
    
    # Hybrid & ensemble related. ONLY FOR EVALUATION:
    --hybrid              Use this flag for the Hybrid approach (with CNN)
    --aux_model_name      Name of the auxiliary model (default: resnet34)    
    --ensemble            Use this flag for the Ensemble
    --ensemble_type       Type of policies aggregation (default: avg)
    --num_models          Number of models in the ensemble (default: 5)    
```

## Citation

If you use our code in your research, please cite our work:
```
@article{dekhovich2025sequential,
  title={Sequential wafer map inspection via feedback loop with reinforcement learning},
  author={Dekhovich, Aleksandr and Soloviev, Oleg and Verhaegen, Michel},
  journal={Expert Systems with Applications},
  volume={275},
  pages={126996},
  year={2025},
  publisher={Elsevier}
}
``` 
