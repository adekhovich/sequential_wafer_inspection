# Sequential wafer map inspection via feedback loop with reinforcement learning

This repository contains the official implementation of Sequential wafer map inspection via feedback loop with reinforcement learning [paper](https://www.sciencedirect.com/science/article/pii/S0957417425006189) by Aleksandr Dekhovich, Oleg Soloviev & Michel Verhaegen.

![Sequential wafer map inspection](https://github.com/adekhovich/sequential_wafer_inspection/blob/main/docs/Figure1.svg)

## Abstract

Wafer map defect recognition is a vital part of the semiconductor manufacturing process that requires a high level of precision. Measurement tools in such manufacturing systems can scan only a small region (patch) of the map at a time. However, this can be resource-intensive and lead to unnecessary additional costs if the full wafer map is measured. Instead, selective sparse measurements of the image save a considerable amount of resources (e.g. scanning time). Therefore, in this work, we propose a feedback loop approach for wafer map defect recognition. The algorithm aims to find sequentially the most informative regions in the image based on previously acquired ones and make a prediction of a defect type by having only these partial observations without scanning the full wafer map. To achieve our goal, we introduce a reinforcement learning-based measurement acquisition process and recurrent neural network-based classifier that takes the sequence of these measurements as an input. Additionally, we employ an ensemble technique to increase the accuracy of the prediction. As a result, we reduce the need for scanned patches by 38% having higher accuracy than the conventional convolutional neural network-based approach on a publicly available WM-811k dataset.


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

Run the code with:
```
python3 src/main.py --train_classifier --train_confidnet --train_patch_selector --seed 0

Possible arguments:
    --data_name                Name of the dataset (default: wm811k)
    --data_path                Path to the dataset (default: ./data/WM811K_labeled.pkl)
    --input_channels           Number of image channels (default: 1)
    --patch_size               Patch size (default: 8)
    --seed                     Random seed (default: 0)
    
    # Classifier related:
    parser.add_argument("--classifier_name", default="gru", type=str, help="Name of the classifiaction model")
    parser.add_argument("--classifier_hidden_dim", default=256, type=int, help="Size of the hidden state")
    parser.add_argument("--classifier_num_layers", default=2, type=int, help="Number of RNN cells")
    parser.add_argument("--classifier_num_epochs", default=240, type=int, help="Number of training epochs for the classifier")
    parser.add_argument("--classifier_lr", default=1e-3, type=float, help="Learning rate for the classifier")
    parser.add_argument("--classifier_wd", default=1e-5, type=float, help="Weight decay for the classifier")
    parser.add_argument("--classifier_optimizer_name", default="Adam", type=str, help="Optimizer to train the classifier")
    parser.add_argument("--classifier_batch_size", default=128, type=int, help="Optimizer to train the classifier") 
    parser.add_argument("--classifier_num_heads", default=8, type=int, help="Number of RNN heads for which the loss is computed") 
    parser.add_argument("--train_classifier", action="store_true", help="")
    parser.add_argument("--supcon", action="store_true", help="Include SupCon loss. Only for ResNet")
    
    # Confidnet related:   
    parser.add_argument("--confidnet_name", default="gru", type=str, help="Name of the confidnet model")
    parser.add_argument("--confidnet_hidden_dim", default=128, type=int, help="Size of the hidden state")
    parser.add_argument("--confidnet_num_layers", default=2, type=int, help="Number of RNN cells")
    parser.add_argument("--confidnet_num_epochs", default=300, type=int, help="Number of training epochs for the confidnet")
    parser.add_argument("--confidnet_lr", default=1e-3, type=float, help="Learning rate for the confidnet")
    parser.add_argument("--confidnet_wd", default=1e-5, type=float, help="Weight decay for the confidnet")
    parser.add_argument("--confidnet_optimizer_name", default="Adam", type=str, help="Optimizer to train the confidnet")
    parser.add_argument("--confidnet_batch_size", default=128, type=int, help="Optimizer to train the confidnet")
    parser.add_argument("--train_confidnet", action="store_true", help="")
    
    # Patch selector related:
    parser.add_argument("--patch_selector_num_iters", default=100, type=int, help="Number of training epochs for the patch_selector")
    parser.add_argument("--patch_selector_lr", default=1e-3, type=float, help="Learning rate for the patch_selector")
    parser.add_argument("--patch_selector_wd", default=1e-5, type=float, help="Weight decay for the patch_selector")
    parser.add_argument("--patch_selector_batch_size", default=16, type=int, help="Optimizer to train the patch_selector")  
    parser.add_argument("--gamma", default=0.99, type=int, help="Discounting factor")  
    parser.add_argument("--train_patch_selector", action="store_true", help="")
    
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
