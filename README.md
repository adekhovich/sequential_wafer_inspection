# Sequential wafer map inspection via feedback loop with reinforcement learning

This repository contains the official implementation of Sequential wafer map inspection via feedback loop with reinforcement learning paper ([link](https://www.sciencedirect.com/science/article/pii/S0957417425006189)) by Aleksandr Dekhovich, Oleg Soloviev & Michel Verhaegen.

![Sequential wafer map inspection](https://github.com/adekhovich/sequential_wafer_inspection//blob/main/docs/Figure1.pdf)

## Abstract

Wafer map defect recognition is a vital part of the semiconductor manufacturing process that requires a high level of precision. Measurement tools in such manufacturing systems can scan only a small region (patch) of the map at a time. However, this can be resource-intensive and lead to unnecessary additional costs if the full wafer map is measured. Instead, selective sparse measurements of the image save a considerable amount of resources (e.g. scanning time). Therefore, in this work, we propose a feedback loop approach for wafer map defect recognition. The algorithm aims to find sequentially the most informative regions in the image based on previously acquired ones and make a prediction of a defect type by having only these partial observations without scanning the full wafer map. To achieve our goal, we introduce a reinforcement learning-based measurement acquisition process and recurrent neural network-based classifier that takes the sequence of these measurements as an input. Additionally, we employ an ensemble technique to increase the accuracy of the prediction. As a result, we reduce the need for scanned patches by 38% having higher accuracy than the conventional convolutional neural network-based approach on a publicly available WM-811k dataset.


## Dataset

The dataset for this research can be downloaded ([here](http://mirlab.org/dataSet/public/)). 


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
