# Deep Generative Inpainting Network (GIN) for Extreme Image Inpainting 
For AIM2020 ECCV Extreme Image Inpainting Track 1 Classic <br> 
This is the Pytorch implementation of our Deep Generative Inpainting Network (GIN) for Extreme Image Inpainting. We have participated in AIM 2020 ECCV Extreme Image Inpainting Challenge. Our GIN is used for reconstructing a completed image with satisfactory visual quality from a randomly masked image. <br><br> 

## Overview
<p align='center'>  
  <img src='examples/architecture.png' width='768'/>
</p>
Our Spatial Pyramid Dilation (SPD) block
<p align='center'>  
  <img src='examples/spd_resnetblk.png' width='768'/>
</p>

## Example of Image Inpainting using our GIN 
- An example from the validation set of the AIM20 ECCV Extreme Image Inpainting Track 1 Classic
- (left: masked image, right: our completed image) 
<p align='center'>  
  <img src='examples/AIM_IC_t1_validation_0_with_holes.png' width='256'/>
  <img src='examples/AIM_IC_t1_validation_0.png' width='256'/>
</p>

## Preparation 
- Our solution is developed using Pytorch 1.5.0 platform 
- We train our model on two NVIDIA GeForce RTX 2080 Ti (with 11GB memory) 
- Apart from Pytorch and related dependencies, 
- Install natsort
```bash
pip install natsort
```
- Install dominate 
```bash
pip install dominate
```
- Install scipy 1.1.0
```bash
pip install scipy==1.1.0
```
- If you would like to use tensorboard for logging, please also install tensorboard and tensorflow 
- Please clone this project: 
```bash
git clone https://github.com/rlct1/gin.git
cd gin
```

## Testing 
- An example of the validation data of this challenge is provided in the `datasets/ade20k/test` folder 
- Please download our trained model for this challenge [here](https://drive.google.com/file/d/1yOtMELWwTBc-PMSY69x1FH8D1anUN7tD/view?usp=sharing) (google drive link), and put it under `checkpoints/gin/`
- For reproducing the test results for this challenge, please put all the testing images under `datasets/ade20k/test/`
- You can test our model by typing: 
```bash
python test_ensemble.py --name gin 
```
- The test results will be stored in `results/test` folder 
- If you would like to test on other datasets, please refer to the file structure in the `datasets/ade20k/test` folder 
- Note that the file structure is for AIM20 IC Track 1 
- You can download our test results for this challenge [here](https://drive.google.com/file/d/1EJgQ3neOA2WkZMmG6uG0GG14VoLYmNFg/view?usp=sharing) (google drive link)

## Training 
- By default, our model is trained using two GPUs 
- Examples of the training images from this challenge is provided in the `datasets/ade20k/train` folder 
- If you would like to train a model using our warm up for initialization, please download our warm up for this challenge [here](https://drive.google.com/file/d/1T3ST-ujhtDZQpWUiagICOAIvBF7CMeYz/view?usp=sharing) (google drive link), and put it under `checkpoints/warmup/`
```bash
python train.py --name yourmodel --continue_train --load_pretrain './checkpoints/warmup' 
```
- If you would like to train a model from scratch, 
```bash
python train.py --name yourmodel 
```
- If you would like to train a model based on your own selection and resources, please refer to the `options/base_options.py` and `options/train_options.py` for details 

## Experiments
Ablation Study
<p align='center'>  
  <img src='examples/ablation_study.png' width='768'/>
</p>
Comparisons
<p align='center'>  
  <img src='examples/comparisons_ffhq_oxford.png' width='768'/>
</p>
Visualization of predicted semantic segmentation map
<p align='center'>  
  <img src='examples/visualization_seg.png' width='768'/>
</p>

## Citation
Thanks for visiting our project page, if it is useful, please cite our paper,
```
@misc{li2020deepgin,
    title={DeepGIN: Deep Generative Inpainting Network for Extreme Image Inpainting},
    author={Chu-Tak Li and Wan-Chi Siu and Zhi-Song Liu and Li-Wen Wang and Daniel Pak-Kong Lun},
    year={2020},
    eprint={2008.07173},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgment 
Our code is developed based on the skeleton of the Pytorch implementation of [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)

