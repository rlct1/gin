# Deep Generative Inpainting Network (GIN) for Extreme Image Inpainting 
For AIM2020 ECCV Extreme Image Inpainting Track 1 Classic <br> 
This is the Pytorch implementation of our Deep Generative Inpainting Network (GIN) for Extreme Image Inpainting. We have participated in AIM 2020 ECCV Extreme Image Inpainting Challenge. Our GIN is used for reconstructing a completed image with satisfactory visual quality from a randomly masked image. <br><br> 

## Example of Image Inpainting using our GIN 
- An example from the validation set of the AIM20 ECCV Extreme Image Inpainting Track 1 Classic
- (left: masked image, right: our completed image) 
<p align='center'>  
  <img src='examples/AIM_IC_t1_validation_0_with_holes.png' width='256'/>
  <img src='examples/test/AIM_IC_t1_validation_0.png' width='256'/>
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
- Our trained model on the modified ADE20K dataset is also provided in the `checkpoints/gin` folder 
- For reproducing the test results for this challenge, please put all the testing images in the `datasets/ade20k/test` folder
- You can test our model by typing: 
```bash
python test_ensemble.py --name gin 
```
- The test results will be stored in `results/test` folder 
- If you would like to test on other datasets, please refer to the file structure in the `datasets/ade20k/test` folder 
- Note that the file structure is for AIM20 IC Track 1 

## Training 
- By default, our model is trained using two GPUs 
- Examples of the training images from this challenge is provided in the `datasets/ade20k/train` folder 
- If you would like to train a model using our warm up for initialization, 
```bash
python train.py --name yourmodel --continue_train --load_pretrain './checkpoints/warmup' 
```
- If you would like to train a model from scratch, 
```bash
python train.py --name yourmodel 
```
- If you would like to train a model based on your own selection and resources, please refer to the `options/base_options.py` and `options/train_options.py` for details 

## Acknowledgment 
Our code is developed based on the skeleton of the Pytorch implementation of [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
