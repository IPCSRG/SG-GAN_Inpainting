# SG-GAN_Inpainting
Code for our paper "[SG-GAN_Inpainting] 

### Introduction


### Requirements

1. Tensorflow = 1.12
2. Python 3
3. NVIDIA GPU + CUDA 9.0
4. Tensorboard


### Installation

1. Clone this repository

   ```bash
   git clone https://github.com/IPCSRG/SG-GAN_Inpainting
   ```
   
### Running

**1.   Image Prepare**

We train our model on Places2 and Celeba dataset.

1. [Places2](http://places2.csail.mit.edu)
2. [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 

generate the image list using script  [`./get_flist.py`](./get_flist.py) for training.

**2.   Training**

To train our model, modify the model config file [inpaint.yaml](inpaint.yaml). You may need to change the path of dataset or the parameters of the networks etc. Then run python train.py \

**3.   Testing**

To output the generated results of the inputs, you can use the [test.py](test.py).  The pre-trained weights can be downloaded from [Places2](), [CelebA-HQ](). Download the checkpoints and save them to './model_logs'

### Citation

We built our code based on  [CA](https://github.com/JiahuiYu/generative_inpainting).If you find this code is helpful for your research, please cite the following paper:

```
@inproceedings{ren2019structureflow,
      author = {Ren, Yurui and Yu, Xiaoming and Zhang, Ruonan and Li, Thomas H. and Liu, Shan and Li, Ge},
      title = {StructureFlow: Image Inpainting via Structure-aware Appearance Flow},
      booktitle={IEEE International Conference on Computer Vision (ICCV)},
      year = {2019}
}
```



### Acknowledgements

We built our code based on [CA](https://github.com/JiahuiYu/generative_inpainting). Part of the code were derived from [Edge-Connect](https://github.com/knazeri/edge-connect) and [CA](https://github.com/JiahuiYu/generative_inpainting). Please consider to cite their papers. 
