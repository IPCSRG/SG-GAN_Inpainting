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

**1.   Datasets**

We train our model on Places2 and Celeba dataset.

1. [Places2](http://places2.csail.mit.edu)
2. [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 

generate the image list using script  [`./get_flist.py`](./get_flist.py) for training.

**2.   Training**

To train our model, modify the model config file [inpaint.yaml](inpaint.yaml). You may need to change the path of dataset or the parameters of the networks etc. Then run python train.py \

**3.   Testing**

To output the generated results of the inputs, you can use the [test.py](test.py).  The pre-trained weights can be downloaded from [Places2](), [CelebA-HQ](). Download the checkpoints and save them to './model_logs'

### Citation

We built our code based on  [CA](https://github.com/JiahuiYu/generative_inpainting). If you find this code is helpful for your research, please cite the following paper:

```
@article{Han18,
  author  = {Han Zhang, Ian J. Goodfellow, Dimitris N. Metaxas, Augustus Odena},
  title   = {Self-Attention Generative Adversarial Networks},
  year    = {2019},
  journal = {International Conference on Machine Learning},
}
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}
```


### Acknowledgements

We built our code based on [CA](https://github.com/JiahuiYu/generative_inpainting). Part of the code were derived from [SAGAN](https://github.com/brain-research/self-attention-gan). Please consider to cite their papers. 
