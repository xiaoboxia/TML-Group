# The code that our team often uses.（PyTorch implementation mainly）

This code repository is provided for the member of TML group led by Prof.Tongliang Liu. It includes the following terms: 
- The reproductions of some papers, covered aspects are noisy-label learning, deep compression, kernel mean estimation and others. I will update it according to the projects I have already finished at all times.
- The PyTorch implementation of the model structures used in experiments.
- Some frequently used code, you can use them for data preprocessing, plot, model evaluation...
- Some useful links that relate to machine learning, computer vision.

This code repository is by Xiaobo Xia. If you are confused, please feel free to let me know or provide your 
valuable advice. Contact me at any time. The email address is xiaoboxia.uni@gmail.com.

## How To Use This Code
You will need:

- [PyTorch](https://PyTorch.org/), version >= 0.4.1, CUDA >= 9.0, python3
- tqdm, numpy, scipy, Image
- The [dataset](https://drive.google.com/open?id=1Tz3W3JVYv2nu-mdM6x33KSnRIY1B7ygQ) (the images and labels have been processed to .npy format)

## Some description for users
- loss.py --The loss function of Co-teaching, Co-teaching Plus, Decoupling, Mentornet, T_revision, MCL, KCL
- plot.py --The code for ploting using a shade error bar
- model --The model structures(Lenet, Lenet_BN, Resnet, Preactivated-Resnet, Wide Resnet, VGG)
- tools --The useful code for dataset loading， dataset division, noisy/complementary label transform, transition matrix estimation, model evaluation...






