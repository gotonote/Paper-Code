
<p align="center">

  <h2 align="center">X-Dyna: Expressive Dynamic Human Image Animation</h2>
  <p align="center">
      <a href="https://boese0601.github.io/">Di Chang</a><sup>1,2</sup>
      ·
      <a href="https://hongyixu37.github.io/homepage/">Hongyi Xu</a><sup>2*</sup>
    ·  
      <a href="https://youxie.github.io/">You Xie</a><sup>2*</sup>
    ·  
      <a href="https://hlings.github.io/">Yipeng Gao</a><sup>1*</sup>
    ·  
      <a href="https://zhengfeikuang.com/">Zhengfei Kuang</a><sup>3*</sup>
    ·  
      <a href="https://primecai.github.io/">Shengqu Cai</a><sup>3*</sup>
    ·  
      <a href="https://zhangchenxu528.github.io/">Chenxu Zhang</a><sup>2*</sup>
    <br>
      <a href="https://guoxiansong.github.io/homepage/index.html">Guoxian Song</a><sup>2</sup>
    ·  
      <a href="https://chaowang.info/">Chao Wang</a><sup>2</sup>
    ·  
      <a href="https://seasonsh.github.io/">Yichun Shi</a><sup>2</sup>
    ·  
      <a href="https://zeyuan-chen.com/">Zeyuan Chen</a><sup>2,5</sup>
    ·  
      <a href="https://shijiezhou-ucla.github.io/">Shijie Zhou</a><sup>4</sup>
    ·  
      <a href="https://scholar.google.com/citations?user=fqubyX0AAAAJ&hl=en">Linjie Luo</a><sup>2</sup>
    <br>
      <a href="https://web.stanford.edu/~gordonwz/">Gordon Wetzstein</a><sup>3</sup>
    ·  
      <a href="https://www.ihp-lab.org/">Mohammad Soleymani</a><sup>1</sup>
    <br>
    <sup>1</sup>Unviersity of Southern California &nbsp;<sup>2</sup>ByteDance Inc. &nbsp; <sup>3</sup>Stanford University &nbsp; 
    <br>
    <sup>4</sup>University of California Los Angeles&nbsp; <sup>5</sup>University of California San Diego
    <br>
    <br>
    <sup>*</sup> denotes equal contribution
    <br>
    </br>
        <a href="https://arxiv.org/abs/2501.10021">
        <img src='https://img.shields.io/badge/arXiv-XDyna-green' alt='Paper PDF'>
        </a>
        <a href='https://x-dyna.github.io/xdyna.github.io/'>
        <img src='https://img.shields.io/badge/Project_Page-XDyna-blue' alt='Project Page'></a>
        <a href='https://huggingface.co/Boese0601/X-Dyna'>
        <img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
     </br>
</p>


-----

This repo is the official pytorch implementation of X-Dyna, which generates temporal-consistent human motions with expressive dynamics.


## 📑 Open-source Plan
- [x] Project Page
- [x] Paper
- [x] Inference code for Dynamics Adapter
- [x] Checkpoints for Dynamics Adapter
- [x] Inference code for S-Face ControlNet
- [x] Checkpoints for S-Face ControlNet
- [ ] Evaluation code (DTFVD, Face-Cos, Face-Det, FID, etc.)
- [ ] Dynamic Texture Eval Data (self-collected from [Pexels](https://www.pexels.com/))
- [ ] Alignment code for inference
- [ ] Gradio Demo


## **Abstract**
We introduce X-Dyna, a novel zero-shot, diffusion-based pipeline for animating a single human image using facial expressions and body movements derived from a driving video, that generates realistic, context-aware dynamics for both the subject and the surrounding environment. Building on prior approaches centered on human pose control, X-Dyna addresses key factors underlying the loss of dynamic details, enhancing the lifelike qualities of human video animations. At the core of our approach is the Dynamics-Adapter, a lightweight module that effectively integrates reference appearance context into the spatial attentions of the diffusion backbone while preserving the capacity of motion modules in synthesizing fluid and intricate dynamic details. Beyond body pose control, we connect a local control module with our model to capture identity-disentangled facial expressions, facilitating accurate expression transfer for enhanced realism in animated scenes. Together, these components form a unified framework capable of learning physical human motion and natural scene dynamics from a diverse blend of human and scene videos. Comprehensive qualitative and quantitative evaluations demonstrate that X-Dyna outperforms state-of-the-art methods, creating highly lifelike and expressive animations.

## **Architecture**

We leverage a pretrained diffusion UNet backbone for controlled human image animation, enabling expressive dynamic details and precise motion control. Specifically, we introduce a dynamics adapter that seamlessly integrates the reference image context as a trainable residual to the spatial attention, in parallel with the denoising process, while preserving the original spatial and temporal attention mechanisms within the UNet. In addition to body pose control via a ControlNet, we introduce a local face control module that implicitly learns facial expression control from a synthesized cross-identity face patch. We train our model on a diverse dataset of human motion videos and natural scene videos simultaneously.

<p align="center">
  <img src="./assets/figures/pipeline.png"  height=400>
</p>

## **Dynamics Adapter**
### **Archtecture Designs for Human Video Animation**
a) IP-Adapter encodes the reference image as an image CLIP embedding and injects the information into the cross-attention layers in SD as the residual. b) ReferenceNet is a trainable parallel UNet and feeds the semantic information into SD via concatenation of self-attention features. c) Dynamics-Adapter encodes the reference image with a partially shared-weight UNet. The appearance control is realized by learning a residual in the self-attention with trainable query and output linear layers. All other components share the same frozen weight with SD.

<p align="center">
  <img src="./assets/figures/Arch_Design.png"  height=250>
</p>


https://github.com/user-attachments/assets/a4a679fd-b8e1-4f9a-ad9c-24adb0ca33eb





## 📈 Results
### Comparison
To evaluate the dynamics texture generation performance of X-Dyna in human video animation, we compare the generation results of X-Dyna with MagicPose (ReferenceNet-based method) and MimicMotion (SVD-based method). For a fair comparison, all generated videos share the same resolution of Height x Width = 896 x 512. 



https://github.com/user-attachments/assets/436a6d6c-9579-446d-831e-6ff2195147c3



https://github.com/user-attachments/assets/5369163a-d0f6-4389-baf4-b77fcd2b7527


https://github.com/user-attachments/assets/0d1f14b3-92ad-4df8-8c34-5f9185be2905



https://github.com/user-attachments/assets/566fd91c-b488-46fc-8841-6f9462b22b26


https://github.com/user-attachments/assets/ac6a8463-0684-469c-b5ba-513697c715d7







### Ablation 
To evaluate the effectiveness of the mix data training in our pipeline, we present a visualized ablation study.


https://github.com/user-attachments/assets/064a6cf7-979d-459f-aa76-32f479d09ecc






## 🎥 More Demos

<table class="center">
  <tr>
    <td><video src="https://github.com/user-attachments/assets/be26533e-c50a-47fa-be64-532480096955" width="16%"></td>
    <td><video src="https://github.com/user-attachments/assets/1b0a3807-783c-4a1a-9d3f-810c00042d90" width="16%"></td>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/dcba53ac-09f0-43d8-aa17-b0af7b663706" width="16%"></td>
    <td><video src="https://github.com/user-attachments/assets/b2a914b2-e635-40f1-98a2-27c9ee92f6e4" width="16%"></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/1c831ef0-2c5d-4713-8cd8-2c7e055c6e93" width="16%"></td>
    <td><video src="https://github.com/user-attachments/assets/25d80816-2a67-46f9-bad4-55012dfaa57e" width="16%"></td>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/998152de-501e-4c00-ac82-e0eca1f04827" width="16%"></td>
    <td><video src="https://github.com/user-attachments/assets/e31bab7f-3922-4b0e-970a-74f7f38a2494" width="16%"></td>
  </tr>
</table>


## 📜 Requirements
* An NVIDIA GPU with CUDA support is required. 
  * We have tested on a single A100 GPU.
  * In our experiment, we used CUDA 11.8.
  * **Minimum**: The minimum GPU memory required is 20GB for generating a single video (batch_size=1) of 16 frames.
  * **Recommended**: We recommend using a GPU with 80GB of memory.
* Operating system: Linux Debian 11 (bullseye)

## 🛠️ Dependencies and Installation

Clone the repository:
```shell
git clone https://github.com/Boese0601/X-Dyna
cd X-Dyna
```

### Installation Guide

We provide an `requirements.txt` file for setting up the environment.

Run the following command on your terminal:
```shell
# 1. Prepare conda environment
conda create -n xdyna python==3.10 

# 2. Activate the environment
conda activate xdyna

# 3. Install dependencies
bash env_torch2_install.sh

# I know it's a bit weird that pytorch is installed with different versions twice in that bash file, but I don't know why it doesn't work if I directly installed the final one (torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118). 
# If you managed to fix this, please open an issue and let me know, thanks. :DDDDD   
# o_O I hate environment and dependencies errors. 
```

## 🧱 Download Pretrained Models
Due to restrictions, we are not able to release the model pre-trained with in-house data. Instead, we re-train our model on public datasets, e.g. [HumanVid](https://github.com/zhenzhiwang/HumanVid), and other human video data for research use, e.g. [Pexels](https://www.pexels.com/). 

We follow the implementation details in our paper and release pretrained weights and other network modules in [this huggingface repository](https://huggingface.co/Boese0601/X-Dyna). After downloading, please put all of them under the [pretrained_weights](pretrained_weights/) folder. 

The Stable Diffusion 1.5 UNet can be found [here](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and place it under [pretrained_weights/initialization/unet_initialization/SD](pretrained_weights/initialization/unet_initialization/SD). 

Your file structure should look like this:

```bash
X-Dyna
|----...
|----pretrained_weights
  |----controlnet
    |----controlnet-checkpoint-epoch-5.ckpt
  |----controlnet_face
    |----controlnet-face-checkpoint-epoch-2.ckpt
  |----unet 
    |----unet-checkpoint-epoch-5.ckpt
  
  |----initialization
    |----controlnets_initialization
      |----controlnet
        |----control_v11p_sd15_openpose
      |----controlnet_face
        |----controlnet2
    |----unet_initialization
      |----IP-Adapter
        |----IP-Adapter
      |----SD
        |----stable-diffusion-v1-5
|----...
``` 

## Inference

### Using Command Line

```bash
cd X-Dyna

bash scripts/inference.sh
```

### More Configurations

We list some explanations of configurations below:

|        Argument        |          Default         |                Description                |
|:----------------------:|:------------------------:|:-----------------------------------------:|
|        `--gpus`        |             0            |             GPU ID for inference          |
|       `--output`       |         ./output         |     Path to save the generated video      |
|   `--test_data_file`   |  ./examples/example.json |     Path to reference and driving data    |
|        `--cfg`         |            7.5           |        Classifier-free guidance scale     |
|       `--height`       |            896           |         Height of the generated video     |
|       `--width`        |            512           |         Width of the generated video      |
|    `--infer_config`    |   ./configs/x_dyna.yaml  |     Path to inference model config file   |
|      `--neg_prompt`    |            None          |        Negative prompt for generation     |
|       `--length`       |            192           |         Length of the generated video     |
|       `--stride`       |             1            |       Stride of driving pose and video    |
|      `--save_fps`      |            15            |           FPS of the generated video      |
|     `--global_seed`    |            42            |                  Random seed              |
|  `--face_controlnet`   |           False          |      Use Face ControlNet for inference    |
|      `--cross_id`      |           False          |                Cross-Identity             |
|  `--no_head_skeleton`  |           False          |      Head skeletons are not visuliazed    |


### Alignment

Appropriate **alignment** between driving video and reference image is necessary for better generation quality. E.g. see examples below:
<br>
From left to right: Reference Image, Extracted Pose from Reference Image, Driving Video, Aligned Driving Pose.
<table class="center">
  <tr>
    <td><video src="https://github.com/user-attachments/assets/67a552b8-99d2-47bb-9de4-a2487e48bc56" width="24%"></td>
    <td><video src="https://github.com/user-attachments/assets/2ca442a6-3da8-456f-b884-ca1e44fb5dba" width="24%"></td>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/2265cfd4-34b9-4b1e-bded-4b87eae6b455" width="24%"></td>
    <td><video src="https://github.com/user-attachments/assets/6fb012d4-034b-43fb-8018-e8725b8ab2ea" width="24%"></td>
  </tr>
</table>


### Examples
We provide some examples of aligned driving videos, human poses and reference images [here](assets/). If you would like to try on your own data, please specify the paths in [this file](examples/example.json).


## 🔗 BibTeX
If you find [X-Dyna](https://arxiv.org/abs/2501.10021) useful for your research and applications, please cite X-Dyna using this BibTeX:

```BibTeX
@article{chang2025x,
  title={X-Dyna: Expressive Dynamic Human Image Animation},
  author={Chang, Di and Xu, Hongyi and Xie, You and Gao, Yipeng and Kuang, Zhengfei and Cai, Shengqu and Zhang, Chenxu and Song, Guoxian and Wang, Chao and Shi, Yichun and others},
  journal={arXiv preprint arXiv:2501.10021},
  year={2025}
}
```


## License

Our code is distributed under the Apache-2.0 license. See `LICENSE.txt` file for more information.


## Acknowledgements

We appreciate the contributions from [AnimateDiff](https://github.com/guoyww/AnimateDiff), [MagicPose](https://github.com/Boese0601/MagicDance), [MimicMotion](https://github.com/tencent/MimicMotion), [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), [MagicAnimate](https://github.com/magic-research/magic-animate), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [ControlNet](https://arxiv.org/abs/2302.05543), [HumanVid](https://github.com/zhenzhiwang/HumanVid), [I2V-Adapter](https://arxiv.org/abs/2312.16693) for their open-sourced research. We appreciate the support from <a href="https://zerg-overmind.github.io/">Quankai Gao</a>, <a href="https://xharlie.github.io/">Qiangeng Xu</a>, <a href="https://ssangx.github.io/">Shen Sang</a>, and <a href="https://tiancheng-zhi.github.io/">Tiancheng Zhi</a> for their suggestions and discussions.

## IP Statement
The purpose of this work is only for research. The images and videos used in these demos are from public sources. If there is any infringement or offense, please get in touch with us (`dichang@usc.edu`), and we will delete it in time.

