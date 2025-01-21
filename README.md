
# <p align="center"> Hand-drawn Sketch-Driven Video Generation: Leveraging Diffusion Model for Controlled Video Creation</p>



[![Awesome](https://awesome.re/badge.svg)](https://github.com/CSfufu/Hand-drawn-Sketch-Driven-Video-Generation-Leveraging-Diffusion-Model-for-Controlled-Video-Creation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/CSfufu/Hand-drawn-Sketch-Driven-Video-Generation-Leveraging-Diffusion-Model-for-Controlled-Video-Creation?color=green)

[[Arxiv Paper](#待更新)]&nbsp;
[[Website Page](https://tankowa.github.io/HuViDPO.github.io/)]&nbsp;
[[Google Drive](https://drive.google.com/drive/folders/1OPGiS5hzGLo8j3FFP-p9aVFlox91dYvC?usp=drive_link)]&nbsp;


<!--
[[Arxiv Paper](https://arxiv.org/abs/2310.10769)]&nbsp;
[[Website Page](https://rq-wu.github.io/projects/LAMP/index.html)]&nbsp;
[[Google Drive](https://drive.google.com/drive/folders/1hIIcpn4WGoM9wVcfbiZTD2fgCzPk7A_X?usp=drive_link)&nbsp;
[[Baidu Disk (pwd: ffsp)](https://pan.baidu.com/s/1y9L2kfUlaHVZGE6B0-vXnA)]&nbsp;
[[Colab Notebook](https://colab.research.google.com/drive/1Cw2e0VFktVjWC5zIKzv2r7D2-4NtH8xm?usp=sharing)]&nbsp;
-->

<!--
![method](assets/method.png)&nbsp;

:rocket: LAMP is a **few-shot-based** method for text-to-video generation. You only need **8~16 videos 1 GPU (> 15 GB VRAM)** for training!! Then you can generate videos with learned motion pattern.
-->

## News

- [2025/1/21] We add Google Drive link about our checkpoints and training data.
- [2025/1/21] We release our checkpoints.
- [2025/1/23] Our code is publicly available.
- [2025/1/24] We have launched our website.

## Abstract

Creating high-quality aesthetic images and video animations typically demands advanced drawing skills beyond ordinary users. While AIGC advancements have enabled automated image generation from sketches, these methods are limited to static images and cannot control video animation generation using hand-drawn sketches. To solve this problem, our method, **Sketch2Video**, is the first to enable the generation of high-quality video animations solely from any number of hand-drawn sketches and simple text prompts, thereby bridging the gap between ordinary users and artists.
Moreover, to address the diverse variations in users' drawing skills, we propose the Abstraction-Level Sketch Control Strategy, which automatically adjusts the guidance strength of sketches during the generation process.
Additionally, to tackle inter-frame inconsistency, we propose an Enhanced SparseCausal-Attention mechanism, significantly improving the spatiotemporal consistency of the generated video animations.

## Our Method
![Description of Image](image/pipeline.png)

Pipeline of our **Sketch2Video**. During the training phase, we train the SC-Attention and Temporal Attention blocks using high-quality, small-scale video datasets that we have searched for by category. This helps improve the spatiotemporal consistency of the generated video animations. During the inference stage, users only need to input their desired prompt along with any number of sketch sequences to generate high-quality video animations tailored to their needs. Specifically, the first frame is processed to generate the corresponding initial image, while the entire sketch sequence is fed into the Inflated T2I-Adapter to extract information, which is then injected into the upsampling layers of the VDM to control the video animation generation process.


## Preparation
### Dependencies and Installation
- Ubuntu > 18.04
- CUDA=11.3
- Others:

```bash
# clone the repo
git clone https://github.com/Tankowa/HuViDPO.git
cd HuViDPO

# create virtual environment
conda create -n HuViDPO python=3.8
conda activate HuViDPO

# install packages
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
pip install xformers==0.0.13
```

### Weights and Data
1. You can download pre-trained T2I diffusion models on Hugging Face.
   In our work, we use [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) as our backbone network. Clone the pretrained weights by `git-lfs` and put them in `./checkpoints`

2. Our checkpoint and training data are listed as follows. You can also collect video data by your own (Suggest websites: [pexels](https://pexels.com/), [frozen-in-time](https://meru.robots.ox.ac.uk/frozen-in-time/)) and put .mp4 files in `./train_data/videos/[motion_name]/` and `./train_data/dpo_videos/[motion_name]/`

3. You can find the training videos and our trained model weights at [[Google Drive](https://drive.google.com/drive/folders/1e409tML98gwouIOxFwcFuGVBSNwsfEtY?usp=share_link)]

4. After deploying the data, run the prepared demo and rate it according to your preferences.
   ```bash
   python give_score.py
   ```

<!--
<table class="center">
<tr>
    <td align="center"> Motion Name </td>
    <td align="center"> Checkpoint Link </td>
    <td align="center"> Training data </td>
</tr>
<tr>
    <td align="center">Birds fly</td>
    <td align="center"><a href="https://pan.baidu.com/s/1nuZVRj-xRqkHySQQ3jCFkw">Baidu Disk (pwd: jj0o)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/10fi8KoBrGJMpLQKhUIaFSQ">Baidu Disk (pwd: w96b)</a></td>
</tr>
<tr>
    <td align="center">Firework</td>
    <td align="center"><a href="https://pan.baidu.com/s/1zJnn5bZpGzChRHJdO9x6WA">Baidu Disk (pwd: wj1p)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1uIyw0Q70svWNM5z7DFYkiQ">Baidu Disk (pwd: oamp)</a></td>
</tr>
<tr>
    <td align="center">Helicopter</td>
    <td align="center"><a href="https://pan.baidu.com/s/1oj6t_VFo9cX0vTZWDq8q3w">Baidu Disk (pwd: egpe)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1MYMjIFyFTiLGEX1w0ees2Q">Baidu Disk (pwd: t4ba)</a></td>
</tr>
<tr>
    <td align="center">Horse run</td>
    <td align="center"><a href="https://pan.baidu.com/s/1lkAFZuEnot4JGruLe6pR3g">Baidu Disk (pwd: 19ld)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1z7FHN-aotdOF2MPUk4lDJg">Baidu Disk (pwd: mte7)</a></td>
</tr>
<tr>
    <td align="center">Play the guitar</td>
    <td align="center"><a href="https://pan.baidu.com/s/1uY47E08_cUofmlmKWfi46A">Baidu Disk (pwd: l4dw)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1cemrtzJtS_Lm8y8nZM9kSw">Baidu Disk (pwd: js26)</a></td>
</tr>
<tr>
    <td align="center">Rain</td>
    <td align="center"><a href="https://pan.baidu.com/s/1Cvsyg7Ld2O0DEK_U__2aXg">Baidu Disk (pwd: jomu)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1hMGrHCLNRDLJQ-4XKk6hZg">Baidu Disk (pwd: 31ug)</a></td>
</tr>
<tr>
    <td align="center">Turn to smile</td>
    <td align="center"><a href="https://pan.baidu.com/s/1UYjWncrxYiAhwpNAafH5WA">Baidu Disk (pwd: 2bkl)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1ErFSm6t-CtYBzsuzxi08dg">Baidu Disk (pwd: l984)</a></td>
</tr>
<tr>
    <td align="center">Waterfall</td>
    <td align="center"><a href="https://pan.baidu.com/s/1tWArxOw6CMceaW_49rIoSA">Baidu Disk (pwd: vpkk)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1hjlqRwa35nZ2pc2D-gIX9A">Baidu Disk (pwd: 2edp)</a></td>
</tr>
<tr>
    <td align="center">All</td>
    <td align="center"><a href="https://pan.baidu.com/s/1vRG7kMCTC7b9YUd4qsSP_A">Baidu Disk (pwd: ifsm)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1h5HrIGWP5OlMqp9gkD9cyQ">Baidu Disk (pwd: 2i2k)</a></td>
</tr>
</table>
-->

## Get Started
### 1. Training
```bash
# Fine-tune base model, you can find train videos at ./train_data/videos/[motion_name]/
CUDA_VISIBLE_DEVICES=X python train_lamp.py --config configs/smile.yaml

# Fine-tune using DPO strategy, you can find train videos at ./train_data/dpo_videos/[motion_name]/
CUDA_VISIBLE_DEVICES=X python new_train.py --config configs/smile-dpo.yaml --weights ./output/smile/diffusion_pytorch_model.bin
```

### 2. Inference
Here is an example command for inference
```bash
# Motion Pattern
CUDA_VISIBLE_DEVICES=X python inference_script_dpo.py --weight ./output/smile/diffusion_pytorch_model.bin --pretrain_weight ./checkpoints/CompVis/stable-diffusion-v1-4 --image_path ./val_data/smile --prompt_path ./val_data/smile/smile.txt --output_path ./output/smile_lora --lora_weights ./output/smile/model_weights_epoch_1.pth

#########################################################################################################
# --weight:           the path of our model(Fine-tune base model)
# --lora_weights:      the path of our model(Fine-tune using DPO strategy)    
# --pretrain_weight:  the path of the pre-trained model (e.g. SDv1.4)
# --first_frame_path: the path of the first frame generated by T2I model (e.g. DPO-XL)
# --prompt:           the input prompt, the default value is aligned with the filename of the first frame
# --output:           output path, default: ./results 
# --height:           video height, default: 320
# --width:            video width, default: 512
# --length            video length, default: 16
# --cfg:              classifier-free guidance, default: 12.5
#########################################################################################################
```


## Visual Examples
### Few-Shot-Based Text-to-Video Generation(We invite everyone to visit our official website to explore additional case studies and experiments. [[Website Page](https://tankowa.github.io/HuViDPO.github.io/)]&nbsp;)
<table class="center">
    <tr>
        <td align="center" style="width: 7%"> Firework</td>
        <td align="center">
            <img src="assets/firework/2.gif">
        </td>
        <td align="center">
            <img src="assets/firework/5.gif">
        </td>
        <td align="center">
            <img src="assets/firework/6.gif">
        </td>
    </tr>
    <tr>
        <td align="center" style="width: 7%"> Helicopter</td>
        <td align="center">
            <img src="assets/helicopter/3.gif">
        </td>
        <td align="center">
            <img src="assets/helicopter/6.gif">
        </td>
        <td align="center">
            <img src="assets/helicopter/7.gif">
        </td>
    </tr>
    <tr>
        <td align="center" style="width: 7%"> Waterfall</td>
        <td align="center">
            <img src="assets/waterfall/1.gif">
        </td>
        <td align="center">
            <img src="assets/waterfall/2.gif">
        </td>
        <td align="center">
            <img src="assets/waterfall/3.gif">
        </td>
    </tr
    <tr>
        <td align="center" style="width: 7%"> Play the guitar</td>
        <td align="center">
            <img src="assets/guitar/2.gif">
        </td>
        <td align="center">
            <img src="assets/guitar/5.gif">
        </td>
        <td align="center">
            <img src="assets/guitar/6.gif">
        </td>
    </tr>
    <tr>
        <td align="center" style="width: 7%"> Birds fly</td>
        <td align="center">
            <img src="assets/birds_fly/1.gif">
        </td>
        <td align="center">
            <img src="assets/birds_fly/5.gif">
        </td>
        <td align="center">
            <img src="assets/birds_fly/6.gif">
        </td>
    </tr>
<table>


## Citation
If you find our repo useful for your research, please cite us:
```
Coming soon！
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

<!--
## Acknowledgement
This repository is maintained by [Lifan Jiang](https://csfufu.life).
The code is built based on [LAMP](https://github.com/RQ-Wu/LAMP). Thanks for the excellent open-source code!!
-->
