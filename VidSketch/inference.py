from tkinter import Image
from inference.pipelines.pipeline_VidSketch import VidSketch
from inference.models.unet import UNet3DConditionModel
from inference.util import save_videos_grid
import torch
import cv2
from PIL import Image, ImageOps
import numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import os
import argparse
from diffusers import T2IAdapter
import cv2
import numpy as np

def calculate_abstractness_score(image_path):
    # 读取图像并转换为灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to read.")
    
    # 将图像二值化
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    
    # 检测轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 连续性分析
    total_area = 0
    total_perimeter = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area > 5 and perimeter > 10:  # 忽略小面积和短周长的轮廓，避免噪声影响
            total_area += area
            total_perimeter += perimeter

    # 动态归一化因子
    max_area = binary_img.shape[0] * binary_img.shape[1]  # 图像的最大可能面积
    max_perimeter = 2 * (binary_img.shape[0] + binary_img.shape[1])  # 图像最大边界长度
    max_score_continuity = max_area / max_perimeter  # 理论上最大连续性值

    # 连续性得分
    if total_perimeter > 0:
        continuity_score = total_area / (total_perimeter + 1e-6)  # 面积和周长的比值
    else:
        continuity_score = 0  # 如果没有轮廓，设定为0

    # 动态归一化
    continuity_score = continuity_score / max_score_continuity  # 归一化到 [0, 1]
    continuity_score = min(continuity_score, 1.0)  # 确保分数不超过 1.0

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    #print("num_labels:",num_labels)
    max_labels = 20  # 假定最大连通域数量为20
    connectivity_score = min(num_labels / max_labels, 1.0)  # 连通域归一化到 [0, 1]

    # 笔画（轮廓）数量分析
    if len(contours) == 0:  # 没有轮廓，直接认为是完全抽象
        abstract_detail = 1.0
    else:
        max_contours = 50  # 假定最大轮廓数量为50
        abstract_detail = min(1 / (len(contours) + 1), 1.0)  # 笔画数量越少，抽象度越高

    # 转换为抽象度分数
    abstract_continuity = 1 - continuity_score  # 连续性越高，抽象度越低
    abstract_connectivity = 1 - connectivity_score  # 连通域越多，抽象度越低

    # 打印调试信息
    print("Continuity Score:", continuity_score, "-> Abstract Continuity:", abstract_continuity)
    print("Connectivity Score:", connectivity_score, "-> Abstract Connectivity:", abstract_connectivity)
    print("Detail (Contours):", len(contours), "-> Abstract Detail:", abstract_detail)
    
    # 综合评分（抽象度）
    combined_score = (
        abstract_continuity * 0.4 +  # 连续性权重
        abstract_connectivity * 0.3 +  # 连通域权重
        abstract_detail * 0.3  # 笔画数量权重
    )
    
    # 最终分数已经在 [0, 1] 范围内，无需额外归一化
    return combined_score



def his_match(src, dst):
    src = src * 255.0
    dst = dst * 255.0
    src = src.astype(np.uint8)
    dst = dst.astype(np.uint8)
    res = np.zeros_like(dst)

    cdf_src = np.zeros((3, 256))
    cdf_dst = np.zeros((3, 256))
    cdf_res = np.zeros((3, 256))
    kw = dict(bins=256, range=(0, 256), density=True)
    for ch in range(3):
        his_src, _ = np.histogram(src[:, :, ch], **kw)
        hist_dst, _ = np.histogram(dst[:, :, ch], **kw)
        cdf_src[ch] = np.cumsum(his_src)
        cdf_dst[ch] = np.cumsum(hist_dst)
        index = np.searchsorted(cdf_src[ch], cdf_dst[ch], side='left')
        np.clip(index, 0, 255, out=index)
        res[:, :, ch] = index[dst[:, :, ch]]
        his_res, _ = np.histogram(res[:, :, ch], **kw)
        cdf_res[ch] = np.cumsum(his_res)
    return res / 255.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default=None, help='Path for of model weights')
    parser.add_argument('--pretrain_weight', type=str, default='./checkpoints/stable-diffusion-v1-4', help='Path for pretrained weight (SD v1.4)')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('--lora_weights', type=str, default=None, help='Output folder')
    parser.add_argument('--image_path', type=str, default=None, help='Output folder')
    parser.add_argument('--sketch_path', type=str, default=None, help='Output folder')
    parser.add_argument('--output_path', type=str, default=None, help='Output folder')
    parser.add_argument('--prompt_path', type=str, default=None, help='Output folder')
    parser.add_argument('--first_frame_path', type=str, default=None, help='The path for first frame image')
    parser.add_argument('-p', '--prompt', type=str, default=None, help='The video prompt. Default value: same to the filename of the first frame image')
    parser.add_argument('-hs', '--height', type=int, default=320, help='video height')
    parser.add_argument('-ws', '--width', type=int, default=512, help='video width')
    parser.add_argument('-l', '--length', type=int, default=9, help='video length')
    parser.add_argument('--cfg', type=float, default=12.5, help='classifier-free guidance scale')
    parser.add_argument('--editing', action="store_true", help='video editing')
    args = parser.parse_args()

    # load weights
    pretrained_model_path = args.pretrain_weight
    my_model_path = args.weight
    unet = UNet3DConditionModel.from_pretrained('/'.join(my_model_path.split('/')[:-1]), subfolder=my_model_path.split('/')[-1], torch_dtype=torch.float16).to('cuda')
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to('cuda')
    
    
    if args.editing:
        ddim_inv_latent = torch.load(f"{'/'.join(my_model_path.split('/')[:-1])}/inv_latents/ddim_latent-500.pt").to(torch.float16)
    else:
        ddim_inv_latent = None

    # build pipeline
    #unet.enable_xformers_memory_efficient_attention()
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to('cuda')
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    generator = torch.Generator(device='cuda')
    
    adapter_name = "/data/jianglifan/TencentARC/t2iadapter_sketch_sd15v2"
    adapter = T2IAdapter.from_pretrained(adapter_name, torch_dtype=torch.float16).to('cuda')
    
    pipe = VidSketch(vae=vae, text_encoder=text_encoder, 
                   tokenizer=tokenizer, unet=unet, 
                   scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
                   ).to("cuda")
    
    pipe.enable_vae_slicing()

    for name, param in unet.named_parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
            print(f"Converted {name} to float32")    

    # 定义要处理的文件夹和文件路径
    prompt = args.prompt
    
    # 指定sketch序列对应的文件夹地址
    folder_path = args.sketch_path
    
    # 初始化一个空列表来存储图片
    images = []
    total_score = 0
    i=0
    # 遍历文件夹中的所有文件
    for filename in sorted(os.listdir(folder_path)):
        #print("filename",filename)
        # 检查文件扩展名是否是图片格式
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 打开图片并进行处理
            image_path = os.path.join(folder_path, filename)
            i=i+1
            score = calculate_abstractness_score(image_path)
            #print(f"Abstractness Score: {score:.2f}")
            total_score+=score
            
            sketch = Image.open(image_path).convert("L")
            sketch = ImageOps.invert(sketch)
            sketch = sketch.resize((512, 320))
            
            # 将处理后的图片存入列表
            images.append(sketch)
            
    final_score = total_score / i
    print("final_score:",final_score)
    
    idx=0
    #用T2I生成好的首帧图像的文件夹路径
    img_fold_path = args.image_path
    #生成视频的保存文件夹地址
    output_folder = args.output_path
    
    for img_path in sorted(os.listdir(img_fold_path)):
        
        full_img_path = os.path.join(img_fold_path, img_path)
        image = cv2.imread(full_img_path)
        image = cv2.resize(image, (512, 320))[:, :, ::-1]
        first_frame_latents = torch.Tensor(image.copy()).to('cuda').type(torch.float16).permute(2, 0, 1).repeat(1, 1, 1, 1)
        first_frame_latents = first_frame_latents / 127.5 - 1.0
        first_frame_latents = vae.encode(first_frame_latents).latent_dist.sample() * 0.18215
        first_frame_latents = first_frame_latents.repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)

        #adapter_conditioning_tau,值越小：适配器的指导力度衰减更快，越早停止对生成过程的影响。值越大：适配器的指导作用在扩散生成的更多步骤中持续存在。
        #adapter_conditioning_scale,值越小：适配器作用较弱，生成结果自由度更高，与草图的匹配度较低。值越大：适配器作用较强，生成结果紧贴草图，但可能导致自由度下降。
        #抽象度评分 (abstractness_score)	adapter_conditioning_tau	adapter_conditioning_scale
        #高抽象度 (接近 1)	0.65 - 0.67	0.80 - 0.81
        #低抽象度 (接近 0)	0.80 - 0.85	0.85 - 0.90
        
        # (3) guidance_scale：
        #     作用：引导条件与生成内容的相关性，值越大越倾向于条件描述。
        #     调整逻辑：
        #     抽象度高的草图：降低 guidance_scale，给模型更多自由生成空间。
        #     抽象度低的草图：适当增加 guidance_scale，确保生成内容更贴合草图主题。
        
        
        #抽象草图其值要调小
        # 视频生成0.65 0.8
        video = pipe(prompt, adapter_conditioning_tau=0.5, adapter_conditioning_scale=0.65, 
                    latents=first_frame_latents,adapter=adapter, image =images, generator=generator, 
                    video_length=args.length, height=args.height, width=args.width, num_inference_steps=50, 
                    guidance_scale=args.cfg, use_inv_latent=False, num_inv_steps=50, ddim_inv_latent=ddim_inv_latent
                    )['videos']
    

        # 对生成的视频帧进行颜色匹配
        for f in range(1, video.shape[2]):
            former_frame = video[0, :, 0, :, :].permute(1, 2, 0).cpu().numpy()
            frame = video[0, :, f, :, :].permute(1, 2, 0).cpu().numpy()
            result = his_match(former_frame, frame)
            result = torch.Tensor(result).type_as(video).to(video.device)
            video[0, :, f, :, :] = result.permute(2, 0, 1)

        idx = idx+1
        save_path = os.path.join(output_folder, f"answer_{idx}.gif")
        
        # 保存视频
        save_videos_grid(video, save_path)
        
        # 打印输出路径
        print(f"Saved video {idx+1} at {save_path}")

if __name__ == '__main__':
    main()