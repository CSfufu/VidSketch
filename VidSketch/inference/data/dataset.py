import decord
decord.bridge.set_bridge('torch')
import torch
from torch.utils.data import Dataset
from einops import rearrange
import random

class SingleVideoDataset(Dataset):
    def __init__(self, video_path: str, prompt: str, tokenizer, width: int = 512, height: int = 512, n_sample_frames: int = 16, sample_frame_rate: int = 1):
        self.video_path = video_path  # 单个视频的路径
        self.prompt = prompt  # 关联的文本提示
        self.width = width  # 视频帧的宽度
        self.height = height  # 视频帧的高度
        self.n_sample_frames = n_sample_frames  # 要采样的帧数
        self.sample_frame_rate = sample_frame_rate  # 采样的帧率
        self.tokenizer = tokenizer  # 用于处理提示词的 tokenizer
        
        
        #print("prompt",prompt)


        # Tokenize prompt (处理提示词，转换为token ids)
        self.prompt_ids = self.tokenizer(
            prompt, 
            max_length=self.tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).input_ids[0]  # 获取 token ids
        
        
        #print("self.prompt_ids:",self.prompt_ids)
        
    def __len__(self):
        # 数据集只有一个视频，因此长度为 1
        return 1

    def __getitem__(self, index):
        # 使用 Decord 加载视频
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)

        # 确定从视频的哪个位置开始采样
        total_frames = len(vr)
        max_start = max(0, total_frames - self.n_sample_frames * self.sample_frame_rate)
        start_idx = random.randint(0, max_start)
        
        # 根据采样帧率采样
        sample_index = list(range(start_idx, total_frames, self.sample_frame_rate))[:self.n_sample_frames]

        # 读取采样的帧
        video = vr.get_batch(sample_index)  # 获取视频帧 [T, H, W, C]

        # 转换为 [T, C, H, W] 以符合 PyTorch 格式
        video = rearrange(video, "f h w c -> f c h w")
        
        # 随机水平翻转作为数据增强
        if random.uniform(0, 1) > 0.5:
            video = torch.flip(video, dims=[3])
        
        # 返回视频帧和对应的 token ids
        return {
            "pixel_values": (video / 127.5 - 1.0),  # 归一化到 [-1, 1]
            "prompt_ids": self.prompt_ids  # 文本提示的 token ids
        }
