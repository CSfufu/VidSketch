import decord
decord.bridge.set_bridge('torch')
import os
import numpy as np
from einops import rearrange
import random
import torch

class DPOdatasets:
    def __init__(
            self,
            video_root_1: str,
            video_root_2: str,
            path: str,
            score_1: float,
            score_2: float,
            width: int = 512,
            height: int = 320,
            n_sample_frames: int = 16,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_name_1 = video_root_1
        self.video_name_2 = video_root_2
        self.score_1 = score_1
        self.score_2 = score_2
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.path = path

    def load_video(self, video_path):
        #video_path = 'train_data/dpo_videos/horse_run/' + video_path
        #model_save_path = os.path.join(validation_data.save_path, f"model_weights_epoch_{epoch + 1}.pth")
        video_path = os.path.join(self.path,video_path)
        #print(video_path)
        vr = decord.VideoReader(video_path, width=self.width, height=self.height)
        start_idx = random.randint(0, len(vr) - self.n_sample_frames * self.sample_frame_rate - 1)
        sample_index = list(range(start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        return video

    def get_videos(self):
        # 加载传入的视频
        video_1 = self.load_video(self.video_name_1)
        video_2 = self.load_video(self.video_name_2)

        # 随机水平翻转
        if random.uniform(0, 1) > 0.5:
            video_1 = torch.flip(video_1, dims=[3])
            video_2 = torch.flip(video_2, dims=[3])

        example = {
            "video_1_pixel_values": (video_1 / 127.5 - 1.0),  # 归一化处理
            "video_2_pixel_values": (video_2 / 127.5 - 1.0),
            "score_1": self.score_1,
            "score_2": self.score_2,
        }

        return example
