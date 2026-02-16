# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import shutil
import imageio
import subprocess
import numpy as np
from typing import Dict, List, Optional, Union
from PIL import Image, ImageDraw

import torch
import torchvision
import torch.distributed as dist
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from safetensors import safe_open
from tqdm import tqdm
from einops import rearrange
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora, convert_motion_lora_ckpt_to_diffusers
from packaging import version as pver

def imread_resize(path, h_max, w_max):
    img = cv2.imread(path)[..., ::-1]
    img = resize_image(img, min(h_max, w_max))
    return img

import cv2


    
def first_video_frame_read_resize_crop(video_path, target_height, target_width):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None
    
    # Read the first frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    if not ret:
        print(f"Error reading the first frame from: {video_path}")
        return None

    # Convert the frame from BGR to RGB
    img = frame[..., ::-1]
    
    # Calculate the aspect ratio of the target dimensions
    target_aspect_ratio = target_width / target_height

    # Get the current dimensions of the image
    h, w, _ = img.shape
    current_aspect_ratio = w / h

    # Determine the cropping dimensions to maintain the aspect ratio
    if current_aspect_ratio > target_aspect_ratio:
        # Image is wider than the target aspect ratio
        new_width = int(target_aspect_ratio * h)
        offset = (w - new_width) // 2
        img = img[:, offset:offset + new_width]
    else:
        # Image is taller than the target aspect ratio
        new_height = int(w / target_aspect_ratio)
        offset = (h - new_height) // 2
        img = img[offset:offset + new_height, :]

    # Resize the image to the target dimensions
    img = cv2.resize(img, (target_width, target_height))

    return img


def imread_resize_crop(path, target_height, target_width):
    # Load the image
    img = cv2.imread(path)[..., ::-1]

    # Calculate the aspect ratio of the target dimensions
    target_aspect_ratio = target_width / target_height

    # Get the current dimensions of the image
    h, w, _ = img.shape
    current_aspect_ratio = w / h

    # Determine the cropping dimensions to maintain the aspect ratio
    if current_aspect_ratio > target_aspect_ratio:
        # Image is wider than the target aspect ratio
        new_width = int(target_aspect_ratio * h)
        offset = (w - new_width) // 2
        img = img[:, offset:offset + new_width]
    else:
        # Image is taller than the target aspect ratio
        new_height = int(w / target_aspect_ratio)
        offset = (h - new_height) // 2
        img = img[offset:offset + new_height, :]

    # Resize the image to the target dimensions
    img = cv2.resize(img, (target_width, target_height))

    return img

def load_images_from_video_to_pil(vid_path, target_size=(512, 512), start_id = 0, vid_len = None, stride = 1):
    images = []
    images_depth = []
    if vid_len is None:
        reader = imageio.get_reader(vid_path)
        vid_len = len(reader)
        print(f"Read {vid_len} frames from video {vid_path}.")
    if vid_len <= 0:
        raise ValueError('vid length should be larger than 0')
    if vid_len is not None:
        reader = imageio.get_reader(vid_path)
    # fps = reader.get_meta_data()['fps']
    cur_vid_len = 0
    if stride == 1:
        for img in reader:
            if cur_vid_len < start_id:
                cur_vid_len += 1
                continue
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            images.append(Image.fromarray(img.clip(0, 255).astype(np.uint8)))
            cur_vid_len += 1
            if cur_vid_len >= start_id + vid_len:
                break
    else:
        # Iterate over the video frames
        for i, img in enumerate(reader):
            if i % stride == 0:  # Start from the second frame (i=1) and stride by 4
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                images.append(Image.fromarray(img.clip(0, 255).astype(np.uint8)))
                cur_vid_len += 1
            if cur_vid_len < start_id:
                cur_vid_len += 1
                continue
            if cur_vid_len >= start_id + vid_len:
                break
    reader.close()

    return images

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)




def save_videos_grid_mp4(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    """
    Save a grid of videos as an MP4 file using MoviePy.

    Args:
        videos (torch.Tensor): Tensor of shape (b, c, t, h, w).
        path (str): Path to save the MP4 file.
        rescale (bool): Whether to rescale from [-1, 1] to [0, 1].
        n_rows (int): Number of rows in the video grid.
        fps (int): Frames per second for the output video.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    frames = []

    # Process frames
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        frames.append(x)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save as MP4 using MoviePy
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(
        path, 
        codec="libx264",
        fps=fps,
        # bitrate="50000k",        # High bitrate for better quality
        # preset="veryslow",       # Optimize for quality
        # audio_codec="aac",       # Audio codec for MP4 compatibility
        # ffmpeg_params=["-crf", "18"]  # Near-lossless compression
    )

    # print(f"Video saved to {path}")

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps, loop=0)

def imread_resize(path, h_max, w_max):
    img = cv2.imread(path)[..., ::-1]
    img = resize_image(img, min(h_max, w_max))
    return img

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H))
    return img

def match_histogram(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.
    """
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # Get the set of unique pixel values and their corresponding indices and counts.
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # Calculate s_k for histogram matching.
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Interpolate pixel values.
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def color_match_frames(reference_image, subsequent_frame):
    # import ipdb; ipdb.set_trace()
    reference_image = (reference_image * 255.0).astype(np.uint8)
    subsequent_frame = (subsequent_frame * 255.0).astype(np.uint8)
    # Convert to LAB color space
    reference_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)
    subsequent_lab = cv2.cvtColor(subsequent_frame, cv2.COLOR_BGR2LAB)

    # Split the LAB channels
    ref_l, ref_a, ref_b = cv2.split(reference_lab)
    sub_l, sub_a, sub_b = cv2.split(subsequent_lab)

    # Match the a and b channels
    matched_a = match_histogram(sub_a, ref_a).astype(np.uint8)
    matched_b = match_histogram(sub_b, ref_b).astype(np.uint8)

    # Combine the L channel from the subsequent frame with matched a and b channels
    matched_lab = cv2.merge((sub_l, matched_a, matched_b))

    # Convert back to BGR color space
    matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
    matched_bgr = matched_bgr / 255.0

    # Save or return the matched image
    return matched_bgr

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


def listdir(path: str) -> List[str]:
    """
    List directory. Supports either hdfs or local path. Returns full path.

    Examples:
        - listdir("hdfs://dir") -> ["hdfs://dir/file1", "hdfs://dir/file2"]
        - listdir("/dir") -> ["/dir/file1", "/dir/file2"]
    """
    files = []

    if path.startswith('hdfs://'):
        pipe = subprocess.Popen(
            args=["hdfs", "dfs", "-ls", path],
            shell=False,
            stdout=subprocess.PIPE)

        for line in pipe.stdout:
            parts = line.strip().split()

            # drwxr-xr-x   - user group  4 file
            if len(parts) < 5:
                continue

            files.append(parts[-1].decode("utf8"))

        pipe.stdout.close()
        pipe.wait()

    else:
        files = [os.path.join(path, file) for file in os.listdir(path)]

    return files