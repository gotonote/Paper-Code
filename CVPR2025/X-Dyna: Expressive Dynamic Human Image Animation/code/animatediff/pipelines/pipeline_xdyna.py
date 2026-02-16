# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

# Copyright (c) 2023 Tune-A-Video Authors.
# Copyright 2024 ByteDance and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by Di Chang on 11/16/2024.
#
# Original file was released under Apache License, Version 2.0, with the full license text
# available at https://github.com/showlab/Tune-A-Video/blob/main/LICENSE
#
# This modified file is released under the same license.

import imageio
import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
import inspect, math
import random
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.transforms import GaussianBlur

from animatediff.models.image_processor import VaeImageProcessor
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput
from animatediff.utils.context import (
    get_context_scheduler,
    get_total_steps
)
from einops import rearrange
from ..models.unet import UNet3DConditionModel
from ..models.controlnet import ControlNetModel
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class XDynaPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class XDynaPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        controlnet: ControlNetModel,
        controlnet_face: ControlNetModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            controlnet_face=controlnet_face,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        if controlnet is None:
            self.use_controlnet = False
        else:
            self.use_controlnet = True
        if controlnet_face is None:
            self.use_face_controlnet = False
        else:
            self.use_face_controlnet = True
            
    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            concat_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return uncond_embeddings, text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
    
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None, do_classifier_free_guidance=True):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
 
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents
        
    def prepare_condition(self, condition, num_videos_per_prompt, device, dtype, do_classifier_free_guidance):
        # prepare conditions for controlnet
        condition = condition.to(device=device, dtype=dtype)
        condition = (condition-condition.min())/(condition.max()-condition.min())
        condition = torch.stack([condition for _ in range(num_videos_per_prompt)], dim=0)
        condition = rearrange(condition, 'b f c h w -> (b f) c h w').clone()
        if do_classifier_free_guidance:
            condition = torch.cat([condition] * 2)
            # condition = torch.cat([torch.zeros_like(condition), condition])
        return condition
    
    def downgrade_input(self, init_latents, generator, device, dtype, video_length=16):
        mask = (
            torch.rand(
                *init_latents.shape,
                generator=generator,
                device=device,
            )
            >= 0.5 # 0.8
        ).to(dtype=dtype)

        blur = GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))
        first_frame_latents = init_latents[:, :, 0:1, :, :].clone()
        init_latents = rearrange(init_latents, "b c f h w -> (b f) c h w")
        init_latents_blur = blur(init_latents)
        init_latents = rearrange(init_latents, "(b f) c h w -> b c f h w", f=video_length)
        init_latents_blur = rearrange(init_latents_blur, "(b f) c h w -> b c f h w", f=video_length)
        init_latents = init_latents * mask + init_latents_blur * (1 - mask)
        init_latents[:, :, 0:1, :, :] = first_frame_latents

        return init_latents

    def select_controlnet_res_samples(self, controlnet_res_samples_cache_dict, context, do_classifier_free_guidance, use_video_controlnet, b, f):
        _down_block_res_samples = []
        _mid_block_res_sample = []
        for i in np.concatenate(np.array(context)):
            _down_block_res_samples.append(controlnet_res_samples_cache_dict[i][0])
            _mid_block_res_sample.append(controlnet_res_samples_cache_dict[i][1])
        down_block_res_samples = [[] for _ in range(len(controlnet_res_samples_cache_dict[i][0]))]
        for res_t in _down_block_res_samples:
            for i, res in enumerate(res_t):
                down_block_res_samples[i].append(res)
        down_block_res_samples = [torch.cat(res) for res in down_block_res_samples]
        mid_block_res_sample = torch.cat(_mid_block_res_sample)
        
        # reshape controlnet output to match the unet3d inputs
        b = b // 2 if do_classifier_free_guidance else b
        _down_block_res_samples = []
        for sample in down_block_res_samples:
            sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=f)
            if do_classifier_free_guidance:
                sample = sample.repeat(2, 1, 1, 1, 1)
            _down_block_res_samples.append(sample)
        down_block_res_samples = _down_block_res_samples
        mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', b=b, f=f)
        if do_classifier_free_guidance:
            mid_block_res_sample = mid_block_res_sample.repeat(2, 1, 1, 1, 1)
            
        return down_block_res_samples, mid_block_res_sample

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        controlnet_condition: list = None,
        controlnet_conditioning_scale: float = 1.0,
        context_frames: int = 16,
        context_stride: int = 1,
        context_overlap: int = 4,
        context_batch_size: int = 1, 
        context_schedule: str = "uniform",
        noise_mode="iid",
        sface_global_condition=None, # s-face
        sface_local_condition=None, # s-face
        repeat_latents=False,
        gaussian_blur=False,
        use_input_cat=False,
        un_cond_image_embeds=None,
        cond_image_embeds=None,
        adzero=False,
        use_controlnet: bool =True,
        mask_controlnet_output=False,
        cross_id=False,
        controlnet_use_cotracker=False,
        cotracker_condition=None,
        pose_embedding=None,
        **kwargs,
    ):
        if cross_id:
            video_length += 1 # for reference first frame
        
        if self.use_face_controlnet:
            controlnet_face = self.controlnet_face
        else:
            controlnet_face = None
        
        if use_controlnet:
            controlnet = self.controlnet
        else:
            controlnet = None


        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        # device = self._execution_device
        device = latents.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale >= 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        uncond_text_embeddings, text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        

        uncond_embeddings = torch.cat([uncond_text_embeddings, un_cond_image_embeds], dim=1)
        cond_embeddings = torch.cat([text_embeddings, cond_image_embeds], dim=1)
        mixed_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

        if controlnet is not None:
            controlnet_condition = self.image_processor.preprocess(controlnet_condition, height=height, width=width)
            if controlnet_use_cotracker and cotracker_condition is not None:
                cotracker_condition = self.image_processor.preprocess(cotracker_condition, height=height, width=width)
                controlnet_condition = torch.cat([controlnet_condition,cotracker_condition],dim=1)
            controlnet_condition = self.prepare_condition(
                    condition=controlnet_condition,
                    device=device,
                    dtype=controlnet.dtype,
                    num_videos_per_prompt=num_videos_per_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )
            # breakpoint()
            controlnet_condition = controlnet_condition.to(device, latents.dtype)
            if do_classifier_free_guidance:
                controlnet_uncond_images, controlnet_cond_images = controlnet_condition.chunk(2)
            else:
                controlnet_uncond_images = controlnet_condition
                controlnet_cond_images = controlnet_condition

        if controlnet_face is not None: 

            controlnet_face_global_condition = self.image_processor.preprocess(sface_global_condition, height=height, width=width)

            controlnet_face_global_condition = self.prepare_condition(
                    condition=controlnet_face_global_condition,
                    device=device,
                    dtype=controlnet_face_global_condition.dtype,
                    num_videos_per_prompt=num_videos_per_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )
            controlnet_face_global_condition = controlnet_face_global_condition.to(device, latents.dtype)

            controlnet_face_local_condition = self.image_processor.preprocess(sface_local_condition, height=height, width=width)

            controlnet_face_local_condition = self.prepare_condition(
                    condition=controlnet_face_local_condition,
                    device=device,
                    dtype=controlnet_face_local_condition.dtype,
                    num_videos_per_prompt=num_videos_per_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )
            controlnet_face_local_condition = controlnet_face_local_condition.to(device, latents.dtype)
            controlnet_face_condition = torch.cat([controlnet_face_local_condition, controlnet_face_global_condition],dim=1)
            if do_classifier_free_guidance:
                controlnet_face_uncond_images, controlnet_face_cond_images = controlnet_face_condition.chunk(2)
            else:
                controlnet_face_uncond_images = controlnet_face_condition
                controlnet_face_cond_images = controlnet_face_condition


        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        if not repeat_latents and not use_input_cat and not adzero:
            noisy_latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
            )
            first_frame_latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                1,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents[:, :, -1:, :, :]
            )
            if noise_mode == "lamp":
                print("using lamp noise")
                for f in range(1, video_length):
                    base_ratio = 0.2
                    noisy_latents[:, :, f:f+1, :, :] = (base_ratio) * noisy_latents[:, :, 0:1, :, :] + (1-base_ratio) * noisy_latents[:, :, f:f+1, :, :]
            elif noise_mode == "mixed":
                print("using mixed noise")
                base_ratio = 1
                base_noise = noisy_latents[:, :, 0:1, :, :]
                for f in range(1, video_length):
                    noisy_latents[:, :, f:f+1, :, :] = np.sqrt(np.square(base_ratio) / (1 + np.square(base_ratio))) * base_noise + np.sqrt(1 / (1 + np.square(base_ratio))) * noisy_latents[:, :, f:f+1, :, :]
            elif noise_mode == "progressive":
                print("using progressive noise")
                base_ratio = 2
                for f in range(1, video_length):
                    noisy_latents[:, :, f:f+1, :, :] = np.sqrt(np.square(base_ratio) / (1 + np.square(base_ratio))) * noisy_latents[:, :, f-1:f, :, :] + np.sqrt(1 / (1 + np.square(base_ratio))) * noisy_latents[:, :, f:f+1, :, :]
            elif noise_mode == "iid":
                print("using iid noise")
            else:
                raise ValueError(f"Unknown noise mode: {noise_mode}")
            noisy_latents[:, :, 0:1, :, :] = first_frame_latents
            latents = noisy_latents
        # channel concat latents and add noise
        else:
            first_frame_latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                1,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents[:, :, -1:, :, :],
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            repeated_latents = first_frame_latents.repeat(1, 1, video_length, 1, 1)
            if gaussian_blur:
                repeated_latents = self.downgrade_input(repeated_latents, generator, device, text_embeddings.dtype, video_length)
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(repeated_latents)
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = []
            for i in range(video_length):
                noisy_latents.append(self.scheduler.add_noise(repeated_latents[:, :, i:i+1, :, :], noise[:, :, i:i+1, :, :], timesteps[:1]))
            noisy_latents = torch.cat(noisy_latents, dim=2)
            # first frame do not add any noise
            noisy_latents[:, :, 0:1, :, :] = first_frame_latents
            latents = noisy_latents
        
        latents_dtype = latents.dtype
        if do_classifier_free_guidance:
            if cross_id:
                controlnet_text_embeddings_c = text_embeddings.repeat_interleave(video_length-1, 0)
                controlnet_text_embeddings_uc = uncond_text_embeddings.repeat_interleave(video_length-1, 0)
            else:
                raise NotImplementedError
        else:
            if cross_id:
                controlnet_text_embeddings_c = text_embeddings.repeat_interleave(video_length-1, 0)
                controlnet_text_embeddings_uc = text_embeddings.repeat_interleave(video_length-1, 0)
            else:
                raise NotImplementedError
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        context_scheduler = get_context_scheduler(context_schedule)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = torch.zeros(
                    (latents.shape[0] * (2 if do_classifier_free_guidance else 1), *latents.shape[1:]),
                    device=latents.device,
                    dtype=latents.dtype,
                    )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents.dtype
                    )

                offset = random.randint(0, context_frames-1)
                context_queue = list(context_scheduler(
                    0, num_inference_steps, latents.shape[2]-1, context_frames, context_stride, context_overlap,True,offset
                    ))
                num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                global_context = []
                for i in range(num_context_batches):
                    global_context.append(context_queue[i*context_batch_size: (i+1)*context_batch_size])
                for context in global_context:
                    assert len(context) == 1
                    # expand the latents if we are doing classifier free guidance
                    ref_latents = latents[:,:,:1,:,:].clone()
                    denoising_latents = latents[:,:,1:,:,:].clone()

                    selected_latents = torch.cat([denoising_latents[:, :, context[0]]])
                    latent_model_input = (
                            torch.cat([ref_latents, selected_latents],dim=2)
                            .to(device)
                            .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                        )


                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    if use_controlnet:
                        b, c, f, h, w = latent_model_input.shape
                        if cross_id:
                            clip_leng = f - 1 # 16
                            controlnet_latent_input = rearrange(latent_model_input[:,:,1:,:,:], "b c f h w -> (b f) c h w")  
                        else:
                            controlnet_latent_input = rearrange(latent_model_input, "b c f h w -> (b f) c h w")  
                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            controlnet_latent_input,
                            t,
                            encoder_hidden_states=torch.cat([controlnet_text_embeddings_uc[context], controlnet_text_embeddings_c[context]]) if do_classifier_free_guidance else controlnet_text_embeddings_c[context],
                            controlnet_cond=torch.cat([controlnet_uncond_images[context], controlnet_cond_images[context]]) if do_classifier_free_guidance else controlnet_cond_images[context],
                            conditioning_scale=controlnet_conditioning_scale,
                            return_dict=False,
                        )


                        if (controlnet_face is not None):  # s-face controlnet 
                            down_block_res_samples_face, mid_block_res_sample_face = self.controlnet_face(
                                    controlnet_latent_input,
                                    t,
                                    encoder_hidden_states=torch.cat([controlnet_text_embeddings_uc[context], controlnet_text_embeddings_c[context]]) if do_classifier_free_guidance else controlnet_text_embeddings_c[context],
                                    controlnet_cond=torch.cat([controlnet_face_uncond_images[context], controlnet_face_cond_images[context]]) if do_classifier_free_guidance else controlnet_face_cond_images[context],
                                    conditioning_scale=controlnet_conditioning_scale,
                                    return_dict=False,
                            )  
                        # reshape controlnet output to match the unet3d inputs
                        _down_block_res_samples = []

                        if self.controlnet_face is not None:
                            for sample, sample_face in zip(down_block_res_samples, down_block_res_samples_face):
                                sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=clip_leng)
                                sample_face = rearrange(sample_face, '(b f) c h w -> b c f h w', b=b, f=clip_leng)
                                B, C, Frame, H, W = sample.shape
                                if cross_id:
                                    sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), sample],dim=2) # b c 17 h w
                                    sample_face = torch.cat([torch.zeros(B,C,1,H,W).to(sample_face.device,sample_face.dtype), sample_face],dim=2) # b c 17 h w
                                    sample_sum = sample + sample_face
                                if mask_controlnet_output:
                                    sample_sum = torch.zeros_like(sample_sum)
                                _down_block_res_samples.append(sample_sum)
                            down_block_res_samples = _down_block_res_samples
                            mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', b=b, f=clip_leng)
                            mid_block_res_sample_face = rearrange(mid_block_res_sample_face, '(b f) c h w -> b c f h w', b=b, f=clip_leng)
                            B, C, Frame, H, W = mid_block_res_sample.shape
                            if cross_id:
                                mid_block_res_sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), mid_block_res_sample],dim=2) # b c 17 h w
                                mid_block_res_sample_face = torch.cat([torch.zeros(B,C,1,H,W).to(sample_face.device,sample_face.dtype), mid_block_res_sample_face],dim=2) # b c 17 h w
                                mid_block_res_sample += mid_block_res_sample_face
                            if mask_controlnet_output:
                                mid_block_res_sample = torch.zeros_like(mid_block_res_sample)
                        else:
                            for sample in down_block_res_samples:
                                sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=clip_leng)
                                B, C, Frame, H, W = sample.shape
                                if cross_id:
                                    sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), sample],dim=2) # b c 17 h w
                                if mask_controlnet_output:
                                    sample = torch.zeros_like(sample)
                                _down_block_res_samples.append(sample)
                            down_block_res_samples = _down_block_res_samples
                            mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', b=b, f=clip_leng)
                            B, C, Frame, H, W = mid_block_res_sample.shape
                            if cross_id:
                                mid_block_res_sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), mid_block_res_sample],dim=2) # b c 17 h w
                            if mask_controlnet_output:
                                mid_block_res_sample = torch.zeros_like(mid_block_res_sample)
                        # predict the noise residual
                        pred = self.unet(latent_model_input, t, 
                        encoder_hidden_states=mixed_embeddings, 
                        down_block_additional_residuals=down_block_res_samples, 
                        mid_block_additional_residual=mid_block_res_sample
                        ).sample.to(dtype=latents_dtype)
                    
                    
                    
                    else:
                        pred = self.unet(latent_model_input, t, encoder_hidden_states=mixed_embeddings).sample.to(dtype=latents_dtype)
                    if do_classifier_free_guidance:
                        pred_uc, pred_c = pred.chunk(2)
                    else:
                        pred_c = pred
                    pred = torch.cat([pred_uc, pred_c]).to(noise_pred.dtype)
                    # for j, c in enumerate(context):
                    assert len(context) == 1
                    new_context = [x + 1 for x in context[0]]
                    noise_pred[:, :, new_context] = noise_pred[:, :, new_context] + pred[:, :, 1:]
                    counter[:, :, new_context] = counter[:, :, new_context] + 1
                    counter[:, :, 0] += 1
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents[:, :, 1:, :, :] = self.scheduler.step(noise_pred[:, :, 1:, :, :], t, latents[:, :, 1:, :, :], **extra_step_kwargs).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        if cross_id:
            latents = latents[:, :, 1:, :, :]
        else:
            raise NotImplementedError


        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return XDynaPipelineOutput(videos=video)
