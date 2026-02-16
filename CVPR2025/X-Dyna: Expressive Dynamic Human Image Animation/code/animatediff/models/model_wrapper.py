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

import torch
from torch import nn
from einops import rearrange
class ModelWrapper(nn.Module):
    def __init__(self, unet, controlnet, controlnet_xbody=None, face_image_proj_model=None):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.controlnet_xbody = controlnet_xbody
        self.face_image_proj_model = face_image_proj_model

    def forward(self,
            timesteps,
            noisy_latents, 
            unet_encoder_hidden_states, 
            encoder_hidden_states,
            controlnet_condition,
            controlnet_xbody_condition=None,
            face_emb=None,
            conditioning_scale=1.0,
            return_dict=False,
            cross_id=False,
        ):
        if (face_emb is not None) and (self.face_image_proj_model is not None): # use face ip
            face_tokens = self.face_image_proj_model(face_emb)  
            unet_encoder_hidden_states = torch.cat([unet_encoder_hidden_states, face_tokens], dim=1)
        b, c, f, h, w = noisy_latents.shape
        if cross_id:
            f = f - 1 # 16
            controlnet_latent_input = rearrange(noisy_latents[:,:,1:,:,:], "b c f h w -> (b f) c h w")  
        else:
            controlnet_latent_input = rearrange(noisy_latents, "b c f h w -> (b f) c h w")  
        down_block_res_samples, mid_block_res_sample = self.controlnet(
                controlnet_latent_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_condition,
                conditioning_scale=conditioning_scale,
                return_dict=return_dict,
        )
        if (controlnet_xbody_condition is not None) and (self.controlnet_xbody is not None):
            down_block_res_samples_xbody, mid_block_res_sample_xbody = self.controlnet_xbody(
                    controlnet_latent_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_xbody_condition,
                    conditioning_scale=conditioning_scale,
                    return_dict=return_dict,
            )            
        # reshape controlnet output to match the unet3d inputs
        _down_block_res_samples = []
        if (controlnet_xbody_condition is not None) and (self.controlnet_xbody is not None):
            for sample, sample_xbody in zip(down_block_res_samples, down_block_res_samples_xbody):
                sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=f)
                sample_xbody = rearrange(sample_xbody, '(b f) c h w -> b c f h w', b=b, f=f)
                B, C, Frame, H, W = sample.shape
                if cross_id:
                    sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), sample],dim=2) # b c 17 h w
                    sample_xbody = torch.cat([torch.zeros(B,C,1,H,W).to(sample_xbody.device,sample_xbody.dtype), sample_xbody],dim=2) # b c 17 h w
                    sample_sum = sample + sample_xbody
                _down_block_res_samples.append(sample_sum)
            down_block_res_samples = _down_block_res_samples
            mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', b=b, f=f)
            mid_block_res_sample_xbody = rearrange(mid_block_res_sample_xbody, '(b f) c h w -> b c f h w', b=b, f=f)
            B, C, Frame, H, W = mid_block_res_sample.shape
            if cross_id:
                mid_block_res_sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), mid_block_res_sample],dim=2) # b c 17 h w
                mid_block_res_sample_xbody = torch.cat([torch.zeros(B,C,1,H,W).to(sample_xbody.device,sample_xbody.dtype), mid_block_res_sample_xbody],dim=2) # b c 17 h w
                mid_block_res_sample += mid_block_res_sample_xbody
        else:
            for sample in down_block_res_samples:
                sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=f)
                B, C, Frame, H, W = sample.shape
                if cross_id:
                    sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), sample],dim=2) # b c 17 h w
                _down_block_res_samples.append(sample)
            down_block_res_samples = _down_block_res_samples
            mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', b=b, f=f)
            B, C, Frame, H, W = mid_block_res_sample.shape
            if cross_id:
                mid_block_res_sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), mid_block_res_sample],dim=2) # b c 17 h w
    
        model_pred = self.unet(noisy_latents, timesteps, unet_encoder_hidden_states, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample).sample
        
        return model_pred

# Copied from CameraCtrl
class PoseAdaptor(nn.Module):
    def __init__(self, unet, pose_encoder):
        super().__init__()
        self.unet = unet
        self.pose_encoder = pose_encoder

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, pose_embedding):
        assert pose_embedding.ndim == 5
        bs = pose_embedding.shape[0]            # b c f h w
        pose_embedding_features = self.pose_encoder(pose_embedding)      # bf c h w
        pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                   for x in pose_embedding_features]
        noise_pred = self.unet(noisy_latents,
                               timesteps,
                               encoder_hidden_states,
                               pose_embedding_features=pose_embedding_features).sample
        return noise_pred


class ModelWrapper_Camera(nn.Module):
    def __init__(self, unet, controlnet, pose_encoder, controlnet_xbody=None):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.pose_encoder = pose_encoder
        self.controlnet_xbody = controlnet_xbody


    def forward(self,
            timesteps,
            noisy_latents, 
            unet_encoder_hidden_states, 
            encoder_hidden_states,
            controlnet_condition,
            pose_embedding,
            controlnet_xbody_condition=None,
            conditioning_scale=1.0,
            return_dict=False,
            cross_id=False,
        ):

        assert pose_embedding.ndim == 5
        bs = pose_embedding.shape[0]            # b c f h w
        pose_embedding_features = self.pose_encoder(pose_embedding)      # bf c h w
        pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                   for x in pose_embedding_features]

        b, c, f, h, w = noisy_latents.shape
        if cross_id:
            f = f - 1 # 16
            controlnet_latent_input = rearrange(noisy_latents[:,:,1:,:,:], "b c f h w -> (b f) c h w")  
        else:
            controlnet_latent_input = rearrange(noisy_latents, "b c f h w -> (b f) c h w")  
        down_block_res_samples, mid_block_res_sample = self.controlnet(
                controlnet_latent_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_condition,
                conditioning_scale=conditioning_scale,
                return_dict=return_dict,
        )
        if (controlnet_xbody_condition is not None) and (self.controlnet_xbody is not None):
            down_block_res_samples_xbody, mid_block_res_sample_xbody = self.controlnet_xbody(
                    controlnet_latent_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_xbody_condition,
                    conditioning_scale=conditioning_scale,
                    return_dict=return_dict,
            )            
        # reshape controlnet output to match the unet3d inputs
        _down_block_res_samples = []
        if (controlnet_xbody_condition is not None) and (self.controlnet_xbody is not None):
            for sample, sample_xbody in zip(down_block_res_samples, down_block_res_samples_xbody):
                sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=f)
                sample_xbody = rearrange(sample_xbody, '(b f) c h w -> b c f h w', b=b, f=f)
                B, C, Frame, H, W = sample.shape
                if cross_id:
                    sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), sample],dim=2) # b c 17 h w
                    sample_xbody = torch.cat([torch.zeros(B,C,1,H,W).to(sample_xbody.device,sample_xbody.dtype), sample_xbody],dim=2) # b c 17 h w
                    sample_sum = sample + sample_xbody
                _down_block_res_samples.append(sample_sum)
            down_block_res_samples = _down_block_res_samples
            mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', b=b, f=f)
            mid_block_res_sample_xbody = rearrange(mid_block_res_sample_xbody, '(b f) c h w -> b c f h w', b=b, f=f)
            B, C, Frame, H, W = mid_block_res_sample.shape
            if cross_id:
                mid_block_res_sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), mid_block_res_sample],dim=2) # b c 17 h w
                mid_block_res_sample_xbody = torch.cat([torch.zeros(B,C,1,H,W).to(sample_xbody.device,sample_xbody.dtype), mid_block_res_sample_xbody],dim=2) # b c 17 h w
                mid_block_res_sample += mid_block_res_sample_xbody
        else:
            for sample in down_block_res_samples:
                sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=f)
                B, C, Frame, H, W = sample.shape
                if cross_id:
                    sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), sample],dim=2) # b c 17 h w
                _down_block_res_samples.append(sample)
            down_block_res_samples = _down_block_res_samples
            mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', b=b, f=f)
            B, C, Frame, H, W = mid_block_res_sample.shape
            if cross_id:
                mid_block_res_sample = torch.cat([torch.zeros(B,C,1,H,W).to(sample.device,sample.dtype), mid_block_res_sample],dim=2) # b c 17 h w

        model_pred = self.unet(noisy_latents, timesteps, unet_encoder_hidden_states, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample, pose_embedding_features=pose_embedding_features).sample
        
        return model_pred
