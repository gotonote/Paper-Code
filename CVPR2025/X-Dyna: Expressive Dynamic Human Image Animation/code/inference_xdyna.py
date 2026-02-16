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
import json
import torch
import random
import imageio
import argparse
import numpy as np
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from animatediff.models.unet import UNet3DConditionModel
from animatediff.utils.util import load_images_from_video_to_pil, imread_resize_crop, save_videos_grid_mp4
from animatediff.models.ip_adapter import Resampler
from animatediff.pipelines.pipeline_xdyna import XDynaPipeline
from animatediff.models.controlnet import ControlNetModel
from decord import VideoReader
from decord import cpu, gpu
import face_alignment
from animatediff.data.nms import nms_precessor
from skimage import img_as_ubyte

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def extract_local_feature_from_single_img(img, fa, nms_processor, last_bbox = None, last_center = None, remove_local=False, real_tocrop=None):
    pred = img
    height, width = np.shape(img)[:2]
    global_viz = np.zeros_like(pred)

    try:
        lmks = fa.get_landmarks_from_image(pred, return_landmark_score=False)[0]
    except:
        print ('undetected faces!!')
        left_landmark,top_landmark,right_landmark,bottom_landmark = last_bbox
        global_viz[top_landmark:bottom_landmark, left_landmark:right_landmark] = pred[top_landmark:bottom_landmark, left_landmark:right_landmark]
        half_size = [int(((right_landmark - left_landmark) / 512) * 64), int(((bottom_landmark - top_landmark) / 512) * 64)]
        left_eye_center, right_eye_center, mouth_center = last_center
        if remove_local:
            local_viz = pred
            local_viz[left_eye_center[1] - half_size[1] : left_eye_center[1] + half_size[1], left_eye_center[0] - half_size[0] : left_eye_center[0] + half_size[0]] = 0
            local_viz[right_eye_center[1] - half_size[1] : right_eye_center[1] + half_size[1], right_eye_center[0] - half_size[0] : right_eye_center[0] + half_size[0]] = 0
            local_viz[mouth_center[1] - half_size[1] : mouth_center[1] + half_size[1], mouth_center[0] - half_size[0] : mouth_center[0]  + half_size[0]] = 0        
        else:
            local_viz = np.zeros_like(pred)
            local_viz[left_eye_center[1] - half_size[1] : left_eye_center[1] + half_size[1], left_eye_center[0] - half_size[0] : left_eye_center[0] + half_size[0]] = pred[left_eye_center[1] - half_size[1] : left_eye_center[1] + half_size[1], left_eye_center[0] - half_size[0] : left_eye_center[0] + half_size[0]]
            local_viz[right_eye_center[1] - half_size[1] : right_eye_center[1] + half_size[1], right_eye_center[0] - half_size[0] : right_eye_center[0] + half_size[0]] = pred[right_eye_center[1] - half_size[1] : right_eye_center[1] + half_size[1], right_eye_center[0] - half_size[0] : right_eye_center[0] + half_size[0]]
            local_viz[mouth_center[1] - half_size[1] : mouth_center[1] + half_size[1], mouth_center[0] - half_size[0] : mouth_center[0]  + half_size[0]] = pred[mouth_center[1] - half_size[1] : mouth_center[1] + half_size[1], mouth_center[0] - half_size[0] : mouth_center[0] + half_size[0]]
        return local_viz, global_viz, last_bbox, last_center
    
    left_eye_center = np.round(np.mean(lmks[43:48], axis=0)).astype(np.int32)
    right_eye_center = np.round(np.mean(lmks[37:42], axis=0)).astype(np.int32)
    mouth_center =  np.round(np.mean(lmks[49:68], axis=0)).astype(np.int32)

    center = [left_eye_center, right_eye_center, mouth_center]
    left_landmark = np.amin(lmks[:,0])
    right_landmark = np.amax(lmks[:,0])
    top_landmark = np.amin(lmks[:,1])
    bottom_landmark = np.amax(lmks[:,1])
    if last_bbox and left_landmark >= last_bbox[0] and top_landmark >= last_bbox[1] and right_landmark <= last_bbox[2] and bottom_landmark <= last_bbox[3]:
        left_landmark,top_landmark,right_landmark,bottom_landmark = last_bbox
    else:
        bbox_land_mark = [left_landmark,top_landmark,right_landmark,bottom_landmark]
        bbox_land_mark_increased = nms_processor.compute_increased_bbox(bbox_land_mark,left_f=1.35,top_f=1.25,right_f=1.35,bot_f=1.45)
        left_landmark,top_landmark,right_landmark,bottom_landmark = bbox_land_mark_increased
        left_landmark = max(0, left_landmark)
        top_landmark = max(0, top_landmark)
        right_landmark = min(width, right_landmark)
        bottom_landmark = min(height, bottom_landmark)


    global_viz[top_landmark:bottom_landmark, left_landmark:right_landmark] = pred[top_landmark:bottom_landmark, left_landmark:right_landmark]

    current_bbox = [left_landmark,top_landmark,right_landmark,bottom_landmark]
    

    if real_tocrop is not None:
        pred = real_tocrop.permute([1, 2, 0]).detach().cpu().numpy()


    half_size = [int(((right_landmark - left_landmark) / 512) * 64), int(((bottom_landmark - top_landmark) / 512) * 64)]
    if remove_local:
        local_viz = pred
        local_viz[left_eye_center[1] - half_size[1] : left_eye_center[1] + half_size[1], left_eye_center[0] - half_size[0] : left_eye_center[0] + half_size[0]] = 0
        local_viz[right_eye_center[1] - half_size[1] : right_eye_center[1] + half_size[1], right_eye_center[0] - half_size[0] : right_eye_center[0] + half_size[0]] = 0
        local_viz[mouth_center[1] - half_size[1] : mouth_center[1] + half_size[1], mouth_center[0] - half_size[0] : mouth_center[0]  + half_size[0]] = 0        
    else:
        local_viz = np.zeros_like(pred)
        local_viz[left_eye_center[1] - half_size[1] : left_eye_center[1] + half_size[1], left_eye_center[0] - half_size[0] : left_eye_center[0] + half_size[0]] = pred[left_eye_center[1] - half_size[1] : left_eye_center[1] + half_size[1], left_eye_center[0] - half_size[0] : left_eye_center[0] + half_size[0]]
        local_viz[right_eye_center[1] - half_size[1] : right_eye_center[1] + half_size[1], right_eye_center[0] - half_size[0] : right_eye_center[0] + half_size[0]] = pred[right_eye_center[1] - half_size[1] : right_eye_center[1] + half_size[1], right_eye_center[0] - half_size[0] : right_eye_center[0] + half_size[0]]
        local_viz[mouth_center[1] - half_size[1] : mouth_center[1] + half_size[1], mouth_center[0] - half_size[0] : mouth_center[0]  + half_size[0]] = pred[mouth_center[1] - half_size[1] : mouth_center[1] + half_size[1], mouth_center[0] - half_size[0] : mouth_center[0] + half_size[0]]

    return local_viz, global_viz, current_bbox, center


def find_best_frame_byheadpose_fa(source_image, driving_video, fa=None):
    if fa is None:
        fa = Landmarks()
    input = cv2.cvtColor( img_as_ubyte(source_image), cv2.COLOR_RGB2BGR)
    _, src_pose_array, _ = fa.inference(input, mode=106)
    if len(src_pose_array) == 0:
        return 0
    min_diff = 1e8
    best_frame = 0

    for i in range(len(driving_video)):
        frame =  cv2.cvtColor(img_as_ubyte(driving_video[i]), cv2.COLOR_RGB2BGR)
        _, drv_pose_array, _ = fa.inference(frame, mode=106)
        if len(drv_pose_array) > 0:
            diff = np.sum(np.abs(np.array(src_pose_array)-np.array(drv_pose_array)))
            if diff < min_diff:
                best_frame = i
                min_diff = diff   
    
    return best_frame

def adjust_driving_video_to_src_image(source_image, driving_video, driving_pose, fa, height, width, best_frame=-1):
    if best_frame == -2:
        return driving_video, driving_pose
    else:
        src = img_as_ubyte(source_image[..., :3])
        if best_frame < 0:
            best_frame = find_best_frame_byheadpose_fa(src, driving_video)

        print ('Best Frame: %d' % best_frame)
        driving = img_as_ubyte(driving_video[best_frame])

        src_lmks = fa.get_landmarks_from_image(src, return_landmark_score=False)
        drv_lmks = fa.get_landmarks_from_image(driving, return_landmark_score=False)

        if (src_lmks is None) or (drv_lmks is None):
            return driving_video, driving_pose
        src_lmks = src_lmks[0]
        drv_lmks = drv_lmks[0]
        src_centers = np.mean(src_lmks, axis=0)
        drv_centers = np.mean(drv_lmks, axis=0)
        edge_src = (np.max(src_lmks, axis=0) - np.min(src_lmks, axis=0))*0.5
        edge_drv = (np.max(drv_lmks, axis=0) - np.min(drv_lmks, axis=0))*0.5

        #matching three points 
        src_point=np.array([[src_centers[0]-edge_src[0],src_centers[1]-edge_src[1]],[src_centers[0]+edge_src[0],src_centers[1]-edge_src[1]],[src_centers[0]-edge_src[0],src_centers[1]+edge_src[1]],[src_centers[0]+edge_src[0],src_centers[1]+edge_src[1]]]).astype(np.float32)
        dst_point=np.array([[drv_centers[0]-edge_drv[0],drv_centers[1]-edge_drv[1]],[drv_centers[0]+edge_drv[0],drv_centers[1]-edge_drv[1]],[drv_centers[0]-edge_drv[0],drv_centers[1]+edge_drv[1]],[drv_centers[0]+edge_drv[0],drv_centers[1]+edge_drv[1]]]).astype(np.float32)
    
        adjusted_driving_video = []
        adjusted_driving_pose = []
        
        for frame in driving_video:
            zoomed=cv2.warpAffine(frame, cv2.getAffineTransform(dst_point[:3], src_point[:3]), (height, width))
            adjusted_driving_video.append(zoomed)

        for frame in driving_pose:
            zoomed=cv2.warpAffine(frame, cv2.getAffineTransform(dst_point[:3], src_point[:3]), (height, width))
            adjusted_driving_pose.append(zoomed)
        
        return adjusted_driving_video, adjusted_driving_pose



    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_weight', type=str, default='stable-diffusion-v1-5', help='Path for pretrained weight (SD v1.5)')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-hs', '--height', type=int, default=256, help='video height')
    parser.add_argument('-ws', '--width', type=int, default=256, help='video width')
    parser.add_argument('-l', '--length', type=int, default=16, help='video length')
    parser.add_argument('-s', '--stride', type=int, default=1, help='stride of control map')
    parser.add_argument('--save_fps', type=int, default=15, help='fps of saved video')
    parser.add_argument('-g', '--global_seed', type=int, default=42, help='random seed')
    parser.add_argument('--cfg', type=float, default=7.5, help='classifier-free guidance scale')
    parser.add_argument('--infer_config', type=str, default='./configs/x_dyna.yaml', help='Path for inference config')
    parser.add_argument('--neg_prompt', type=str, default="", help='The negative prompt')
    parser.add_argument('--pretrained_image_encoder_path', type=str, default='', help='Path for pretrained image encoder')
    parser.add_argument('--pretrained_ipadapter_path', type=str, default='', help='Path for pretrained ipadapter encoder')
    parser.add_argument('--test_data_file', nargs='?', type=str, default='', help='Path for testing data file')
    parser.add_argument('--pretrained_unet_path', nargs='?', type=str, default=None, help='Path for unet file')
    parser.add_argument('--pretrained_controlnet_path', nargs='?', type=str, default=None, help='Path for controlnet file')
    parser.add_argument('--pose_controlnet_initialization_path', nargs='?', type=str, default=None, help='Path for pose controlnet initialization')
    parser.add_argument('--face_controlnet_initialization_path', nargs='?', type=str, default=None, help='Path for face controlnet initialization')
    parser.add_argument('--use_controlnet', action='store_true', help='if ControlNet is used for inference.')
    parser.add_argument('--face_controlnet', action='store_true', help='if S-Face ControlNet is used for inference.')
    parser.add_argument('--pretrained_face_controlnet_path', nargs='?', type=str, default=None, help='Path for S-Face Controlnet file')
    parser.add_argument('--no_head_skeleton', action='store_true', help='if head skeleton is not used during training.')
    parser.add_argument('--cross_id', action='store_true', help='if cross identity trained.')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    pretrained_model_path = args.pretrain_weight
    pretrained_image_encoder_path = args.pretrained_image_encoder_path
    pretrained_ipadapter_path = args.pretrained_ipadapter_path
    inference_config = OmegaConf.load(args.infer_config)
    global_seed = args.global_seed
    pretrained_unet_path = args.pretrained_unet_path
    pretrained_controlnet_path = args.pretrained_controlnet_path
    face_controlnet = args.face_controlnet
    pretrained_face_controlnet_path = args.pretrained_face_controlnet_path


    unet = UNet3DConditionModel.from_pretrained_ip(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).to(torch.float16).to('cuda')
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to('cuda')
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to('cuda')
    clip_image_processor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_image_encoder_path, torch_dtype=torch.float16).to('cuda')
    image_proj_model = Resampler(
            dim=768,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=16,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=768,
            ff_mult=4
        )
    image_proj_model.load_state_dict(torch.load(pretrained_ipadapter_path, "cpu")["image_proj"], strict=True)
    image_proj_model.to(torch.float16).to('cuda')
    print("Load pretrained clip image encoder and ipadapter model successfully")
    

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    image_proj_model.requires_grad_(False)



    # builds pipeline
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs))
    if pretrained_controlnet_path is None:
        print("Please specify controlnet path")
        raise NotImplementedError
    else:
        if (pretrained_unet_path != "") and (pretrained_unet_path != None):
            print(f"from pretrained unet: {pretrained_unet_path}")
            pretrained_unet_path = torch.load(pretrained_unet_path, map_location="cpu")
            state_dict = pretrained_unet_path["state_dict"] if "state_dict" in pretrained_unet_path else pretrained_unet_path
            state_dict = {k.replace("module.", "") if "module." in k else k: v for k, v in state_dict.items()}
            m, u = unet.load_state_dict(state_dict, strict=False)
            print(f"unet missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0
        
        controlnet = ControlNetModel.from_pretrained(args.pose_controlnet_initialization_path)
        if pretrained_controlnet_path.endswith(".ckpt"):
            print(f"from pretrained controlnet: {pretrained_controlnet_path}")
            pretrained_controlnet_path = torch.load(pretrained_controlnet_path, map_location="cpu")
            controlnet_state_dict = pretrained_controlnet_path["state_dict"] if "state_dict" in pretrained_controlnet_path else pretrained_controlnet_path
            controlnet_state_dict = {k.replace("module.", "") if "module." in k else k: v for k, v in controlnet_state_dict.items()}
            m, u = controlnet.load_state_dict(controlnet_state_dict, strict=True)
            print(f"controlnet missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0 and len(m) == 0
        
        if face_controlnet:
            print(f"from pretrained S-Face controlnet: {pretrained_face_controlnet_path}")
            if pretrained_face_controlnet_path.endswith(".ckpt"):
                controlnet_face = ControlNetModel.from_pretrained(args.face_controlnet_initialization_path)
                pretrained_face_controlnet_path = torch.load(pretrained_face_controlnet_path, map_location="cpu")
                controlnet_face_state_dict = pretrained_face_controlnet_path["state_dict"] if "state_dict" in pretrained_face_controlnet_path else pretrained_face_controlnet_path
                controlnet_face_state_dict = {k.replace("module.", "") if "module." in k else k: v for k, v in controlnet_face_state_dict.items()}
                m, u = controlnet_face.load_state_dict(controlnet_face_state_dict, strict=True)
                print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
                assert len(u) == 0 and len(m) == 0
            else:
                print("Please specify the s-face controlnet path with a ckpt file.")
                raise NotImplementedError
    
        
        if is_xformers_available():
            # unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
            if face_controlnet:
                controlnet_face.enable_xformers_memory_efficient_attention()
        print(f'Constructing pipeline')
        pipe = XDynaPipeline(
        unet=unet, 
        controlnet=controlnet,
        vae=vae, 
        tokenizer=tokenizer, 
        text_encoder=text_encoder, 
        scheduler=noise_scheduler,
        controlnet_face=controlnet_face if face_controlnet else None,
        )

    

    pipe.to(torch.float16).to('cuda')
    pipe.enable_vae_slicing()
    device = torch.device("cuda")
    print('Pipeline loading done')
    

    if args.test_data_file != None:
        assert os.path.exists(args.test_data_file), "Testing data file does not exists!!!"
        with open(args.test_data_file, "r") as f:
            test_data = json.load(f)
    else:
        print("INFO: Please specify your image and control path in the test_data_file json.")
        raise NotImplementedError
    
    if face_controlnet:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True, device='cuda')
        _nms_precessor = nms_precessor([args.height, args.width]) 
    
    
    for step, data_pair in enumerate(test_data):
        with torch.cuda.amp.autocast(enabled=True):
            img_root = data_pair['image_path']
            prompt = data_pair['caption']
            if args.no_head_skeleton:
                control_root = data_pair['target_no_head_poses']
                if face_controlnet:
                    driven_video_root = data_pair['target_videos']
            else:
                print("We use no head skeleton for openpose.")
                raise NotImplementedError

            img_path = img_root
            control_path = control_root
            print("Control Path: ",control_path)
            print("Reference Image Path: ", img_path)
            samples=[]
            samples_gen=[]
            validation_image_control = load_images_from_video_to_pil(
                control_path, 
                target_size = (args.width, args.height), 
                vid_len=args.length, 
                stride=args.stride if not (("c1" in control_path) or ("c2" in control_path) or ("c3" in control_path) or ("c4" in control_path)) else 1,
                )
            length_driven = len(validation_image_control)
            if args.neg_prompt is not None:
                neg_prompt = args.neg_prompt
            else:
                neg_prompt = ""
            print('Prompt: ', prompt)
            print('Negative Prompt: ', neg_prompt)
            # Get ip-adapter image embeddings
            image = Image.open(img_path).convert("RGB")
            image = clip_image_processor(images=image, return_tensors="pt").pixel_values.to('cuda').to(torch.float16)
            with torch.no_grad():
                clip_image_embeds = image_encoder(image, output_hidden_states=True).hidden_states[-2]
                clip_image_embeds = image_proj_model(clip_image_embeds).to(device="cuda", dtype=torch.float16)
                un_cond_image_embeds = image_encoder(torch.zeros_like(image).to(image.device).to(torch.float16), output_hidden_states=True).hidden_states[-2]
                un_cond_image_embeds = image_proj_model(un_cond_image_embeds).to(device="cuda", dtype=torch.float16)
            
                print("Using seed {} for generation".format(global_seed))
                generator = torch.Generator(device="cuda").manual_seed(global_seed)
                # Get first frame latents as usual
                image = imread_resize_crop(img_path, args.height, args.width)
                ref_name = os.path.splitext(os.path.basename(img_path))[0]
                if face_controlnet:
                    driven_video_path = driven_video_root
                    tgt_pose = VideoReader(control_path, ctx=cpu(0))
                    tgt_video = VideoReader(driven_video_path, ctx=cpu(0))
                    w,h,start_w,start_h,target_w,target_h = _nms_precessor.crop2ratio(image)
                    image = cv2.resize(image[start_h:(start_h+target_h),start_w:(start_w+target_w)], (args.width, args.height), interpolation=cv2.INTER_AREA)
                    
                    cv2.imwrite(f"{args.output}/{ref_name}_reference.jpg", image[..., ::-1])
                    
                    driving_video = []
                    driving_pose = []
                    
                    for frame_id in range(length_driven):
                        frame = tgt_video[frame_id].asnumpy()
                        pose = tgt_pose[frame_id].asnumpy()
                        w,h,start_w,start_h,target_w,target_h = _nms_precessor.crop2ratio(frame)
                        cropped_frame = cv2.resize(frame[start_h:(start_h+target_h),start_w:(start_w+target_w)],(args.width, args.height))
                        driving_video.append(cropped_frame)
                        cropped_pose = cv2.resize(pose[start_h:(start_h+target_h),start_w:(start_w+target_w)],(args.width, args.height))
                        driving_pose.append(cropped_pose)
                    
                    adjusted_driving_video, adjusted_driving_pose = adjust_driving_video_to_src_image(image, driving_video, driving_pose, fa, args.width, args.height, best_frame=-2)
                    
                    control_local = []
                    control_global = []
                    last_bbox = [0,0,0,0]
                    last_center = [np.zeros(2, dtype=int),np.zeros(2, dtype=int),np.zeros(2, dtype=int)]
                    for frame_id in range(len(adjusted_driving_video)):
                        frame = adjusted_driving_video[frame_id]
                        pose = adjusted_driving_pose[frame_id]
                        control_l, control_g, cur_bbox, center = extract_local_feature_from_single_img(frame, fa, _nms_precessor, last_bbox = last_bbox, last_center = last_center)
                        last_center = center
                        control_local.append(control_l)
                        control_global.append(control_g)

                    validation_image_control = [Image.fromarray(pil_image.astype(np.uint8)) for pil_image in driving_pose]
                    validation_sface_global_control = [Image.fromarray(pil_global_image.astype(np.uint8)) for pil_global_image in control_global]
                    validation_sface_local_control = [Image.fromarray(pil_local_image.astype(np.uint8)) for pil_local_image in control_local]
                    driving_video_vis = np.array(driving_video)
                    driving_video_vis = driving_video_vis / 255.0
                    driving_video_vis = rearrange(driving_video_vis, "(b f) h w c -> b c f h w",f=length_driven)
                    driving_video_vis = torch.from_numpy(driving_video_vis)
                    samples.append(driving_video_vis)
                    control_global_vis = np.array([np.array(validation_sface_global_control[i]).copy() for i in range(len(validation_sface_global_control))])
                    control_global_vis = control_global_vis / 255.0
                    control_global_vis = rearrange(control_global_vis, "(b f) h w c -> b c f h w",f=length_driven)
                    control_global_vis = torch.from_numpy(control_global_vis)
                    samples.append(control_global_vis)
                    control_local_vis = np.array([np.array(validation_sface_local_control[i]).copy() for i in range(len(validation_sface_local_control))])
                    control_local_vis = control_local_vis / 255.0
                    control_local_vis = rearrange(control_local_vis, "(b f) h w c -> b c f h w",f=length_driven)
                    control_local_vis = torch.from_numpy(control_local_vis)
                    samples.append(control_local_vis)
                
                control_vis = np.array([np.array(validation_image_control[i]).copy() for i in range(len(validation_image_control))])
                control_vis = control_vis / 255.0
                control_vis = rearrange(control_vis, "(b f) h w c -> b c f h w",f=length_driven)
                control_vis = torch.from_numpy(control_vis)
                samples.append(control_vis)
                


                first_frame_latents = torch.Tensor(image.copy()).to('cuda').type(torch.float16).permute(2, 0, 1).repeat(1, 1, 1, 1)
                first_frame_latents = first_frame_latents / 127.5 - 1.0
                first_frame_latents = vae.encode(first_frame_latents).latent_dist.sample(generator) * 0.18215
                first_frame_latents = first_frame_latents.repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4).type(torch.float16)
                # video generation
                video = pipe(prompt=prompt, generator=generator, 
                            pose_embedding=None, # This arg is for CameraCtrl, which is not used in our model.
                            latents=first_frame_latents, 
                            video_length=length_driven, height=image.shape[0], width=image.shape[1], 
                            num_inference_steps=25, guidance_scale=args.cfg, 
                            noise_mode="iid", negative_prompt=neg_prompt, 
                            repeat_latents=True, gaussian_blur=True, strength=1.0,
                            cond_image_embeds=clip_image_embeds,
                            un_cond_image_embeds=un_cond_image_embeds,
                            controlnet_condition=validation_image_control,
                            sface_global_condition=validation_sface_global_control if face_controlnet else None, 
                            sface_local_condition=validation_sface_local_control if face_controlnet else None,
                            use_controlnet=args.use_controlnet,
                            cross_id=args.cross_id, 
                            ).videos
            samples.append(video)
            samples_gen.append(video)
            samples = torch.concat(samples)
            samples_gen = torch.concat(samples_gen)
            ref_img_name = os.path.splitext(os.path.basename(img_path))[0]
            control_name = control_path.split("/")[-2]
            save_path = f"{args.output}/{ref_img_name}_{control_name}.mp4"
            save_gen_path = f"{args.output}/gen/{ref_img_name}_{control_name}.mp4"

            if face_controlnet:
                save_videos_grid_mp4(samples, save_path, n_rows=5, fps=args.save_fps)
            else:
                save_videos_grid_mp4(samples, save_path, n_rows=2, fps=args.save_fps)
            save_videos_grid_mp4(samples_gen, save_gen_path, n_rows=1, fps=args.save_fps)
            
            
if __name__ == '__main__':
    main()