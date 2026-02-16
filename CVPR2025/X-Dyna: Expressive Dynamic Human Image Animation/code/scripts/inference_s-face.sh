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

export TORCH_USE_CUDA_DSA=1
if [[ ! -d "logs" ]]; then
  mkdir logs
fi

################
gpus=${1-0}
job_name=${2-xdyna_infer}
pretrain_weight=${3-"SD/stable-diffusion-v1-5"} # path to pretrained SD1.5
output_dir=${4-"./output"} # save path
infer_config=${5-"configs/x_dyna.yaml"} # inference model config
pretrained_image_encoder_path=${6-"IP-Adapter/IP-Adapter/models/image_encoder"} # path to pretrained IP-Adapter image clip
pretrained_ipadapter_path=${7-"IP-Adapter/IP-Adapter/models/ip-adapter-plus_sd15.bin"} # path to pretrained IP-Adapter
pretrained_root_path=${8-"./pretrained_weights/initialization/unet_initialization"} # path to SD and IP-Adapter initialization root
test_data_file=${9-"examples/example.json"} # path to testing data file, used for batch inference
pose_controlnet_initialization_path=${10-"./pretrained_weights/initialization/controlnets_initialization/controlnet/control_v11p_sd15_openpose"}
face_controlnet_initialization_path=${11-"./pretrained_weights/initialization/controlnets_initialization/controlnet_face/controlnet2"}
pretrained_unet_path=${12-"./pretrained_weights/unet/"} # path to pretrained dynamics-adapter, motion module and unet
pretrained_controlnet_path=${13-"./pretrained_weights/controlnet/"} # path to pretrained pose controlnet
pretrained_face_controlnet_path=${14-"./pretrained_weights/controlnet_face/"} # path to pretrained s-face controlnet
################

echo 'start job:' ${job_name}

now=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=logs/${job_name}_${now}.log
echo 'log file: ' ${LOG_FILE}


# cfg 
export CUDA_VISIBLE_DEVICES=${gpus}
python inference_xdyna.py \
    --output ${output_dir} \
    --pretrain_weight "${pretrained_root_path}/${pretrain_weight}" \
    --length 192 \
    --height 896 --width 512 \
    --cfg 7.5 --infer_config ${infer_config} \
    --pretrained_image_encoder_path "${pretrained_root_path}/${pretrained_image_encoder_path}" \
    --pretrained_ipadapter_path "${pretrained_root_path}/${pretrained_ipadapter_path}" \
    --neg_prompt "" \
    --test_data_file ${test_data_file} \
    --pose_controlnet_initialization_path ${pose_controlnet_initialization_path} \
    --face_controlnet_initialization_path ${face_controlnet_initialization_path} \
    --face_controlnet \
    --pretrained_face_controlnet_path ${pretrained_face_controlnet_path} \
    --pretrained_unet_path ${pretrained_unet_path} \
    --pretrained_controlnet_path ${pretrained_controlnet_path} \
    --cross_id \
    --use_controlnet \
    --no_head_skeleton \
    --global_seed 40 \
    --stride 2 \
    --save_fps 15 \
    2>&1 | tee ${LOG_FILE}


