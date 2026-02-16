# Copyright 2024 ByteDance and/or its affiliates.
#
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

# Install Anaconda or Miniconda
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 xformers --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install face_alignment
pip install pyvideoreader
pip install imageio[ffmpeg] 
pip install moviepy
pip install diffusers==0.24.0 
pip install joblib
pip install scikit-image
pip install visdom
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install hydra-core --upgrade
pip install omegaconf opencv-python einops visdom tqdm scipy plotly scikit-learn imageio[ffmpeg] gradio trimesh huggingface_hub
pip uninstall numpy -y
pip install numpy==1.26.3
pip uninstall xformers -y
pip uninstall torch torchvision torchaudio -y
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.20
pip install huggingface_hub==0.25.2
