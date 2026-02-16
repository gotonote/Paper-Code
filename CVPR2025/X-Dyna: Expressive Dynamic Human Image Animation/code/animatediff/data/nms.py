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

class nms_precessor():
    def __init__(self,resolution) -> None:
        self.OPENPOSE_FACE_INDEX = list(range(23, 91))
        self.resolution = resolution
 
    def crop2ratio(self,src_img,driving_pose=None,plot_type=None):
        h,w = src_img.shape[0],src_img.shape[1]
        if w/h < self.resolution[1]/self.resolution[0]:
            target_w = w
            target_h = int(target_w/self.resolution[1]*self.resolution[0])
        else:
            target_h = h
            target_w = int(target_h/self.resolution[0]*self.resolution[1])

        start_w = (w-target_w)//2
        start_h=(h-target_h)//4
        return w,h,start_w,start_h,target_w,target_h
    
    def compute_increased_bbox(self,bbox,left_f=0.35,top_f=0.8,right_f=0.35,bot_f=0.2,max_x=None,max_y=None):
        left, top, right, bot = bbox
        width = right - left
        height = bot - top
        center_x = (left+right)*0.5
        center_y = (top+bot)*0.5
        if max_x is None:
            max_x = self.resolution[1]
        if max_y is None:
            max_y = self.resolution[0]
        length = max(width,height)
        left = int(center_x - left_f * length)
        top = int(center_y - top_f * length)
        right = int(center_x + right_f * length)
        bot = int(center_y + bot_f * length)
        return (left, top, right, bot)