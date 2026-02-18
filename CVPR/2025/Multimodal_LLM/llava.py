"""
Multimodal LLM (LLaVA) 实现

本实现为第三方复现，仅供参考
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM
from transformers import AutoTokenizer


class CLIPVisionEncoder(nn.Module):
    """CLIP Vision Encoder"""
    
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
    
    def forward(self, images):
        outputs = self.model(images)
        return outputs.last_hidden_state


class Projector(nn.Module):
    """Vision-Language Projector"""
    
    def __init__(self, vision_hidden_size, language_hidden_size, projector_type="linear"):
        super().__init__()
        
        if projector_type == "linear":
            self.projector = nn.Linear(vision_hidden_size, language_hidden_size)
        elif projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, language_hidden_size),
                nn.GELU(),
                nn.Linear(language_hidden_size, language_hidden_size)
            )
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")
    
    def forward(self, x):
        return self.projector(x)


class LLaVA(nn.Module):
    """
    Large Language and Vision Assistant
    
    端到端的多模态大语言模型
    """
    
    def __init__(
        self,
        vision_encoder_name="openai/clip-vit-large-patch14",
        llm_name="lmsys/vicuna-13b-v1.5",
        mm_projector_type="mlp",
        freeze_llm=False,
        freeze_vision_encoder=True
    ):
        super().__init__()
        
        # 视觉编码器
        self.vision_encoder = CLIPVisionEncoder(vision_encoder_name)
        
        # 投影层
        self.mm_projector = Projector(
            self.vision_encoder.hidden_size,
            5120,  # Vicuna hidden size
            mm_projector_type
        )
        
        # 语言模型
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.llm_hidden_size = self.llm.config.hidden_size
        
        # 调整 projector 输出维度
        self.mm_projector = Projector(
            self.vision_encoder.hidden_size,
            self.llm_hidden_size,
            mm_projector_type
        )
        
        # 冻结部分参数
        if freeze_vision_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
        
        # 图像 tokens 生成器
        self.num_image_tokens = 256
    
    def forward(self, images, input_ids, attention_mask=None):
        """
        前向传播
        
        Args:
            images: [B, C, H, W] 图像
            input_ids: [B, L] token ids
            attention_mask: [B, L] attention mask
        """
        batch_size = images.shape[0]
        
        # 视觉编码
        image_features = self.vision_encoder(images)
        
        # 投影到语言空间
        image_embeds = self.mm_projector(image_features)
        
        # 获取文本 embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 创建图像位置标记 (作为 placeholder)
        image_tokens = torch.zeros(
            batch_size, self.num_image_tokens, self.llm_hidden_size,
            device=image_embeds.device,
            dtype=image_embeds.dtype
        )
        
        # 在文本前添加图像 embeddings
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # 调整 attention mask
        if attention_mask is not None:
            image_mask = torch.ones(
                batch_size, self.num_image_tokens,
                device=attention_mask.device
            )
            attention_mask = torch.cat([image_mask, attention_mask], dim=1)
        
        # LLM 前向
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return outputs.logits
    
    def generate(self, images, prompts, max_new_tokens=256):
        """
        生成文本回复
        
        Args:
            images: 输入图像
            prompts: 文本提示
        
        Returns:
            generated_text: 生成的文本
        """
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5")
        
        # Tokenize prompts
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(images.device)
        attention_mask = inputs.attention_mask.to(images.device)
        
        # Forward
        with torch.no_grad():
            outputs = self.forward(images, input_ids, attention_mask)
        
        # Greedy decoding
        generated_ids = torch.argmax(outputs, dim=-1)
        
        # Decode
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return generated_text


def create_llava_model():
    """创建 LLaVA 模型"""
    model = LLaVA(
        vision_encoder_name="openai/clip-vit-large-patch14",
        llm_name="lmsys/vicuna-7b-v1.5",  # Use smaller model for demo
        mm_projector_type="mlp"
    )
    return model
