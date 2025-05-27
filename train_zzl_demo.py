from transformers import Trainer, TrainingArguments,DefaultDataCollator,DataCollatorWithPadding
from transformers import VisionEncoderDecoderConfig, AutoModelForCausalLM, AutoModel
from modules.surya_order.config import MBartOrderConfig, VariableDonutSwinConfig
from modules.surya_order.decoder import MBartOrder
from modules.surya_order.encoder import VariableDonutSwinModel
from modules.surya_order.encoderdecoder import OrderVisionEncoderDecoderModel
from modules.surya_order.processor import OrderImageProcessor
from modules.surya_order.settings import settings
from PIL import Image
import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset  
from typing import List, Dict
from datasets import Dataset as HFDataset
import torch.nn as nn
from transformers import Trainer, AdamW
from torch.optim.lr_scheduler import LinearLR

class ReadingOrderDataset(Dataset):    
    def __init__(self,image_dir,bbox_dir,order_dir):
        super().__init__()  
        
        # 目录校验
        for dir_path in [image_dir, bbox_dir, order_dir]:
            if not os.path.isdir(dir_path):
                raise ValueError(f"目录不存在: {dir_path}")
        # 初始化类属性
        self.image_dir = image_dir
        self.bbox_dir = bbox_dir
        self.order_dir = order_dir
        
        # 构建样本索引
        self.samples = self._build_sample_list()
        
        # 创建Hugging Face Dataset实例
        self.hf_dataset = self._create_hf_dataset()
    
    def _build_sample_list(self) -> List[Dict]:
        """扫描目录并构建有效样本列表"""
        samples = []
        
        # 扫描图像文件 (支持.jpg, .png, .jpeg)
        image_files = [
            f for f in os.listdir(self.image_dir) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        
        # 构建样本映射
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            
            # 构建完整路径
            bbox_path = os.path.join(self.bbox_dir, f"{base_name}.npy")
            order_path = os.path.join(self.order_dir, f"{base_name}.txt")
            image_path = os.path.join(self.image_dir, img_file)
            
            # 验证文件存在性
            if all(os.path.exists(p) for p in [bbox_path, order_path]):
                samples.append({
                    "image_path": image_path,
                    "bbox_path": bbox_path,
                    "order_path": order_path,
                })
            else:
                missing_files = [
                    p for p in [bbox_path, order_path] 
                    if not os.path.exists(p)
                ]
                print(f"忽略不完整样本 {img_file}，缺失文件: {', '.join(missing_files)}")
        return samples
    
    def _create_hf_dataset(self) -> HFDataset:
        """创建Hugging Face Dataset实例"""
        df = pd.DataFrame(self.samples)
        return HFDataset.from_pandas(df)
    
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """通过Hugging Face Dataset获取数据"""
        example = self.hf_dataset[idx]
        try:
            image = Image.open(example["image_path"]).convert("RGB")
            bboxes = np.load(example["bbox_path"], allow_pickle=True).tolist()
            
            with open(example["order_path"], "r") as f:
                raw_order = f.read().strip().strip('[]')
                target_order = [int(x) for x in raw_order.split(',')]
            result = {"image": image,"bboxes": bboxes,"target_order": target_order}
            return [image,bboxes,target_order] #如果返回dict，会走到trainer()底层的过滤逻辑
        except Exception as e:
            print(f"加载样本失败，样本信息：{str(example)} : {str(e)}")
            return None
        
class ReadingOrderDataCollator(DefaultDataCollator):
    def __init__(self, processor, model_dtype):
        self.processor = processor
        self.model_dtype = model_dtype
        
    def __call__(self, features):
        # 获取原始数据
        images = [sample[0] for sample in features if sample is not None]
        bboxes_list = [sample[1] for sample in features if sample is not None]
        target_order = [torch.tensor(sample[2], dtype=torch.long) for sample in features  if sample is not None]
        batch_size = len(features)


        # 使用处理器预处理
        model_inputs = self.processor(images=images,boxes=bboxes_list)

        # 获取预处理的数据
        pixel_values_tensor = torch.tensor(
            np.stack(model_inputs["pixel_values"]), 
            dtype=self.model_dtype
        )
        input_boxes_mask_tensor =  torch.tensor(model_inputs["input_boxes_mask"], dtype=torch.bool)  # model_inputs["input_boxes_mask"]example:[[0,0,1,1,...],[1,1,1,1...]]
        input_boxes_tensor = torch.tensor(model_inputs["input_boxes"], dtype=self.model_dtype)
        input_boxes_counts = model_inputs["input_boxes_counts"]  #[[padding_boxes_count,max_boxes_count],[]],example:[[2,12],[0,12]]
        input_boxes_counts_tensor = torch.tensor(input_boxes_counts, dtype=torch.long)

        # 动态计算最大序列长度
        boxes_counts_list = [counts[1]-counts[0]-1 for counts in input_boxes_counts] #不含padding和sep的box数量
        sep_pos_list = [counts[1]-1 for counts in input_boxes_counts]
        max_seq_len = max(pos + 1 + num for pos, num in zip(sep_pos_list, boxes_counts_list))

        decoder_input_boxes = torch.full(
            (batch_size, max_seq_len, 4),
            self.processor.token_pad_id,
            dtype=torch.long
        )
        
        decoder_input_boxes_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.bool,
        )
        
        labels = torch.full(
            (batch_size, max_seq_len),
            -100,
            dtype=torch.long
        )

        offset = self.processor.box_size["height"] + 1
        

        
        for i in range(batch_size):
            # 批量填充前置内容
            pos = sep_pos_list[i]
            boxes_num = boxes_counts_list[i]
            decoder_input_boxes[i, :pos+1] = input_boxes_tensor[i, :pos+1]
            decoder_input_boxes_mask[i, :pos+1] = input_boxes_mask_tensor[i, :pos+1]

            # 批量填充目标区域
            target_tokens = (target_order[i].unsqueeze(-1) + offset).repeat(1, 1, 4)
            decoder_input_boxes[i, pos + 1:pos + 1 + boxes_num ] = target_tokens
            decoder_input_boxes_mask[i, pos + 1:pos + 1 + boxes_num] = True
            labels[i, pos:pos + boxes_num] = target_order[i]

        return {
            "pixel_values": pixel_values_tensor,
            "decoder_input_boxes": decoder_input_boxes,
            "decoder_input_boxes_mask": decoder_input_boxes_mask,
            "decoder_input_boxes_counts": input_boxes_counts_tensor,
            "labels": labels
        }
    

class ReadingOrderTrainer(Trainer):        
    def compute_loss(self, model, inputs,return_outputs=False):
        # 提取标签与模型输出
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits
        vocab_size = model.config.decoder.vocab_size
        # 自定义损失计算
        loss_fct = nn.CrossEntropyLoss(ignore_index = -100)
        loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def load_model_and_processor(checkpoint,device,dtype,box_size,max_tokens):
    """加载模型和预处理器"""
    config = VisionEncoderDecoderConfig.from_pretrained(checkpoint)
    
    # 自定义解码器配置
    decoder_config = vars(config.decoder)
    decoder = MBartOrderConfig(**decoder_config)
    config.decoder = decoder
    
    # 自定义编码器配置
    encoder_config = vars(config.encoder)
    encoder = VariableDonutSwinConfig(**encoder_config)
    config.encoder = encoder
    
    # 注册自定义模型
    AutoModel.register(MBartOrderConfig, MBartOrder)
    AutoModelForCausalLM.register(MBartOrderConfig, MBartOrder)
    AutoModel.register(VariableDonutSwinConfig, VariableDonutSwinModel)

    model = OrderVisionEncoderDecoderModel.from_pretrained(checkpoint, config=config, torch_dtype=dtype)
    model = model.to(device)

   # 加载预处理器
    processor = OrderImageProcessor.from_pretrained(checkpoint)
    processor.size = settings.ORDER_IMAGE_SIZE
    processor.token_sep_id = max_tokens + box_size + 1
    processor.token_pad_id = max_tokens + box_size + 2
    processor.max_boxes = settings.ORDER_MAX_BOXES - 1
    processor.box_size = {"height": box_size, "width": box_size}

    return model,processor



def main():
    checkpoint = '/home/zzl/surya_deng/models/vikp/surya_order'
    box_size = 1024
    max_tokens = 256
    train_image_dir = "/home/zzl/surya_deng/pichulipaixu/datasets_eval"
    train_bbox_dir = "/home/zzl/surya_deng/pichulipaixu/datasets_eval"
    train_order_dir = "/home/zzl/surya_deng/pichulipaixu/datasets_eval"
    eval_image_dir = "/home/zzl/surya_deng/pichulipaixu/datasets_eval"
    eval_bbox_dir = "/home/zzl/surya_deng/pichulipaixu/datasets_eval"
    eval_order_dir = "/home/zzl/surya_deng/pichulipaixu/datasets_eval"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dtype = settings.MODEL_DTYPE
    dtype = torch.float32
    print(f"使用的设备: {device}")
    model, processor = load_model_and_processor(checkpoint = checkpoint,device = device,dtype = dtype,box_size = box_size,max_tokens = max_tokens)
    
    train_dataset = ReadingOrderDataset(image_dir = train_image_dir, bbox_dir = train_bbox_dir, order_dir = train_order_dir)
    eval_dataset = ReadingOrderDataset(image_dir = eval_image_dir, bbox_dir = eval_bbox_dir, order_dir = eval_order_dir)
    training_args = TrainingArguments(
        output_dir="/home/zzl/surya_deng/results_zzl",
        # logging_steps=1,
        # logging_strategy="steps",
        per_device_train_batch_size=2,
        evaluation_strategy="epoch",
        num_train_epochs=3,
        # learning_rate=5e-5,  # 进一步降低学习率
        max_grad_norm=0.5,  # 更严格的梯度裁剪
        warmup_steps=1000,
        # adam_beta1=0.8,
        # adam_beta2=0.95,
        # adam_epsilon=1e-8,
        weight_decay=0.01,
        fp16=False,  # 确认禁用混合精度
    )
    # optimizer = AdamW(model.parameters(), lr=2e-5)
    # scheduler = LinearLR(optimizer, total_iters=100)
    trainer = ReadingOrderTrainer(
        model=model,
        args=training_args,
        data_collator= ReadingOrderDataCollator(processor=processor,model_dtype=dtype),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # optimizers=(optimizer, scheduler)
    )
    trainer.train()
    

if __name__ == "__main__":
    main()