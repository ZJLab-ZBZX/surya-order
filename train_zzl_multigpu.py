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
from transformers import Trainer
import math
import argparse

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
        vocab_size = model.config.decoder.vocab_size if hasattr(model, "config") else model.module.config.decoder.vocab_size
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

   # 加载预处理器
    processor = OrderImageProcessor.from_pretrained(checkpoint)
    processor.size = settings.ORDER_IMAGE_SIZE
    processor.token_sep_id = max_tokens + box_size + 1
    processor.token_pad_id = max_tokens + box_size + 2
    processor.max_boxes = settings.ORDER_MAX_BOXES - 1
    processor.box_size = {"height": box_size, "width": box_size}

    return model,processor



def parse_args():
    parser = argparse.ArgumentParser(description="训练阅读顺序模型")
    
    # 模型相关参数
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--box_size", type=int, default=1024, help="框大小设置")
    parser.add_argument("--max_tokens", type=int, default=256, help="最大token数")
    
    # 数据路径参数
    parser.add_argument("--train_image_dir", type=str, required=True, help="训练图像目录")
    parser.add_argument("--train_bbox_dir", type=str, help="训练边界框目录")
    parser.add_argument("--train_order_dir", type=str,help="训练顺序标注目录")
    parser.add_argument("--eval_image_dir", type=str, required=True, help="评估图像目录")
    parser.add_argument("--eval_bbox_dir", type=str,help="评估边界框目录")
    parser.add_argument("--eval_order_dir", type=str,help="评估顺序标注目录")
    
    # 训练参数
    parser.add_argument("--num_train_epochs", type=int, default=30, help="训练epoch数")
    # parser.add_argument("--eval_steps", type=int, default=10, help="评估步数")
    parser.add_argument("--train_batch_size", type=int, default=10, help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    # 训练控制参数
    parser.add_argument("--eval_batch_size", type=int, default=10, help="评估批次大小")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志间隔步数")
    parser.add_argument("--dataloader_num_workers", type=int, default=64, help="数据加载工作线程数")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"使用的设备: {device}")

    # 加载模型和处理器
    model, processor = load_model_and_processor(
        checkpoint=args.checkpoint,
        device=device,
        dtype=dtype,
        box_size=args.box_size,
        max_tokens=args.max_tokens
    )

    # 数据集准备
    args.train_bbox_dir = args.train_bbox_dir or args.train_image_dir
    args.train_order_dir = args.train_order_dir or args.train_image_dir
    args.eval_bbox_dir = args.eval_bbox_dir or args.eval_image_dir
    args.eval_order_dir = args.eval_order_dir or args.eval_image_dir
    train_dataset = ReadingOrderDataset(
        image_dir=args.train_image_dir,
        bbox_dir=args.train_bbox_dir,
        order_dir=args.train_order_dir
    )
    
    eval_dataset = ReadingOrderDataset(
        image_dir=args.eval_image_dir,
        bbox_dir=args.eval_bbox_dir,
        order_dir=args.eval_order_dir
    )

    # 计算训练参数
    # train_file_lines = len(train_dataset)
    # total_training_samples = train_file_lines * args.num_train_epochs
    # total_batch_size = args.train_batch_size * int(os.environ.get("WORLD_SIZE", 1))
    # max_steps = math.ceil(total_training_samples / total_batch_size)
    # print(f"total_training_samples:{total_training_samples},total_batch_size:{total_batch_size},max_steps:{max_steps}, args.num_train_epochs:{args.num_train_epochs}")
    # print(int(os.environ.get("WORLD_SIZE", 1)))

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        do_train=True,
        do_eval=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs = args.num_train_epochs,
        overwrite_output_dir=True,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        learning_rate=args.learning_rate,
        metric_for_best_model="eval_loss",
        fp16=True,
        max_steps=max_steps,
        dataloader_num_workers=args.dataloader_num_workers
    )
    print(f"训练参数:{training_args}，训练数据集数量:{train_file_lines}")
    trainer = ReadingOrderTrainer(
        model=model,
        args=training_args,
        data_collator= ReadingOrderDataCollator(processor=processor,model_dtype=dtype),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir,"best_model"))

    

if __name__ == "__main__":
    main()
    # 命令行执行：python -m torch.distributed.launch --nproc_per_node=4 train_zzl_multigpu.py
