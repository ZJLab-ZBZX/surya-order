import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from functools import partial
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler

from tqdm import tqdm

# 导入自定义模块
from modules.surya_order.processor import OrderImageProcessor
from modules.surya_order.settings import settings

from copy import deepcopy

def find_sep_positions(batch_bboxes, sep_token_id):
    """查找<SEP>标记位置"""
    batch_size = batch_bboxes.size(0)
    sep_positions = []
    for i in range(batch_size):
        for j in range(batch_bboxes.size(1)):
            if batch_bboxes[i, j, 0] == sep_token_id:
                sep_positions.append(j)
                break
        else:
            sep_positions.append(batch_bboxes.size(1) - 1)
    return sep_positions

class OrderDataset(Dataset):
    """阅读顺序数据加载Dataset类"""
    def __init__(self, data_paths, data_type='custom'):
        self.data_paths = data_paths
        self.valid_samples = self._filter_invalid_paths()
        self.data_type = data_type

    def _filter_invalid_paths(self):
        """过滤无效路径"""
        valid_indices = []
        for i, (img_path, json_path) in tqdm(enumerate(self.data_paths),desc='Check valid dataset'):
            try:
                # with Image.open(img_path) as img:
                #     img.verify()
                with open(json_path, 'r') as f:
                    json.load(f)
                valid_indices.append(i)
            except Exception as e:
                print(f"[WARNING] 跳过无效样本 ({i}): {img_path} | {json_path} - {str(e)}")

        if not valid_indices:
            raise ValueError("没有找到有效样本")

        return {
            "data_paths": [(self.data_paths[i][0], self.data_paths[i][1]) for i in valid_indices]
        }

    def __len__(self):
        return len(self.valid_samples["data_paths"])

    def __getitem__(self, idx):
        try:
            img_path, json_path = self.valid_samples["data_paths"][idx]
            # pil_img = Image.open(img_path).convert("RGB")

            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            if self.data_type == 'custom':
                xyxy_bboxes = json_data['bboxes']
                remapped_orders = json_data.get('mathpix_order_fix',list(range(len(xyxy_bboxes))))
                labels = json_data.get('label',list(range(len(xyxy_bboxes))))
            else:
                # 提取 bbox 和 order
                layout_rects = [item["bbox"] for item in json_data["page_layout_list"]]
                orders = [int(item["order"]) for item in json_data["page_layout_list"]]

                # 将 bbox 从嵌套列表格式转换为 xyxy 格式
                xyxy_bboxes = []
                for bbox in layout_rects:
                    (_x1, _y1), (_x2, _y2) = bbox[1], bbox[3]
                    x1, x2 = sorted([_x1, _x2])
                    y1, y2 = sorted([_y1, _y2])
                    xyxy_bboxes.append([x1, y1, x2, y2])

                # 将 orders 映射为从 0 开始的连续整数
                unique_orders = sorted(set(orders))
                order_mapping = {order: idx for idx, order in enumerate(unique_orders)}
                remapped_orders = [order_mapping[order] for order in orders]

            result = {
                'image': img_path,
                'bboxes': xyxy_bboxes,  # 使用 xyxy 格式的 bbox
                'mathpix_orders': remapped_orders,
                'image_path': img_path,
                'json_path': json_path,
                'label': labels
            }

            # 只更新 result 中不存在的键
            for key, value in json_data.items():
                if key not in result:
                    result[key] = value

            return result
        except Exception as e:
            print(f"[ERROR] 加载样本 {idx} 失败: {str(e)}")
            return None

def custom_collate_fn(batch, processor, dtype=torch.float32):
    """自定义collate函数"""
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None

    images = [Image.open(sample['image']).convert("RGB") for sample in batch]
    bboxes_list = [sample['bboxes'] for sample in batch]

    # 图像处理
    model_inputs = processor(images=images, boxes=bboxes_list)

    # 转换为张量
    pixel_values = torch.tensor(np.stack(model_inputs["pixel_values"]), dtype=dtype)
    input_boxes_mask = [torch.tensor(mask, dtype=torch.bool) for mask in model_inputs["input_boxes_mask"]]

    # 使用JSON中的顺序
    target_orders = [torch.tensor(sample['mathpix_orders'], dtype=torch.long) for sample in batch]
    num_boxes_list = [len(sample['mathpix_orders']) for sample in batch]

    # 处理序列
    batch_bboxes = torch.tensor(model_inputs["input_boxes"], dtype=torch.long)
    sep_pos_list = find_sep_positions(batch_bboxes, processor.token_sep_id)
    max_seq_len = max(pos + 1 + num for pos, num in zip(sep_pos_list, num_boxes_list))
    batch_size = len(images)

    # 初始化张量
    decoder_input_boxes = torch.full((batch_size, max_seq_len, 4), processor.token_pad_id, dtype=torch.long)
    decoder_input_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)

    for i in range(batch_size):
        sep_pos = sep_pos_list[i]
        target_order = target_orders[i]
        num_boxes = num_boxes_list[i]

        decoder_input_boxes[i, :sep_pos + 1] = batch_bboxes[i, :sep_pos + 1]
        decoder_input_mask[i, :sep_pos + 1] = input_boxes_mask[i][:sep_pos + 1]

        offset = processor.box_size["height"] + 1
        target_tokens = (target_order.unsqueeze(-1) + offset).repeat(1, 4)
        decoder_input_boxes[i, sep_pos + 1:sep_pos + 1 + num_boxes] = target_tokens
        decoder_input_mask[i, sep_pos + 1:sep_pos + 1 + num_boxes] = True
        labels[i, sep_pos:sep_pos + num_boxes] = target_order

    return {
        "pixel_values": pixel_values,
        "decoder_input_boxes": decoder_input_boxes,
        "decoder_input_boxes_mask": decoder_input_mask,
        "decoder_input_boxes_counts": torch.tensor(model_inputs["input_boxes_counts"], dtype=torch.long),
        "labels": labels,
        "num_boxes_list": num_boxes_list,
        "sep_pos_list": sep_pos_list,
        "image_paths": [sample['image_path'] for sample in batch],
    }

def predict_collate_fn(batch,processor,dtype=torch.float16):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # 解压原始数据
    images = [Image.open(sample['image']).convert("RGB") for sample in batch]
    bboxes = [item['bboxes'] for item in batch]
    orders = [item['mathpix_orders'] for item in batch]
    labels = [item['label'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    json_paths = [item['json_path'] for item in batch]
    
    # 批量预处理
    model_inputs = processor(images=deepcopy(images), boxes=deepcopy(bboxes))
    
    # 转换为Tensor并移动到设备
    pixel_values = torch.tensor(
        np.array(model_inputs["pixel_values"]),
        dtype=dtype
    )
    
    input_boxes = torch.from_numpy(
        np.array(model_inputs["input_boxes"], dtype=np.int32)
    )
    
    input_boxes_mask = torch.from_numpy(
        np.array(model_inputs["input_boxes_mask"], dtype=np.int32)
    )
    
    input_boxes_counts = torch.tensor(
        np.array(model_inputs["input_boxes_counts"]),
        dtype=torch.long
    )
    result = {
        "pixel_values": pixel_values,
        "decoder_input_boxes": input_boxes,
        "decoder_input_boxes_mask": input_boxes_mask,
        "decoder_input_boxes_counts": input_boxes_counts,
        'images': images,
        'bboxes': bboxes,
        'mathpix_orders':orders,
        'label':labels,
        'image_paths':image_paths
    }
    return result

class BBoxLengthSampler(Sampler):
    """根据bbox数量分桶的sampler，确保每个batch内bbox长度相近"""
    def __init__(self, dataset, batch_size, num_buckets=10, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 统计每个样本的bbox数量
        self.bbox_counts = []
        for i in tqdm(range(len(dataset)),desc='Init for BBoxLengthSampler'):
            sample = dataset[i]
            if sample is not None:
                self.bbox_counts.append(len(sample['bboxes']))
            else:
                self.bbox_counts.append(0)
        
        # 创建分桶
        self.num_buckets = num_buckets
        self._create_buckets()
    
    def _create_buckets(self):
        """根据bbox数量创建分桶"""
        counts = np.array(self.bbox_counts)
        percentiles = np.linspace(0, 100, self.num_buckets + 1)
        self.bucket_limits = np.percentile(counts[counts > 0], percentiles)
        
        # 分配样本到桶
        self.buckets = [[] for _ in range(self.num_buckets)]
        for idx, count in enumerate(self.bbox_counts):
            if count == 0:
                continue
            bucket_idx = np.searchsorted(self.bucket_limits, count, side='right') - 1
            bucket_idx = max(0, min(bucket_idx, self.num_buckets - 1))
            self.buckets[bucket_idx].append(idx)

        if self.shuffle:
            for bucket in self.buckets:
                np.random.shuffle(bucket)
        
        # 从每个桶中轮流取样本
        self.batches = []
        for bucket in self.buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    self.batches.append(batch)
    
    def __iter__(self):      
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return sum(len(bucket) // self.batch_size for bucket in self.buckets)
        else:
            return sum((len(bucket) + self.batch_size - 1) // self.batch_size 
            for bucket in self.buckets)

class DistributedSamplerWrapper(DistributedSampler):
    """
    支持自定义batch sampler的分布式包装器
    特点:
    1. 按全局批次索引分配，而非样本索引
    2. 保证各进程的批次shuffle一致性
    3. 支持动态epoch设置
    """
    
    def __init__(self, sampler, num_replicas=None, rank=None, shuffle=True):
        super().__init__(sampler.dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.sampler = sampler
        self.epoch = 0  # 用于控制shuffle的随机种子
        # 生成所有批次的列表 (需要保证各进程生成顺序一致)
        self.all_batches = list(self.sampler)
        # 计算需要填充的元素数量
        remainder = len(self.all_batches) % 4
        if remainder != 0:
            num_fill = 4 - remainder
        else:
            num_fill = 0

        # 使用最后一个元素填充
        if num_fill > 0 and len(self.all_batches) > 0:
            fill_element = self.all_batches[-1]
            self.all_batches += [fill_element] * num_fill

    def __iter__(self):
        # 按全局批次索引分配
        selected_batches = [
            batch 
            for idx, batch in enumerate(self.all_batches)
            if idx % self.num_replicas == self.rank
        ]

        # # 保证shuffle的跨进程一致性
        # if self.shuffle:
        #     g = torch.Generator()
        #     g.manual_seed(self.epoch)
        #     indices = torch.randperm(len(selected_batches), generator=g).tolist()
        #     selected_batches = [selected_batches[i] for i in indices]

        return iter(selected_batches)

    def __len__(self):
        # 计算当前rank应处理的批次数
        total_batches = len(self.sampler)
        return (total_batches + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, epoch):
        """同步随机种子"""
        self.epoch = epoch
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

def load_dataset_sampler(text_file, rank, world_size, batch_size=4, is_validation=False, data_type='custom'):
    """
    从文本文件加载数据集并返回 DataLoader。

    Args:
        text_file (str): 包含图像路径和JSON路径的文本文件路径。
        processor: 数据处理器，用于处理图像和边界框。
        batch_size (int, optional): 批量大小。默认为 4。
        pin_memory (bool, optional): 是否启用 pin_memory。默认为 True。

    Returns:
        DataLoader: 数据加载器。
    """
    # 从文本文件中读取路径
    with open(text_file, 'r') as f:
        lines = f.readlines()

    data_paths = []
    for line in lines:
        img_path, json_path = line.strip().split()
        data_paths.append((img_path, json_path))

    if not data_paths:
        raise ValueError("未找到有效训练数据")

    # 创建数据集与数据加载器
    dataset = OrderDataset(data_paths,data_type=data_type)

    if is_validation:
        shuffle = False
        drop_last = True
    else:
        shuffle = True
        drop_last = True
        

    base_sampler = BBoxLengthSampler(
            dataset, 
            batch_size=batch_size, 
            num_buckets=2,
            shuffle = shuffle,
            drop_last = drop_last
        )

    sampler = DistributedSamplerWrapper(
            base_sampler,
            num_replicas=world_size,
            rank=rank
        )

    return dataset,sampler

def load_dataset(text_file, processor, rank, world_size, batch_size=4, num_workers=4, pin_memory=True, is_validation=False, data_type='custom'):
    """
    从文本文件加载数据集并返回 DataLoader。

    Args:
        text_file (str): 包含图像路径和JSON路径的文本文件路径。
        processor: 数据处理器，用于处理图像和边界框。
        batch_size (int, optional): 批量大小。默认为 4。
        pin_memory (bool, optional): 是否启用 pin_memory。默认为 True。

    Returns:
        DataLoader: 数据加载器。
    """
    # 从文本文件中读取路径
    with open(text_file, 'r') as f:
        lines = f.readlines()

    data_paths = []
    for line in lines:
        img_path, json_path = line.strip().split()
        data_paths.append((img_path, json_path))

    if not data_paths:
        raise ValueError("未找到有效训练数据")

    # 创建数据集与数据加载器
    dataset = OrderDataset(data_paths,data_type=data_type)

    if is_validation:
        shuffle = False
        drop_last = True
    else:
        shuffle = True
        drop_last = True
        

    base_sampler = BBoxLengthSampler(
            dataset, 
            batch_size=batch_size, 
            num_buckets=2,
            shuffle = shuffle,
            drop_last = drop_last
        )

    sampler = DistributedSamplerWrapper(
            base_sampler,
            num_replicas=world_size,
            rank=rank
        )
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(custom_collate_fn, processor=processor),
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return dataloader, sampler