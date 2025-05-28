# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import amp
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import logging
import argparse
import time
import psutil
import GPUtil
from datetime import datetime
# from modules.surya_order.settings import settings

# 导入数据预处理模块
from dataset_bucket import load_dataset
from model import load_model, load_processor

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

RESULT_DIR = './result_v3'

# 日志配置
def setup_logger(log_dir, rank=0):
    """设置日志记录器，将日志输出到文件和控制台"""
    if rank != 0:  # 只在主进程中记录日志
        return None
    
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取当前时间戳作为日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除已有handlers
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器并添加到handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 早停机制的类
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def get_gpu_memory(rank):
    """获取指定GPU的 mem情况"""
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            return gpus[rank].memoryUsed
        except:
            return -1
    return 0

def get_cpu_memory():
    """获取CPU mem情况"""
    return psutil.virtual_memory().used / (1024 ** 3)  # GB

class TrainingMonitor:
    """训练监控器，用于记录训练过程中的各种指标"""
    def __init__(self, log_dir, rank=0):
        self.rank = rank
        self.log_dir = log_dir
        self.batch_logs = []
        self.epoch_logs = []
        
        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            self.batch_log_file = os.path.join(log_dir, "batch_loss.csv")
            self.epoch_log_file = os.path.join(log_dir, "epoch_metrics.csv")
            
            # 创建并初始化CSV文件
            with open(self.batch_log_file, 'w') as f:
                f.write("epoch,batch_idx,loss,gpu_memory,cpu_memory,time\n")
                
            with open(self.epoch_log_file, 'w') as f:
                f.write("epoch,train_loss,val_loss,time,gpu_memory,cpu_memory\n")
    
    def log_batch(self, epoch, batch_idx, loss, gpu_memory, cpu_memory, time):
        """记录batch级别的指标"""
        if self.rank != 0:
            return
            
        self.batch_logs.append({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'loss': loss,
            'gpu_memory': gpu_memory,
            'cpu_memory': cpu_memory,
            'time': time
        })
        
        # 每100个batch写入一次文件
        # if len(self.batch_logs) >= 100:
        self._write_batch_logs()
    
    def log_epoch(self, epoch, train_loss, val_loss, time, gpu_memory, cpu_memory):
        """记录epoch级别的指标"""
        if self.rank != 0:
            return
            
        self.epoch_logs.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time': time,
            'gpu_memory': gpu_memory,
            'cpu_memory': cpu_memory
        })
        
        self._write_epoch_logs()
        self._write_batch_logs()  # 确保所有batch日志都被写入
        
        # 绘制当前的训练曲线
        self._plot_training_metrics()
    
    def _write_batch_logs(self):
        """将batch日志写入CSV文件"""
        if not self.batch_logs:
            return
            
        with open(self.batch_log_file, 'a') as f:
            for log in self.batch_logs:
                f.write(f"{log['epoch']},{log['batch_idx']},{log['loss']},{log['gpu_memory']},{log['cpu_memory']},{log['time']}\n")
                
        self.batch_logs = []
    
    def _write_epoch_logs(self):
        """将epoch日志写入CSV文件"""
        if not self.epoch_logs:
            return
            
        with open(self.epoch_log_file, 'a') as f:
            for log in self.epoch_logs:
                f.write(f"{log['epoch']},{log['train_loss']},{log['val_loss']},{log['time']},{log['gpu_memory']},{log['cpu_memory']}\n")
    
    def _plot_training_metrics(self):
        """绘制训练指标图表"""
        try:
            # 读取数据
            epoch_df = pd.read_csv(self.epoch_log_file)
            batch_df = pd.read_csv(self.batch_log_file)
            
            # 创建图表目录
            plot_dir = os.path.join(self.log_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # 绘制Epoch损失曲线
            plt.figure(figsize=(12, 6))
            plt.plot(epoch_df['epoch'], epoch_df['train_loss'], label='train loss')
            if 'val_loss' in epoch_df.columns and not epoch_df['val_loss'].isna().all():
                plt.plot(epoch_df['epoch'], epoch_df['val_loss'], label='val loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Train and Val Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, "epoch_loss.png"))
            plt.close()
            
            # 绘制Batch损失曲线
            plt.figure(figsize=(12, 6))
            plt.plot(batch_df['batch_idx'], batch_df['loss'])
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title('Batch Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, "batch_loss.png"))
            plt.close()
            
            # 绘制资源使用情况
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(epoch_df['epoch'], epoch_df['gpu_memory'], label='GPU mem (MB)')
            plt.xlabel('Epoch')
            plt.ylabel('Mem')
            plt.title('GPU mem')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(epoch_df['epoch'], epoch_df['cpu_memory'], label='CPU mem (GB)')
            plt.xlabel('Epoch')
            plt.ylabel(' mem')
            plt.title('CPU mem')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "memory_usage.png"))
            plt.close()
            
        except Exception as e:
            print(f"绘制图表时出错: {e}")

def train_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, monitor, rank, init_batch_idx, USE_AMP=False):
    model.train()
    epoch_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")
    start_time = time.time()
    
    for batch_idx, batch in enumerate(progress):
        if batch is None: continue
        
        batch_start_time = time.time()
        
        optimizer.zero_grad(set_to_none=True)
        
        # 获取资源使用情况
        gpu_memory = get_gpu_memory(rank)
        cpu_memory = get_cpu_memory()
        
        # AMP自动管理混合精度
        with amp.autocast('cuda', enabled=USE_AMP, dtype=torch.float16):
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                decoder_input_boxes=batch["decoder_input_boxes"].to(device),
                decoder_input_boxes_mask=batch["decoder_input_boxes_mask"].to(device),
                decoder_input_boxes_counts=batch["decoder_input_boxes_counts"].to(device),
            )
            logits = outputs.logits.transpose(1, 2)
            loss = criterion(logits, batch["labels"].to(device))

        # 根据USE_AMP选择反向传播方式
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_loss += loss.item()
        progress.set_postfix(loss=loss.item())
        
        batch_time = time.time() - batch_start_time
        
        # 记录batch级别的监控数据
        monitor.log_batch(epoch, init_batch_idx+batch_idx, loss.item(), gpu_memory, cpu_memory, batch_time)
    
    avg_loss = epoch_loss / len(dataloader)
    epoch_time = time.time() - start_time
    
    # 获取资源使用情况
    gpu_memory = get_gpu_memory(rank)
    cpu_memory = get_cpu_memory()

    final_batch_idx = init_batch_idx+batch_idx
    
    return avg_loss, gpu_memory, cpu_memory, epoch_time, final_batch_idx

def validate_epoch(model, dataloader, criterion, device, rank, world_size):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    # 确保所有进程的进度条不会冲突，仅在rank 0显示
    if rank == 0:
        progress = tqdm(dataloader, desc="Validation", unit="batch")
    else:
        progress = dataloader

    with torch.no_grad():
        for batch in progress:
            if batch is None:
                continue
            # 确保数据移动到对应设备
            # batch = {k: v.to(device) for k, v in batch.items()}
            
            # 计算损失
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                decoder_input_boxes=batch["decoder_input_boxes"].to(device),
                decoder_input_boxes_mask=batch["decoder_input_boxes_mask"].to(device),
                decoder_input_boxes_counts=batch["decoder_input_boxes_counts"].to(device),
            )
            logits = outputs.logits.transpose(1, 2)
            loss = criterion(logits, batch["labels"].to(device))
            batch_size = batch["labels"].to(device).size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    # 同步所有进程的损失和样本数
    total_loss = torch.tensor(total_loss, device=device, dtype=torch.float32)
    total_samples = torch.tensor(total_samples, device=device, dtype=torch.float32)
    
    # 确保分布式环境已初始化
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    else:
        # 如果未初始化分布式环境，则直接计算平均值
        avg_loss = total_loss.item() / total_samples.item()
        return avg_loss

    avg_loss = total_loss.item() / total_samples.item()
    return avg_loss

def main(rank, world_size, use_multi_gpu, args):
    # 创建保存监控数据的目录
    monitor_dir = os.path.join(RESULT_DIR, "monitoring")
    os.makedirs(monitor_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(monitor_dir, rank)
    if rank == 0:
        logger.info(f"监控数据将保存到: {monitor_dir}")
    
    # 参数定义
    checkpoint = args.checkpoint  # 模型检查点路径
    train_dataset_path = args.train_dataset_path  # 数据集路径
    batch_size = args.batch_size  # 每个 GPU 的批次大小
    USE_AMP = args.use_amp  # 是否使用混合精度训练
    TRAIN_EPOCHS = args.epochs  # 训练轮数
    SAVE_EPOCH_INTERVAL = args.save_interval  # 每隔多少轮保存一次模型
    LEARNING_RATE = args.lr  # 学习率
    WEIGHT_DECAY = args.weight_decay  # 权重衰减
    patience = args.patience  # 早停机制的耐心值
    use_validation = args.use_validation  # 是否使用验证集

    if use_multi_gpu:
        setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 加载模型与处理器
    processor = load_processor(checkpoint)
    model = load_model(checkpoint,dtype=torch.float32)

    model = model.to(device)

    if use_multi_gpu:
        # 将模型包装为 DDP
        model = DDP(model, device_ids=[rank])

    # 加载数据集时传入 rank 和 world_size
    dataloader, train_sampler = load_dataset(
        train_dataset_path, processor, rank, world_size,
        batch_size = batch_size, 
        num_workers= 2, 
        is_validation=False
    )
    if use_validation:
        val_dataloader, val_sampler = load_dataset(
        args.val_dataset_path, processor, rank, world_size,
        batch_size= 4,
        num_workers= 1,
        is_validation=True
        )

    # 初始化优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 初始化GradScaler（仅在USE_AMP=True时）
    scaler = amp.GradScaler(enabled=USE_AMP)

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=patience)
    
    # 初始化训练监控器
    monitor = TrainingMonitor(monitor_dir, rank)

    # 训练循环
    if rank == 0:
        logger.info(f"\n开始训练 | 设备: {device} | 混合精度: {USE_AMP} | 批次大小: {batch_size*world_size}")
    
    # 初始化batch_idx
    init_batch_idx = 0

    for epoch in range(TRAIN_EPOCHS):
        train_sampler.set_epoch(epoch)  # 设置 epoch 以正确 shuffle 数据
        
        # 训练一个epoch
        train_loss, gpu_memory, cpu_memory, epoch_time, final_batch_idx = train_epoch(
            model, dataloader, optimizer, criterion, scaler, device, epoch, monitor, rank, init_batch_idx, USE_AMP=USE_AMP
        )

        init_batch_idx = final_batch_idx

        if use_validation:
            # 所有进程参与验证
            val_loss = validate_epoch(model, val_dataloader, criterion, device, rank, world_size)
        else:
            val_loss = None

        # 记录epoch级别的监控数据
        monitor.log_epoch(epoch, train_loss, val_loss, epoch_time, gpu_memory, cpu_memory)
        
        if rank == 0:
            # 处理验证损失的格式化问题
            if val_loss is not None:
                val_loss_str = f"{val_loss:.4f}"
            else:
                val_loss_str = "N/A"

            logger.info(f"Epoch {epoch+1}/{TRAIN_EPOCHS} | 平均损失: {train_loss:.4f} | 验证损失: {val_loss_str} | 耗时: {epoch_time:.2f}s | GPU内存: {gpu_memory}MB | CPU内存: {cpu_memory:.2f}GB")

            # 检查早停机制
            if use_validation:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    logger.info("早停机制触发，训练提前终止。")
                    break

            # 保存检查点
            if (epoch + 1) % SAVE_EPOCH_INTERVAL == 0:
                save_path = os.path.join(RESULT_DIR, f"order_model_epoch{epoch+1}")
                if use_multi_gpu:
                    model.module.save_pretrained(save_path)
                else:
                    model.save_pretrained(save_path)
                if USE_AMP:
                    torch.save(scaler.state_dict(), os.path.join(save_path, "scaler_state.pth"))
                logger.info(f"模型检查点已保存至: {save_path}")

    # 保存最终模型
    if rank == 0:
        final_save_path = os.path.join(RESULT_DIR, "order_model_final")
        if use_multi_gpu:
            model.module.save_pretrained(final_save_path)
        else:
            model.save_pretrained(final_save_path)
        logger.info(f"\n训练完成！模型已保存至: {final_save_path}")
        
        # 生成最终的训练报告
        monitor._plot_training_metrics()
        logger.info(f"训练监控数据和图表已保存至: {monitor_dir}")

    if use_multi_gpu:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--checkpoint", type=str, default="./models/vikp/surya_order", help="模型检查点路径")
    parser.add_argument("--train_dataset_path", type=str, default="/root/surya/train_temp/dataset_train_v2.txt", help="训练数据集路径")
    parser.add_argument("--val_dataset_path", type=str, default='/root/surya/train_temp/dataset_val_v2.txt', help="验证数据集路径（可选）")
    parser.add_argument("--batch_size", type=int, default=8, help="每个 GPU 的批次大小")
    parser.add_argument("--use_amp", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--save_interval", type=int, default=1, help="每隔多少轮保存一次模型")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--patience", type=int, default=2, help="早停机制的耐心值")
    parser.add_argument("--use_multi_gpu", action="store_true", help="是否使用多卡训练")
    parser.add_argument("--use_validation", action="store_true", help="是否使用验证集")

    args = parser.parse_args()

    use_multi_gpu = args.use_multi_gpu
    world_size = torch.cuda.device_count() if use_multi_gpu else 1
    
    # 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"使用 {world_size} 个 GPU 进行训练")
    if use_multi_gpu:
        mp.spawn(main, args=(world_size, use_multi_gpu, args), nprocs=world_size, join=True)
    else:
        main(0, 1, use_multi_gpu, args)