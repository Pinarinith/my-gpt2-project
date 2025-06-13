import torch
import argparse
import json
import os
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from train import train_single_model, evaluate_model, hyperparam_search
from test import test_only
from infer import run_inference
from dataset import prepare_dataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MiniGPT训练脚本')

    # 数据相关参数
    parser.add_argument('--data_path', type=str, default='data/tinyshakespeare.txt',
                        help='训练数据文件路径')
    parser.add_argument('--block_size', type=int, default=128,
                        help='上下文长度（序列长度）')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批量大小')

    # 模型结构参数
    parser.add_argument('--n_layer', type=int, default=4,
                        help='Transformer层数')
    parser.add_argument('--n_head', type=int, default=4,
                        help='注意力头数')
    parser.add_argument('--n_embd', type=int, default=256,
                        help='嵌入维数')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout率')

    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='评估间隔（步数），默认为训练集大小的一半')

    # 保存和加载
    parser.add_argument('--model_save_dir', type=str, default='models',
                        help='模型保存目录')
    parser.add_argument('--model_name', type=str, default='minigpt_model',
                        help='模型保存名称（不含扩展名）')
    parser.add_argument('--load_model', type=str, default=None,
                        help='加载预训练模型路径')
    parser.add_argument('--load_config', type=str, default=None,
                        help='加载配置文件路径')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出log的路径')

    # 运行模式
    parser.add_argument('--mode', type=str, choices=['train_eval', 'infer', 'test'],
                        default='train_eval',
                        help='运行模式：train_eval（训练+评估）、infer（推理生成）、test（测试）')
    parser.add_argument('--hyperparam_search', action='store_true',
                        help='是否进行超参数搜索')
    parser.add_argument('--config_file', type=str, default=None,
                        help='超参数搜索配置文件路径')

    # 推理相关参数
    parser.add_argument('--seed_text', type=str, default="ROMEO:",
                        help='推理时的起始文本')
    parser.add_argument('--max_new_tokens', type=int, default=500,
                        help='生成的最大新token数量')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='生成时的温度参数，控制随机性')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k采样参数，设为0或None表示不使用')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='生成样本的数量')

    # 设备设置
    parser.add_argument('--device', type=str, default='auto',
                        help='使用的设备：cuda、cpu或auto')

    # 其他设置
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--verbose', action='store_true',
                        help='是否打印详细信息')

    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    """获取运行设备"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg

    if device == 'cuda' and not torch.cuda.is_available():
        print("警告：CUDA不可用，切换到CPU")
        device = 'cpu'

    return device




def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 获取设备
    device = get_device(args.device)
    print(f"使用设备: {device}")

    # 准备数据集
    print(f"加载数据集: {args.data_path}")

    if args.mode == 'infer':
        # 推理模式只需要vocab映射
        _, _, _, char_to_idx, idx_to_char = prepare_dataset(args.data_path)
        vocab_size = len(char_to_idx)
        print(f"词汇表大小: {vocab_size}")

        # 运行推理
        run_inference(args, char_to_idx, idx_to_char, vocab_size, device)
        return

    # 其他模式需要完整的数据加载
    train_ds, val_ds, test_ds, char_to_idx, idx_to_char = prepare_dataset(args.data_path)
    vocab_size = len(char_to_idx)
    print(f"训练集大小: {len(train_ds)}")
    print(f"验证集大小: {len(val_ds)}")
    print(f"测试集大小: {len(test_ds)}")
    print(f"词汇表大小: {vocab_size}")

    # 创建数据加载器
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 根据模式执行相应操作
    if args.mode == 'train_eval':
        if args.hyperparam_search:
            model, config = hyperparam_search(args, train_dataloader, val_dataloader, vocab_size, device)
        else:
            model, config = train_single_model(args, train_dataloader, val_dataloader, vocab_size, device)

    elif args.mode == 'test':
        test_only(args, test_dataloader, vocab_size, device)


if __name__ == "__main__":
    main()