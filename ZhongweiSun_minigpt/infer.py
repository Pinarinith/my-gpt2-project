import os
import json
import torch
from datetime import datetime

from model import MiniGPT


def load_model_for_inference(model_path, config_path, vocab_size, device):
    """加载训练好的模型用于推理"""
    # 加载配置
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # 从嵌套结构中获取模型配置
    config = config_data.get("config", {})

    # 构建模型结构
    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=config.get("block_size", 128),  # 如果配置中没有block_size，使用默认值128
        n_layer=config.get("n_layer", 4),  # 提供默认值以防配置缺失
        n_head=config.get("n_head", 4),
        n_embd=config.get("n_embd", 256),
        dropout=config.get("dropout", 0.1)
    ).to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"模型已从 {model_path} 加载")
    print(f"模型配置: {config}")
    return model, config


def generate_text_sample(model, start_string, max_new_tokens, temperature, top_k, char_to_idx, idx_to_char, device):
    """生成文本样本"""
    model.eval()

    # 将起始字符串转换为索引
    try:
        start_indices = [char_to_idx[s] for s in start_string]
    except KeyError as e:
        print(f"警告：起始文本中包含未知字符 {e}，将被跳过")
        start_indices = [char_to_idx[s] for s in start_string if s in char_to_idx]
        if not start_indices:
            print("错误：起始文本不包含任何已知字符")
            return ""

    input_tensor = torch.tensor([start_indices], dtype=torch.long, device=device)

    # 生成文本
    with torch.no_grad():
        generated_indices = model.generate(
            input_tensor,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None
        )

    # 转换回字符
    generated_chars = [idx_to_char[idx.item()] for idx in generated_indices[0]]
    return "".join(generated_chars)


def save_inference_results(output_dir, results_data, args):
    """保存推理结果到文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存详细结果到JSON文件
    json_filename = f"inference_results_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    print(f"详细结果已保存到: {json_path}")

    # 保存纯文本输出
    txt_filename = f"generated_text_{timestamp}.txt"
    txt_path = os.path.join(output_dir, txt_filename)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"MiniGPT 文本生成结果\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"起始文本: '{args.seed_text}'\n")
        f.write(f"温度: {args.temperature}\n")
        f.write(f"Top-k: {args.top_k if args.top_k > 0 else 'None'}\n")
        f.write(f"最大新token数: {args.max_new_tokens}\n")
        f.write("=" * 80 + "\n\n")

        for i, sample in enumerate(results_data['samples']):
            f.write(f"=== 生成样本 {i + 1} ===\n")
            f.write(f"参数: 温度={sample['temperature']}, Top-k={sample['top_k']}\n")
            f.write(f"生成文本:\n{sample['text']}\n")
            f.write("-" * 80 + "\n\n")

    print(f"文本结果已保存到: {txt_path}")

    # 保存每个样本到单独的文件
    samples_dir = os.path.join(output_dir, f"samples_{timestamp}")
    os.makedirs(samples_dir, exist_ok=True)

    for i, sample in enumerate(results_data['samples']):
        sample_filename = f"sample_{i + 1}_temp{sample['temperature']}_topk{sample['top_k']}.txt"
        sample_path = os.path.join(samples_dir, sample_filename)

        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample['text'])

    print(f"单独样本文件已保存到: {samples_dir}")

    return json_path, txt_path, samples_dir


def run_inference(args, char_to_idx, idx_to_char, vocab_size, device):
    """运行推理模式"""
    # 确定模型和配置文件路径
    if args.load_model and args.load_config:
        model_path = args.load_model
        config_path = args.load_config
        # 使用模型文件所在的目录作为输出目录
        output_dir = os.path.dirname(model_path)
    else:
        # 使用默认的最佳模型路径
        model_path = os.path.join(args.model_save_dir, "best_minigpt_model.pth")
        config_path = os.path.join(args.model_save_dir, "best_minigpt_config.json")
        # 使用模型保存目录作为输出目录
        output_dir = args.model_save_dir

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        return
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在: {config_path}")
        return

    # 加载模型
    model, config = load_model_for_inference(model_path, config_path, vocab_size, device)

    print(f"\n开始文本生成...")
    print(f"起始文本: '{args.seed_text}'")
    print(f"温度: {args.temperature}")
    print(f"Top-k: {args.top_k if args.top_k > 0 else 'None'}")
    print(f"最大新token数: {args.max_new_tokens}")
    print(
        f"模型配置: 层数={config.get('n_layer')}, 头数={config.get('n_head')}, 嵌入维度={config.get('n_embd')}, dropout={config.get('dropout')}")
    print("-" * 80)

    # 准备结果数据结构
    results_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'config_path': config_path,
            'seed_text': args.seed_text,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'max_new_tokens': args.max_new_tokens,
            'num_samples': args.num_samples,
            'model_config': config
        },
        'samples': []
    }

    # 生成多个样本
    for i in range(args.num_samples):
        generated_text = generate_text_sample(
            model,
            start_string=args.seed_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            device=device
        )

        print(f"\n=== 生成样本 {i + 1} ===")
        print(generated_text)
        print("-" * 80)

        # 添加到结果数据
        results_data['samples'].append({
            'sample_id': i + 1,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'text': generated_text
        })

    # 如果只生成一个样本，可以尝试不同的参数组合
    if args.num_samples == 1:
        print("\n=== 额外样本（不同参数） ===")

        # 样本2：更低温度
        print(f"\n--- 低温度样本 (温度=0.5) ---")
        generated_text_2 = generate_text_sample(
            model,
            start_string=args.seed_text,
            max_new_tokens=args.max_new_tokens,
            temperature=0.5,
            top_k=args.top_k,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            device=device
        )
        print(generated_text_2)

        results_data['samples'].append({
            'sample_id': 2,
            'temperature': 0.5,
            'top_k': args.top_k,
            'text': generated_text_2,
            'note': '低温度样本'
        })

        # 样本3：无top-k限制
        print(f"\n--- 无Top-k限制样本 ---")
        generated_text_3 = generate_text_sample(
            model,
            start_string=args.seed_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=0,  # 不使用top-k
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            device=device
        )
        print(generated_text_3)

        results_data['samples'].append({
            'sample_id': 3,
            'temperature': args.temperature,
            'top_k': 0,
            'text': generated_text_3,
            'note': '无Top-k限制样本'
        })

    # 保存结果到模型文件目录
    json_path, txt_path, samples_dir = save_inference_results(output_dir, results_data, args)

    print(f"\n推理完成！结果已保存到模型目录:")
    print(f"- 模型目录: {output_dir}")
    print(f"- JSON格式: {json_path}")
    print(f"- 文本格式: {txt_path}")
    print(f"- 单独样本: {samples_dir}")


def save_single_sample(text, output_dir, filename_prefix, temperature, top_k):
    """保存单个样本到文件"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_temp{temperature}_topk{top_k}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

    return filepath