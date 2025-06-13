import os
import json
import torch
from tqdm import tqdm
from model import create_model_from_config
import copy
from datetime import datetime


@torch.no_grad()
def test_model(model, dataloader, device, output_file=None):
    """使用多种标准测试模型"""
    model.eval()

    # 初始化测试指标
    total_loss = 0
    total_tokens = 0
    total_correct_predictions = 0
    total_perplexity = 0
    batch_count = 0

    # 用于计算更详细的指标
    all_losses = []
    token_accuracies = []

    print("开始模型测试...")
    progress_bar = tqdm(dataloader, desc="Testing", leave=True)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size, seq_len = inputs.shape

        # 前向传播
        logits, loss = model(inputs, targets)

        # 1. 基本损失统计
        batch_loss = loss.item()
        total_loss += batch_loss
        all_losses.append(batch_loss)

        # 2. Token级别准确率
        predictions = torch.argmax(logits, dim=-1)
        # 只计算非padding token的准确率（假设padding token为0）
        valid_mask = targets != 0  # 根据实际情况调整padding token值
        if valid_mask.sum() > 0:
            correct_predictions = (predictions == targets) & valid_mask
            token_accuracy = correct_predictions.sum().item() / valid_mask.sum().item()
            token_accuracies.append(token_accuracy)
            total_correct_predictions += correct_predictions.sum().item()
            total_tokens += valid_mask.sum().item()

        # 3. 困惑度计算
        # 困惑度 = exp(平均交叉熵损失)
        perplexity = torch.exp(loss).item()
        total_perplexity += perplexity

        batch_count += 1

        # 更新进度条
        avg_loss = total_loss / batch_count
        avg_perplexity = total_perplexity / batch_count
        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'PPL': f'{avg_perplexity:.2f}'
        })

    # 计算最终指标
    test_results = {}

    # 1. 平均损失
    test_results['average_loss'] = total_loss / batch_count

    # 2. 总体Token准确率
    test_results['token_accuracy'] = total_correct_predictions / total_tokens if total_tokens > 0 else 0

    # 3. 平均困惑度
    test_results['average_perplexity'] = total_perplexity / batch_count

    # 4. 损失分布统计
    test_results['loss_statistics'] = {
        'min_loss': min(all_losses),
        'max_loss': max(all_losses),
        'std_loss': torch.tensor(all_losses).std().item(),
        'median_loss': torch.tensor(all_losses).median().item()
    }

    # 5. Token准确率分布统计（如果有数据）
    if token_accuracies:
        test_results['accuracy_statistics'] = {
            'min_accuracy': min(token_accuracies),
            'max_accuracy': max(token_accuracies),
            'std_accuracy': torch.tensor(token_accuracies).std().item(),
            'median_accuracy': torch.tensor(token_accuracies).median().item()
        }

    # 6. 序列级别指标
    test_results.update(calculate_sequence_metrics(model, dataloader, device))

    # 7. 生成质量指标（采样测试）
    test_results.update(calculate_generation_metrics(model, dataloader, device))

    # 打印测试结果到控制台和文件
    print_test_results(test_results, output_file)

    # 保存详细的测试结果到JSON文件
    if output_file:
        save_detailed_results(test_results, output_file)

    return test_results['average_loss']  # 保持与原函数签名一致


@torch.no_grad()
def calculate_sequence_metrics(model, dataloader, device, max_samples=100):
    """计算序列级别的指标"""
    model.eval()

    sequence_accuracies = []
    exact_matches = 0
    total_sequences = 0

    # 只取前max_samples个样本进行详细分析
    sample_count = 0

    for inputs, targets in dataloader:
        if sample_count >= max_samples:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs, targets)
        predictions = torch.argmax(logits, dim=-1)

        batch_size = inputs.shape[0]

        for i in range(batch_size):
            if sample_count >= max_samples:
                break

            pred_seq = predictions[i]
            target_seq = targets[i]

            # 计算序列准确率（排除padding）
            valid_mask = target_seq != 0  # 根据实际情况调整
            if valid_mask.sum() > 0:
                seq_correct = (pred_seq == target_seq) & valid_mask
                seq_accuracy = seq_correct.sum().item() / valid_mask.sum().item()
                sequence_accuracies.append(seq_accuracy)

                # 完全匹配检查
                if seq_correct.all():
                    exact_matches += 1

                total_sequences += 1
                sample_count += 1

    return {
        'sequence_metrics': {
            'average_sequence_accuracy': sum(sequence_accuracies) / len(
                sequence_accuracies) if sequence_accuracies else 0,
            'exact_match_rate': exact_matches / total_sequences if total_sequences > 0 else 0,
            'total_sequences_analyzed': total_sequences
        }
    }


@torch.no_grad()
def calculate_generation_metrics(model, dataloader, device, num_samples=10):
    """计算生成质量指标"""
    model.eval()

    # 获取一些输入样本进行生成测试
    generation_results = []

    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_samples:
            break

        inputs = inputs.to(device)
        batch_size, seq_len = inputs.shape

        # 取第一个样本的前半部分作为prompt
        prompt_len = seq_len // 2
        prompt = inputs[0:1, :prompt_len]  # 取第一个样本

        # 生成续写
        generated = generate_sequence(model, prompt, max_new_tokens=seq_len - prompt_len, device=device)

        # 计算生成序列的统计信息
        generated_part = generated[0, prompt_len:]  # 只看新生成的部分

        # 计算生成序列的多样性（unique token比率）
        unique_tokens = len(torch.unique(generated_part))
        total_tokens = len(generated_part)
        diversity_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0

        generation_results.append({
            'diversity_ratio': diversity_ratio,
            'generated_length': total_tokens
        })

    # 计算平均指标
    if generation_results:
        avg_diversity = sum(r['diversity_ratio'] for r in generation_results) / len(generation_results)
        avg_length = sum(r['generated_length'] for r in generation_results) / len(generation_results)
    else:
        avg_diversity = 0
        avg_length = 0

    return {
        'generation_metrics': {
            'average_diversity_ratio': avg_diversity,
            'average_generated_length': avg_length,
            'num_generation_samples': len(generation_results)
        }
    }


def generate_sequence(model, prompt, max_new_tokens=50, temperature=1.0, device='cpu'):
    """生成序列的辅助函数"""
    model.eval()
    prompt = prompt.to(device)

    for _ in range(max_new_tokens):
        # 获取当前序列的logits
        with torch.no_grad():
            logits, _ = model(prompt)

        # 取最后一个位置的logits
        logits = logits[:, -1, :] / temperature

        # 采样下一个token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 添加到序列中
        prompt = torch.cat([prompt, next_token], dim=1)

        # 如果序列过长，截断前面的部分（滑动窗口）
        if prompt.shape[1] > model.block_size:
            prompt = prompt[:, -model.block_size:]

    return prompt


def print_test_results(results, output_file=None):
    """打印测试结果到控制台和文件"""
    # 生成格式化的结果文本
    result_text = generate_result_text(results)

    # 打印到控制台
    print(result_text)

    # 如果指定了输出文件，也写入文件
    if output_file:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_text)

        print(f"\n测试结果已保存到: {output_file}")


def generate_result_text(results):
    """生成格式化的结果文本"""
    lines = []

    # 添加时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"测试时间: {timestamp}")
    lines.append("")

    lines.append("=" * 60)
    lines.append("测试结果摘要")
    lines.append("=" * 60)

    # 基本指标
    lines.append(f"平均损失: {results['average_loss']:.4f}")
    lines.append(f"Token准确率: {results['token_accuracy']:.4f} ({results['token_accuracy'] * 100:.2f}%)")
    lines.append(f"平均困惑度: {results['average_perplexity']:.2f}")

    # 损失统计
    loss_stats = results['loss_statistics']
    lines.append("")
    lines.append("损失分布:")
    lines.append(f"  最小值: {loss_stats['min_loss']:.4f}")
    lines.append(f"  最大值: {loss_stats['max_loss']:.4f}")
    lines.append(f"  中位数: {loss_stats['median_loss']:.4f}")
    lines.append(f"  标准差: {loss_stats['std_loss']:.4f}")

    # Token准确率统计
    if 'accuracy_statistics' in results:
        acc_stats = results['accuracy_statistics']
        lines.append("")
        lines.append("Token准确率分布:")
        lines.append(f"  最小值: {acc_stats['min_accuracy']:.4f}")
        lines.append(f"  最大值: {acc_stats['max_accuracy']:.4f}")
        lines.append(f"  中位数: {acc_stats['median_accuracy']:.4f}")
        lines.append(f"  标准差: {acc_stats['std_accuracy']:.4f}")

    # 序列级别指标
    seq_metrics = results.get('sequence_metrics', {})
    if seq_metrics:
        lines.append("")
        lines.append("序列级别指标:")
        lines.append(f"  平均序列准确率: {seq_metrics['average_sequence_accuracy']:.4f}")
        lines.append(
            f"  完全匹配率: {seq_metrics['exact_match_rate']:.4f} ({seq_metrics['exact_match_rate'] * 100:.2f}%)")
        lines.append(f"  分析序列数: {seq_metrics['total_sequences_analyzed']}")

    # 生成质量指标
    gen_metrics = results.get('generation_metrics', {})
    if gen_metrics:
        lines.append("")
        lines.append("生成质量指标:")
        lines.append(f"  平均词汇多样性: {gen_metrics['average_diversity_ratio']:.4f}")
        lines.append(f"  平均生成长度: {gen_metrics['average_generated_length']:.1f}")
        lines.append(f"  生成样本数: {gen_metrics['num_generation_samples']}")

    lines.append("=" * 60)

    return "\n".join(lines)


def save_detailed_results(results, output_file):
    """保存详细的测试结果到JSON文件"""
    # 创建JSON输出文件名
    json_file = output_file.replace('.txt', '.json') if output_file.endswith('.txt') else output_file + '.json'

    # 添加时间戳到结果中
    results_with_timestamp = {
        'timestamp': datetime.now().isoformat(),
        'test_results': results
    }

    # 保存到JSON文件
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_with_timestamp, f, ensure_ascii=False, indent=2)

    print(f"详细测试结果已保存到JSON文件: {json_file}")


def test_only(args, test_dataloader, vocab_size, device):
    """仅评估模式"""
    if not args.load_model or not args.load_config:
        raise ValueError("评估模式需要指定 --load_model 和 --load_config")

    # 加载配置
    with open(args.load_config, 'r') as f:
        config = json.load(f)

    # 创建并加载模型
    model = create_model_from_config(vocab_size, args.block_size, config, device)
    model.load_state_dict(torch.load(args.load_model, map_location=device))

    print(f"已加载模型: {args.load_model}")
    print(f"模型配置: {config}")

    # 如果没有指定输出文件，使用默认文件名
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"logs/test_results_{timestamp}.txt"

    # 评估
    test_model(model, test_dataloader, device, args.output_file)
    return
