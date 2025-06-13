import os
import json
import torch
from tqdm import tqdm
from model import create_model_from_config, load_hyperparam_configs, save_model_and_config
import copy


def get_config_folder_name(config):
    """根据配置生成文件夹名称"""
    return f"n_layer_{config['n_layer']}_n_head_{config['n_head']}_n_embd_{config['n_embd']}_dropout_{config['dropout']}_lr_{config['learning_rate']}"


def train_model(model, train_dataloader, val_dataloader, optimizer, device, epochs=10, eval_interval=500, save_dir=None,
                config=None):
    """模型训练函数"""
    model.to(device)
    model.train()

    # 用于追踪最佳模型
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    best_batch = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", leave=False)

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()

            # 更新进度条
            progress_bar.set_postfix(loss=loss.item())

            # 定期评估验证集损失
            if (batch_idx + 1) % eval_interval == 0:
                val_loss = evaluate_model(model, val_dataloader, device)
                print(f"  Batch {batch_idx + 1}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}")

                # 如果是最佳模型，保存状态
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    best_batch = batch_idx + 1
                    print(f"    * 新的最佳验证损失: {best_val_loss:.4f}")

                    # 保存最佳模型
                    if save_dir and config:
                        best_model_path = os.path.join(save_dir, "best_model.pth")
                        torch.save(best_model_state, best_model_path)

                        # 保存最佳模型的训练信息
                        best_info = {
                            'val_loss': best_val_loss,
                            'epoch': best_epoch,
                            'batch': best_batch,
                            'config': config
                        }
                        best_info_path = os.path.join(save_dir, "best_model_info.json")
                        with open(best_info_path, 'w') as f:
                            json.dump(best_info, f, indent=4)

                model.train()  # 确保回到训练模式

    print("Training complete.")
    print(f"最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch}, Batch {best_batch})")

    return model, best_model_state, best_val_loss


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """模型评估函数"""
    model.eval()
    total_loss = 0
    total_batches = 0
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False, ncols=80)

    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        total_loss += loss.item()
        total_batches += 1
        progress_bar.update()

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    return avg_loss


def train_single_model(args, train_dataloader, val_dataloader, vocab_size, device):
    """训练单个模型"""
    config = {
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_embd': args.n_embd,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate
    }

    if args.verbose:
        print(f"训练配置: {config}")

    # 创建模型特定的保存目录
    config_folder = get_config_folder_name(config)
    model_save_dir = os.path.join(args.model_save_dir, config_folder)
    os.makedirs(model_save_dir, exist_ok=True)

    # 创建模型
    if args.load_model and args.load_config:
        # 加载预训练模型
        with open(args.load_config, 'r') as f:
            loaded_config = json.load(f)
        model = create_model_from_config(vocab_size, args.block_size, loaded_config, device)
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"已加载预训练模型: {args.load_model}")
    else:
        model = create_model_from_config(vocab_size, args.block_size, config, device)

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # 训练
    eval_interval = args.eval_interval or len(train_dataloader) // 2
    trained_model, best_model_state, best_val_loss = train_model(
        model, train_dataloader, val_dataloader, optimizer, device,
        args.epochs, eval_interval, model_save_dir, config
    )

    # 保存最终训练完成的模型
    final_model_path, final_config_path = save_model_and_config(
        trained_model, config, model_save_dir, "final_model"
    )
    print(f"最终模型已保存至: {final_model_path}")
    print(f"配置已保存至: {final_config_path}")

    # 保存最佳模型（如果存在）
    if best_model_state is not None:
        best_model = create_model_from_config(vocab_size, args.block_size, config, device)
        best_model.load_state_dict(best_model_state)
        best_model_path, best_config_path = save_model_and_config(
            best_model, config, model_save_dir, "best_model"
        )
        print(f"最佳模型已保存至: {best_model_path}")

    return trained_model, config


def hyperparam_search(args, train_dataloader, val_dataloader, vocab_size, device):
    """超参数搜索"""
    configs = load_hyperparam_configs(args.config_file)

    overall_best_val_loss = float('inf')
    overall_best_config = None
    overall_best_model_state = None
    overall_best_save_dir = None

    print(f"开始超参数搜索，共{len(configs)}个配置")

    # 创建总的保存目录
    base_save_dir = os.path.join(args.model_save_dir, "hyperparam_search")
    os.makedirs(base_save_dir, exist_ok=True)

    results_summary = []

    for i, config in enumerate(configs):
        print(f"\n配置 {i + 1}/{len(configs)}: {config}")

        # 为每个配置创建专门的文件夹
        config_folder = get_config_folder_name(config)
        config_save_dir = os.path.join(base_save_dir, f"config_{i + 1:02d}_{config_folder}")
        os.makedirs(config_save_dir, exist_ok=True)

        # 创建模型
        model = create_model_from_config(vocab_size, args.block_size, config, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

        # 训练
        eval_interval = args.eval_interval or len(train_dataloader) // 2
        trained_model, best_model_state, best_val_loss = train_model(
            model, train_dataloader, val_dataloader, optimizer, device,
            args.epochs, eval_interval, config_save_dir, config
        )

        # 最终评估
        final_val_loss = evaluate_model(trained_model, val_dataloader, device)
        print(f"配置 {i + 1} - 最佳验证损失: {best_val_loss:.4f}, 最终验证损失: {final_val_loss:.4f}")

        # 保存最终模型
        final_model_path, final_config_path = save_model_and_config(
            trained_model, config, config_save_dir, "final_model"
        )

        # 记录结果
        result = {
            'config_id': i + 1,
            'config': config,
            'best_val_loss': best_val_loss,
            'final_val_loss': final_val_loss,
            'save_dir': config_save_dir,
            'final_model_path': final_model_path,
            'final_config_path': final_config_path
        }
        results_summary.append(result)

        # 更新全局最佳模型
        if best_val_loss < overall_best_val_loss:
            overall_best_val_loss = best_val_loss
            overall_best_config = config
            overall_best_model_state = best_model_state
            overall_best_save_dir = config_save_dir
            print(f"发现新的全局最佳模型，验证损失: {overall_best_val_loss:.4f}")

    # 保存全局最佳模型
    if overall_best_model_state is not None:
        overall_best_model = create_model_from_config(vocab_size, args.block_size, overall_best_config, device)
        overall_best_model.load_state_dict(overall_best_model_state)

        global_best_dir = os.path.join(base_save_dir, "global_best")
        os.makedirs(global_best_dir, exist_ok=True)

        model_path, config_path = save_model_and_config(
            overall_best_model, overall_best_config, global_best_dir, "global_best_model"
        )
        print(f"\n全局最佳模型已保存至: {model_path}")
        print(f"全局最佳配置已保存至: {config_path}")
        print(f"全局最佳验证损失: {overall_best_val_loss:.4f}")

    # 保存搜索结果摘要
    summary_path = os.path.join(base_save_dir, "search_results_summary.json")
    search_summary = {
        'total_configs': len(configs),
        'overall_best_val_loss': overall_best_val_loss,
        'overall_best_config': overall_best_config,
        'results': results_summary
    }
    with open(summary_path, 'w') as f:
        json.dump(search_summary, f, indent=4)
    print(f"搜索结果摘要已保存至: {summary_path}")

    return overall_best_model, overall_best_config


