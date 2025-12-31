"""
DeBERTa模型训练 - 独立运行版本
适用于钢铁领域数据集训练
"""

# ==================== 警告忽略 ====================
import warnings
import numpy as np

# 忽略特定警告
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.warnings.filterwarnings('ignore')
import logging
import torch
import glob
import os
import json
import gc
import time
import shutil
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset, Dataset, DatasetDict
from tokenizers.normalizers import BertNormalizer

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DebertaV2ForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import numpy as np
np.warnings.filterwarnings('ignore')

# ==================== 全局配置区域 ====================

# 数据路径配置
TRAIN_FILE = r"..\data\train_data.txt"
VAL_FILE = r"..\data\val_data.txt"
OUTPUT_DIR = r"model_results_save"

# 基础配置
SEED = 666
PRETRAINED_MODEL = "..\deberta-v3-base"
CACHE_DIR = None

# 词表和序列配置
VOCAB_SIZE = 50000
MAX_SEQ_LENGTH = 256

# 训练配置
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 100

# 学习率策略
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.3
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = 'cosine'

# MLM配置
MLM_PROBABILITY = 0.15

# Early Stopping配置
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_THRESHOLD = 0.00001

# 功能开关
SKIP_NORMALIZATION = False  # 是否跳过文本标准化
SKIP_TOKENIZER_TRAINING = False  # 是否使用预训练tokenizer
USE_PRETRAINED_WEIGHTS = False  # 是否使用预训练权重

# 训练优化配置
GRADIENT_CHECKPOINTING = False  # 梯度检查点
MAX_GRAD_NORM = 0.5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-6

# 保存和日志配置
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 5
LOGGING_STEPS = 50



# 检查并删除现有文件夹
if os.path.exists(OUTPUT_DIR):
    print(f"删除现有文件夹: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)





# ==================== 日志配置 ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 辅助函数 ====================

def print_config():
    """打印当前配置"""
    print("\n" + "=" * 80)
    print("📋 DeBERTa训练配置")
    print("=" * 80)
    print(f"训练数据:          {TRAIN_FILE}")
    print(f"验证数据:          {VAL_FILE}")
    print(f"输出目录:          {OUTPUT_DIR}")
    print(f"预训练模型:        {PRETRAINED_MODEL}")
    print("-" * 80)
    print(f"词表大小:          {VOCAB_SIZE}")
    print(f"最大序列长度:      {MAX_SEQ_LENGTH}")
    print(f"训练批次大小:      {BATCH_SIZE}")
    print(f"梯度累积步数:      {GRADIENT_ACCUMULATION_STEPS}")
    print(f"有效批次大小:      {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"训练轮数:          {NUM_EPOCHS}")
    print("-" * 80)
    print(f"学习率:            {LEARNING_RATE}")
    print(f"Warmup比例:        {WARMUP_RATIO}")
    print(f"学习率调度器:      {LR_SCHEDULER_TYPE}")
    print(f"权重衰减:          {WEIGHT_DECAY}")
    print(f"Mask比例:          {MLM_PROBABILITY}")
    print("-" * 80)
    print(f"使用预训练权重:    {USE_PRETRAINED_WEIGHTS}")
    print(f"跳过标准化:        {SKIP_NORMALIZATION}")
    print(f"跳过tokenizer训练: {SKIP_TOKENIZER_TRAINING}")
    print(f"Early Stop耐心值:  {EARLY_STOPPING_PATIENCE}")
    print("=" * 80 + "\n")


def check_environment():
    """检查运行环境"""
    logger.info("🔍 检查运行环境...")

    # 检查CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"训练设备: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 检查文件
    if not Path(TRAIN_FILE).exists():
        raise FileNotFoundError(f"训练数据文件不存在: {TRAIN_FILE}")
    if not Path(VAL_FILE).exists():
        raise FileNotFoundError(f"验证数据文件不存在: {VAL_FILE}")

    logger.info(f"✓ 训练数据: {TRAIN_FILE}")
    logger.info(f"✓ 验证数据: {VAL_FILE}")

    # 检查预训练模型
    if not Path(PRETRAINED_MODEL).exists():
        raise FileNotFoundError(f"预训练模型不存在: {PRETRAINED_MODEL}")
    logger.info(f"✓ 预训练模型: {PRETRAINED_MODEL}")

    return device


def setup_directories():
    """创建输出目录"""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = output_dir / 'tokenizer'
    tokens_dir = output_dir / 'tokens'
    model_dir = output_dir / 'model_checkpoints'
    final_model_dir = output_dir / 'final_model'

    for d in [tokenizer_dir, tokens_dir, model_dir, final_model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"✓ 输出目录已创建: {output_dir}")

    return output_dir, tokenizer_dir, tokens_dir, model_dir, final_model_dir


def cleanup_memory():
    """清理内存和GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("✓ 内存清理完成")


# ==================== 核心训练函数 ====================

from transformers import TrainerCallback
import json
import os


class TrainingHistoryCallback(TrainerCallback):
    def __init__(self, history_file):
        super().__init__()
        self.history_file = history_file
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        self._create_new_history()

    def _create_new_history(self):
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rates': [],
            'epochs': [],
            'steps': [],
            'train_logs': [],
            'eval_logs': [],
            'log_history': []
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # 首先保存到完整日志历史
        self.history['log_history'].append(logs.copy())

        # 记录所有类型的日志
        if 'loss' in logs and 'eval_loss' not in logs:
            # 训练日志
            train_log = logs.copy()
            self.history['train_logs'].append(train_log)
            self.history['train_loss'].append(train_log.get('loss'))

            # 记录学习率
            if 'learning_rate' in logs:
                self.history['learning_rates'].append(logs['learning_rate'])

            # 记录epoch
            if 'epoch' in logs:
                self.history['epochs'].append(logs['epoch'])

            # 记录step
            if 'step' in logs:
                self.history['steps'].append(logs['step'])

        elif 'eval_loss' in logs:
            # 评估日志
            eval_log = logs.copy()
            self.history['eval_logs'].append(eval_log)
            self.history['eval_loss'].append(eval_log['eval_loss'])

        self._save_history()

    def _save_history(self):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存历史文件时出错: {e}")

def count_training_history_files(folder_path):
    """
    统计文件夹中包含'training_history'的JSON文件数量
    """
    pattern = os.path.join(folder_path, "*training_history*.json")
    matching_files = glob.glob(pattern)

    return len(matching_files)


training_history_num = count_training_history_files(OUTPUT_DIR)

# 创建训练历史文件路径
history_file = os.path.join(OUTPUT_DIR, f'training_history_{training_history_num}.json')
history_callback = TrainingHistoryCallback(history_file)


# 优化器检查回调
class OptimizerCheckCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        """在训练步骤开始时检查优化器（此时优化器已初始化）"""
        trainer = kwargs.get('trainer')
        if trainer and hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            # 只在第一次步骤时打印
            if state.global_step == 0:
                print("\n=== 优化器状态检查 ===")
                print("优化器参数组:")
                for i, param_group in enumerate(trainer.optimizer.param_groups):
                    print(f"  第{i}组 - 学习率: {param_group['lr']}")
                print(f"优化器类型: {type(trainer.optimizer).__name__}")
                print("====================\n")


# 训练监控回调
class TrainingMonitorCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        eval_results = kwargs.get('metrics', {})
        eval_loss = eval_results.get('eval_loss', float('inf'))


class ResumeTrainingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        # 使用正确的方式检查是否从检查点恢复
        if hasattr(state, 'resume_from_checkpoint') and state.resume_from_checkpoint is not None:
            print(f"🔄 从检查点恢复训练: {state.resume_from_checkpoint}")
            print(f"📊 恢复位置: 第 {state.global_step} 步, 第 {state.epoch:.2f} 轮")
        else:
            print("🚀 开始新的训练")




def normalize_text_file(input_file, output_file):
    """标准化文本文件"""
    logger.info(f"正在标准化 {input_file}...")

    normalizer = BertNormalizer(
        lowercase=False,
        strip_accents=True,
        clean_text=True,
        handle_chinese_chars=True
    )

    with open(input_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()

    normalized_texts = [normalizer.normalize_str(text.strip()) for text in tqdm(texts)]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(normalized_texts))

    logger.info(f"标准化完成，保存到 {output_file}")
    return output_file


def train_tokenizer(train_file, val_file, save_dir, vocab_size):
    """训练新的tokenizer"""
    logger.info("正在训练新的tokenizer...")

    train_file_str = str(train_file)
    val_file_str = str(val_file)
    raw_datasets = load_dataset("text", data_files={"train": train_file_str, "val": val_file_str})
    logger.info(f"数据集加载完成: {raw_datasets}")

    def get_training_corpus():
        batch_size = 1000
        for i in range(0, len(raw_datasets["train"]), batch_size):
            yield raw_datasets["train"][i: i + batch_size]["text"]
        for i in range(0, len(raw_datasets["val"]), batch_size):
            yield raw_datasets["val"][i: i + batch_size]["text"]

    old_tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, cache_dir=CACHE_DIR)
    logger.info(f"原tokenizer词表大小: {len(old_tokenizer)}")

    training_corpus = get_training_corpus()
    tokenizer = old_tokenizer.train_new_from_iterator(
        text_iterator=training_corpus,
        vocab_size=vocab_size
    )

    logger.info(f"新tokenizer词表大小: {len(tokenizer)}")
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Tokenizer已保存到 {save_dir}")

    return tokenizer


def tokenize_dataset(train_file, val_file, tokenizer_dir, save_dir, max_seq_length):
    """对数据集进行tokenization"""
    logger.info("正在进行tokenization...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    start_tok = tokenizer.convert_tokens_to_ids('[CLS]')
    sep_tok = tokenizer.convert_tokens_to_ids('[SEP]')

    def full_sent_tokenize(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            sents = f.read().strip().split('\n')

        logger.info(f"处理 {len(sents)} 个句子...")
        tok_sents = [tokenizer(s, padding=False, truncation=False)['input_ids']
                     for s in tqdm(sents, desc="Tokenizing")]

        for s in tok_sents:
            if len(s) > 0:
                s.pop(0)

        res = [[]]
        l_curr = 0

        for s in tok_sents:
            l_s = len(s)
            idx = 0
            while idx < l_s - 1:
                if l_curr == 0:
                    res[-1].append(start_tok)
                    l_curr = 1
                s_end = min(l_s, idx + max_seq_length - l_curr) - 1
                res[-1].extend(s[idx:s_end] + [sep_tok])
                idx = s_end
                if len(res[-1]) == max_seq_length:
                    res.append([])
                l_curr = len(res[-1])

        for s in res[:-1]:
            assert s[0] == start_tok and s[-1] == sep_tok
            assert len(s) == max_seq_length

        attention_mask = []
        for s in res:
            attention_mask.append([1] * len(s) + [0] * (max_seq_length - len(s)))

        return {'input_ids': res, 'attention_mask': attention_mask}

    import pandas as pd
    df_train = pd.DataFrame(full_sent_tokenize(train_file))
    df_val = pd.DataFrame(full_sent_tokenize(val_file))

    tokenized_datasets = DatasetDict({
        'train': Dataset.from_pandas(df_train),
        'val': Dataset.from_pandas(df_val)
    })

    logger.info(f"Tokenized数据集: {tokenized_datasets}")
    logger.info(f"训练样本数: {len(tokenized_datasets['train'])}")
    logger.info(f"验证样本数: {len(tokenized_datasets['val'])}")

    tokenized_datasets.save_to_disk(save_dir)
    logger.info(f"Tokenized数据集已保存到 {save_dir}")

    return tokenized_datasets


def train_model(tokenizer_dir, tokens_dir, model_save_dir, final_save_dir):
    """训练DeBERTa模型"""
    logger.info("正在训练模型...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    if USE_PRETRAINED_WEIGHTS:
        logger.info("使用预训练权重初始化模型 (推荐方式)")
        model = DebertaV2ForMaskedLM.from_pretrained(
            PRETRAINED_MODEL,
            cache_dir=CACHE_DIR,
        )
        model.resize_token_embeddings(len(tokenizer))
    else:
        logger.info("从头训练模型 (不推荐，数据量太小)")
        config = AutoConfig.from_pretrained(PRETRAINED_MODEL, cache_dir=CACHE_DIR)
        model = DebertaV2ForMaskedLM(config=config)
        model.resize_token_embeddings(len(tokenizer))

    model_size = sum(t.numel() for t in model.parameters())
    logger.info(f"模型参数量: {model_size / 1000 ** 2:.1f}M")

    dataset_train = Dataset.load_from_disk(Path(tokens_dir) / 'train')
    dataset_val = Dataset.load_from_disk(Path(tokens_dir) / 'val')
    logger.info(f"训练集大小: {len(dataset_train)}, 验证集大小: {len(dataset_val)}")

    dataset_train.set_format(type='torch', columns=['input_ids'])
    dataset_val.set_format(type='torch', columns=['input_ids'])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY
    )

    total_steps = (len(dataset_train) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    logger.info(f"总训练步数: {total_steps}")
    logger.info(f"Warmup步数: {warmup_steps}")
    logger.info(f"有效batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

    training_args = TrainingArguments(
        output_dir=model_save_dir,
        overwrite_output_dir=True,
        eval_strategy='steps',
        eval_steps=EVAL_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        adam_epsilon=ADAM_EPSILON,
        max_grad_norm=MAX_GRAD_NORM,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        num_train_epochs=NUM_EPOCHS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        save_strategy='steps',
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        logging_strategy='steps',
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        seed=SEED,
        data_seed=SEED,
        fp16=False,
        optim='adamw_torch',
        report_to='none',
        disable_tqdm=True,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=tokenizer,
        callbacks=[early_stopping,
               history_callback,
               OptimizerCheckCallback(),
               TrainingMonitorCallback(),
               ResumeTrainingCallback()],
    )

    logger.info("=" * 80)
    logger.info("开始训练...")
    logger.info("=" * 80)

    result = trainer.train()

    logger.info(f"\n训练完成!")
    logger.info(f"训练时间: {result.metrics['train_runtime']:.2f}秒")
    logger.info(f"样本/秒: {result.metrics['train_samples_per_second']:.2f}")

    trainer.save_model(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    logger.info(f"最终模型已保存到 {final_save_dir}")

    logger.info("=" * 80)
    logger.info("最终评估...")
    logger.info("=" * 80)
    eval_results = trainer.evaluate()

    logger.info(f"\n最终评估结果:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value:.4f}")

    with open(final_save_dir / 'eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)

    return eval_results


# ==================== 主函数 ====================

def main():
    """主训练流程"""
    start_time = time.time()

    print("\n" + "=" * 80)
    print("DeBERTa高熵合金模型训练")
    print("=" * 80)

    # 1. 设置随机种子
    set_seed(SEED)
    logger.info(f"随机种子设置为: {SEED}")

    # 2. 打印配置
    print_config()

    # 3. 环境检查
    device = check_environment()

    # 4. 创建目录
    output_dir, tokenizer_dir, tokens_dir, model_dir, final_model_dir = setup_directories()

    # 5. 清理内存
    cleanup_memory()

    # 6. 文本标准化（可选）
    if not SKIP_NORMALIZATION:
        train_norm_file = output_dir / 'train_normalized.txt'
        val_norm_file = output_dir / 'val_normalized.txt'

        normalize_text_file(TRAIN_FILE, train_norm_file)
        normalize_text_file(VAL_FILE, val_norm_file)

        train_file = train_norm_file
        val_file = val_norm_file
    else:
        logger.info("跳过文本标准化步骤")
        train_file = TRAIN_FILE
        val_file = VAL_FILE

    # 7. Tokenizer处理
    if not SKIP_TOKENIZER_TRAINING:
        train_tokenizer(train_file, val_file, tokenizer_dir, VOCAB_SIZE)
    else:
        logger.info("使用预训练tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, cache_dir=CACHE_DIR)
        tokenizer.save_pretrained(tokenizer_dir)

    # 8. Tokenization
    tokenize_dataset(train_file, val_file, tokenizer_dir, tokens_dir, MAX_SEQ_LENGTH)

    # 9. 训练模型
    eval_results = train_model(tokenizer_dir, tokens_dir, model_dir, final_model_dir)

    # 10. 训练完成
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    print("\n" + "=" * 80)
    print("✅ 训练流程全部完成!")
    print("=" * 80)
    print(f"总运行时间: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    print(f"最终模型保存位置: {final_model_dir}")
    print(f"最终验证损失: {eval_results.get('eval_loss', 'N/A'):.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()