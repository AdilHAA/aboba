import os
import re
import ast
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import pandas as pd 
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score
from seqeval.scheme import IOB2
import evaluate


LABEL_LIST = [
	'O',
	'B-TYPE', 'I-TYPE',
	'B-BRAND', 'I-BRAND',
	'B-VOLUME', 'I-VOLUME',
	'B-PERCENT', 'I-PERCENT',
]
LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}

DEFAULT_MODEL_CANDIDATES = [
	'sergeyzh/Berta',
	'ai-forever/ruBert-base',
	'ai-forever/ruElectra-base',
	'DeepPavlov/rubert-base-cased',
]

MAX_LENGTH = 128


def read_dataset_csv(path: str) -> pd.DataFrame:
	df = pd.read_csv(path, sep=';', dtype=str)
	df = df.fillna('')
	return df


def parse_annotation(annotation_str: str) -> List[Tuple[int, int, str]]:
	try:
		parsed = ast.literal_eval(annotation_str)
		if isinstance(parsed, list):
			return [(int(s), int(e), str(t)) for s, e, t in parsed]
	except Exception:
		pass
	return []


def _compute_word_ids_from_offsets(text: str, offsets: List[Tuple[int, int]]) -> Tuple[List[int], List[Tuple[int, int]]]:
	words = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
	word_ids: List[int] = []
	for s, e in offsets:
		if s == 0 and e == 0:
			word_ids.append(None)
			continue
		best_overlap = 0
		best_idx = None
		for i, (ws, we) in enumerate(words):
			if we <= s:
				continue
			if ws >= e:
				break
			overlap = min(e, we) - max(s, ws)
			if overlap > best_overlap:
				best_overlap = overlap
				best_idx = i
		word_ids.append(best_idx)
	return word_ids, words


def tokenize_and_label_tokens(text, entities, tokenizer):
    # entities: список (start, end, tag), покрывающий весь текст без дыр, в порядке возрастания start
    # Пример тега: 'O', 'B-PER', 'I-PER' и т.п.  Для пословной BIO метка применяется ко всем сабвордам слова.
    seq = []
    labels = []

    for s, e, tag in entities:
        # Кодируем ровно тот фрагмент текста, который размечен
        tok_ids = tokenizer.encode(text[s:e], add_special_tokens=False)
        seq.extend(tok_ids)
        lab_id = LABEL_TO_ID[tag]
        labels.extend([lab_id] * len(tok_ids))

    # Добавляем спец‑токены только если они определены у данного токенизатора
    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if cls_id is not None and sep_id is not None:
        seq = [cls_id] + seq + [sep_id]
        labels = [-100] + labels + [-100]  # игнор спец‑токенов в лоссе

    # Усечение до MAX_LENGTH (как в исходном пайплайне)
    if len(seq) > MAX_LENGTH:
        seq = seq[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    attn_mask = [1] * len(seq)

    return {
        "input_ids": seq,
        "labels": labels,
        "attention_mask": attn_mask
    }


class NerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, pad_to: int = 32):
        self.samples = []
        self.pad_to = pad_to
        self.tokenizer = tokenizer
        skipped = 0

        # Подготовим pad_token_id
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            # Опционально можно назначить пад‑токен здесь:
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # pad_id = tokenizer.pad_token_id
            # На крайний случай используем 0, но лучше настроить пад‑токен в токенизаторе.
            pad_id = 0

        for _, row in df.iterrows():
            text = row['sample']
            if not text or len(text.strip()) == 0:
                skipped += 1
                continue

            entities = parse_annotation(row['annotation'])
            try:
                # Ожидается, что функция уже возвращает целочисленные списки одинаковой длины
                result = tokenize_and_label_tokens(text, entities, tokenizer)
                input_ids = result["input_ids"]
                attention_mask = result["attention_mask"]
                label_ids = result["labels"]

                # Приведение длины к pad_to
                if len(input_ids) != len(label_ids) or len(input_ids) != len(attention_mask):
                    skipped += 1
                    continue

                if len(input_ids) > self.pad_to:
                    input_ids = input_ids[:self.pad_to]
                    attention_mask = attention_mask[:self.pad_to]
                    label_ids = label_ids[:self.pad_to]
                elif len(input_ids) < self.pad_to:
                    pad_len = self.pad_to - len(input_ids)
                    input_ids = input_ids + [pad_id] * pad_len
                    attention_mask = attention_mask + [0] * pad_len
                    label_ids = label_ids + [-100] * pad_len  # падды игнорируются в лоссе

                if len(input_ids) == len(label_ids) and len(input_ids) > 0:
                    self.samples.append({
                        'input_ids': torch.tensor(input_ids, dtype=torch.long),
                        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                        'labels': torch.tensor(label_ids, dtype=torch.long),
                    })
                else:
                    skipped += 1
            except Exception as e:
                print(f"Error processing sample: {str(e)}")
                skipped += 1

        if skipped > 0:
            print(f"Skipped {skipped} samples due to errors")
        print(f"Successfully processed {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


@dataclass
class ModelConfig:
	model_name: str
	output_dir: str = "models/ner"
	learning_rate: float = 3e-5
	num_train_epochs: int = 4
	per_device_train_batch_size: int = 64
	per_device_eval_batch_size: int = 64
	warmup_ratio: float = 0.03
	weight_decay: float = 0.01
	logging_steps: int = 50
	eval_steps: int = 200
	save_steps: int = 400
	metric_for_best_model: str = "eval_f1"
	load_best_model_at_end: bool = True
	save_total_limit: int = 3
	seed: int = 42
	fp16: bool = True  # Для экономии памяти
	early_stopping_patience: int = 3


def compute_metrics(eval_pred):
	predictions, labels = eval_pred
	predictions = np.argmax(predictions, axis=2)

	true_predictions = [
		[ID_TO_LABEL[p] for p, l in zip(prediction, label) if l != -100]
		for prediction, label in zip(predictions, labels)
	]
	true_labels = [
		[ID_TO_LABEL[l] for l in label if l != -100]
		for label in labels
	]

	filtered_predictions = []
	filtered_labels = []
	for pred, label in zip(true_predictions, true_labels):
		if len(pred) > 0 and len(label) > 0:
			filtered_predictions.append(pred)
			filtered_labels.append(label)

	if not filtered_predictions:
		return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

	metric = evaluate.load("seqeval")
	results = metric.compute(predictions=filtered_predictions, references=filtered_labels, mode="strict", scheme="IOB2")
	return {
		"precision": results.get("overall_precision", 0.0),
		"recall": results.get("overall_recall", 0.0),
		"f1": results.get("overall_f1", 0.0),
		"accuracy": results.get("overall_accuracy", 0.0),
	}


from peft import LoraConfig, TaskType, get_peft_model

def _guess_lora_targets(model) -> list:
    names = [n for n, _ in model.named_modules()]
    # Находим первый подходящий пресет по именам слоёв
    presets = [
        ["q_proj", "k_proj", "v_proj", "o_proj"],  # многие T5/LLM
        ["q", "k", "v", "o"],                      # T5-подобные
        ["query", "key", "value", "dense"],        # BERT/Roberta/ELECTRA
    ]
    for patt in presets:
        if any(any(p in n for n in names) for p in patt):
            return patt
    # Фолбэк: попытка для BERT-семейства
    return ["query", "key", "value", "dense"]

def train_model_with_trainer(train_df: pd.DataFrame, valid_df: pd.DataFrame, cfg: ModelConfig):
    print(f"Training model: {cfg.model_name}")

    # 1) Токенайзер
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # 2) Базовая модель
    base_model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    # 3) LoRA-конфиг и обёртка
    target_modules = _guess_lora_targets(base_model)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.TOKEN_CLS,
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_cfg)
    # Для снижения VRAM (особенно T5-подобные)
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    # 4) Датасеты
    train_ds = NerDataset(train_df, tokenizer)
    valid_ds = NerDataset(valid_df, tokenizer)
    if len(train_ds) == 0 or len(valid_ds) == 0:
        return None, None, {'f1': 0.0}

    # 5) Коллатор
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    # 6) Аргументы обучения (8-bit AdamW совместим с Trainer)
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        metric_for_best_model=cfg.metric_for_best_model,
        load_best_model_at_end=cfg.load_best_model_at_end,
        greater_is_better=True,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        fp16=cfg.fp16,
        remove_unused_columns=False,
        report_to="none",      # при необходимости замените на "wandb"
        optim="adamw_8bit",    # 8‑битный оптимизатор
    )

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    # 8) Обучение и валидация
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    # 9) Слияние LoRA → базовая модель и сохранение
    merged_model = trainer.model
    if hasattr(merged_model, "merge_and_unload"):
        merged_model = merged_model.merge_and_unload()  # возвращает обычную Transformers‑модель
    merged_model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metrics = {
        'f1': eval_result.get('eval_f1', 0.0),
        'precision': eval_result.get('eval_precision', 0.0),
        'recall': eval_result.get('eval_recall', 0.0),
        'accuracy': eval_result.get('eval_accuracy', 0.0),
        'train_loss': train_result.training_loss,
        'eval_loss': eval_result.get('eval_loss', 0.0),
    }
    return merged_model, tokenizer, metrics


def select_best_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, candidates: List[str] = None, base_output_dir: str = "models") -> Tuple[str, Dict[str, Any]]:
	"""Выбор лучшей модели с использованием Trainer"""
	if candidates is None:
		candidates = DEFAULT_MODEL_CANDIDATES

	print(f"Selecting best model from {len(candidates)} candidates")

	best_score, best_model = -1.0, None
	best_metrics = {}

	for i, model_name in enumerate(candidates):
		print(f"\nEvaluating model {i+1}/{len(candidates)}: {model_name}")

		# Создаем отдельную папку для каждой модели
		model_output_dir = f"{base_output_dir}/candidate_{i}_{model_name.replace('/', '_')}"

		cfg = ModelConfig(
			model_name=model_name,
			output_dir=model_output_dir,
			num_train_epochs=3,  # Меньше эпох для быстрого сравнения
			early_stopping_patience=2
		)

		model, tokenizer, metrics = train_model_with_trainer(train_df, valid_df, cfg)

		if model is None:
			continue

		score = metrics['f1']
		print(f"Model {model_name} F1 Score: {score:.4f}")

		if score > best_score:
			best_score, best_model, best_metrics = score, model_name, metrics
			print(f"New best model: {model_name} with F1 score {score:.4f}")

	if best_model is None:
		print("No models could be trained successfully, falling back to first candidate")
		best_model = candidates[0]

	return best_model, best_metrics


def train_and_save(data_path: str, output_dir: str, model_name: str = None):
	"""Основная функция обучения и сохранения модели"""

	print(f"Loading dataset from {data_path}")
	df = read_dataset_csv(data_path)
	print(f"Dataset loaded, total samples: {len(df)}")

	# Валидация данных
	df = df[df['sample'].notna() & (df['sample'] != '')].reset_index(drop=True)
	print(f"After filtering empty samples: {len(df)}")

	if len(df) < 10:
		raise ValueError("Too few samples in dataset")

	train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)
	print(f"Train samples: {len(train_df)}, Validation samples: {len(valid_df)}")

	if model_name is None:
		print("No specific model provided, selecting best model...")
		best_model_name, selection_metrics = select_best_model(train_df, valid_df, base_output_dir=output_dir + "_selection")
		model_name = best_model_name
		print(f"Selected best model: {model_name}")
	else:
		print(f"Using specified model: {model_name}")

	# Финальное обучение с лучшей моделью
	print("\nStarting final training with full parameters...")
	cfg = ModelConfig(
		model_name=model_name,
		output_dir=output_dir,
		num_train_epochs=10,
		per_device_train_batch_size=64,
		learning_rate=1e-4,
		early_stopping_patience=3
	)

	model, tokenizer, final_metrics = train_model_with_trainer(train_df, valid_df, cfg)

	if model is None or tokenizer is None:
		raise ValueError(f"Failed to train model {model_name}")

	# Сохранение метаданных
	meta = {
		'model_name': model_name,
		'metrics': final_metrics,
		'config': {
			'max_length': MAX_LENGTH,
			'label_list': LABEL_LIST,
			'train_samples': len(train_df),
			'valid_samples': len(valid_df),
		}
	}

	with open(os.path.join(output_dir, 'meta.json'), 'w', encoding='utf-8') as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	print(f"Model saved to {output_dir}")
	print("Training completed successfully!")

	return model, tokenizer, final_metrics


def _predict_normalized_labels_word_level(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification
) -> Tuple[List[Tuple[int, int]], List[str]]:
    device = next(model.parameters()).device

    # Единая токенизация с оффсетами и word_ids
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        return_tensors='pt',
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
        is_split_into_words=False,
    )

    offsets = enc["offset_mapping"][0].tolist()
    try:
        word_ids = enc.word_ids(0)
    except Exception:
        word_ids = None

    # На устройство, offset_mapping оставляем на CPU
    batch = {k: v.to(device) for k, v in enc.items() if k != "offset_mapping"}

    model.eval()
    with torch.no_grad():
        logits = model(**batch).logits[0].cpu()  # [T, C]
        # argmax по токенам можно не брать тут, т.к. агрегируем логи по словам

    # Fallback, если токенизатор не даёт word_ids
    if word_ids is None or all(w is None for w in word_ids):
        word_ids, _ = _compute_word_ids_from_offsets(text, offsets)

    # Сопоставление: слово -> индексы токенов (без спец‑токенов с (0,0))
    valid_word_ids = [w for w in word_ids if w is not None]
    num_words = (max(valid_word_ids) + 1) if valid_word_ids else 0
    word_to_token_indices: Dict[int, List[int]] = {}
    for ti, wid in enumerate(word_ids):
        if wid is None:
            continue
        if ti < len(offsets) and tuple(offsets[ti]) == (0, 0):
            continue
        word_to_token_indices.setdefault(wid, []).append(ti)

    # Предсказываем класс на уровне слова: суммируем логи по сабвордам слова
    # и берём класс с максимальным логитом; BIO префикс отбрасываем для "базы"
    word_base_types: List[Optional[str]] = [None] * num_words
    for w in range(num_words):
        idxs = word_to_token_indices.get(w, [])
        if not idxs:
            continue
        summed = logits[idxs].sum(dim=0)  # [C]
        lid = int(torch.argmax(summed).item())
        lstr = ID_TO_LABEL[lid]
        if lstr == "O":
            word_base_types[w] = None
        else:
            base = lstr.split("-", 1)[-1]
            word_base_types[w] = base

    # Формируем BIO на уровне слов: первое слово сущности -> B-base, последующие подряд -> I-base
    word_labels: List[str] = []
    for w in range(num_words):
        base = word_base_types[w]
        if base is None:
            word_labels.append("O")
        else:
            prev_base = word_base_types[w - 1] if w > 0 else None
            tag = "B" if base != prev_base else "I"
            word_labels.append(f"{tag}-{base}")

    # Преобразуем в токенные метки по вашей схеме "1 слово — 1 класс":
    # все сабворды первого слова сущности получают B-base, все сабворды следующих слов -> I-base
    normalized_labels: List[str] = []
    for ti, wid in enumerate(word_ids):
        # Спец‑токены и позиции без слова метятся как O
        if wid is None or (ti < len(offsets) and tuple(offsets[ti]) == (0, 0)):
            normalized_labels.append("O")
            continue
        wl = word_labels[wid]
        if wl == "O":
            normalized_labels.append("O")
        else:
            # Для первого слова сущности: B-base на всех сабвордах слова
            # Для последующих слов сущности: I-base на всех сабвордах слова
            normalized_labels.append(wl)

    return offsets, normalized_labels


def decode_predictions(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification
) -> List[Tuple[int, int, str]]:
    # Получаем токенные offsets и токенные метки, сформированные пословной логикой:
    # каждому слову предсказывается класс, а его сабвордам назначается одна и та же BIO-метка
    offsets, normalized_labels = _predict_normalized_labels_word_level(text, tokenizer, model)

    # Восстанавливаем соответствие "токен -> слово" от тех же offsets
    # Если токенизатор отдаёт word_ids, их лучше использовать напрямую в _predict..., но здесь делаем fallback-совместимость
    word_ids, word_spans = _compute_word_ids_from_offsets(text, offsets)

    # Оставим только слова, которые реально покрыты токенами усечённой последовательности (исключаем спец-токены и (0,0))
    covered_word_to_token_indices: Dict[int, List[int]] = {}
    for ti, wid in enumerate(word_ids):
        if wid is None:
            continue
        if ti < len(offsets) and tuple(offsets[ti]) == (0, 0):
            continue
        covered_word_to_token_indices.setdefault(wid, []).append(ti)

    # Коллекция "индексы слов, которые имеют хоть один валидный токен"
    covered_word_indices = sorted(covered_word_to_token_indices.keys())

    # Если окно усекло текст, не возвращаем слова вне окна, чтобы не генерировать фиктивные 'O' на невидимых фрагментах
    # Строим тип для каждого покрытого слова по первой ненулевой метке токенов слова
    word_types: Dict[int, str] = {}
    for w in covered_word_indices:
        assigned = "O"
        for ti in covered_word_to_token_indices[w]:
            lbl = normalized_labels[ti] if ti < len(normalized_labels) else "O"
            if lbl != "O":
                assigned = lbl.split("-", 1)[-1]  # base
                break
        if assigned != "O":
            word_types[w] = assigned

    # Формируем список пословных сущностей в BIO по покрытым словам
    # Первое слово цепочки base -> B-base, последующие подряд слова с тем же base -> I-base, иначе 'O'
    entities: List[Tuple[int, int, str]] = []
    prev_base: Optional[str] = None
    for w in covered_word_indices:
        start_char, end_char = word_spans[w]
        if start_char >= end_char:
            continue
        base = word_types.get(w)
        if base is None:
            entities.append((start_char, end_char, "O"))
            prev_base = None
        else:
            tag = "B" if base != prev_base else "I"
            entities.append((start_char, end_char, f"{tag}-{base}"))
            prev_base = base

    return entities


def decode_token_labels(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification
) -> List[Tuple[int, int, str]]:
    # Предсказания по вашей логике: 1 слово — 1 класс (BIO на уровне слов),
    # развернутые на токенные метки
    offsets, normalized_labels = _predict_normalized_labels_word_level(text, tokenizer, model)

    tokens_with_labels: List[Tuple[int, int, str]] = []
    for i, (s, e) in enumerate(offsets):
        # Пропускаем спец‑токены fast‑токенизатора (офсеты (0,0))
        if s == 0 and e == 0:
            continue
        label = normalized_labels[i] if i < len(normalized_labels) else 'O'
        tokens_with_labels.append((s, e, label))

    return tokens_with_labels


def infer_file(model_dir: str, input_csv: str, output_csv: str):
    from peft import PeftModel
    """Инференция на файле"""
    print(f"Loading model from {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    LABEL_LIST = ['O','B-TYPE', 'I-TYPE','B-BRAND', 'I-BRAND','B-VOLUME', 'I-VOLUME','B-PERCENT', 'I-PERCENT',]
    
    LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
    ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}
    
    base = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=9, id2label=ID_TO_LABEL, label2id=LABEL_TO_ID)
    model = PeftModel.from_pretrained(base, model_dir) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Loading input data from {input_csv}")
    df = read_dataset_csv(input_csv)
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        text = row['sample']
        if not text or len(text.strip()) == 0:
            results.append({'sample': text, 'annotation': '[]'})
            continue
    
        try:
            entities = decode_predictions(text, tokenizer, model)
            results.append({'sample': text, 'annotation': str(entities)})
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            results.append({'sample': text, 'annotation': '[]'})
    
    print(f"Saving results to {output_csv}")
    pd.DataFrame(results).to_csv(output_csv, sep=';', index=False)
    print("Inference completed successfully!")


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='NER training with HuggingFace Trainer')
	parser.add_argument('--train', action='store_true', help='Train mode')
	parser.add_argument('--data', type=str, default='dataset/Датасет/train.csv', help='Training data path')
	parser.add_argument('--out', type=str, default='models/ner', help='Output directory for model')
	parser.add_argument('--model', type=str, default=None, help='Specific model to use')
	parser.add_argument('--infer', action='store_true', help='Inference mode')
	parser.add_argument('--input', type=str, default='dataset/Датасет/train.csv', help='Input data for inference')
	parser.add_argument('--output', type=str, default='submission.csv', help='Output file for predictions')

	args = parser.parse_args()

	if args.train:
		train_and_save(args.data, args.out, args.model)

	if args.infer:
		infer_file(args.out, args.input, args.output)
