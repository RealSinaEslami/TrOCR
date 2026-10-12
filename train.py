import os
import evaluate
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)

# ----------------- تنظیمات اصلی -----------------
MODEL_NAME = r"microsoft/trocr-base-handwritten"
DATA_DIR = r"C:\Users\Amir\Desktop\HamrahTel\TrOCR\dataset\concatinated"
OUTPUT_DIR = "./trocr_farsi_finetuned_3"

BATCH_SIZE = 8 
EPOCHS = 30
LR = 5e-5
MAX_TARGET_LEN = 64
EARLY_STOP_VAL = 10
# ------------------------------------------------


def clean_persian_text(text: str) -> str:
    """نرمال‌سازی متن برای سنجش دقیق و منصفانه CER"""
    text = str(text).strip()
    text = text.replace("ي", "ی").replace("ك", "ک")
    # حذف اعراب و نشانه‌ها
    for diacritic in ["ً", "ٌ", "ٍ", "َ", "ُ", "ِ", "ّ", "ْ", "ـ"]:
        text = text.replace(diacritic, "")
    text = text.replace("\u200c", " ")  # تبدیل نیم‌فاصله به فاصله
    return " ".join(text.split())


class PersianHandwritingDataset(Dataset):
    def __init__(self, root_dir, processor, max_target_len=MAX_TARGET_LEN, transform=None):
        self.root_dir = root_dir
        self.processor = processor
        self.max_target_len = max_target_len
        self.transform = transform

        # 1. تعیین مسیر تصاویر (یا داخل پوشه images یا مستقیماً در root_dir)
        images_sub = os.path.join(root_dir, "images")
        self.images_dir = images_sub if os.path.isdir(images_sub) else root_dir

        # 2. پیدا کردن فایل CSV در این پوشه
        csv_candidates = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(".csv")
        ]
        if not csv_candidates:
            raise FileNotFoundError(f"هیچ فایل CSV در مسیر {root_dir} پیدا نشد!")

        csv_path = csv_candidates[0]
        try:
            self.df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            self.df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # دو ستون اول: ستون اول نام/مسیر فایل تصویر و ستون دوم متن
        img_col = self.df.columns[0]
        text_col = self.df.columns[1]

        self.df = self.df.dropna(subset=[img_col, text_col]).reset_index(drop=True)

        # استخراج مسیر معتبر تصاویر و متن‌ها
        self.image_paths = []
        self.texts = []

        valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]

        for _, row in self.df.iterrows():
            raw_img_name = str(row[img_col]).strip()
            text_val = str(row[text_col]).strip()

            # بررسی پسوند و مسیر فایل
            candidates = [os.path.join(self.images_dir, raw_img_name)]
            if not any(raw_img_name.lower().endswith(ext) for ext in valid_exts):
                for ext in valid_exts:
                    candidates.append(os.path.join(self.images_dir, raw_img_name + ext))

            found_path = None
            for p in candidates:
                if os.path.exists(p):
                    found_path = p
                    break

            if found_path:
                self.image_paths.append(found_path)
                self.texts.append(text_val)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"هیچ تصویری در دایرکتوری {self.images_dir} منطبق بر CSV پیدا نشد!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        text = self.texts[idx]

        # اعمال Data Augmentation در صورت تعریف شدن (فقط روی Train)
        if self.transform is not None:
            image = self.transform(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # قرار دادن -100 روی توکن‌های پدینگ برای صرف‌نظر در محاسبه Loss
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long)
        }


cer_metric = evaluate.load("cer")

def make_compute_metrics(processor):
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        # نرمال‌سازی هر دو رشته برای ارزیابی منصفانه و دقیق
        clean_preds = [clean_persian_text(p) for p in pred_str]
        clean_labels = [clean_persian_text(l) for l in label_str]

        cer = cer_metric.compute(predictions=clean_preds, references=clean_labels)
        return {"cer": cer}

    return compute_metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading TrOCR Processor & Model...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # افزودن کاراکترها و نشانه‌های فارسی به توکنایزر
    persian_chars = [
        'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ', 'س', 'ش',
        'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ه', 'ی', ' ',
        '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹',
        '!', '؟', '،', '.', ':', '؛', '(', ')'
    ]
    num_added_toks = processor.tokenizer.add_tokens(persian_chars)
    if num_added_toks > 0:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        print(f"Added {num_added_toks} new Persian tokens to tokenizer.")

    # تنظیمات مدل دیکودر
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = len(processor.tokenizer)
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    # تنظیمات استراتژی تولید متن (Generation Config)
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = MAX_TARGET_LEN
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 1.0
    model.generation_config.repetition_penalty = 1.2
    model.generation_config.num_beams = 4

    # تعریف ترنسفورم‌ها با زمینه سفید (fill=255) برای دست‌نوشته‌ها
    train_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=(-4, 4), fill=255),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, shear=(-3, 3, -3, 3), fill=255),
    ])

    train_path = os.path.join(DATA_DIR, "train")
    test_path = os.path.join(DATA_DIR, "test")

    train_ds = PersianHandwritingDataset(
        root_dir=train_path,
        processor=processor,
        max_target_len=MAX_TARGET_LEN,
        transform=train_transforms   # فقط داده آموزش آگمنت می‌شود
    )
    eval_ds = PersianHandwritingDataset(
        root_dir=test_path,
        processor=processor,
        max_target_len=MAX_TARGET_LEN,
        transform=None               # تست بدون تغییر می‌ماند
    )

    print(f"Train samples: {len(train_ds)} | Test samples: {len(eval_ds)}")

    total_steps = (len(train_ds) // BATCH_SIZE) * EPOCHS
    warmup_steps = int(0.08 * total_steps)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        dataloader_num_workers=0,
        predict_with_generate=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_VAL)],
    )

    print("--- Starting TrOCR Training ---")
    trainer.train()

    print("--- Saving Final Model and Processor ---")
    final_save_path = os.path.join(OUTPUT_DIR, "final_best")
    trainer.save_model(final_save_path)
    processor.save_pretrained(final_save_path)
    print(f"Model saved successfully to: {final_save_path}")


if __name__ == "__main__":
    main()
