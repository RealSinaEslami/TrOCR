import os
import glob
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ================== تنظیمات مدل و تصاویر ==================
# آدرس چک‌پوینت مدل آموزش‌دیده
MODEL_PATH = r"best weights\checkpoint-25000"

# مسیر پوشه‌ای که عکس‌های تست داخل آن هستند
# (می‌توانید مستقیم آدرس پوشه test دیتاست یا یک پوشه دلخواه را بگذارید)
TEST_IMAGES_DIR = r"C:\Users\Sina\Desktop\HamrahTel\TrOCR\dataset\check"

# حداکثر تعداد تصاویری که می‌خواهید نمایش داده شود (مثلاً 20 عدد یا None برای همه)
MAX_SAMPLES_TO_TEST = 10
# ========================================================


def predict(image: Image.Image, model, processor, device: str) -> str:
    """دریافت تصویر و استخراج متن فارسی"""
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=64,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.0,
            early_stopping=True,
        )

    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return pred_text.strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading finetuned model from: {MODEL_PATH} ...")
    
    # پردازشگر را اگر داخل چک‌پوینت نبود، از مدل پایه یا پوشه اصلی می‌خواند
    try:
        processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    except Exception:
        print("Processor not found in checkpoint folder, falling back to base model...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    print("Model loaded successfully.\n" + "=" * 50)

    # پیدا کردن تمام عکس‌ها در پوشه تست
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, "**", ext), recursive=True))

    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"هیچ تصویری در مسیر {TEST_IMAGES_DIR} پیدا نشد!")
        return

    if MAX_SAMPLES_TO_TEST:
        image_paths = image_paths[:MAX_SAMPLES_TO_TEST]

    print(f"Found {len(image_paths)} images to test:\n")

    # اجرای تست روی تک تک تصاویر
    for idx, img_path in enumerate(image_paths, 1):
        try:
            image = Image.open(img_path).convert("RGB")
            predicted_text = predict(image, model, processor, device)

            file_name = os.path.basename(img_path)
            print(f"[{idx}/{len(image_paths)}] File: {file_name}")
            print(f"  --> Predicted: {predicted_text}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()
