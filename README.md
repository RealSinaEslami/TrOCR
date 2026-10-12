# TrOCR Fine-Tuning for Persian Text Recognition

This repository focuses on fine-tuning Microsoft's TrOCR model for Persian (Farsi) text recognition using PyTorch and Hugging Face Transformers.

## Overview

TrOCR is an end-to-end Optical Character Recognition (OCR) model that combines a Vision Transformer encoder with a Transformer text decoder to generate text from images.

The goal of this project is to adapt a pretrained TrOCR model to Persian text recognition, including handwritten text, using a custom image-text dataset.

## Features

- Fine-tuning a pretrained Microsoft TrOCR model.
- Training on custom Persian image-text pairs.
- Validation and evaluation using Character Error Rate (CER).
- Persian text normalization for more consistent evaluation.
- Saving trained model checkpoints for inference.

## Pretrained Model

The starting checkpoint can be configured in the training script.

For example:

```python
MODEL_NAME = "microsoft/trocr-base-handwritten"
```

Alternatively, `microsoft/trocr-base-stage1` can be used when starting from the Stage 1 checkpoint rather than the handwriting-fine-tuned checkpoint.

- [TrOCR Base Handwritten](https://huggingface.co/microsoft/trocr-base-handwritten)
- [TrOCR Base Stage 1](https://huggingface.co/microsoft/trocr-base-stage1)

## Installation

Install the main dependencies:

```bash
pip install torch torchvision transformers datasets evaluate pillow pandas numpy editdistance
```

Use a PyTorch installation compatible with your system and CUDA environment.

## Dataset

The dataset should contain images and their corresponding ground-truth transcriptions.

A typical CSV format is:

```csv
image_path,text
dataset/images/0001.png,نمونه متن فارسی
dataset/images/0002.png,این یک متن نمونه است
```

Prepare separate training, validation, and test splits. Ensure that image paths are valid and that each transcription matches its corresponding image.

## Training

Run the training script from the repository root. For example:

```bash
python train.py
```

Configure the model checkpoint, dataset paths, batch size, learning rate, number of epochs, and maximum target sequence length according to the training script.

## Evaluation

Character Error Rate (CER) measures the character-level edit distance between predicted and reference text:

\[
CER = \frac{S + D + I}{N}
\]

Where:

- `S`: Number of substitutions
- `D`: Number of deletions
- `I`: Number of insertions
- `N`: Number of characters in the reference text

A lower CER indicates better recognition performance.

For Persian text, consistent normalization is important. Common normalization operations include:

- Replacing Arabic `ي` with Persian `ی`.
- Replacing Arabic `ك` with Persian `ک`.
- Handling diacritics consistently.
- Normalizing whitespace.

Apply the same normalization rules to predictions and reference labels before computing CER.

## Inference

After training, load the saved checkpoint to recognize text in a new image:

```python
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

checkpoint_path = "./trocr_farsi_finetuned/final_best"

processor = TrOCRProcessor.from_pretrained(checkpoint_path)
model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

image = Image.open("path/to/image.png").convert("RGB")

pixel_values = processor(
    images=image,
    return_tensors="pt"
).pixel_values.to(device)

with torch.no_grad():
    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=64
    )

prediction = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print(prediction)
```

Update the checkpoint and image paths to match your environment.

## Reproducibility

For reliable comparisons, record the pretrained checkpoint, dataset splits, random seed, training hyperparameters, library versions, normalization rules, and final test CER.

## References

- [TrOCR Research Paper](https://arxiv.org/abs/2109.10282)
- [Microsoft UniLM / TrOCR Repository](https://github.com/microsoft/unilm/tree/master/trocr)
- [Hugging Face TrOCR Documentation](https://huggingface.co/docs/transformers/model_doc/trocr)

## License

Review the licenses of the pretrained model, datasets, and dependencies before redistributing model weights or data. Add the appropriate license for this repository before publishing.
