import os
import json
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
IMAGE_DIR = r"C:\Users\Tejas Kakade\OneDrive\Desktop\QWen model dpo\data\images"
SFT_OUT = "data/sft/train.jsonl"
DPO_OUT = "data/dpo/train.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")

# ─────────────────────────────
# CREATE DIRS
# ─────────────────────────────
os.makedirs("data/sft", exist_ok=True)
os.makedirs("data/dpo", exist_ok=True)

# ─────────────────────────────
# LOAD BLIP
# ─────────────────────────────
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)

# ─────────────────────────────
# HELPERS
# ─────────────────────────────
def clean_caption(text: str) -> str:
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)  # remove repetition
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def generate_caption(img_path: Path) -> str:
    image = Image.open(img_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return clean_caption(caption)

def strong_description(caption: str) -> str:
    return (
        f"{caption.capitalize()}. "
        "Clean visual appearance with good composition and clarity. "
        "Well-suited for professional listings and marketing use."
    )

def weak_description(caption: str) -> str:
    return (
        f"{caption.capitalize()}. "
        "Basic description with limited detail and no promotional context."
    )

# ─────────────────────────────
# MAIN LOOP
# ─────────────────────────────
images = [
    p for p in Path(IMAGE_DIR).iterdir()
    if p.suffix.lower() in VALID_EXT
]

print(f"Found {len(images)} images")

with open(SFT_OUT, "w", encoding="utf-8") as sft_f, \
     open(DPO_OUT, "w", encoding="utf-8") as dpo_f:

    for img in tqdm(images, desc="Building dataset"):
        caption = generate_caption(img)
        rel_path = f"images/{img.name}"

        chosen = strong_description(caption)
        rejected = weak_description(caption)

        # ───── SFT ─────
        sft_sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Create a short, SEO-friendly product description based on the image."},
                        {"type": "image", "image": rel_path}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": chosen}
                    ]
                }
            ]
        }
        sft_f.write(json.dumps(sft_sample) + "\n")

        # ───── DPO ─────
        dpo_sample = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Create a short, SEO-friendly product description based on the image."},
                        {"type": "image", "image": rel_path}
                    ]
                }
            ],
            "chosen": chosen,
            "rejected": rejected
        }
        dpo_f.write(json.dumps(dpo_sample) + "\n")

print("✅ Clean SFT and DPO datasets created")
print(f"→ {SFT_OUT}")
print(f"→ {DPO_OUT}")
