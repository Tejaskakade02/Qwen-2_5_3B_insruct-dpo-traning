import os
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    Qwen2VLForConditionalGeneration,
)
from trl import DPOTrainer, DPOConfig
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_ROOT = "data"                 # root folder for images
SFT_DATA = "data/sft/train.jsonl"
DPO_DATA = "data/dpo/train.jsonl"

assert torch.cuda.is_available(), "❌ CUDA not available"
DEVICE = "cuda"

# ============================================================
# LOAD PROCESSOR
# ============================================================
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

# ============================================================
# LOAD MODEL (GPU)
# ============================================================
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": DEVICE},
    trust_remote_code=True
)
model.train()

# ============================================================
# IMAGE LOADER (SAFE)
# ============================================================
def load_image(rel_path: str):
    full_path = os.path.join(IMAGE_ROOT, rel_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Missing image: {full_path}")
    return Image.open(full_path).convert("RGB")

# ============================================================
# SFT DATA COLLATOR
# ============================================================
def sft_collator(batch):
    texts, images = [], []

    for sample in batch:
        messages = sample["messages"]

        user_content = messages[0]["content"]
        instruction = next(x["text"] for x in user_content if x["type"] == "text")
        image_path = next(x["image"] for x in user_content if x["type"] == "image")
        answer = messages[1]["content"][0]["text"]

        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        texts.append(
            processor.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
        images.append(load_image(image_path))

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs   # ✅ CPU tensors only

# ============================================================
# SFT TRAINING
# ============================================================
print("\n🔥 Starting SFT training...\n")

sft_dataset = load_dataset("json", data_files=SFT_DATA, split="train")

sft_args = TrainingArguments(
    output_dir="outputs/sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=10,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
    gradient_checkpointing=True,
)

sft_trainer = Trainer(
    model=model,
    args=sft_args,
    train_dataset=sft_dataset,
    data_collator=sft_collator,
)

sft_trainer.train()
sft_trainer.save_model("outputs/sft")

print("\n✅ SFT completed\n")

# ============================================================
# LOAD SFT MODEL FOR DPO
# ============================================================
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "outputs/sft",
    torch_dtype=torch.bfloat16,
    device_map={"": DEVICE},
    trust_remote_code=True
)
model.train()

# ============================================================
# DPO DATA COLLATOR
# ============================================================
def dpo_collator(batch):
    prompts, chosen, rejected, images = [], [], [], []

    for sample in batch:
        user = sample["prompt"][0]["content"]
        instruction = next(x["text"] for x in user if x["type"] == "text")
        image_path = next(x["image"] for x in user if x["type"] == "image")

        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        prompts.append(
            processor.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        chosen.append(sample["chosen"])
        rejected.append(sample["rejected"])
        images.append(load_image(image_path))

    return {
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected,
        "images": images,
    }

# ============================================================
# DPO TRAINING
# ============================================================
print("\n🔥 Starting DPO training...\n")

dpo_dataset = load_dataset("json", data_files=DPO_DATA, split="train")

dpo_args = DPOConfig(
    output_dir="outputs/dpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    beta=0.1,
    logging_steps=10,
    bf16=True,
    remove_unused_columns=False,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_args,
    train_dataset=dpo_dataset,
    data_collator=dpo_collator,
)

dpo_trainer.train()
dpo_trainer.save_model("outputs/dpo")

print("\n🎉 SFT → DPO pipeline finished successfully on GPU")
