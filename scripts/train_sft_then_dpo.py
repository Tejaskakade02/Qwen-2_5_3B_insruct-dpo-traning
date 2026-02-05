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
import os

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# ─────────────────────────────────
# LOAD PROCESSOR + MODEL (CORRECT)
# ─────────────────────────────────
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

# ─────────────────────────────────
# SFT DATA COLLATOR
# ─────────────────────────────────
def vl_data_collator(batch):
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
                    {"type": "text", "text": instruction}
                ]
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        text = processor.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False
        )

        texts.append(text)
        images.append(Image.open(os.path.join("data", image_path)).convert("RGB"))

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    )

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# ─────────────────────────────────
# SFT TRAINING
# ─────────────────────────────────
print("\n🔥 Starting SFT training...\n")

sft_dataset = load_dataset(
    "json",
    data_files="data/sft/train.jsonl",
    split="train"
)

sft_args = TrainingArguments(
    output_dir="outputs/sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=5,
    bf16=torch.cuda.is_available(),
    remove_unused_columns=False,
    report_to="none"
)

sft_trainer = Trainer(
    model=model,
    args=sft_args,
    train_dataset=sft_dataset,
    data_collator=vl_data_collator
)

sft_trainer.train()
sft_trainer.save_model("outputs/sft")

print("\n✅ SFT completed\n")

# ─────────────────────────────────
# LOAD SFT MODEL FOR DPO
# ─────────────────────────────────
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "outputs/sft",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

# ─────────────────────────────────
# DPO DATA COLLATOR
# ─────────────────────────────────
def dpo_vl_collator(batch):
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
                    {"type": "text", "text": instruction}
                ]
            }
        ]

        prompt = processor.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )

        prompts.append(prompt)
        chosen.append(sample["chosen"])
        rejected.append(sample["rejected"])
        images.append(Image.open(os.path.join("data", image_path)).convert("RGB"))

    return {
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected,
        "images": images,
    }

# ─────────────────────────────────
# DPO TRAINING
# ─────────────────────────────────
print("\n🔥 Starting DPO training...\n")

dpo_dataset = load_dataset(
    "json",
    data_files="data/dpo/train.jsonl",
    split="train"
)

dpo_args = DPOConfig(
    output_dir="outputs/dpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    beta=0.1,
    logging_steps=5,
    bf16=torch.cuda.is_available(),
    remove_unused_columns=False,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_args,
    train_dataset=dpo_dataset,
    data_collator=dpo_vl_collator
)

dpo_trainer.train()
dpo_trainer.save_model("outputs/dpo")

print("\n🎉 SFT → DPO pipeline finished successfully")
