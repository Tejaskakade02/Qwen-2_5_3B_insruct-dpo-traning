import os
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    Qwen2VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model
from PIL import Image

# ============================================================
# GPU + TORCH SETTINGS (VERY IMPORTANT)
# ============================================================
assert torch.cuda.is_available(), "❌ GPU not available"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda"
DTYPE = torch.bfloat16

print("✅ GPU:", torch.cuda.get_device_name(0))

# ============================================================
# CONFIG
# ============================================================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

IMAGE_ROOT = "data"
SFT_DATA = "data/sft/train.jsonl"

OUTPUT_DIR = "outputs/qwen2_vl_lora_15gb"

# ============================================================
# LOAD PROCESSOR
# ============================================================
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

# ============================================================
# LOAD MODEL
# ============================================================
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True
)

# ============================================================
# 🔥 FREEZE VISION ENCODER (CRITICAL FOR 15GB)
# ============================================================
for param in model.model.visual.parameters():
    param.requires_grad = False

# ============================================================
# APPLY LORA (LOW-RANK, LOW-VRAM)
# ============================================================
lora_config = LoraConfig(
    r=4,                     # 🔥 VERY IMPORTANT
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()

# ============================================================
# IMAGE LOADER (DOWNSCALE TO SAVE VRAM)
# ============================================================
def load_image(rel_path: str):
    full_path = os.path.join(IMAGE_ROOT, rel_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(full_path)

    img = Image.open(full_path).convert("RGB")
    img = img.resize((224, 224))   # 🔥 HUGE MEMORY SAVER
    return img

# ============================================================
# SFT COLLATOR (CPU ONLY)
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
    return inputs

# ============================================================
# LOAD DATASET
# ============================================================
dataset = load_dataset("json", data_files=SFT_DATA, split="train")

# ============================================================
# TRAINING ARGS (15GB SAFE)
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,        # 🔥 DO NOT INCREASE
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=2,
)

# ============================================================
# TRAIN
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=sft_collator,
)

print("\n🔥 Starting SFT (15GB GPU SAFE)...\n")
trainer.train()

# ============================================================
# SAVE LORA MODEL
# ============================================================
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Training complete. Model saved to: {OUTPUT_DIR}")

