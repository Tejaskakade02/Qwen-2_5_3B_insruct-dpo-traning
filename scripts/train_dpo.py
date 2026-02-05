from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import DPOTrainer, DPOConfig

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)

dataset = load_dataset(
    "json",
    data_files="data/dpo/train.jsonl",
    split="train"
)

config = DPOConfig(
    output_dir="outputs/qwen2.5-vl-dpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=config,
    train_dataset=dataset,
    tokenizer=processor
)

trainer.train()
trainer.save_model()
