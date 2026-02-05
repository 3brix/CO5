# -*- coding: utf-8 -*-
"""
Task LLM Continued Training Solution

Created on Wed November 19 07:51:46 2025

@author: agha
"""

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig

max_seq_length = 1024 
load_in_4bit = True  # quantization method

# uses LoRA and quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2b-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)
# defaults provide acceptable results
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # hyperparameter to play?
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    bias="none",
    lora_alpha = 16,   # larger nr more weight to the adapter/lower more weight for the original matrix
    lora_dropout = 0,
    random_state=1234,
    use_rslora=False,
    loftq_config=None,
    use_gradient_checkpointing = "unsloth",
)


lm_prompt = """{}"""
EOS_TOKEN = tokenizer.eos_token # each sequence should be ended with this special character, so LLM learns when to stop...


def format_dataset_lm():
    with open('contents/shakespeare.txt', 'r', encoding='utf-8') as f:
        full_text = f.read()
    # chunk
    chunk_size = 1000 
    texts = []
    
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i : i + chunk_size]
        # format
        formatted_text = lm_prompt.format(chunk) + EOS_TOKEN
        texts.append(formatted_text)

    # sanity check
    print(f"Final Dataset size: {len(texts)} chunks")
    
    if len(texts) == 0:
        raise ValueError("Dataset is still empty. Check your file content!")

    return Dataset.from_dict({"text": texts})

dataset = format_dataset_lm()


# Config the trainer  --> check because low ram
sftConfig = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    # max_steps = 1,
    num_train_epochs=2, #play around with, too much epoch chatastrophic forgetting (output being worse then the previous one)
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=1234,
    output_dir="outputs",
)


# Trainer object
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = sftConfig
)

t_obj = trainer.train()

# Switch to inference mode  --> backprop stopped, creating input: text and input, generate output
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    lm_prompt.format("HAMLET: ")
], return_tensors = "pt").to("cuda")


text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


# if you have benchmarks to your data, you can use it to evaluate the output

"""my output: <bos>HAMLET:

I am content to be a villain's son.


POLIXENES:

I am content to be a villain's son.


ROMEO:

I am content to be a villain's son.


FRIAR LAURENCE:

I am content to be a villain's son.


ROMEO:

I am content to be a villain's son.


FRIAR LAURENCE:

I am content to be a villain's son.


ROMEO:

I am content to be a villain's son.


FRIAR LAURENCE:

I am content to be a villain' """
