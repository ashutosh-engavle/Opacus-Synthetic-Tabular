import gc
import os
import numpy as np
import pandas as pd
import torch
from transformers import (AutoTokenizer,AutoModelForCausalLM)
from datasets import Dataset
import random
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from tqdm import tqdm
import torch.optim as optim
from opacus.utils.batch_memory_manager import BatchMemoryManager

def process_data(df, tokenizer, shuffle=False, max_length=750):
    columns = df.columns.tolist()
    df = df.reset_index(drop=True)
    ds = Dataset.from_pandas(df)

    # Function to combine data in an ordered format
    def combine_data_ordered(sample):
        concat = ""
        for col in columns:
            concat += "%s is %s, " % (col, str(sample[col]).strip())
        return {"concat": concat}

    def combine_data_shuffled(sample):
        concat = ""
        for col in random.sample(columns, k=len(columns)):
            concat += "%s is %s, " % (col, str(sample[col]).strip())

        return {"concat": concat}

    if shuffle:
        combined_ds = ds.map(combine_data_shuffled)
    else:
        combined_ds = ds.map(combine_data_ordered)
    combined_ds = combined_ds.remove_columns(ds.column_names)

    def tokenizer_function(sample):
        result = tokenizer(
            sample["concat"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenizer_ds = combined_ds.map(tokenizer_function, batched=True)
    tokenizer_ds = tokenizer_ds.remove_columns("concat")
    tokenizer_ds.set_format("torch")

    return tokenizer_ds


def train_model(
    data,
    experiment_name="Titanic",
    checkpoint="distilgpt2",
    epochs=10,
    batch_size=1,
    epsilon=10,
):
    """
    Function to train a GPT-2 model on the Titanic dataset.

    Args:
    csv_file_path (str): Path to the CSV file containing the dataset.
    experiment_name (str): Name of the experiment for saving checkpoints.
    checkpoint (str): Model checkpoint to use (e.g., 'distilgpt2').
    shuffle (bool): Whether to shuffle the data.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.

    Returns:
    str: Path to the saved model.
    """

    gc.collect()
    torch.cuda.empty_cache()

    model_name = (
        f"{experiment_name}_Epochs{epochs}_DPSGD_Epsilon{EPSILON}_{checkpoint}.pt"
    )
    model_path = f"models/{model_name}"
    if os.path.exists(model_path):
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        model.load_state_dict(torch.load(model_path))
        return model

    # Initialize the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    train_tokenizer_ds = process_data(data, tokenizer)
    print(train_tokenizer_ds)

    data_loader = DataLoader(train_tokenizer_ds, batch_size=len(data), drop_last=True)
    DELTA = 1 / len(data)
    EPSILON = epsilon

    privacy_engine = PrivacyEngine()
    model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        # noise_multiplier=1.0,  # Adjust as needed
        # max_grad_norm=1.0      # Adjust as needed
        target_delta=DELTA,
        target_epsilon=EPSILON,
        epochs=epochs,
        max_grad_norm=0.1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, epochs + 1):
        losses = []

        with BatchMemoryManager(
            data_loader=train_dataloader,
            max_physical_batch_size=batch_size,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            for step, batch in enumerate(tqdm(memory_safe_data_loader)):
                optimizer.zero_grad()

                # the batch has the format [input_ids, attention_mask, labels]
                batch = {k: v.to(device) for k, v in batch.items()}
                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": batch["labels"],
                }

                outputs = model(
                    **inputs
                )  # output = loss, logits, hidden_states, attentions

                loss = outputs[0]
                loss.backward()
                losses.append(loss.item())

                optimizer.step()

                # if step > 0 and step % 139 == 0:
        train_loss = np.mean(losses)
        eps = privacy_engine.get_epsilon(DELTA)

        print(
            f"Epoch: {epoch} | "
            f"Train loss: {train_loss:.3f} | "
            f"ɛ: {eps:.2f}"
        )

    # Save the model
    original_model = AutoModelForCausalLM.from_pretrained(checkpoint)
    original_model.load_state_dict(model._module.state_dict())
    torch.save(original_model.state_dict(), model_path)

    return original_model

if __name__ == "__main__":
    train_model(data=pd.read_csv("titanic_processed_train.csv"), # or use titanic_processed.csv if goal is to share the synthetic data online
                experiment_name="Titanic",
                epsilon=10,
                epochs=10)