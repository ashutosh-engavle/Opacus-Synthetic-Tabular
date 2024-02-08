import os
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.asyncio import tqdm as tqdm_async

def generate_synthetic_data_llms(
    n_samples=1000,
    model=None,
    experiment_name="Titanic_Epochs10",
    checkpoint="distilgpt2",
    batch_size=50,
    EPSILON=10,
    columns=None,
    label_col="WakeUpPain",
):
    file_path = f"synth_data_DP/{experiment_name}_Samples{str(n_samples)}_Epsilon{str(EPSILON)}.csv"
    print(file_path)
    # If data is generated return it
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    
    if not columns:
        columns = pd.read_csv("titanic_processed_train.csv").columns.tolist()

    model_name = f"{experiment_name}_DPSGD_Epsilon{EPSILON}_{checkpoint}.pt"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.load_state_dict(torch.load(f"models/{model_name}"))
    model.to(device)

    temperature = 0.7
    population = ["0", "1"]
    weights = [0.75, 0]
    weights[1] = 1 - weights[0]
    k = batch_size

    generated_data = []

    for i in tqdm_async(range(int(n_samples / k) + 1)):
        start_words = random.choices(population, weights, k=k)
        start = [label_col + " is " + str(s) + "," for s in start_words]
        # print(start)
        start_token = torch.tensor(tokenizer(start)["input_ids"]).to(device)

        gens = model.generate(
            input_ids=start_token,
            min_length=500,
            max_length=750,
            do_sample=True,
            temperature=temperature,
            pad_token_id=50256,
        )  # input_ids=test_ids

        for g in gens:
            generated_data.append(g)

    decoded_data = [tokenizer.decode(gen) for gen in generated_data]
    decoded_data = [d.replace("<|endoftext|>", "") for d in decoded_data]
    decoded_data = [d.replace("\\n", " ") for d in decoded_data]
    decoded_data = [d.replace("\\r", "") for d in decoded_data]
    print(decoded_data[:3])

    df_gen = pd.DataFrame(columns=columns)
    print(df_gen)

    for g in tqdm_async(decoded_data):
        features = g.split(",")

        td = dict.fromkeys(columns)

        # Transform all features back to tabular data
        for f in features:
            values = f.strip().split(" ")

            if values[0] in columns:
                td[values[0]] = [values[-1]]

        td = {key: [value] for key, value in td.items()}
        df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)

    df_gen = df_gen.dropna(how="all")
    df_gen = df_gen.dropna(subset=[label_col])
    for col in df_gen.columns:
        # Unwrap the list, fill NaN where the list is None
        df_gen[col] = df_gen[col].apply(
            lambda x: x[0] if isinstance(x, list) and x else None
        )

        # Convert to numeric, if possible
        df_gen[col] = pd.to_numeric(df_gen[col], errors="coerce")

    df_gen = df_gen.dropna(axis=1, how="all")
    df_gen = df_gen.dropna(axis=0, how="any")

    for col in df_gen.select_dtypes(include=["float64", "int64"]).columns:
        mean_value = df_gen[col].mean()
        df_gen[col].fillna(mean_value, inplace=True)
    print(df_gen)

    df_gen = df_gen.reset_index(drop=True)
    df_gen.to_csv(file_path, index=False)
    return df_gen

if __name__ == '__main__':
    generate_synthetic_data_llms()
