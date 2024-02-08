
# Opacus-Synthetic-Tabular
Opacus-Synthetic-Tabular is a tool for generating synthetic tabular data with differential privacy, leveraging Opacus for secure, privacy-preserving AI models.

A good example is the [Titanic dataset from kaggle](https://www.kaggle.com/competitions/titanic "Titanic") . The project is structured into four main steps, outlined below.

## Steps Overview

1. **Preprocess Titanic Data (`preprocess_titanic.py`)**: This script preprocesses the Titanic dataset for synthetic data generation. It includes splitting the data into training and test sets, where the training set is used in Step 2, and the test set is used in Step 4.

2. **Train with DP-SGD (`train_with_DPSGD.py`)**: This script trains a GPT-2 model with Differential Privacy using Stochastic Gradient Descent (DP-SGD) and the Opacus library. The training utilizes the preprocessed data from Step 1.

3. **Generate Synthetic Data (`generate_synthetic_data.py`)**: After training the model, this script generates synthetic data resembling the original Titanic data distribution. The generated synthetic data is saved for use in Step 4.

4. **Evaluate Synthetic Data (`titanic_test.py`)**: This script evaluates the utility of the synthetic data by training and testing a RandomForestClassifier. The classifier is trained on the synthetic data and tested using the original test set from Step 1.

## Requirements

- Python 3.10.8 or similar
- modules in requirements.txt
```bash
pip install -r requirements.txt
```

## Usage

1. **Preprocess the data**:
    ```bash
    python preprocess_titanic.py
    ```
Will generate titanic_processed.csv, titanic_processed_train.csv and titanic_processed_test.csv

2. **Train the model with DP-SGD**:
    ```bash
    python train_with_DPSGD.py
    ```
Uses titanic_processed_train.csv to train and save the model into models\\{experiment_name}_Epochs{epochs}_DPSGD_Epsilon{EPSILON}_{checkpoint}.pt

3. **Generate synthetic data**:
    ```bash
    python generate_synthetic_data.py
    ```
Uses the saved model to generate synthetic data and saves it to synth_data_DP\\{experiment_name}_Samples{str(n_samples)}_Epsilon{str(EPSILON)}.csv

4. **Test the synthetic data**:
    ```bash
    python titanic_test.py
    ```
Uses the trained synthetic data to train a RandomForestClassifier and test it using titanic_processed_test.csv from step 1.

## Inspiration

This project was inspired by the research paper titled "Language Models are Realistic Tabular Data Generators" by Vadim Borisov, Kathrin Sessler, Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci, which explores the generation of synthetic data using GPT-2 models. The paper demonstrates the potential of GPT-2 in generating realistic datasets, serving as a foundational concept for our approach. However, our project extends the idea by incorporating differential privacy to enhance data privacy.

For more details, you can read the original paper here: [Language Models are Realistic Tabular Data Generators](https://openreview.net/forum?id=cEygmQNOeI).

## Contributing

Feel free to fork the project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
