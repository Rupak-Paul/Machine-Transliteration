# CS6910-Assignment-3
This is [CS6910](http://www.cse.iitm.ac.in/~miteshk/CS6910.html) course assignment-3 at IIT Madras. [Here](https://wandb.ai/cs6910-dl-assignments/assignment%203/reports/Assignment-3--Vmlldzo3NTUwNzY4?accessToken=cb5ahfcp8eisq1oe6ixumae10ttzpp16rtdbtsfm30le7l9zgdqko388iasvrh93) you will find detailed information about the assignment. In this assignment, I created a **Recurrent Neural Network** to transliterate a word from **English** to **Bengali** (how we type while chatting with our friends on WhatsApp etc). Different types of cells such as **vanilla RNN**, **LSTM**, **GRU** have been implemented to improve the accuracy of the model. Along with that, an **attention network** has been added in the model to further increase the accuracy of the model.

The model has been trained by using [Aksharantar dataset](https://drive.google.com/file/d/1tGIO4-IPNtxJ6RQMmykvAfY_B0AaLY5A/view?usp=drive_link) released by [AI4Bharat](https://ai4bharat.org/).

[Here](https://wandb.ai/cs23m056/CS23M056_DL_Assignment_3/reports/CS6910-Assignment-3--Vmlldzo3OTgzODE2) is the detailed wandb report for this assignment.

## Dependencies
In this assignment, I used these libraies:
```
import torch
import random
import pandas as pd
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
```

If you don't have these packages installed then use this command to install:
```
pip install pytorch
pip install pandas
```

## Supported Command Line Arguments

If you want to **train** and **use** this model for your **dataset** then download the **train.py** file. Hyperparameters of the model, train and test dataset path can be mentioned using command line arguments. Here are the supported command line arguments:

| Argument                    | Short Option | Type   | Default                                    | Description                                   |
|-----------------------------|--------------|--------|--------------------------------------------|-----------------------------------------------|
| `--train_dataset_path`      | `-trainPath` | `str`  | `aksharantar_sampled/ben/ben_train.csv`    | Path to the training dataset                  |
| `--validation_dataset_path` | `-vaildPath` | `str`  | `aksharantar_sampled/ben/ben_valid.csv`    | Path to the validation dataset                |
| `--test_dataset_path`       | `-testPath`  | `str`  | `aksharantar_sampled/ben/ben_test.csv`     | Path to the test dataset                      |
| `--epochs`                  | `-ep`        | `int`  | `15`                                       | Number of epochs                              |
| `--batch_size`              | `-bs`        | `int`  | `32`                                       | Batch size                                    |
| `--cell_type`               | `-ct`        | `str`  | `LSTM`                                     | Type of RNN cell (e.g., LSTM, GRU)            |
| `--embedding_size`          | `-es`        | `int`  | `128`                                      | Size of the embeddings                        |
| `--hidden_size`             | `-hs`        | `int`  | `256`                                      | Size of the hidden layers                     |
| `--encoder_layer`           | `-el`        | `int`  | `3`                                        | Number of encoder layers                      |
| `--decoder_layer`           | `-dl`        | `int`  | `3`                                        | Number of decoder layers                      |
| `--dropout`                 | `-dp`        | `float`| `0.2`                                      | Dropout rate                                  |
| `--bidirectional`           | `-bd`        | `str`  | `Yes`                                      | Use bidirectional RNN (Yes/No)                |
| `--attention`               | `-atn`       | `str`  | `Yes`                                      | Use attention mechanism (Yes/No)              |
