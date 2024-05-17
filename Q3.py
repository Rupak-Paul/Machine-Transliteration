import torch
import random
import pandas as pd
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# This class represents vocabulary of a languages used for transliteration
class Vocabulary:
    SOS = '$' # start delimiter
    EOS = '.' # end delimiter
    PAD = '-' # null delimiter
    SOS_ID = 0 # indexed representation of start delimiter
    EOS_ID = 1 # indexed representation of end delimiter
    PAD_ID = 2 # indexed representation of null delimiter
    
    # Creates and returns the object of Vocabulary for the given language dataset
    @staticmethod
    def createVocabulary(dataset, name):
        newVocabulary = Vocabulary(name)
        
        for word in dataset:
            newVocabulary.addWord(word)
        
        return newVocabulary
    
    def __init__(self, name):
        self.name = name # stores name of the language
        self.char2index = {self.SOS:0, self.EOS:1, self.PAD:2} # stores alphabet of the language to corresponding index mapping
        self.char2count = {} # stores count of a particular alphabet
        self.index2char = {0:self.SOS, 1:self.EOS, 2:self.PAD} # stores index to corresponding alphabet mapping
        self.n_chars = 3 # stores total number of alphabet present in the language

    # Adding a word to the Vocabulary
    def addWord(self, word):
        for char in word:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.char2count[char] = 1
                self.index2char[self.n_chars] = char
                self.n_chars += 1
            else:
                self.char2count[char] += 1    

    # Function to prepare the dataset by converting characters to their corresponding indexed representations
    # Returns padded tensor containing the indexed representations of characters
    def prepareDataset(self, dataset):
        preparedDataset = []
        
        for word in dataset:
            chars = []
            
            chars.append(self.char2index[self.SOS]) # appending SOS token at the start of the word
            for c in word:
                chars.append(self.char2index[c])
            chars.append(self.char2index[self.EOS]) # appending EOS token at the end of the word

            # Convert the list of characters to a tensor and store it in the prepared dataset list
            preparedDataset.append(torch.tensor(chars, dtype=torch.long))
        
        # Pad the sequences in the prepared dataset to ensure equal length and convert it to a tensor     
        return pad_sequence(preparedDataset, padding_value=self.char2index[self.PAD], batch_first=True).to(device=device)


# This class represents a encoder used for transliteration
class Encoder(nn.Module):
    def __init__(self, embeddingSize, noOfLayer, hiddenSize, cellType, dropout, inputVocabularySize):
        super(Encoder, self).__init__()  
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(inputVocabularySize, embeddingSize)
        self.cell = getattr(nn, cellType)(embeddingSize, hiddenSize, noOfLayer, dropout=dropout)

    # Defining forward propagation through the encoder
    def forward(self, x):
        embeddedRep = self.dropout(self.embedding(x))
        
        if isinstance(self.cell, nn.LSTM):
            outputs, (hidden, cellState) = self.cell(embeddedRep)
            cellState = (cellState[-1,:,:]).unsqueeze(0)
        else:
            outputs, hidden = self.cell(embeddedRep)
            cellState = None
        
        hidden = (hidden[-1,:,:]).unsqueeze(0)

        return hidden, cellState


# This class represents a decoder used for transliteration
class Decoder(nn.Module):
    def __init__(self, embeddingSize, noOfLayer, hiddenSize, cellType, dropout, inputVocabularySize, outputVocabularySize):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(inputVocabularySize, embeddingSize)
        self.cell = getattr(nn, cellType)(embeddingSize, hiddenSize, noOfLayer, dropout=dropout)
        self.fc = nn.Linear(hiddenSize, outputVocabularySize)
    
    # Defiining forward propagation through the decoder
    def forward(self, x, hidden, cellState):
        embeddedRep = self.dropout(self.embedding(x.unsqueeze(0)))
        
        if isinstance(self.cell, nn.LSTM):
            outputs, (hidden, cellState) = self.cell(embeddedRep, (hidden, cellState))
        else:
            outputs, hidden = self.cell(embeddedRep, hidden)

        outputs = self.fc(outputs).squeeze(0)

        return outputs, hidden, cellState


# This class represents a RNN model that can do transliteration
class TransliterationModel(nn.Module):
    def __init__(self, embeddingSize, noOfEncoderLayer, noOfDecoderLayer, hiddenSize, cellType, dropout, inputVocabularySize, outputVocabularySize):
        super(TransliterationModel, self).__init__()
        self.embeddingSize = embeddingSize
        self.noOfEncoderLayer = noOfEncoderLayer
        self.noOfDecoderLayer = noOfDecoderLayer
        self.hiddenSize = hiddenSize
        self.cellType = cellType
        self.dropout = dropout
        self.inputVocabularySize = inputVocabularySize
        self.outputVocabularySize = outputVocabularySize
        
        # creating encoder and decoder
        self.encoder = Encoder(embeddingSize, noOfEncoderLayer, hiddenSize, cellType, dropout, inputVocabularySize)
        self.decoder = Decoder(embeddingSize, noOfDecoderLayer, hiddenSize, cellType, dropout, outputVocabularySize, outputVocabularySize)

        # creating optimizer and loss function to train the model
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.lossFunction = nn.CrossEntropyLoss()
        
    # Defining forward propagation for the model
    def forward(self, x, y, teacher_force_ratio = 0.5):
        outputs = torch.zeros(y.shape[0], x.shape[1], self.outputVocabularySize).to(device)

        # passing the input through encoder
        hidden, cell = self.encoder(x)
        hidden = hidden.repeat(self.noOfDecoderLayer, 1, 1)
        if self.cellType == "LSTM":
            cell = cell.repeat(self.noOfDecoderLayer, 1, 1)

        # passing the output of the encoder through decoder
        decoderInput = y[0]
        for i in range(1, y.shape[0]):
            output, hidden, cell = self.decoder(decoderInput, hidden, cell)
            outputs[i] = output

            if random.random() < teacher_force_ratio:
                decoderInput = y[i]
            else:
                decoderInput = output.argmax(dim=1)

        return outputs

    # Train the model using train dataset for the given number of epochs
    def trainModel(self, train_dataloader, val_dataloader, noOfEpochs):
        for epoch in range(noOfEpochs):
            self.train()
            
            for batch_idx, (x, y) in enumerate(train_dataloader):
                # forward propagation
                y_hat = self(x.T.to(device), y.T.to(device))
                
                # backpropagating the loss
                self.optimizer.zero_grad()
                self.lossFunction(y_hat[1:].reshape(-1, y_hat.shape[2]), y.T.to(device)[1:].reshape(-1)).backward()

                # clip gradients to prevent exploding gradients problem
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                
                # update the model parameters using the optimizer
                self.optimizer.step()

            # calculating losses and accuracies
            train_loss, train_accuracy = self.evaluateModel(train_dataloader)
            val_loss, val_accuracy = self.evaluateModel(val_dataloader)
            
            # logging losses and accuracies to console
            print("Epoch: ", epoch+1)
            print("Train loss, accuracy: ", train_loss, ", ", train_accuracy)
            print("Val loss, accuracy: ", val_loss, ", ", val_accuracy)
            
            # logging losses and accuracies to wandb
            wandb.log({
                "epochs": epoch+1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
        
    # Calculates and returns loss and accuracy for the given dataset using current parameters of the model
    def evaluateModel(self, dataloader):
        correctPrediction = 0
        totalLoss = 0
        
        self.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                # forward pass through the model
                y_hat = self(x.T.to(device), y.T.to(device), teacher_force_ratio=0.0)
                
                # batch wise accumulating loss and correct predictions
                correctPrediction += torch.logical_or(y_hat.argmax(dim=2) == y.T.to(device), y.T.to(device) == Vocabulary.PAD_ID).all(dim=0).sum().item()
                totalLoss += self.lossFunction(y_hat[1:].reshape(-1, y_hat.shape[2]), y.T.to(device)[1:].reshape(-1)).item()
                
        accuracy = correctPrediction / (len(dataloader)*dataloader.batch_size)
        loss = totalLoss / len(dataloader)
        
        return loss, accuracy                
 
          
############################################################################################################################   

# hyperparameter of the model  
noOfEpochs = 15
batch_size = 32
embeddingSize = 128
noOfEncoderLayer = 3
noOfDecoderLayer = 3
hiddenSize = 256
cellType = "LSTM"
dropout = 0.2      

# loading dataset
train_dataset = pd.read_csv('aksharantar_sampled/ben/ben_train.csv', sep=',', header=None).values
val_dataset = pd.read_csv('aksharantar_sampled/ben/ben_valid.csv', sep=',', header=None).values

# building vocabulary
englishVocabulary = Vocabulary.createVocabulary(train_dataset[:,0], "English")
bengaliVocabulary = Vocabulary.createVocabulary(train_dataset[:,1], "Bengali")

# preparing dataset
x_train = englishVocabulary.prepareDataset(train_dataset[:,0])
y_train = bengaliVocabulary.prepareDataset(train_dataset[:,1])

x_val = englishVocabulary.prepareDataset(val_dataset[:,0])
y_val = bengaliVocabulary.prepareDataset(val_dataset[:,1])

# creating dataloader
train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

# creating the model
model = TransliterationModel(embeddingSize, noOfEncoderLayer, noOfDecoderLayer, hiddenSize, cellType, dropout, englishVocabulary.n_chars, bengaliVocabulary.n_chars).to(device)

# training the model
model.trainModel(train_dataloader, val_dataloader, noOfEpochs)

# creating test dataset
test_dataset = pd.read_csv('aksharantar_sampled/ben/ben_test.csv', sep=',', header=None).values
x_test = englishVocabulary.prepareDataset(test_dataset[:,0])
y_test = bengaliVocabulary.prepareDataset(test_dataset[:,1])
test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# calculating test accuracy
test_loss, test_accuracy = model.evaluateModel(test_dataloader)

# logging test accuracy on console
print("Test accuracy: ", test_accuracy)