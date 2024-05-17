import torch
import random
import pandas as pd
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocabulary:
    SOS = '$'
    EOS = '.'
    PAD = '-'
    SOS_ID = 0
    EOS_ID = 1
    PAD_ID = 2
    
    @staticmethod
    def createVocabulary(dataset, name):
        newVocabulary = Vocabulary(name)
        
        for word in dataset:
            newVocabulary.addWord(word)
        
        return newVocabulary
    
    def __init__(self, name):
        self.name = name
        self.char2index = {self.SOS:0, self.EOS:1, self.PAD:2}
        self.char2count = {}
        self.index2char = {0:self.SOS, 1:self.EOS, 2:self.PAD}
        self.n_chars = 3

    def addWord(self, word):
        for char in word:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.char2count[char] = 1
                self.index2char[self.n_chars] = char
                self.n_chars += 1
            else:
                self.char2count[char] += 1    

    def prepareDataset(self, dataset):
        preparedDataset = []
        
        for word in dataset:
            chars = []
            
            chars.append(self.char2index[self.SOS])
            for c in word:
                chars.append(self.char2index[c])
            chars.append(self.char2index[self.EOS])

            preparedDataset.append(torch.tensor(chars, dtype=torch.long))
            
        return pad_sequence(preparedDataset, padding_value=self.char2index[self.PAD], batch_first=True).to(device=device)


class Encoder(nn.Module):
    def __init__(self, embeddingSize, noOfLayer, hiddenSize, cellType, dropout, inputVocabularySize):
        super(Encoder, self).__init__()  
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(inputVocabularySize, embeddingSize)
        self.cell = getattr(nn, cellType)(embeddingSize, hiddenSize, noOfLayer, dropout=dropout)

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


class Decoder(nn.Module):
    def __init__(self, embeddingSize, noOfLayer, hiddenSize, cellType, dropout, inputVocabularySize, outputVocabularySize):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(inputVocabularySize, embeddingSize)
        self.cell = getattr(nn, cellType)(embeddingSize, hiddenSize, noOfLayer, dropout=dropout)
        self.fc = nn.Linear(hiddenSize, outputVocabularySize)
    
    def forward(self, x, hidden, cellState):
        embeddedRep = self.dropout(self.embedding(x.unsqueeze(0)))
        
        if isinstance(self.cell, nn.LSTM):
            outputs, (hidden, cellState) = self.cell(embeddedRep, (hidden, cellState))
        else:
            outputs, hidden = self.cell(embeddedRep, hidden)

        outputs = self.fc(outputs).squeeze(0)

        return outputs, hidden, cellState


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
        
        self.encoder = Encoder(embeddingSize, noOfEncoderLayer, hiddenSize, cellType, dropout, inputVocabularySize)
        self.decoder = Decoder(embeddingSize, noOfDecoderLayer, hiddenSize, cellType, dropout, outputVocabularySize, outputVocabularySize)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.lossFunction = nn.CrossEntropyLoss()
        
    def forward(self, x, y, teacher_force_ratio = 0.5):
        outputs = torch.zeros(y.shape[0], x.shape[1], self.outputVocabularySize).to(device)

        hidden, cell = self.encoder(x)
        hidden = hidden.repeat(self.noOfDecoderLayer, 1, 1)
        if self.cellType == "LSTM":
            cell = cell.repeat(self.noOfDecoderLayer, 1, 1)

        decoderInput = y[0]
        for i in range(1, y.shape[0]):
            output, hidden, cell = self.decoder(decoderInput, hidden, cell)
            outputs[i] = output

            if random.random() < teacher_force_ratio:
                decoderInput = y[i]
            else:
                decoderInput = output.argmax(dim=1)

        return outputs

    def trainModel(self, train_dataloader, val_dataloader, noOfEpochs):
        for epoch in range(noOfEpochs):
            self.train()
            
            for batch_idx, (x, y) in enumerate(train_dataloader):
                y_hat = self(x.T.to(device), y.T.to(device))

                self.optimizer.zero_grad()
                self.lossFunction(y_hat[1:].reshape(-1, y_hat.shape[2]), y.T.to(device)[1:].reshape(-1)).backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                self.optimizer.step()

            train_loss, train_accuracy = self.evaluateModel(train_dataloader)
            val_loss, val_accuracy = self.evaluateModel(val_dataloader)
            
            print("Epoch: ", epoch)
            print("Train loss: ", train_loss)
            print("Validation loss: ", val_loss)
            print("Train accuracy: ", train_accuracy)
            print("Validation accuracy: ", val_accuracy)
            print("")
        
    def evaluateModel(self, dataloader):
        correctPrediction = 0
        totalLoss = 0
        
        self.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                y_hat = self(x.T.to(device), y.T.to(device), teacher_force_ratio=0.0)
                
                correctPrediction += torch.logical_or(y_hat.argmax(dim=2) == y.T.to(device), y.T.to(device) == Vocabulary.PAD_ID).all(dim=0).sum().item()
                totalLoss += self.lossFunction(y_hat[1:].reshape(-1, y_hat.shape[2]), y.T.to(device)[1:].reshape(-1)).item()
                
        accuracy = correctPrediction / (len(dataloader)*dataloader.batch_size)
        loss = totalLoss / len(dataloader)
        
        return loss, accuracy        
 
          
############################################################################################################################   
  
noOfEpochs = 5
batch_size = 32
embeddingSize = 64
noOfEncoderLayer = 2
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


model = TransliterationModel(embeddingSize, noOfEncoderLayer, noOfDecoderLayer, hiddenSize, cellType, dropout, englishVocabulary.n_chars, bengaliVocabulary.n_chars).to(device)
model.trainModel(train_dataloader, val_dataloader, noOfEpochs)