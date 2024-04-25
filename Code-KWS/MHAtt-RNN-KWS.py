import torch
import torch.nn as nn
import copy
import soundfile
import librosa
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from read_writehtk import htkread, writehtk 
from flask import Flask, request
import flask
from flask_cors import CORS
import requests
import json
import sys, os
import code
import datetime
from pydub import AudioSegment
import math
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.optim as optim


np.random.seed(0)
torch.manual_seed(42)


class KWS(nn.Module):
    def __init__(self, num_classes,hidden_dim, num_layers, in_channel=1, n_head=4):
        super(KWS, self).__init__()
        self.num_classes = num_classes
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_channel, 10, (5,1), stride=(1,1), dilation=(1,1), padding='same')
        self.BN1 = nn.BatchNorm2d(10)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(10, 1, (5,1), stride=(1,1), dilation=(1,1), padding='same')
        self.BN2 = nn.BatchNorm2d(1)
        self.relu2 = nn.ReLU(inplace=True)
        self.gru = nn.GRU(input_size=40, hidden_size=self.hidden_dim, num_layers=self.num_layers, bidirectional=True,batch_first=True) 
        self.q_emb = nn.Linear(self.hidden_dim<<1, (self.hidden_dim<<1)*self.n_head)
        self.dropout = nn.Dropout(0.1)
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim*self.num_layers*self.n_head, self.hidden_dim*self.n_head),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim*self.n_head, self.hidden_dim*self.num_layers),
            nn.Linear(self.hidden_dim*self.num_layers,self.num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)  
        x  = self.conv1(x)
        x  = self.BN1(x)
        x =  self.relu1(x)
        x = self.conv2(x) 
        x = self.BN2(x)
        x = self.relu2(x)
     
        x = x.reshape(x.size(0),-1,x.size(1)*x.size(3)) 
        x,hidden = self.gru(x)  
        
        middle = x.size(1)//2   
        mid_feature = x[:,middle,:] 
        
        multiheads = []
        queries = self.q_emb(mid_feature).view(self.n_head, batch_size, -1, self.hidden_dim<<1) 
        for query in queries:
            att_weights = torch.bmm(query,x.transpose(1, 2)) 
            att_weights = F.softmax(att_weights, dim=-1) 
            multiheads.append(torch.bmm(att_weights, x).view(batch_size,-1)) 
        
        x = torch.cat(multiheads, dim=-1) 
        x = self.dropout(x)
        
        x = self.fc(x)
        
        return x


def readBatchScp(batch_scp):
    X_batch = []
    y_train = []
    for x in batch_scp:
          try: 
               X = torch.from_numpy(htkread(x.split(",")[0].strip('\n') ))
               y = int(x.split(",")[2].strip('\n') )
               X_batch.append(X)
               y_train.append(y)
          except:
              continue
    return X_batch, y_train

class CustomDataset_train(Dataset):
    def __init__(self, time_series, labels):
        """
        This class creates a torch dataset.

        """
        self.time_series = time_series
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        time_serie = self.time_series[idx]  # .clone().detach()
        label = self.labels[idx]  # label

        return (time_serie, label)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def predict(outputs):
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    return predictions


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def melspectrogram(xdata, samplerate, n_fft, hop_length, win_length, n_mels):
     S = librosa.feature.melspectrogram(y=xdata, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length)
     S=librosa.power_to_db(S)  #(40, 101) - (melfilter, frame)
     return  torch.FloatTensor(S)

app = Flask(__name__)
CORS(app, resources={r"/model": {"origins": "*"}})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

batch_size = 512
num_classes = 101
hidden_dim = 128
num_layers = 2

model = KWS(num_classes, hidden_dim, num_layers)
model.to(device)

# Training loop
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
criterion = CrossEntropyLoss()
#print(model)

n = count_parameters(model)
#print("Number of parameters: %s" % n)

@app.route('/model', methods=['POST'])
def run_model():

# Server code
   data = request.files
   data = data.get('audio')
   print(data)
   data.save("./myfile")
   audio = AudioSegment.from_file('./myfile')
   audio.export('./myfile.wav', format='wav')
# -------------------------------------------------

   PATH='model_epoch_50.model'
   model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
   model.eval()

   keyword = np.genfromtxt('keyWordList',dtype='str')
   n_fft, n_mels = 1024, 40
#    n_fft, n_mels = 1440, 40
   xdata, samplerate = soundfile.read('myfile.wav')
   print(samplerate)
#    xdata, samplerate = soundfile.read('FBNWB01B0004I014.wav')
   xdata = np.squeeze(np.asanyarray(xdata))
   xdata = xdata/np.amax(np.abs(xdata))  # normalize the signal

   hop_length = int(samplerate/100) # 10ms frame shift
   win_length = int(np.floor(samplerate*30/1000)) #% 30ms frame length
   #make 1-sec duration
   if len(xdata) < samplerate:
            zero_padding = np.zeros(samplerate - xdata.shape[0])
            xdata = copy.deepcopy(np.hstack((xdata, zero_padding)))
   else:
            xdata = copy.deepcopy(xdata[:samplerate])
   X = melspectrogram(xdata, samplerate, n_fft, hop_length, win_length, n_mels)
   input_ids = X.to(device)
#    X = np.random.rand(98,40)
#    X = X.astype(np.float32)
#    input_ids = torch.from_numpy(X).to(device)
   
   outputs = model(input_ids.unsqueeze(0).unsqueeze(1).permute(0,1,3,2)) ##torch.Size([1, 1, 98, 40])
   outputs = torch.softmax(outputs, dim=1)
   predictions = predict(outputs) 
   print('Identified Keyword:', keyword[predictions.cpu().detach().numpy()[0]])
   outputs = outputs.cpu().detach().numpy()
   print('Details scores')
   print('.................................')
   for i in np.arange(0, outputs.shape[1]):
       print('%s: %.2f' %(keyword[i], outputs[0][i]))
       
   response = flask.jsonify({'prediction': keyword[predictions[0]],'outputs': outputs.tolist()})
   response.headers.add('Access-Control-Allow-Origin', '*')
   return response


if __name__ == "__main__":
   app.run(port=5000)

