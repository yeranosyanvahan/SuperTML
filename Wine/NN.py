import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.nn import *
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns


class NN(Module):
    def __init__(self, model, cut, add, gradients=-1):
        super(NN, self).__init__()
        self.net = Sequential(*list(model.children())[:-cut])
        self.NN = add
        params = list(model.parameters())

        for p in params[:-gradients]:
            p.requires_grad = False

        self.model = model
        
    def score(self,X_test,Y_test):
        return self.accuracy(np.array(self.predict(X_test)),Y_test)
        
    def accuracy(self, out, labels):
        return sum(out == labels) / len(labels)

    def predict_proba(self, test, device='cpu'):
        return self.to(device)(torch.from_numpy(test).to(device, dtype=torch.float))

    def predict(self, test, device='cpu'):
        pred = self.predict_proba(test, device)
        _, values = torch.max(pred, dim=1)
        return values

    def confusion_matrix(self, X, Y, heatmap=False, device='cpu'):
        if heatmap:
            sns.heatmap(self.confusion_matrix(X, Y, device=device), annot=True)
        return confusion_matrix(Y, self.predict(X, device=device))

    def save(self, path='model'):
        torch.save(self.state_dict(), path)

    def load(self, path='model'):
        self.load_state_dict(torch.load(path))

    def optimization(self, criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer

    def TRAIN(self, train_data, validation_data=None, epochs=10, batch_size=32, verbose=1, random_seed=1, device='cpu'):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)

        optimizer, criterion = self.optimizer, self.criterion

        Train_stats = []
        Validation_stats = []

        for epoch in range(epochs):

            Names = ['Epoch', 'Accuracy', 'Loss']

            Stats = {}

            Stats['Accuracy'] = 0

            Stats = []
            for i, (inputs, labels) in enumerate(train_data):
                self.train()
                inputs = inputs.to(device, dtype=torch.float);
                optimizer.zero_grad()
                outputs = self(inputs)
                labels = labels.to(device).long()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                average = 'micro'
                _, pred = torch.max(outputs, dim=1)
                Stats.append([epoch + 1, self.accuracy(pred, labels).item(), loss.item()])
            Stats = {key: val for key, val in zip(Names, np.mean(np.asarray(Stats).T, axis=1))}

            Train_stats.append(Stats)

            if verbose:
                print("-----      TRAIN RESULTS      -----")
                print(Train_stats[-1])

            Stats = []

            if validation_data != None:

                for i, (inputs, labels) in enumerate(validation_data):
                    self.eval()
                    inputs = inputs.to(device, dtype=torch.float);
                    outputs = self(inputs)
                    labels = labels.to(device).long()
                    loss = criterion(outputs, labels)

                    _, pred = torch.max(outputs, dim=1)
                    Stats.append([epoch + 1, self.accuracy(pred, labels).item(), loss.item()])

                Validation_stats.append({key: val for key, val in zip(Names, np.mean(np.asarray(Stats).T, axis=1))})

                if verbose:
                    print("-----      Validation  RESULTS     -----")
                    print(Validation_stats[-1])
        self.Train_stats = Train_stats
        self.Validation_stats = Validation_stats
        return Train_stats, Validation_stats

    def plot(self, metric):

        Validation = pd.DataFrame(self.Validation_stats)
        Train = pd.DataFrame(self.Train_stats)

        ax = Train.plot(x='Epoch', y=metric, label='Train', figsize=(20, 10))
        Validation.plot(x='Epoch', y=metric, ax=ax, label='Validation')
        ax.set_ylabel(metric)
        ax.grid()
        ax.locator_params(axis="x", nbins=25)

    def forward(self, X):
        A = self.net(X)
        A = A.reshape(A.shape[:2])
        return self.NN(A)