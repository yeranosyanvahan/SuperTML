import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.nn import *
import pandas as pd
import seaborn as sns


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from matplotlib.patches import Patch
import itertools
from collections import deque
import seaborn as sns
from sklearn.metrics import log_loss, roc_curve,confusion_matrix, auc

class ClassificationReport:
    def __init__(self,model):
        self.model=model

    def plot_ROC(self,classnames=None):
        classes=np.max(self.labels)
        if(classnames == None):
            classnames=[f"Class{c}" for c in range(classes+1)]
        plt.figure(figsize=(20,10))
        for c in range(classes+1):
         tpr, fpr, _ = roc_curve(self.labels==c, self.prob[:,c]) 
         plt.plot(tpr, fpr, label = classnames[c])
         plt.title("ROC")
         plt.xlabel("False Positive Rate")
         plt.ylabel("True Positive Rate")
         plt.legend()
         plt.grid()
         plt.locator_params(axis="x", nbins=25)
            
    def print_AUC(self):
        classes=np.max(self.labels)
        aucs=[]
        for c in range(classes+1):
         tpr, fpr, _ = roc_curve(self.labels==c, self.prob[:,c]) 
         print(f"Clas_{c}:",auc(tpr,fpr))
            
    def AUC(self):
        classes=np.max(self.labels)
        aucs=[]
        for c in range(classes+1):
         tpr, fpr, _ = roc_curve(self.labels==c, self.prob[:,c]) 
         aucs.append(auc(tpr,fpr))
        return aucs
    
    def fit(self,X,Y):
        prob=np.array(self.model.predict(X))
        if(len(prob.shape)==1):
            prob=np.array([1-prob,prob]).T
        self.prob=prob
        self.prediction=np.argmax(prob,axis=1)
        self.labels=np.array(Y).reshape(-1)
    
    def cross_entropy(self):
        return log_loss(self.labels,self.prob)
    
    def accuracy(self):
        return sum(self.prediction ==  self.labels) / len(self.labels)
    
    def confusion_matrix(self):
        return confusion_matrix(self.labels, self.prediction)      
    
    def heatmap(self):
        sns.heatmap(self.confusion_matrix(), annot=True)
    
    def mosaic_plot(self):
        conf_matrix=self.confusion_matrix()
        n_classes=len(conf_matrix)
        
        if(n_classes==3):
             pallet = [
                '#0000f0', 
                '#f00000',
                '#00f000'
             ]
        else:
             pallet = [
                '#101010', 
                '#f0f0f0'
             ]

        class_lists = [range(n_classes)]*2
        mosaic_tuples = tuple(itertools.product(*class_lists))

        res_list = conf_matrix[0]
        for i, l in enumerate(conf_matrix):
            if i == 0:
                pass
            else:
                tmp = deque(l)
                tmp.rotate(-i)
                res_list=np.append(res_list,tmp)
        data = {t:res_list[i] for i,t in enumerate(mosaic_tuples)}

        fig, ax = plt.subplots(figsize=(11, 10))
        plt.rcParams.update({'font.size': 16})

        font_color = '#2c3e50'
        
        colors = deque(pallet[:n_classes])
        all_colors = []
        for i in range(n_classes):
            if i > 0:
                colors.rotate(-1)
            all_colors=np.append(all_colors,colors)

        props = {(str(a), str(b)):{'color':all_colors[i]} for i,(a, b) in enumerate(mosaic_tuples)}

        labelizer = lambda k: ''

        p = mosaic(data, labelizer=labelizer, properties=props, ax=ax)

        title_font_dict = {
            'fontsize': 20,
            'color' : font_color,
        }
        axis_label_font_dict = {
            'fontsize': 16,
            'color' : font_color,
        }

        ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
        ax.axes.yaxis.set_ticks([])
        ax.tick_params(axis='x', which='major', labelsize=14)

        ax.set_title('Classification Report', fontdict=title_font_dict, pad=25)
        ax.set_xlabel('Observed Class', fontdict=axis_label_font_dict, labelpad=10)
        ax.set_ylabel('Predicted Class', fontdict=axis_label_font_dict, labelpad=35)

        legend_elements = [Patch(facecolor=all_colors[i], label='Class {}'.format(i)) for i in range(n_classes)]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1,1.018), fontsize=16)

        plt.tight_layout()
        plt.show()


        
        
class NN(Module):
    def __init__(self, model, cut, add, gradients=-1):
        super(NN, self).__init__()
        self.net = Sequential(*list(model.children())[:-cut])
        self.NN = add
        params = list(model.parameters())

        for p in params[:-gradients]:
            p.requires_grad = False

        self.model = model
        
 
    def accuracy(self, out, labels):
        return sum(out == labels) / len(labels)

    def predict(self, test, device='cpu'):
        return self.to(device)(torch.from_numpy(test).to(device, dtype=torch.float)).detach().numpy()

    def predict_classes(self, test, device='cpu'):
        pred = self.predict(test, device)
        _, values = torch.max(pred, dim=1)
        return values

    def save(self, path='model'):
        torch.save(self.state_dict(), path)

    def load(self, path='model',device='cpu'):
        self.load_state_dict(torch.load(path,map_location=torch.device(device)))

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