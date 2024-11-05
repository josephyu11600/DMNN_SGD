#%% IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import Kmeans

#%% CLASSES

class Spherical_Denrite(nn.Module):
    def __init__(self, c, r):
        super().__init__()
        self.centroid = nn.Parameter(torch.tensor(c, dtype=torch.float32), requires_grad=False)
        self.radii    = nn.Parameter(torch.tensor(r, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        # print(x.shape)
        # print(self.centroid.shape)
        x = self.radii - torch.sqrt(torch.pow(x - self.centroid,2).sum(1))
        return x
    
class Soft_Maximum(nn.Module):
    def __init__(self,beta=1.0):
        super().__init__()
        self.beta = beta
        self.softmax = nn.Softmax(dim=1) 
    def forward(self, x):
        softmax = self.softmax(self.beta * x)
        return x * softmax
    

class DMNN(nn.Module):
    def __init__(self, data,classes=2):
        super().__init__()
        self.classes = classes
        self.spherical_dendrites = []
        self.output_nodes = []
        self.initialize_spherical_dendrites(data)
        self.soft_maximum  = Soft_Maximum()
        self.softmax       = nn.Softmax(dim=1) 

    def forward(self, inp):

        inp = inp.reshape(inp.shape[0], -1)

        dendrite_cluster = []
        for i, dc in enumerate(self.spherical_dendrites):
            synapse_cluster = []
            for j, dendrite in enumerate(self.spherical_dendrites[i]):
                x = self.spherical_dendrites[i][j](inp)
                synapse_cluster.append(x)
            print(synapse_cluster[0].shape)
            x   = torch.stack(synapse_cluster, dim=1)
            print(x.shape)
            out = self.soft_maximum(x)
            print(out.shape)
            dendrite_cluster.append(out)
        print(len(dendrite_cluster))
        for c in range(self.classes):
            node_activations = []
            for i in range(len(dendrite_cluster)):
                if i == 0:
                    x = self.output_nodes[c][i](dendrite_cluster[i])
                else:
                    x += self.output_nodes[c][i](dendrite_cluster[i])
            node_activations.append(x)
        print(len(node_activations))
        out = torch.stack(node_activations, dim=1)
        out = self.softmax(out)
        return out
    
    def initialize_spherical_dendrites(self,data):
        for class_no in range(self.classes): 
            dendrite_cluster = []
            class_data = data['train_signals'][data['train_labels'] == class_no,...] #EDIT dictionary referencing :) 
            class_data = class_data.reshape(class_data.shape[0],-1)
            print(class_data.shape)
            # class_data = class_data[:1000,] #EDIT remove
            print(class_data.shape)
            n = Kmeans.find_optimal_clusters(data=class_data,max_k=1000)
            centroids, radii = Kmeans.calculate_centroids_and_radii(data=class_data, n_clusters=n)
            for centroid, radius in zip(centroids, radii):
                dendrite_cluster.append(Spherical_Denrite(centroid, radius))
            self.spherical_dendrites.append(dendrite_cluster)

        for c in range(self.classes):
            node = []
            for dendrite_cluster in self.spherical_dendrites:
                weight_kl = nn.Linear(len(dendrite_cluster),1)
                node.append(weight_kl)
            self.output_nodes.append(node)


#%% LOCAL TEST


if __name__ == "__main__":

    import numpy as np
    import sys
    import os
    sys.path.append('../..')
    import DMNN_SGD.configuration as configuration

    
    Config = configuration.Config()

    print(Config.data_path, Config.train_file)

    file_name = os.path.join(Config.data_path, Config.train_file)
    data = np.load(file_name)
    keys = list(data.keys())
    net  = DMNN(data)
    inp  = torch.tensor(data['train_signals'][:,...],dtype=torch.float32)
    print(net(inp))
    print('__EX__')


# %%
