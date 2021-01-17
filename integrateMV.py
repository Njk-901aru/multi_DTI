import time, argparse, gc, os

import numpy as np
import cupy as cp

from rdkit import Chem

from feature import *
import SCFPfunctions as Mf
import pickle

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Reporter, report, report_scope
from chainer import Link, Chain, ChainList, training
from chainer.datasets import tuple_dataset
from chainer.training import extensions
from chainer.optimizer_hooks import WeightDecay
#-------------------------------------------------------------
    #Network definition
class CNN(chainer.Chain):

    def __init__(self, prosize, plensize, batchsize, s1, sa1, s2, sa2, s3, sa3, j1, pf1, ja1, j2, pf2, ja2, j3, pf3, ja3, n_hid3, n_hid4, n_hid5, n_out):
        
        # prosize, plensize_20 = size of protein one hot feature matrix
        # k1, s1, f1 = window-size, stride-step, No. of filters of first SMILES-CNN convolution layer
        # ka1, sa1 = window-size, stride-step of first SMILES-CNN average-pooling layer
        # k2, s2, f2 = window-size, stride-step, No. of filters of second SMILES-CNN convolution layer
        # ka2, sa2 = window-size, stride-step of second SMILES-CNN average-pooling layer
        # k3, s3, f3 = window-size, stride-step, No. of filters of third SMILES-CNN convolution layer
        # ka3, sa3 = window-size, stride-step of third SMILES-CNN average-pooling layer
        # j1, s1, pf1 = window-size, stride-step, No. of filters of first protein-CNN convolution layer
        # ja1, sa1 = window-size, stride-step of first protein-CNN average-pooling layer
        # j2, s2, pf2 = window-size, stride-step, No. of filters of second protein-CNN convolution layer
        # ja2, sa2 = window-size, stride-step of second protein-CNN average-pooling layer
        # j3, s3, pf3 = window-size, stride-step, No. of filters of third protein-CNN convolution layer
        # ja3, sa3 = window-size, stride-step of third protein-CNN average-pooling layer
        
        super(CNN, self).__init__(
            conv1_chem=L.Convolution2D(1, f1, (k1, clensize), stride=s1, pad = (k1//2,0)),
            bn1=L.BatchNormalization(f1),
            conv2_chem=L.Convolution2D(f1, f2, (k2, 1), stride=s2, pad = (k2//2,0)),
            bn2=L.BatchNormalization(f2),
            conv1_pro=L.Convolution2D(1, pf1, (j1, plensize), stride=s1, pad = (j1//2,0)),
            bn1_pro=L.BatchNormalization(pf1),
            conv2_pro=L.Convolution2D(pf1, pf2, (j2, 1), stride=s2, pad = (j2//2,0)),
            bn2_pro=L.BatchNormalization(pf2),
            conv3_pro=L.Convolution2D(pf2, pf3, (j3, 1), stride=s3, pad = (j3//2,0)),
            bn3_pro=L.BatchNormalization(pf3),       
            fc3=L.Linear(None, n_hid3),
            fc4=L.Linear(None, n_hid4),
            fc5=L.Linear(None, n_hid5),
            fc3_pro=L.Linear(None, n_hid3),
            fc4_pro=L.Linear(None, n_hid4),
            fc5_pro=L.Linear(None, n_hid5),
            fc6=L.Linear(None, n_out)
        )
        self.n_hid3, self.n_hid4, self.n_hid5, self.n_out = n_hid3, n_hid4, n_hid5, n_out
        self.prosize, self.plensize = prosize, plensize
        self.s1, self.sa1, self.s2, self.sa2, self.s3, self.sa3 = s1, sa1, s2, sa2, s3, sa3
        self.j1, self.ja1, self.j2, self.ja2, self.j3, self.ja3 = j1, ja1, j2, ja2, j3, ja3 

        self.m1 = (self.prosize+(self.j1//2*2)-self.j1)//self.s1+1
        self.m2 = (self.m1+(self.ja1//2*2)-self.ja1)//self.sa1+1
        self.m3 = (self.m2+(self.j2//2*2)-self.j2)//self.s2+1
        self.m4 = (self.m3+(self.ja2//2*2)-self.ja2)//self.sa2+1
        self.m5 = (self.m4+(self.j3//2*2)-self.j3)//self.s3+1
        self.m6 = (self.m5+(self.ja3//2*2)-self.ja3)//self.sa3+1
        
    def __call__(self, ecfp, sequences, n2vc, n2vp, interactions):
        z = self.cos_similarity(ecfp, sequences, n2vc, n2vp)
        Z = self.fc6(z)
        loss = F.sigmoid_cross_entropy(Z, interactions)
        accuracy = F.binary_accuracy(Z, interactions) 
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss
    
    def predict_pro(self, seq):
        h = F.dropout(F.leaky_relu(self.bn1_pro(self.conv1_pro(seq))), ratio=0.2) # 1st conv
        h = F.average_pooling_2d(h, (self.ja1,1), stride=self.sa1, pad=(self.ja1//2,0)) # 1st pooling
        h = F.dropout(F.leaky_relu(self.bn2_pro(self.conv2_pro(h))), ratio=0.2) # 2nd conv
        h = F.average_pooling_2d(h, (self.ja2,1), stride=self.sa2, pad=(self.ja2//2,0)) # 2nd pooling
        h = F.dropout(F.leaky_relu(self.bn3_pro(self.conv3_pro(h))), ratio=0.2) # 3rd conv
        h = F.average_pooling_2d(h, (self.ja3,1), stride=self.sa3, pad=(self.ja3//2,0)) # 3rd pooling
        h_pro = F.max_pooling_2d(h, (self.m6,1)) # grobal max pooling, fingerprint
        #print(h_pro.shape)
        h_pro = F.dropout(F.leaky_relu(self.fc3_pro(h_pro)), ratio=0.2)# fully connected_1
        #print(h_pro.shape)
        return h_pro
    
    def cos_similarity(self, fp, seq, n2c, n2p):
        x_compound = fp
        x_protein = self.predict_pro(seq)
        x_compound = self.fc4(F.concat((x_compound, n2c)))
        x_compound = F.dropout(F.leaky_relu(x_compound), ratio=0.2)
        x_compound = F.dropout(F.leaky_relu(self.fc5(x_compound)), ratio=0.2)
        x_protein = self.fc4_pro(F.concat((x_protein, n2p)))
        x_protein = F.dropout(F.leaky_relu(x_protein), ratio=0.2)
        #print(x_protein.shape)
        x_protein = F.dropout(F.leaky_relu(self.fc5_pro(x_protein)), ratio=0.2)
        #print(x_protein.shape)
        
        y = x_compound * x_protein
        
        return y 

