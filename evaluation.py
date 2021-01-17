#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pickle
import time, argparse, gc, os

import numpy as np
import cupy as cp

from rdkit import Chem

#from feature import *
import integrateMV as MV

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
 # featurevector size
plensize= 20
#-------------------------------------------------------------
def main():

    START = time.time()

    import argparse as arg
    parser = arg.ArgumentParser(description='ecfpWD_n2v')
    parser.add_argument('--gpu', '-g', type=int, default=1, help='GPU ID')
    parser.add_argument('--batchsize', '-b', type=int, default=100, help='minibatch')
    parser.add_argument('--epoch', '-e', type=int, default=150, help='number of max iteration to evaluate')
    parser.add_argument('--s1', type=int, default=1)
    parser.add_argument('--sa1', type=int, default=1)
    parser.add_argument('--s2', type=int, default=1)
    parser.add_argument('--sa2', type=int, default=1)
    parser.add_argument('--s3', type=int, default=1)
    parser.add_argument('--sa3', type=int, default=1)
    parser.add_argument('--j1', type=int, default=33)
    parser.add_argument('--pf1', type=int, default=64)
    parser.add_argument('--ja1', type=int, default=17)
    parser.add_argument('--j2', type=int, default=23)
    parser.add_argument('--pf2', type=int, default=64)
    parser.add_argument('--ja2', type=int, default=11)
    parser.add_argument('--j3', type=int, default=33)
    parser.add_argument('--pf3', type=int, default=32)
    parser.add_argument('--ja3', type=int, default=17)
    parser.add_argument('--n_hid3', type=int, default=70)
    parser.add_argument('--n_hid4', type=int, default=80)
    parser.add_argument('--n_hid5', type=int, default=60)
    parser.add_argument('--n_out', type=int, default=1)
    parser.add_argument('--prosize', type=int, default=5762)
    parser.add_argument('--input', '-i', default='./dataset/hard_dataset')
    parser.add_argument('--output', '-o', default='./result/hard_dataset')
    parser.add_argument('--place', '-p', default='./ecfpN2vc_mSGD/pattern')
    parser.add_argument('--frequency', type=int, default=1)
    args = parser.parse_args(args=[])

    for i in range(5):
        print('Making Scoring Dataset...')
        ecfp = np.load(args.input+'/cv_'+str(i)+'/test_fingerprint.npy')
        ecfp = np.asarray(ecfp, dtype='float32').reshape(-1,1024)
        file_interactions=np.load(args.input+'cv_'+str(i)+'/test_interaction.npy')
        print('Loading labels: test_interaction.npy')
        file_sequences=np.load(args.input+'cv_'+str(i)+'/test_reprotein.npy')
        print('Loading sequences: test_protein.npy')

        cID = np.load(args.input+'cv_'+str(i)+'/test_chemIDs.npy')
        print('Loading chemIDs: test_chemIDs.npy')
        with open(args.input+'cv_'+str(i)+'/test_proIDs.txt') as f:
            pID = [s.strip() for s in f.readlines()]
        print('Loading proIDs: test_proIDs.txt')

        n2v_c, n2v_p = [], []
        with open('./data_multi/modelpp.pickle', mode='rb') as f:
            modelpp = pickle.load(f)
        with open('./data_multi/modelcc.pickle', mode='rb') as f:
            modelcc = pickle.load(f)
        for j in cID:
            n2v_c.append(modelcc.wv[str(j)])
        for k in pID:
            n2v_p.append(modelpp.wv[k])

        interactions = np.asarray(file_interactions, dtype='int32').reshape(-1,1)
        sequences = np.asarray(file_sequences, dtype='float32').reshape(-1,1,args.prosize,plensize)
        n2vc = np.asarray(n2v_c, dtype='float32').reshape(-1,128)
        n2vp = np.asarray(n2v_p, dtype='float32').reshape(-1,128)

        # reset memory
        del file_interactions, file_sequences, n2v_c, n2v_p, cID, pID, modelcc, modelpp
        gc.collect()

        borders = [len(interactions) * l // 100 for l in range(100+1)]

        def cupy(x):
            return cp.array(x)

        print(interactions.shape, ecfp.shape, sequences.shape, n2vc.shape, n2vp.shape)
        #-------------------------------
        print('Evaluater is  running...')

        #-------------------------------

        # Set up a neural network to evaluate
        model = MV.CNN(args.prosize, plensize, args.batchsize, args.s1, args.sa1, args.s2, args.sa2, args.s3, args.sa3, args.j1, args.pf1, args.ja1, args.j2, args.pf2, args.ja2, args.j3, args.pf3, args.ja3, args.n_hid3, args.n_hid4, args.n_hid5, args.n_out)
        model.compute_accuracy = False
        model.to_gpu(args.gpu)

        f = open(args.output+'/'+args.place+str(i)+'/evaluation_'+str(i)+'.csv', 'w')

        #-------------------------------
        print("epoch","TP","FN","FP","TN","Loss","Accuracy","B_accuracy","Sepecificity","Precision","Recall","F-measure","AUC","AUPR", sep="\t")
        f.write("epoch,TP,FN,FP,TN,Loss,Accuracy,B_accuracy,Sepecificity,Precision,Recall,F-measure,AUC,AUPR\n")

        for epoch in range(args.frequency, args.epoch+1 ,args.frequency):
            pred_score,loss =[],[]

            with cp.cuda.Device(args.gpu):
                serializers.load_npz(args.output+'/'+args.place+str(i)+'/model_snapshot_' + str(epoch), model)

            for m in range(100):
                with cp.cuda.Device(args.gpu):
                    with chainer.using_config('train', False):
                        x_gpu = cupy(ecfp[borders[m]:borders[m+1]])
                        y_gpu = cupy(sequences[borders[m]:borders[m+1]])
                        n2vc_gpu = cupy(n2vc[borders[m]:borders[m+1]])
                        n2vp_gpu = cupy(n2vp[borders[m]:borders[m+1]])
                        t_gpu = cupy(interactions[borders[m]:borders[m+1]])
                        pred_tmp_gpu = model.cos_similarity(Variable(x_gpu), Variable(y_gpu), Variable(n2vc_gpu), Variable(n2vp_gpu))
                        pred_tmp_gpu = model.fc6(pred_tmp_gpu)
                        pred_tmp_gpu = F.sigmoid(pred_tmp_gpu)
                        pred_tmp = pred_tmp_gpu.data.get()
                        #pred_auc = pred_tmp_auc.data.get()
                        loss_tmp = model(Variable(x_gpu),Variable(y_gpu),Variable(n2vc_gpu),Variable(n2vp_gpu),Variable(t_gpu)).data.get()
                pred_score.extend(pred_tmp.reshape(-1).tolist())
                #pred_Auc.extend(pred_auc.reshape(-1).tolist())
                loss.append(loss_tmp.tolist())

            loss = np.mean(loss)
            pred_score = np.array(pred_score).reshape(-1,1)
            pred = 1*(pred_score >=0.5)
            shape_interactions = np.array(interactions).reshape(-1, 1)
            count_TP= np.sum(np.logical_and(shape_interactions == pred, pred == 1)*1)
            count_FP = np.sum(np.logical_and(shape_interactions != pred, pred == 1)*1)
            count_FN = np.sum(np.logical_and(shape_interactions != pred, pred == 0)*1)
            count_TN = np.sum(np.logical_and(shape_interactions == pred, pred == 0)*1)

            Accuracy = (count_TP + count_TN)/(count_TP+count_FP+count_FN+count_TN)
            Sepecificity = count_TN/(count_TN + count_FP)
            Precision = count_TP/(count_TP+count_FP)
            Recall = count_TP/(count_TP+count_FN)
            Fmeasure = 2*Recall*Precision/(Recall+Precision)
            B_accuracy = (Sepecificity+Recall)/2

            AUC = metrics.roc_auc_score(shape_interactions, pred_score, average = 'weighted')
            AUPR = metrics.average_precision_score(shape_interactions, pred_score, average = 'weighted')

            print(epoch,count_TP,count_FN,count_FP,count_TN,loss,Accuracy,B_accuracy,Sepecificity,Precision,Recall,Fmeasure,AUC,AUPR, sep="\t")
            text = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}\n'.format(
                epoch,count_TP,count_FN,count_FP,count_TN,loss,Accuracy,B_accuracy,Sepecificity,Precision,Recall,Fmeasure,AUC,AUPR)
            f.write(text)

        f.close()
        del borders, model, f, sequences, interactions, x_gpu, y_gpu, t_gpu, n2vc_gpu, n2vp_gpu, pred_tmp, loss_tmp, count_TP,count_FN,count_FP,count_TN,loss,Accuracy,B_accuracy,Sepecificity,Precision,Recall,Fmeasure,AUC,AUPR
        gc.collect()

#-------------------------------
if __name__ == '__main__':
    main()
