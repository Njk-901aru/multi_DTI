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
    parser.add_argument('--frequency', type=int, default=1)
    args = parser.parse_args(args=[])

    print(args.gpu)
    print('GPU: ', args.gpu)
    print('# Minibatch-size: ', args.batchsize)
    print('')

    #-------------------------------
    # GPU check
    xp = np
    if args.gpu >= 0:
        print('GPU mode')
        #xp = cp

    #-------------------------------
    # Loading SMILEs
    for i in range(5):
        #i = i+4
        print('Making Training dataset...')
        ecfp = xp.load(args.input+'/cv_'+str(i)+'/train_fingerprint.npy')
        ecfp = xp.asarray(ecfp, dtype='float32').reshape(-1,1024)

        file_interactions=xp.load(args.input+'/cv_'+str(i)+'/train_interaction.npy')
        print('Loading labels: train_interaction.npy')
        cID = xp.load(args.input+'/cv_'+str(i)+'/train_chemIDs.npy')
        print('Loading chemIDs: train_chemIDs.npy')
        with open(args.input+'/cv_'+str(i)+'/train_proIDs.txt') as f:
            pID = [s.strip() for s in f.readlines()]
        print('Loading proIDs: train_proIDs.txt')
        n2v_c, n2v_p = [], []
        with open('./data_multi/modelpp.pickle', mode='rb') as f:
            modelpp = pickle.load(f)
        with open('./data_multi/modelcc.pickle', mode='rb') as f:
            modelcc = pickle.load(f)
        for j in cID:
            n2v_c.append(modelcc.wv[str(j)])
        for k in pID:
            n2v_p.append(modelpp.wv[k])
        interactions = xp.asarray(file_interactions, dtype='int32').reshape(-1,args.n_out)
        n2vc = np.asarray(n2v_c, dtype='float32').reshape(-1,128)
        n2vp = np.asarray(n2v_p, dtype='float32').reshape(-1,128)
        #reset memory
        del n2v_c, n2v_p, cID, pID, modelcc, modelpp, file_interactions
        gc.collect()

        file_sequences=xp.load(args.input+'/cv_'+str(i)+'/train_reprotein.npy')
        print('Loading sequences: train_reprotein.npy', flush=True)
        sequences = xp.asarray(file_sequences, dtype='float32').reshape(-1,1,args.prosize,plensize)
        # reset memory
        del file_sequences
        gc.collect()

        print(interactions.shape, ecfp.shape, sequences.shape, n2vc.shape, n2vp.shape, flush=True)

        print('Now concatenating...', flush=True)
        train_dataset = datasets.TupleDataset(ecfp, sequences, n2vc, n2vp, interactions)
        n = int(0.8 * len(train_dataset))
        train_dataset, valid_dataset = train_dataset[:n], train_dataset[n:]
        print('train: ', len(train_dataset), flush=True)
        print('valid: ', len(valid_dataset), flush=True)

        print('pattern: ', i, flush=True)
        output_dir = args.output+'/'+'ecfpN2vc_mSGD'+'/'+'pattern'+str(i)
        os.makedirs(output_dir)

        #-------------------------------
        #reset memory again
        del n, sequences, interactions, ecfp, n2vc, n2vp
        gc.collect()

        #-------------------------------
        # Set up a neural network to train
        print('Set up a neural network to train', flush=True)
        model = MV.CNN(args.prosize, plensize, args.batchsize, args.s1, args.sa1, args.s2, args.sa2, args.s3, args.sa3, args.j1, args.pf1, args.ja1, args.j2, args.pf2, args.ja2, args.j3, args.pf3, args.ja3, args.n_hid3, args.n_hid4, args.n_hid5, args.n_out)
        #-------------------------------
        # Make a specified GPU current
        if args.gpu >= 0:
            chainer.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu()  # Copy the model to the GPU
        #-------------------------------
        # Setup an optimizer
        optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
        optimizer.setup(model)
        #-------------------------------
        # L2 regularization(weight decay)
        for param in model.params():
            if param.name != 'b':  # バイアス以外だったら
                param.update_rule.add_hook(WeightDecay(0.00001))  # 重み減衰を適用
        #-------------------------------
        # Set up a trainer
        print('Trainer is setting up...', flush=True)

        train_iter = chainer.iterators.SerialIterator(train_dataset, batch_size= args.batchsize, shuffle=True)
        test_iter = chainer.iterators.SerialIterator(valid_dataset, batch_size= args.batchsize, repeat=False, shuffle=True)
        updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=output_dir)
        # Evaluate the model with the test dataset for each epoch
        trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
        # Take a snapshot for each specified epoch
        trainer.extend(extensions.snapshot_object(model, 'model_snapshot_{.updater.epoch}'), trigger=(args.frequency,'epoch'))
        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport(trigger=(1, 'epoch'), log_name='log_epoch'))
        trainer.extend(extensions.LogReport(trigger=(10, 'iteration'), log_name='log_iteration'))
        # Print selected entries of the log to stdout
        trainer.extend(extensions.PrintReport( ['epoch', 'elapsed_time','main/loss', 'validation/main/loss','main/accuracy','validation/main/accuracy']))
        # Print some results
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
        # Print a progress bar to stdout
        trainer.extend(extensions.ProgressBar())

        # Run the training
        trainer.run()

        END = time.time()
        print('Nice, your Learning Job is done.　Total time is {} sec．'.format(END-START))

        del model, train_iter, test_iter, updater, trainer
        gc.collect()

#-------------------------------
# Model Fegure

#-------------------------------
if __name__ == '__main__':
    main()
