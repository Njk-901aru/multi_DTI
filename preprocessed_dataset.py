#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import argparse as arg
import pandas as pd
import pubchempy as pcp
import numpy as np
from rdkit import Chem
from pyensembl import EnsemblRelease

parser = arg.ArgumentParser(description='ecfpWD_n2v')
parser.add_argument('--input', '-i', default='./dataset_hard')
parser.add_argument('--data', '-d', default='/train')
args = parser.parse_args(args=[])

for i in range(5):
    #load protein-compound interaction dataset
    data = pd.read_csv(args.input+'/cv_'+str(i)+args.data+'.csv')

    #save labels
    label = np.array(data['label'], dtype='int32')
    np.save(args.input+'/cv_'+str(i)+args.data+'_interaction.npy', label)

    #save Ensembl protein ID (ENSP) for applying to node2vec
    with open(args.input+'/cv_'+str(i)+args.data+'_proIDs.txt', mode='w') as f:
        f.write('\n'.join(data['protein']))

    #save pubchem ID for applying to node2vec
    cid = np.array(data['chemical'], dtype='int32')
    np.save(args.input+'/cv_'+str(i)+args.data+'_chemIDs.npy', cid)

    #convert pubchem ID to CanonicalSMILES
    c_id = data.chemical.tolist()
    pcp.download('CSV', args.input+'/cv_'+str(i)+'/ismilesref.csv', c_id, operation='property/IsomericSMILES', overwrite=True)
    smileb =pd.read_csv(args.input+'/cv_'+str(i)+'/ismilesref.csv')
    smib = []
    for j in smileb['IsomericSMILES']:
        smib.append(Chem.MolToSmiles(Chem.MolFromSmiles(j), kekuleSmiles=False, isomericSmiles=True))
    with open(args.input+'/cv_'+str(i)+args.data+'.smiles', mode='w') as f:
        f.write('\n'.join(smib))
    #convert CanonicalSMILES to ecfp
    file_smiles = args.input+'/cv_'+str(i)+args.data+'.smiles'
    smi = Chem.SmilesMolSupplier(file_smiles,delimiter='\t',titleLine=False)
    mols = [mol for mol in smi if mol is not None]
    fp = []

    for mol in mols:
        morganfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
        fp.append(morganfp)
    fp = np.array(fp, dtype='float32')
    np.save(args.input+'/cv_'+str(i)+args.data+'_fingerprint.npy', fp)

    #get amino acid seq from Ensembl protein ID (ENSP) and convert them to onehot vectors
    with open(args.input+'/cv_'+str(i)+args.data+'_proIDs.txt') as f:
        pID = [s.strip() for s in f.readlines()]
    plen = len(pID)
    ens = EnsemblRelease(93) #release 93 uses human reference genome GRCh38
    toseq = []
    for j in pID:
        seq = ens.protein_sequence(j) #get amino acid seq from ENSP using pyensembl
        toseq.append(seq)

    amino_acid = 'ACDEFGHIKLMNPQRSTVWY' #defeine universe of possible input values
    char_to_int = dict((c, n) for n, c in enumerate(amino_acid)) #defeiine a mapping of chars to integers
    int_to_char = dict((n, c) for n, c in enumerate(amino_acid))
    integer_encoded = []
    for l in range(len(toseq)):
        integer_encoded.append([char_to_int[char] for char in toseq[l]]) #integer encode input data

    Max = 5762
    onehot_tr = np.empty((plen, Max, 20), dtype='float32')
    for j in range(len(integer_encoded)):
        b_onehot = np.identity(20, dtype='float32')[integer_encoded[j]]
        differ_tr = Max - len(integer_encoded[j])
        b_zeros = np.zeros((differ_tr, 20), dtype='float32')
        onehot_tr[j] = np.vstack((b_onehot, b_zeros))
    np.save(args.input+'/cv_'+str(i)+args.data+'_reprotein.npy', ontr)


# In[ ]:
