#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from neural_network.NeuralNetwork   import *
from neural_network.activation_fn   import *
from training.DataContainer import *
from training.DataProvider  import *

'''usage: python3 predict_testset.py @run_*.inp
   to be changed in script: file paths to checkpoint files (models) and output file name (last line)
   also keep in mind to save the mae, rmse values printed at the end of the calculation.
   
   script is used to predict the test set and calculate metrics as MAE(E), RMSE(E), MAE(F), RMSE(F), ...
   BE CAREFUL: THE values that are output are in eV!!!
'''

#define command line arguments
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument("--restart", type=str, default=None,  help="restart training from a specific folder")
parser.add_argument("--num_features",             type=int,   help="dimensionality of feature vectors")
parser.add_argument("--num_basis",                type=int,   help="number of radial basis functions")
parser.add_argument("--num_blocks",               type=int,   help="number of interaction blocks")
parser.add_argument("--num_residual_atomic",      type=int,   help="number of residual layers for atomic refinements")
parser.add_argument("--num_residual_interaction", type=int,   help="number of residual layers for the message phase")
parser.add_argument("--num_residual_output",      type=int,   help="number of residual layers for the output blocks")
parser.add_argument("--cutoff",                   type=float, help="cutoff distance for short range interactions")
parser.add_argument("--use_electrostatic",        type=int,   help="use electrostatics in energy prediction (0/1)")
parser.add_argument("--use_dispersion",           type=int,   help="use dispersion in energy prediction (0/1)")
parser.add_argument("--grimme_s6", default=None, type=float, help="grimme s6 dispersion coefficient")
parser.add_argument("--grimme_s8", default=None, type=float, help="grimme s8 dispersion coefficient")
parser.add_argument("--grimme_a1", default=None, type=float, help="grimme a1 dispersion coefficient")
parser.add_argument("--grimme_a2", default=None, type=float, help="grimme a2 dispersion coefficient")
parser.add_argument("--dataset",                  type=str,   help="file path to dataset")
parser.add_argument("--num_train",                type=int,   help="number of training samples")
parser.add_argument("--num_valid",                type=int,   help="number of validation samples")
parser.add_argument("--seed",                     type=int,   help="seed for splitting dataset into training/validation/test")
parser.add_argument("--max_steps",                type=int,   help="maximum number of training steps")
parser.add_argument("--learning_rate",            type=float, help="learning rate used by the optimizer")
parser.add_argument("--ema_decay",                type=float, help="exponential moving average decay used by the trainer")
parser.add_argument("--keep_prob",                type=float, help="keep probability for dropout regularization of rbf layer")
parser.add_argument("--l2lambda",                 type=float, help="lambda multiplier for l2 loss (regularization)")
parser.add_argument("--nhlambda",                 type=float, help="lambda multiplier for non-hierarchicality loss (regularization)")
parser.add_argument("--decay_steps",              type=int,   help="decay the learning rate every N steps by decay_rate")
parser.add_argument("--decay_rate",               type=float, help="factor with which the learning rate gets multiplied by every decay_steps steps")
parser.add_argument("--batch_size",               type=int,   help="batch size used per training step")
parser.add_argument("--valid_batch_size",         type=int,   help="batch size used for going through validation_set")
parser.add_argument('--force_weight',             type=float, help="this defines the force contribution to the loss function relative to the energy contribution (to take into account the different numerical range)")
parser.add_argument('--charge_weight',            type=float, help="this defines the charge contribution to the loss function relative to the energy contribution (to take into account the different numerical range)")
parser.add_argument('--dipole_weight',            type=float, help="this defines the dipole contribution to the loss function relative to the energy contribution (to take into account the different numerical range)")
parser.add_argument('--summary_interval',         type=int,   help="write a summary every N steps")
parser.add_argument('--validation_interval',      type=int,   help="check performance on validation set every N steps")
parser.add_argument('--save_interval',            type=int,   help="save progress every N steps")
parser.add_argument('--record_run_metadata',      type=int,   help="records metadata like memory consumption etc.")

#if no command line arguments are present, config file is parsed
config_file='config.txt'
if len(sys.argv) == 1:
    if os.path.isfile(config_file):
        args = parser.parse_args(["@"+config_file])
    else:
        args = parser.parse_args(["--help"])
else:
    args = parser.parse_args()

#file paths to checkpoint files
checkpoints = "../models_clPhOH/clphoh.meta.mp2.631g.3000_a",


#load dataset
data = DataContainer(args.dataset)

#generate DataProvider (splits dataset into training, validation and test set based on seed)
data_provider = DataProvider(data, args.num_train, args.num_valid, args.batch_size, args.valid_batch_size, seed=args.seed)

#create neural network
nn = NeuralNetwork(F=args.num_features,           
                   K=args.num_basis,                
                   sr_cut=args.cutoff,              
                   num_blocks=args.num_blocks, 
                   num_residual_atomic=args.num_residual_atomic,
                   num_residual_interaction=args.num_residual_interaction,
                   num_residual_output=args.num_residual_output,
                   use_electrostatic=(args.use_electrostatic==1),
                   use_dispersion=(args.use_dispersion==1),
                   s6=args.grimme_s6,
                   s8=args.grimme_s8,
                   a1=args.grimme_a1,
                   a2=args.grimme_a2,
                   activation_fn=shifted_softplus, 
                   seed=None,
                   scope="neural_network")


#create placeholders for feeding data
Eref      = tf.placeholder(tf.float32, shape=[None, ], name="Eref")
Fref      = tf.placeholder(tf.float32, shape=[None,3], name="Fref") 
Z         = tf.placeholder(tf.int32,   shape=[None, ], name="Z")     
Dref      = tf.placeholder(tf.float32, shape=[None,3], name="Dref") 
Qref      = tf.placeholder(tf.float32, shape=[None, ], name="Qref")   
R         = tf.placeholder(tf.float32, shape=[None,3], name="R")      
idx_i     = tf.placeholder(tf.int32,   shape=[None, ], name="idx_i") 
idx_j     = tf.placeholder(tf.int32,   shape=[None, ], name="idx_j") 
batch_seg = tf.placeholder(tf.int32,   shape=[None, ], name="batch_seg") 

#model energy/forces
Ea, Qa, Dij, nhloss = nn.atomic_properties(Z, R, idx_i, idx_j)
energy, forces = nn.energy_and_forces_from_atomic_properties(Ea, Qa, Dij, Z, R, idx_i, idx_j, Qref, batch_seg)
#total charge
Qtot = tf.segment_sum(Qa, batch_seg)
#dipole moment vector
QR = tf.stack([Qa*R[:,0], Qa*R[:,1], Qa*R[:,2]],1)
D  = tf.segment_sum(QR, batch_seg)

#function to calculate l2 loss, mean squared error, mean absolute error between two values
def calculate_errors(val1, val2):
    with tf.name_scope("calculate_errors"):
        delta = val1-val2
        mse   = tf.reduce_mean(delta**2)
        mae   = tf.reduce_mean(tf.abs(delta))
        loss  = mse #mean squared error loss
        #loss = mae #mean absolute error loss
    return loss, mse, mae

#function to calculate the mae and mse with numpy (no tensorstuff)
def calculate_mae_mse(val1, val2):
    delta = val1-val2
    mse = np.mean(delta**2)
    mae = np.mean(np.abs(delta))
    return mae, mse

#calculate errors
_, emse, emae = calculate_errors(Eref, energy)
_, fmse, fmae = calculate_errors(Fref, forces)
_, dmse, dmae = calculate_errors(Dref, D)
_, qmse, qmae = calculate_errors(Qref, Qtot)

#helper function to fill a feed dictionary
def fill_feed_dict(data):
    feed_dict = { 
        Eref:      data["E"], 
        Fref:      data["F"],
        Z:         data["Z"], 
        Dref:      data["D"],
        Qref:      data["Q"],
        R:         data["R"],
        idx_i:     data["idx_i"],
        idx_j:     data["idx_j"],
        batch_seg: data["batch_seg"] 
    }
    return feed_dict

#helper function to print errors
def print_errors(sess, get_data, data_count, data_name, checkpoints):
    #print("\n" + data_name + " ("+ str(data_count) + "):")

    emse_avg = {}
    emae_avg = {}
    fmse_avg = {}
    fmae_avg = {}
    qmse_avg = {}
    qmae_avg = {}
    dmse_avg = {}
    dmae_avg = {}
    for checkpoint in checkpoints:
        emse_avg[checkpoint] = 0.0
        emae_avg[checkpoint] = 0.0
        fmse_avg[checkpoint] = 0.0
        fmae_avg[checkpoint] = 0.0
        qmse_avg[checkpoint] = 0.0
        qmae_avg[checkpoint] = 0.0
        dmse_avg[checkpoint] = 0.0
        dmae_avg[checkpoint] = 0.0    
    emse_avg["ensemble"] = 0.0
    emae_avg["ensemble"] = 0.0
    fmse_avg["ensemble"] = 0.0
    fmae_avg["ensemble"] = 0.0
    qmse_avg["ensemble"] = 0.0
    qmae_avg["ensemble"] = 0.0
    dmse_avg["ensemble"] = 0.0
    dmae_avg["ensemble"] = 0.0     

    #in case we have only one checkpoint, the nn does not need to be restored
    if len(checkpoints) == 1:
        nn.restore(sess, checkpoints[0])

    with open(data_name+".dat", "w") as f:

        for i in range(data_count):
            print(i)
            data = get_data(i)
            feed_dict = fill_feed_dict(data)
            
            #calculate average prediction of ensemble
            Eavg = 0
            Favg = 0
            Davg = 0
            Qavg = 0
            num = 1
            for checkpoint in checkpoints:
                if len(checkpoints) > 1:
                    nn.restore(sess, checkpoint)
                Etmp, Ftmp, Dtmp, Qtmp = sess.run([energy, forces, D, Qtot], feed_dict=feed_dict)
                #compute errors for this checkpoint
                emae_tmp, emse_tmp = calculate_mae_mse(Etmp, data["E"])
                fmae_tmp, fmse_tmp = calculate_mae_mse(Ftmp, data["F"])
                dmae_tmp, dmse_tmp = calculate_mae_mse(Dtmp, data["D"])
                qmae_tmp, qmse_tmp = calculate_mae_mse(Qtmp, data["Q"])
                #add to average errors for this checkpoint
                emae_avg[checkpoint] += (emae_tmp-emae_avg[checkpoint])/(i+1)
                emse_avg[checkpoint] += (emse_tmp-emse_avg[checkpoint])/(i+1)
                fmae_avg[checkpoint] += (fmae_tmp-fmae_avg[checkpoint])/(i+1)
                fmse_avg[checkpoint] += (fmse_tmp-fmse_avg[checkpoint])/(i+1)
                qmae_avg[checkpoint] += (qmae_tmp-qmae_avg[checkpoint])/(i+1)
                qmse_avg[checkpoint] += (qmse_tmp-qmse_avg[checkpoint])/(i+1)
                dmae_avg[checkpoint] += (dmae_tmp-dmae_avg[checkpoint])/(i+1)
                dmse_avg[checkpoint] += (dmse_tmp-dmse_avg[checkpoint])/(i+1)
                #update ensemble predictions
                Eavg += (Etmp-Eavg)/num
                Favg += (Ftmp-Favg)/num  
                Davg += (Dtmp-Davg)/num 
                Qavg += (Qtmp-Qavg)/num    
                num += 1   

            #calculate errors
            emae_tmp, emse_tmp = calculate_mae_mse(Eavg, data["E"])
            fmae_tmp, fmse_tmp = calculate_mae_mse(Favg, data["F"])
            dmae_tmp, dmse_tmp = calculate_mae_mse(Davg, data["D"])
            qmae_tmp, qmse_tmp = calculate_mae_mse(Qavg, data["Q"])

            #add to average errors of ensemble
            emae_avg["ensemble"] += (emae_tmp-emae_avg["ensemble"])/(i+1)
            emse_avg["ensemble"] += (emse_tmp-emse_avg["ensemble"])/(i+1)
            fmae_avg["ensemble"] += (fmae_tmp-fmae_avg["ensemble"])/(i+1)
            fmse_avg["ensemble"] += (fmse_tmp-fmse_avg["ensemble"])/(i+1)
            qmae_avg["ensemble"] += (qmae_tmp-qmae_avg["ensemble"])/(i+1)
            qmse_avg["ensemble"] += (qmse_tmp-qmse_avg["ensemble"])/(i+1)
            dmae_avg["ensemble"] += (dmae_tmp-dmae_avg["ensemble"])/(i+1)
            dmse_avg["ensemble"] += (dmse_tmp-dmse_avg["ensemble"])/(i+1)

            f.write(str(data["E"][0])+"  "+str(Eavg) + "\n")

    #print results
    print("RESULTS:\n")
    ############################################## UNCOMMENT FOR THE RESULTS OF AN ENSEMBLE
    #checkpoints.append("ensemble")
    for checkpoint in checkpoints:
        print("kcal/mol , " + str(checkpoint))
        if not np.isnan(emae_avg[checkpoint]):
            print("EMAE:,",  emae_avg[checkpoint]*23.06035)
        if not np.isnan(emse_avg[checkpoint]):
            print("ERMSE:,", np.sqrt(emse_avg[checkpoint])*23.06035)
        if not np.isnan(fmae_avg[checkpoint]):
            print("FMAE:,",  fmae_avg[checkpoint]*23.06035)
        if not np.isnan(fmse_avg[checkpoint]):
            print("FRMSE:,", np.sqrt(fmse_avg[checkpoint])*23.06035)
        #if not np.isnan(qmae_avg[checkpoint]):
            #print("QMAE:,",  qmae_avg[checkpoint])
        #if not np.isnan(qmse_avg[checkpoint]):
            #print("QRMSE:,", np.sqrt(qmse_avg[checkpoint]))
        #if not np.isnan(dmae_avg[checkpoint]):
            #print("DMAE:,",  dmae_avg[checkpoint])
        #if not np.isnan(dmse_avg[checkpoint]):
            #print("DRMSE:,", np.sqrt(dmse_avg[checkpoint]))
        print()


#create tensorflow session
with tf.Session() as sess:

    #calculate errors on test data
    print_errors(sess, data_provider.get_test_data,  data_provider.ntest, "clphoh.meta.mp2.631g.3000", checkpoints)    
