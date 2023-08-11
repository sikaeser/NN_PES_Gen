#!/bin/bash
#
#$ -cwd
#$ -S /bin/bash
#$ -m e
#$ -pe smp 1
#$ -l gpu=1
#$ -M silvan.kaeser@stud.unibas.ch ##########ADAPT
#$ -N nn-run_silvan  ############ADAPT

#input file
input=run_systemnameXY.inp ##############ADAPT

#restart folder (in case a run is restarted)
restart=

#datasets folder
datasets=datasets
#neural network code
neural_network=neural_network
#training handling code
training=training
#atomlabels file
atom_labels=atom_labels.tsv
#pythonscript for training
train=train.py
#environment
envi=env-gpu #GPU ############ADAPT eventually
#envi=env     #CPU
 
startfolder=`pwd`
scratch=/scratch/$USER.$JOB_ID

#create scratch folder
if [ -d "$scratch" ]; then
   echo "scratch directory exists already"
else
   echo "creating scratch directory"
   mkdir $scratch
fi

#copy existing restart folder if present
if [ -d "$restart" ]; then
   cp -r $restart $scratch
fi

#link/copy data to scratch folder and go there
cp -r $train $scratch
cp -r $input $scratch
cp -r $atom_labels $scratch
cp -r $neural_network $scratch
cp -r $training $scratch
ln -s /home/kaeser/machinelearning/NeuralNetwork/$datasets $scratch/$datasets  #######ADAPT
cd $scratch

#make necessary folders and load environment
source $HOME/$envi/bin/activate

#run actual jobs
export CUDA_VISIBLE_DEVICES=0
./$train @$input 
cp -r $scratch $startfolder

#remove scratch folder
#cd $startfolder
#rm -r $scratch

