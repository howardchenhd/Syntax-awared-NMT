#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu0,floatX=float32

#cd $PBS_O_WORKDIR
#
python ./translate_gpu.py -n -k 5 \
        ./model_hal.npz  \
	$HOME/data/zh2en/tree/corpus.ch.pkl \
	$HOME/data/zh2en/tree/corpus.en.pkl \
	$HOME/data/zh2en/devntest/MT02/MT02.src\
        $HOME/data/zh2en/devntest/MT02/MT02.ce.tree\
	--saveto ./MT02.trans.en

python ./translate_gpu.py -n -k 5 \
        ./model_hal.npz  \
        $HOME/data/zh2en/tree/corpus.ch.pkl \
        $HOME/data/zh2en/tree/corpus.en.pkl \
        $HOME/data/zh2en/devntest/MT03/MT03.src\
        $HOME/data/zh2en/devntest/MT03/MT03.ce.tree\
        --saveto ./MT03.trans.en

python ./translate_gpu.py -n -k 5 \
        ./model_hal.npz  \
        $HOME/data/zh2en/tree/corpus.ch.pkl \
        $HOME/data/zh2en/tree/corpus.en.pkl \
        $HOME/data/zh2en/devntest/MT04/MT04.src\
        $HOME/data/zh2en/devntest/MT04/MT04.ce.tree\
        --saveto ./MT04.trans.en

python ./translate_gpu.py -n -k 5 \
        ./model_hal.npz  \
        $HOME/data/zh2en/tree/corpus.ch.pkl \
        $HOME/data/zh2en/tree/corpus.en.pkl \
        $HOME/data/zh2en/devntest/MT05/MT05.src\
        $HOME/data/zh2en/devntest/MT05/MT05.ce.tree\
        --saveto ./MT05.trans.en

python ./translate_gpu.py -n -k 5 \
        ./model_hal.npz  \
        $HOME/data/zh2en/tree/corpus.ch.pkl \
        $HOME/data/zh2en/tree/corpus.en.pkl \
        $HOME/data/zh2en/devntest/MT06/MT06.src\
        $HOME/data/zh2en/devntest/MT06/MT06.ce.tree\
        --saveto ./MT06.trans.en
