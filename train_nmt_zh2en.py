import numpy
import os

import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     maxlen=50,
                     batch_size=32,
                     valid_batch_size=32,
                     validFreq=100,
                     dispFreq=100,
                     saveFreq=1000,
                     sampleFreq=1000,
                     datasets=['/home/chenhd/data/zh2en/tree/corpus.ch',
                               '/home/chenhd/data/zh2en/tree/corpus.en'],
                     valid_datasets=['/home/chenhd/data/zh2en/devntest/MT02/MT02.src',
                                     '/home/chenhd/data/zh2en/devntest/MT02/reference0'],
                     dictionaries=['/home/chenhd/data/zh2en/tree/corpus.ch.pkl',
                                   '/home/chenhd/data/zh2en/tree/corpus.en.pkl'],
                     treeset=['/home/chenhd/data/zh2en/tree/corpus.ch.tree',
                              '/home/chenhd/data/zh2en/devntest/MT02/MT02.ce.tree'],
                     use_dropout=params['use-dropout'][0],
                     # shuffle_each_epoch=True,
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_hal.npz'],
        'dim_word': [512],
        'dim': [1024],
        'n-words': [30000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [True]})
