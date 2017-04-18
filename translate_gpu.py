'''
Translates a source file using a translation model.
'''
import argparse
import theano
import numpy
import cPickle as pkl
import sys

from nmt import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)

# from multiprocessing import Process, Queue


def translate_model(queue, rqueue, mask_left, mask_right, write_mask, model, options, k, normalize):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    #params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model) #, params)
    tparams = init_tparams(params)

    # word index
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    def _translate(seq, left, right, write):
        # sample given an input sequence and obtain scores
        print left.shape, right.shape, write.shape, len(seq)
        sample, score = gen_sample(tparams, f_init, f_next,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   left[:, :, None], right[:, :, None], write, 
                                   options, trng=trng, k=k, maxlen=200,
                                   stochastic=False, argmax=False)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        return sample[sidx]

    for idx, [x, l, r, w] in enumerate(zip(queue, mask_left, mask_right, write_mask)):
        # req = queue.get()
        if x is None:
            break

        print idx
        seq = _translate(x, l, r, w)

        rqueue.append(seq)

    return


def main(model, dictionary, dictionary_target, source_file, tree_file, saveto, k=5,
         normalize=False, n_process=5, chr_level=False):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'
    

    rqueue = []
    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw
    
    def _seqs2words_sent(cc):
        ww = []
        for w in cc:
            if w == 0:
                break
            ww.append(word_idict_trg[w])
        return ' '.join(ww)     

    def _send_jobs(fname, tname):
        queue = []
        mask_left = []
        mask_right = []
        write_mask = []
        len_x = []
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                if chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)

                x += [0]
                len_x.append(len(x))
                # queue.put((idx, x))
                queue.append(x)
        print 'source file loaded!'
        with open(tname, 'r') as f:
            for idx, line in enumerate(f):
                tree_actions = line.strip().split()
                mask_l = numpy.zeros((len(tree_actions), len_x[idx] + len(tree_actions))).astype('float32')
                mask_r = numpy.zeros((len(tree_actions), len_x[idx] + len(tree_actions))).astype('float32')
                wrt_mask = numpy.zeros((len(tree_actions), len_x[idx] + len(tree_actions))).astype('float32')
                
                # print mask_l.shape

                idx_act = 0
                for tree_act in tree_actions:
                    wrt_mask[idx_act][(eval(tree_act)[2] - 201) + len_x[idx]] = 1.
                    if eval(tree_act)[0] > 200:
                        mask_l[idx_act][(eval(tree_act)[0] - 201) + len_x[idx]] = 1.
                    else:
                        mask_l[idx_act][eval(tree_act)[0] - 1] = 1.
                    if eval(tree_act)[1] > 200:
                        mask_r[idx_act][(eval(tree_act)[1] - 201) + len_x[idx]] = 1.
                    else:
                        mask_r[idx_act][eval(tree_act)[1] - 1] = 1.
                    idx_act += 1
                    # print idx_act

                # mask_left.put((idx, mask_l))
                # mask_right.put((idx, mask_r))
                # write_mask.put((idx, wrt_mask))
                mask_left.append(mask_l)
                mask_right.append(mask_r)
                write_mask.append(wrt_mask)
                    
        print 'tree file loaded!'
 
        return idx+1, queue, mask_left, mask_right, write_mask


    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        out_idx = 0
        for idx in xrange(n_samples):
            trans[idx] = rqueue[idx]
            if numpy.mod(idx, 10) == 0:
                print 'Sample ', (idx+1), '/', n_samples, ' Done'
            while out_idx < n_samples and trans[out_idx] != None:
                yield trans[out_idx]
                out_idx += 1


    print 'Translating ', source_file, '...'
    n_samples, queue, mask_l, mask_r, wtr_mask = _send_jobs(source_file, tree_file)                # return the number of sentences
    print 'sentence: ', n_samples   
    translate_model(queue, rqueue, mask_l, mask_r, wtr_mask, model, options, k, normalize)
    print 'translated!'
    for i, trans in enumerate(_retrieve_jobs(n_samples)):
        samples = trans
        saveto.write(_seqs2words_sent(samples) + "\n")
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-c', action="store_true", default=False)
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('source_tree', type=str)
    parser.add_argument('--saveto', '-s', type=argparse.FileType('w'),
                        default=sys.stdout, metavar='PATH',
                        help="Output file (default: standard output)")    

    args = parser.parse_args()

    main(args.model, args.dictionary, args.dictionary_target, args.source, args.source_tree, 
         args.saveto, k=args.k, normalize=args.n,
         chr_level=args.c)
