'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

from data_iterator import TextIterator

from treedata_iterator import TreeTextIterator

profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout trng = RandomStream(123)
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params=None):
    pp = numpy.load(path)
    if not params:
        params = OrderedDict()
        for kk, vv in pp.iteritems():
            if kk not in ['zipped_params', 'history_errs', 'uidx']:
                params[kk] = pp[kk]
    else:
        for kk, vv in params.iteritems():
            if kk not in pp:
                warnings.warn('%s is not in the archive' % kk)
                continue
            params[kk] = pp[kk]

    return params


def load_ftparams(path, params):
    pp = numpy.load(path)
    for kk, vv in pp.iteritems():
        if kk not in params:
            warnings.warn('%s is not in the archive' % kk)
            continue
        if kk.startswith('ff_') or kk.startswith('decoder'):
            continue
        params[kk] = pp[kk]
        print kk

    return params

# layers: 'name': ('parameter initializer', 'feed-forward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'gru_cover': ('param_init_gru_cover', 'gru_cover_layer'),
          'tree_gru': ('param_init_tgru', 'tgru_layer'),          # bottom-up tree encoder
          'td_tgru': ('param_init_td_tgru', 'td_tgru_layer'),     # top-down tree encoder
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)      # svd, return u=>[ndim * ndim]
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask

def prepare_treedata(seqs_x, seqs_y, seqs_tree, maxlen=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]
    lengths_tree = [len(s) for s in seqs_tree]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_tree = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_tree = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y, s_tree, l_tree in zip(lengths_x, seqs_x, lengths_y, seqs_y, seqs_tree, lengths_tree):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_tree.append(s_tree)
                new_lengths_tree.append(l_tree)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_tree = new_lengths_tree
        seqs_tree = new_seqs_tree
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    maxlen_tree = numpy.max(lengths_tree)

    mask_left = numpy.zeros((maxlen_tree, maxlen_x + maxlen_tree, n_samples)).astype('float32')
    mask_right = numpy.zeros((maxlen_tree, maxlen_x + maxlen_tree, n_samples)).astype('float32')

    tree_mask = numpy.zeros((maxlen_tree, n_samples)).astype('float32')

    write_mask = numpy.zeros((maxlen_tree, maxlen_x + maxlen_tree)).astype('float32')
    # end of tree
    eot = numpy.zeros((maxlen_x + maxlen_tree, n_samples)).astype('float32')

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    for idx, [tree] in enumerate(zip(seqs_tree)):  # n_samples
        if tree != '[1,1,201]':
            tree_mask[:lengths_tree[idx], idx] = 1.
            eot[(maxlen_x + lengths_tree[idx] - 1), idx] = 1.
        #tree = tree.split()
        for idx_action, [tree_action] in enumerate(zip(tree)):  # tree_timesteps
            write_mask[idx_action][(eval(tree_action)[2] - 201) + maxlen_x] = 1.
            if eval(tree_action)[0] <= 200:
                mask_left[idx_action][eval(tree_action)[0] - 1][idx] = 1.
            else:
                mask_left[idx_action][(eval(tree_action)[0] - 201) + maxlen_x][idx] = 1.
            if eval(tree_action)[1] <= 200:
                mask_right[idx_action][eval(tree_action)[1] - 1][idx] = 1.
            else:
                mask_right[idx_action][(eval(tree_action)[1] - 201) + maxlen_x][idx] = 1.

    return x, x_mask, y, y_mask, mask_left, mask_right, tree_mask, write_mask, eot

# feed-forward layer: affine transformation + point-wise non-linearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):  # 512 1024
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)        # nin * 2dim
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)            # dim * 2dim
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


#state_below [[n_timesteps, n_samples, options['dim_word']]
def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:             # mini-batch
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated     W: W_r for reset gate W_u for update gate
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences (m_, x_, xx_)|outputs-info (h_)| non-seqs (U, Ux)
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)         # h_ : hidden state  U: U_r for reset gate U_u for update gate
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))     # first dim for reset gate
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))     # last dim for update gate

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_   # if mask = 1 h = h herwise h = h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]  # [ matrix[n_samples * dim] ]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


#Tree-LSTM
def param_init_tgru(options, params, prefix='tgru', nin=None, dim=None, dimctx=None):
    if dimctx is None:
        dimctx = options['dim_ctx']

    # reset gate left and reset gate right
    U_l = numpy.concatenate([ortho_weight(dimctx),
                             ortho_weight(dimctx)], axis=1)
    params[_p(prefix, 'U_l')] = U_l
    params[_p(prefix, 'b')] = numpy.zeros((2 * dimctx,)).astype('float32')
    U_r = numpy.concatenate([ortho_weight(dimctx),
                           ortho_weight(dimctx)], axis=1)
    params[_p(prefix, 'U_r')] = U_r

    # left and right forget gates
    U_lf = numpy.concatenate([ortho_weight(dimctx),
                              ortho_weight(dimctx),
                              ortho_weight(dimctx)], axis=1)
    # U_lf = ortho_weight(dimctx)
    params[_p(prefix, 'U_lf')] = U_lf
    params[_p(prefix, 'b_f')] = numpy.zeros((3 * dimctx,)).astype('float32')
    U_rf = numpy.concatenate([ortho_weight(dimctx),
                              ortho_weight(dimctx),
                              ortho_weight(dimctx)], axis=1)
    # U_rf = ortho_weight(dimctx)
    params[_p(prefix, 'U_rf')] = U_rf

    # for memory cell
    U_lc = ortho_weight(dimctx)
    params[_p(prefix, 'U_lc')] = U_lc
    U_rc = ortho_weight(dimctx)
    params[_p(prefix, 'U_rc')] = U_rc
    params[_p(prefix, 'b_c')] = numpy.zeros((dimctx,)).astype('float32')
   
    # gate for head lexicalization
    #W_zl = ortho_weight(nin)
    #params[_p(prefix, 'W_zl')] = W_zl
    #W_zr = ortho_weight(nin)
    #params[_p(prefix, 'W_zr')] = W_zr
    #params[_p(prefix, 'b_z')] = numpy.zeros((nin,)).astype('float32')

    #U_x = numpy.concatenate([norm_weight(nin, dimctx),
    #                       norm_weight(nin, dimctx)], axis=1)
    #params[_p(prefix, 'U_x')] = U_x

    #U_xu = numpy.concatenate([norm_weight(nin, dimctx),
    #                          norm_weight(nin, dimctx),
    #                          norm_weight(nin, dimctx)], axis=1)
    #params[_p(prefix, 'U_xu')] = U_xu

    #U_xh = norm_weight(nin, dimctx)
    #params[_p(prefix, 'U_xh')] = U_xh   
 
    # U_pl = ortho_weight(dimctx)
    # params[_p(prefix, 'U_pl')] = U_pl
    # U_pr = ortho_weight(dimctx)
    # params[_p(prefix, 'U_pr')] = U_pr
    return params


# mask_left/mask_right : [tree_timesteps * (tree_timesteps + x_timesteps) * nsamples]
# ctx_/memory : [(tree_timesteps + x_timesteps) * nsamples * dimctx]
def tgru_layer(tparams, mask_left, options, prefix='tgru', mask=None, mask_right=None, write_mask=None, context=None,
               # stateblew=None,
               **kwargs):
    nsteps = mask_left.shape[0]
    if mask_left.ndim == 3:             # mini-batch
        n_samples = mask_left.shape[2]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'U_l')].shape[0]
    nin = options['dim_word']

    if mask is None:
        mask = tensor.alloc(1., mask_left.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]  # fix row, col n*dim : (n+1)*dim

    # step function to be used by scan
    # arguments    | sequences (m_, x_, xx_)|outputs-info (h_)| non-seqs (U, Ux)
    def _step_slice(m_, left_, right_, write, ctx_,   # statex, 
                    root, U_l, U_r, b, U_lf, U_rf, b_f, U_lc, U_rc, b_c, 
                    # W_zl, W_zr, b_z, 
                    # U_x, U_xu, U_xh
                   ):
        #leftx_ = leftx.sum(0)                    # n_samples * dimctx
        
        leftx_ = (left_[:, :, None] * ctx_).sum(0)
        rightx_ = (right_[:, :, None] * ctx_).sum(0)

        # left_in = (left_[:, :, None] * statex).sum(0)
        # right_in = (right_[:, :, None] * statex).sum(0)

        # in_gate = tensor.nnet.sigmoid(tensor.dot(left_in, W_zl) + tensor.dot(right_in, W_zr) + b_z) 
        # state_t = in_gate * left_in + (1. - in_gate) * right_in


        # reset gates
        r_gates = tensor.dot(leftx_, U_l) + tensor.dot(rightx_, U_r) + b  # + tensor.dot(state_t, U_x)

        r_l = tensor.nnet.sigmoid(_slice(r_gates, 0, dim))
        r_r = tensor.nnet.sigmoid(_slice(r_gates, 1, dim))

        # update gate
        u_gates = tensor.dot(leftx_, U_lf) + tensor.dot(rightx_, U_rf) + b_f    # + tensor.dot(state_t, U_xu)
                
        # u_gates = tensor.nnet.sigmoid(u_gates)
        u_l = tensor.nnet.sigmoid(_slice(u_gates, 0, dim))
        u_r = tensor.nnet.sigmoid(_slice(u_gates, 1, dim))
        u_last = tensor.nnet.sigmoid(_slice(u_gates, 2, dim))        

        # memory cell update
        h = tensor.tanh(r_l * tensor.dot(leftx_, U_lc) + r_r * tensor.dot(rightx_, U_rc) + b_c)   # + tensor.dot(state_t, U_xh))
        
        # h = u_gates * (tensor.dot(leftx_, U_pl) + tensor.dot(rightx_, U_pr)) + (1. - u_gates) * h

        h = u_l * leftx_ + u_r * rightx_ + u_last * h

        h = m_[:, None] * h   # [n_samples * 1] * [n_samples * dimctx] => [n_samples * dimctx]
        
        # state_t = m_[:, None] * state_t       
        # root = m_[:, None] * state_t + (1. - m_[:, None]) * root
        root = m_[:, None] * h + (1. - m_[:, None]) * root

        # update ctx_, memery_,
        ctx_ = ctx_ + write[:, None, None] * h
      
        # statex = statex + write[:, None, None] * state_t

        return ctx_, root   #, statex

    # prepare scan arguments
    seqs = [mask, mask_left, mask_right, write_mask]

    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U_l')],
                   tparams[_p(prefix, 'U_r')],
                   tparams[_p(prefix, 'b')],
                   tparams[_p(prefix, 'U_lf')],
                   tparams[_p(prefix, 'U_rf')],
                   tparams[_p(prefix, 'b_f')],
                   tparams[_p(prefix, 'U_lc')],
                   tparams[_p(prefix, 'U_rc')],
                   tparams[_p(prefix, 'b_c')]
                   #tparams[_p(prefix, 'W_zl')],
                   #tparams[_p(prefix, 'W_zr')],
                   #tparams[_p(prefix, 'b_z')],
                   #tparams[_p(prefix, 'U_x')],
                   #tparams[_p(prefix, 'U_xu')],
                   #tparams[_p(prefix, 'U_xh')]
                   ]

    context = concatenate([context, tensor.alloc(0., write_mask.shape[0], n_samples, dim)], axis=0)
    
    # state_init = concatenate([stateblew, tensor.alloc(0., write_mask.shape[0], n_samples, nin)], axis=0)
   
    #root = tensor.alloc(0., n_samples, nin)
    root = tensor.alloc(0., n_samples, dim)    

    # print 'tgru', context.ndim
    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[context, root],   # state_init
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    # rval = [rval]
    # print 'rval', rval[0].ndim
    return rval[0][-1], rval[1][-1]  #, rval[2][-1]


def param_init_td_tgru(options, params, prefix='tgru', nin=None, dim=None, dimctx=None):
    if dimctx is None:
        dimctx = options['dim_ctx']

    # reset gate left and reset gate right
    U_lh = numpy.concatenate([ortho_weight(dimctx),
                             ortho_weight(dimctx)], axis=1)
    params[_p(prefix, 'U_lh')] = U_lh
    
    U_rh = numpy.concatenate([ortho_weight(dimctx),
                             ortho_weight(dimctx)], axis=1)
    params[_p(prefix, 'U_rh')] = U_rh

    # nin => dimctx
    # U_l = numpy.concatenate([norm_weight(nin, dimctx),
    #                         norm_weight(nin, dimctx)], axis=1)
    U_l = numpy.concatenate([ortho_weight(dimctx),
                             ortho_weight(dimctx)], axis=1)

    params[_p(prefix, 'U_l')] = U_l

    U_r = numpy.concatenate([ortho_weight(dimctx),
                             ortho_weight(dimctx)], axis=1)
    params[_p(prefix, 'U_r')] = U_r

    params[_p(prefix, 'b_l')] = numpy.zeros((2 * dimctx,)).astype('float32')
    params[_p(prefix, 'b_r')] = numpy.zeros((2 * dimctx,)).astype('float32')
   
    Ux = ortho_weight(dimctx)
    params[_p(prefix, 'Ux')] = Ux
    Ux_r = ortho_weight(dimctx)
    params[_p(prefix, 'Ux_r')] = Ux_r
    
    U_lx = ortho_weight(dimctx)
    params[_p(prefix, 'U_lx')] = U_lx
    U_rx = ortho_weight(dimctx)
    params[_p(prefix, 'U_rx')] = U_rx
    params[_p(prefix, 'b_lx')] = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_rx')] = numpy.zeros((dimctx,)).astype('float32')

    return params


def td_tgru_layer(tparams, mask_l, options, prefix='tgru', mask_right=None, write_mask=None,
               stateblew=None, root=None, eot=None,  
               **kwargs):
    nsteps = mask_l.shape[0]
    if mask_l.ndim == 3:             # mini-batch
        n_samples = mask_l.shape[2]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'U_lh')].shape[0]
    nin = options['dim_word']

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]  # fix row, col n*dim : (n+1)*dim

    def init_context(ctx_, statex, left_, right_, write, rt, init_ctx, eot, U_lh, U_l, b_l, Ux, U_lx, b_lx, 
                     U_rh, U_r, b_r, Ux_r, U_rx, b_rx):
        h_ = init_ctx
        state = rt
        # left   
        preact = tensor.dot(h_, U_lh)         # h_ : hidden state  U: U_r for reset gate U_u for update gate
        preact += tensor.dot(state, U_l) + b_l

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))     # first dim for reset gate
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))     # last dim for update gate

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + tensor.dot(state, U_lx) + b_lx

        # hidden state proposal
        h_l = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h_l = u * h_ + (1. - u) * h_l
        ctx_ = ctx_ + eot[:, :, None] * h_l

        ctx_td = td_layer(tparams, left_, options, prefix, right_, write, ctx_, statex,
                           U_lh, U_l, b_l, Ux, U_lx, b_lx, 
                           U_rh, U_r, b_r, Ux_r, U_rx, b_rx)       

        return ctx_td

        # prepare scan arguments

    _step = init_context
    shared_vars = [tparams[_p(prefix, 'U_lh')],
                   tparams[_p(prefix, 'U_l')],
                   tparams[_p(prefix, 'b_l')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'U_lx')],
                   tparams[_p(prefix, 'b_lx')],
                   tparams[_p(prefix, 'U_rh')],
                   tparams[_p(prefix, 'U_r')],
                   tparams[_p(prefix, 'b_r')],
                   tparams[_p(prefix, 'Ux_r')],
                   tparams[_p(prefix, 'U_rx')],
                   tparams[_p(prefix, 'b_rx')]
                   ]
    context = tensor.alloc(0., write_mask.shape[1], n_samples, dim)
    init_ctx = tensor.alloc(0., n_samples, dim)
    rval = _step(*([context, stateblew, mask_l, mask_right, write_mask, root, init_ctx, eot] + shared_vars)) 

    return rval


def td_layer(tparams, mask_left, options, prefix, mask_right, write_mask, context,
               stateblew, U_lh, U_l, b_l, Ux, U_lx, b_lx, U_rh, U_r, b_r, Ux_r, U_rx, b_rx, 
               **kwargs):
    nsteps = mask_left.shape[0]
    if mask_left.ndim == 3:             # mini-batch
        n_samples = mask_left.shape[2]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'U_lh')].shape[0]
    nin = options['dim_word']


    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]  # fix row, col n*dim : (n+1)*dim

    # step function to be used by scan
    # arguments    | sequences (m_, x_, xx_)|outputs-info (h_)| non-seqs (U, Ux)
    def _step_slice(left_, right_, write, ctx_, statex, U_lh, U_l, b_l, Ux, U_lx, b_lx, U_rh, U_r, b_r, Ux_r, U_rx, b_rx):
        # leftx_ = leftx.sum(0)                    # n_samples * dimctx
        h_ = (write[:, None, None] * ctx_).sum(0)

        left_in = (left_[:, :, None] * statex).sum(0)
        right_in = (right_[:, :, None] * statex).sum(0)


        # left   
        preact = tensor.dot(h_, U_lh)         # h_ : hidden state  U: U_r for reset gate U_u for update gate
        preact += tensor.dot(left_in, U_l) + b_l

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))     # first dim for reset gate
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))     # last dim for update gate

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + tensor.dot(left_in, U_lx) + b_lx

        # hidden state proposal
        h_l = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h_l = u * h_ + (1. - u) * h_l
        # h = m_[:, None] * h + (1. - m_)[:, None] * h_   # if mask = 1 h = h herwise h = h_

        
        # right
        preact = tensor.dot(h_, U_rh)         # h_ : hidden state  U: U_r for reset gate U_u for update gate
        preact += tensor.dot(right_in, U_r) + b_r

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))     # first dim for reset gate
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))     # last dim for update gate

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux_r)
        preactx = preactx * r
        preactx = preactx + tensor.dot(right_in, U_rx) + b_rx

        # hidden state proposal
        h_r = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h_r = u * h_ + (1. - u) * h_r
        # h = m_[:, None] * h + (1. - m_)[:, None] * h_ 

        ctx_ = ctx_ + left_[:, :, None] * h_l + right_[:, :, None] * h_r
    
        

        return ctx_

    # prepare scan arguments
    seqs = [mask_left, mask_right, write_mask]

    _step = _step_slice
    shared_vars = [U_lh, U_l, b_l, Ux, U_lx, b_lx, U_rh, U_r, b_r, Ux_r, U_rx, b_rx]

    # print 'tgru', context.ndim
    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=context,
                                non_sequences=[stateblew] + shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    # print 'rval', rval[0].ndim
    return rval[0][-1]
 

# Conditional GRU layer with Attention    nin = dim_word = (512)  dim = 1024 dimctx = 2*dim =(2048)
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    # U_cw = ortho_weight(dimctx)
    # params[_p(prefix, 'U_cw')] = U_cw
    # U_ct = ortho_weight(dimctx)
    # params[_p(prefix, 'U_ct')] = U_ct

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None, init_cover=None,
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'W_comb_att')].shape[0]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
     
    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) +\
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    # seqs = [m_ = mask, x_ = state_below_, xx_ = state_belowx]
    # outputs_info = [h_ = init_state, ctx_, alpha_]
    # non_seqs = [pctx_,  context = cc_]+shared_vars
    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):                # , U_cw, U_ct
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_  # n_samples * dim

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)    # n_samples * 2dim
        pctx__ = pctx_ + pstate_[None, :, :]    # n_timesetp * n_samples * 2dim + 1 * n_samples * 2dim
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)            # n_timesetp * n_samples * 2dim
        alpha = tensor.dot(pctx__, U_att)+c_tt  # n_timestep * n_smaples * 1
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]]) # n_timestep * n_smaples
        alpha = tensor.exp(alpha)

        if context_mask:                       # context_mask = x_mask:[n_timestep * n_samples]
            alpha = alpha * context_mask       # [n_timestep * n_samples]

        alpha = alpha / alpha.sum(0, keepdims=True)      #alignment weight
            # cc_ = context:[n_timestep * n_samples *2dim]; alpha[:, :, None] => [n_timestep * n_samples * 1]
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)          # current context [n_samples *2dim]
        #    ctx_ = concatenate([ctx_, ctx_], axis=1)
             
        preact2 = tensor.dot(h1, U_nl)+b_nl    # U_nl:[dim * 2dim]; preact2=> [n_samples * 2dim]
        preact2 += tensor.dot(ctx_, Wc)        # Wc: [2dim * 2dim]; preact2=> [n_samples * 2dim]
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)           # r2 => [n_samples * dim]
        u2 = _slice(preact2, 1, dim)           # u2 => [n_samples * dim]

        # print 'Wcx:', Wcx.shape
        preactx2 = tensor.dot(h1, Ux_nl)+bx_nl # Ux_nl:[dim * dim]; preactx2=> [n_samples * dim]
        preactx2 *= r2                         # [n_samples * dim]
        preactx2 += tensor.dot(ctx_, Wcx)      # Wcx: [(dimctx=2dim) * dim] preactx2 => [n_samples * dim]

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1  # h2=>[n_samples * dim]

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    # seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]


    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


def param_init_gru_cover(options, params, prefix='gru_cover',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    dim_cov = options['dim_cov']

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    # coverage update by chend
    Uc_nl = numpy.concatenate([norm_weight(dim, dim_cov),
                              norm_weight(dim, dim_cov)], axis=1)
    params[_p(prefix, 'Uc_nl')] = Uc_nl
    params[_p(prefix, 'bc_nl')] = numpy.zeros((2 * dim_cov,)).astype('float32')

    U_ncov = ortho_weight(dim_cov)
    params[_p(prefix, 'U_ncov')] = U_ncov
    params[_p(prefix, 'b_ncov')] = numpy.zeros((dim_cov,)).astype('float32')

    # attention: coverage -> attention
    W_cov_att = norm_weight(dim_cov, dimctx)
    params[_p(prefix, 'W_cov_att')] = W_cov_att

    # coverage: context -> gates
    Wc_cov = norm_weight(dimctx, 2*dim_cov)
    params[_p(prefix, 'Wc_cov')] = Wc_cov
    b_cov = numpy.zeros((2*dim_cov,)).astype('float32')
    params[_p(prefix, 'b_cov')] = b_cov

    # coverage: alpha -> gates
    Wa_cov = norm_weight(1, 2*dim_cov)
    params[_p(prefix, 'Wa_cov')] = Wa_cov
    # coverage: coverage -> gates
    U_cov = numpy.concatenate([ortho_weight(dim_cov), ortho_weight(dim_cov)], axis=1)
    params[_p(prefix, 'U_cov')] = U_cov

    # coverage: hidden -> coverage
    U_hc = norm_weight(dim, dim_cov)
    params[_p(prefix, 'U_hc')] = U_hc

    # coverage: context -> coverage
    W_hc = norm_weight(dimctx, dim_cov)
    params[_p(prefix, 'W_hc')] = W_hc
    b_hc = numpy.zeros((dim_cov,)).astype('float32')
    params[_p(prefix, 'b_hc')] = b_hc

    # coverage: alpha -> coverage
    W_ac = norm_weight(1, dim_cov)
    params[_p(prefix, 'W_ac')] = W_ac
    ###########################

    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    if options['use_tree_cover']:
        U_left = numpy.concatenate([ortho_weight(dim_cov), ortho_weight(dim_cov)], axis=1)
        params[_p(prefix, 'U_left')] = U_left
        U_right = numpy.concatenate([ortho_weight(dim_cov), ortho_weight(dim_cov)], axis=1)
        params[_p(prefix, 'U_right')] = U_right
        U_nleft = ortho_weight(dim_cov)
        params[_p(prefix, 'U_nleft')] = U_nleft
        U_nright = ortho_weight(dim_cov)
        params[_p(prefix, 'U_nright')] = U_nright

        Wa_l = norm_weight(1, 2 * dim_cov)
        params[_p(prefix, 'Wa_l')] = Wa_l
        Wa_r = norm_weight(1, 2 * dim_cov)
        params[_p(prefix, 'Wa_r')] = Wa_r

        Wac_l = norm_weight(1, dim_cov)
        params[_p(prefix, 'Wac_l')] = Wac_l
        Wac_r = norm_weight(1, dim_cov)
        params[_p(prefix, 'Wac_r')] = Wac_r

    return params


def coverage_tree_update(mask_left, coverage, attention, mask_right, write_mask, prefix='cover_tree',
                             mask=None, **kwargs):
    assert coverage, 'Coverage must be provided'
    assert mask_left, 'Tree must be provided'
    nsteps = mask_left.shape[0]
    if mask is None:
        mask = tensor.alloc(1., mask_left.shape[0], 1)

    def _step_slice(m_, left_, right_, write_, l_child, r_child, a_l, a_r, cover_, att_):
        leftc_ = (left_[:, :, None] * cover_).sum(0)
        rightc_ = (right_[:, :, None] * cover_).sum(0)

        lefta_ = (left_[:, :, None] * att_).sum(0)
        righta_ = (right_[:, :, None] * att_).sum(0)

        # parent
        # pt_ = (write_[:, None, None] * cover_).sum(0)

        leftc_ = m_[:, None] * leftc_
        rightc_ = m_[:, None] * rightc_
        # pt_ = m_[:, None] * pt_

        # write into
        l_child += write_[:, None, None] * leftc_
        r_child += write_[:, None, None] * rightc_
        # parent += left_[:, :, None] * pt_
        # parent += right_[:, :, None] * pt_

        a_l += write_[:, None, None] * lefta_
        a_r += write_[:, None, None] * righta_

        return l_child, r_child, a_l, a_r  # parent


    _step = _step_slice
    seqs = [mask, mask_left, mask_right, write_mask]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[tensor.alloc(0., coverage.shape[0], coverage.shape[1], coverage.shape[2]),
                                              tensor.alloc(0., coverage.shape[0], coverage.shape[1], coverage.shape[2]),
                                              tensor.alloc(0., coverage.shape[0], coverage.shape[1], 1),
                                              tensor.alloc(0., coverage.shape[0], coverage.shape[1], 1)],
                                non_sequences=[coverage, attention],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)

    return rval


def gru_cover_layer(tparams, state_below, options, prefix='gru',
                      mask=None, context=None, one_step=False,
                      init_memory=None, init_state=None, init_cover=None,
                      context_mask=None,
                      mask_left=None, tree_mask=None, mask_right=None, write_mask=None,
                      **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]
    # added by chenhd
    dim_cov = options['dim_cov']

    #######################
    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) +\
        tparams[_p(prefix, 'b_att')]

    # n_timstep * n_samples * 2dim_cov
    pctx_cov = tensor.dot(context, tparams[_p(prefix, 'Wc_cov')]) +\
        tparams[_p(prefix, 'b_cov')]

    pctx_hc = tensor.dot(context, tparams[_p(prefix, 'W_hc')]) +\
        tparams[_p(prefix, 'b_hc')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    # seqs = [m_ = y_mask, x_ = state_below_, xx_ = state_belowx]
    # outputs_info = [h_ = init_state, ctx_, alpha_, coverage_]
    # non_seqs = [pctx_,  context = cc_]+shared_vars
    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, coverage_, pctx_, pctx_cov, pctx_hc, cc_,
                    U, U_cov, Wa_cov, W_ac, Wc, W_comb_att, W_cov_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Uc_nl, Ux_nl, U_ncov, U_hc, b_nl, bc_nl, bx_nl, b_ncov):
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_  # n_samples * dim

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)  # n_samples * 2dim
        pstatec_ = tensor.dot(coverage_, W_cov_att)  # n_timestep * n_samples * 2dim added by chend
        pctx__ = pctx_ + pstate_[None, :, :] + pstatec_  # n_timesetp * n_samples * 2dim + 1 * n_samples * 2dim
        # pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)            # n_timesetp * n_samples * 2dim
        alpha = tensor.dot(pctx__, U_att)+c_tt  # n_timestep * n_smaples * 1
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]]) # n_timestep * n_smaples
        alpha = tensor.exp(alpha)
        if context_mask:                       # context_mask = x_mask:[n_timestep * n_samples]
            alpha = alpha * context_mask       # [n_timestep * n_samples]
        alpha = alpha / alpha.sum(0, keepdims=True)      #alignment weight
        # cc_ = context:[n_timestep * n_samples *2dim]; alpha[:, :, None] => [n_timestep * n_samples * 1]
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)          # current context [n_samples *2dim]

        # coverage update  by chenhd
        preact_cov = tensor.dot(coverage_, U_cov)                # n_timestep * n_samples * 2dim_cov
        preact_cov += (tensor.dot(h1, Uc_nl)+bc_nl)              # + n_samples * 2dim_cov
        preact_cov += tensor.dot(alpha[:, :, None],  Wa_cov)     # + n_timestep * n_samples * 2dim_cov
        preact_cov += pctx_cov                                   # + n_timestep * n_samples * 2dim_cov
        preact_cov = tensor.nnet.sigmoid(preact_cov)

        r_cov = _slice(preact_cov, 0, dim_cov)           # r_cov => [n_timestep * n_samples * dim_cov]
        u_cov = _slice(preact_cov, 1, dim_cov)           # u_cov => [n_timestep * n_samples * dim_cov]

        preactx_cov = tensor.dot(coverage_, U_ncov) + b_ncov # [n_timestep * n_samples * dim_cov]
        preactx_cov *= r_cov
        preactx_cov += tensor.dot(h1, U_hc)                  # + n_samples * dim_cov
        preactx_cov += pctx_hc                               # + n_timestep * n_samples * dim_cov
        preactx_cov += tensor.dot(alpha[:, :, None],  W_ac)  # + [n_timestep * n_samples * 1] * [1 * dim_cov]

        coverage_new = tensor.tanh(preactx_cov)

        coverage_new = u_cov * coverage_ + (1. - u_cov) * coverage_new

        if context_mask:
            coverage_new = coverage_new * context_mask[:, :, None]

        coverage_new = m_[:, None] * coverage_new + (1. - m_)[:, None] * coverage_
        #############################

        preact2 = tensor.dot(h1, U_nl)+b_nl    # U_nl:[dim * 2dim]; preact2=> [n_samples * 2dim]
        preact2 += tensor.dot(ctx_, Wc)        # Wc: [2dim * 2dim]; preact2=> [n_samples * 2dim]
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)           # r2 => [n_samples * dim]
        u2 = _slice(preact2, 1, dim)           # u2 => [n_samples * dim]

        preactx2 = tensor.dot(h1, Ux_nl)+bx_nl        # Ux_nl:[dim * dim]; preactx2=> [n_samples * dim]
        preactx2 *= r2                                # [n_samples * dim]
        preactx2 += tensor.dot(ctx_, Wcx)             # Wcx: [(dimctx=2dim) * dim] preactx2 => [n_samples * dim]

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1  # h2=>[n_samples * dim]

        return h2, ctx_, alpha.T, coverage_new  # pstate_, preact, preactx, r, u

    def _step_tree_slice(m_, x_, xx_, h_, ctx_, alpha_, coverage_, pctx_, pctx_cov, pctx_hc, cc_,
                         mask_l, t_mask, mask_r, wrt_mask,
                         U, U_cov, Wa_cov, W_ac, Wc, W_comb_att, W_cov_att, U_att, c_tt, Ux, Wcx,
                         U_nl, Uc_nl, Ux_nl, U_ncov, U_hc, b_nl, bc_nl, bx_nl, b_ncov,
                         U_left, U_right, Wa_l, Wa_r,
                         U_nleft, U_nright, Wac_l, Wac_r
                         ):
            preact1 = tensor.dot(h_, U)
            preact1 += x_
            preact1 = tensor.nnet.sigmoid(preact1)

            r1 = _slice(preact1, 0, dim)
            u1 = _slice(preact1, 1, dim)

            preactx1 = tensor.dot(h_, Ux)
            preactx1 *= r1
            preactx1 += xx_

            h1 = tensor.tanh(preactx1)

            h1 = u1 * h_ + (1. - u1) * h1
            h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_  # n_samples * dim

            # attention
            pstate_ = tensor.dot(h1, W_comb_att)  # n_samples * 2dim
            pstatec_ = tensor.dot(coverage_, W_cov_att)  # n_timestep * n_samples * 2dim added by chend
            pctx__ = pctx_ + pstate_[None, :, :] + pstatec_  # n_timesetp * n_samples * 2dim + 1 * n_samples * 2dim
            # pctx__ += xc_
            pctx__ = tensor.tanh(pctx__)  # n_timesetp * n_samples * 2dim
            alpha = tensor.dot(pctx__, U_att) + c_tt  # n_timestep * n_smaples * 1
            alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])  # n_timestep * n_smaples
            alpha = tensor.exp(alpha)
            if context_mask:  # context_mask = x_mask:[n_timestep * n_samples]
                alpha = alpha * context_mask  # [n_timestep * n_samples]
            alpha = alpha / alpha.sum(0, keepdims=True)  # alignment weight
            # cc_ = context:[n_timestep * n_samples *2dim]; alpha[:, :, None] => [n_timestep * n_samples * 1]
            ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context [n_samples *2dim]

            # coverage update  by chenhd
            cover_last = coverage_tree_update(mask_left, coverage_, alpha[:, :, None], mask_right, write_mask, prefix=prefix,
                                              mask=tree_mask, **kwargs)
            preact_cov = tensor.dot(coverage_, U_cov)  # n_timestep * n_samples * 2dim_cov

            preact_cov += tensor.dot(cover_last[0][-1], U_left)
            preact_cov += tensor.dot(cover_last[1][-1], U_right)

            preact_cov += tensor.dot(cover_last[2][-1], Wa_l)
            preact_cov += tensor.dot(cover_last[3][-1], Wa_r)

            preact_cov += (tensor.dot(h1, Uc_nl) + bc_nl)  # + n_samples * 2dim_cov
            preact_cov += tensor.dot(alpha[:, :, None], Wa_cov)  # + n_timestep * n_samples * 2dim_cov
            preact_cov += pctx_cov  # + n_timestep * n_samples * 2dim_cov
            preact_cov = tensor.nnet.sigmoid(preact_cov)

            r_cov = _slice(preact_cov, 0, dim_cov)  # r_cov => [n_timestep * n_samples * dim_cov]
            u_cov = _slice(preact_cov, 1, dim_cov)  # u_cov => [n_timestep * n_samples * dim_cov]

            preactx_cov = tensor.dot(coverage_, U_ncov) + b_ncov  # [n_timestep * n_samples * dim_cov]
            preactx_cov *= r_cov
            preactx_cov += tensor.dot(h1, U_hc)  # + n_samples * dim_cov
            preactx_cov += pctx_hc  # + n_timestep * n_samples * dim_cov
            preactx_cov += tensor.dot(alpha[:, :, None], W_ac)  # + [n_timestep * n_samples * 1] * [1 * dim_cov]

            preactx_cov += tensor.dot(cover_last[0][-1], U_nleft)
            preactx_cov += tensor.dot(cover_last[1][-1], U_nright)

            preactx_cov += tensor.dot(cover_last[2][-1], Wac_l)
            preactx_cov += tensor.dot(cover_last[3][-1], Wac_r)

            coverage_new = tensor.tanh(preactx_cov)

            coverage_new = u_cov * coverage_ + (1. - u_cov) * coverage_new

            if context_mask:
                coverage_new = coverage_new * context_mask[:, :, None]

            coverage_new = m_[:, None] * coverage_new + (1. - m_)[:, None] * coverage_
            #############################

            preact2 = tensor.dot(h1, U_nl) + b_nl  # U_nl:[dim * 2dim]; preact2=> [n_samples * 2dim]
            preact2 += tensor.dot(ctx_, Wc)  # Wc: [2dim * 2dim]; preact2=> [n_samples * 2dim]
            preact2 = tensor.nnet.sigmoid(preact2)

            r2 = _slice(preact2, 0, dim)  # r2 => [n_samples * dim]
            u2 = _slice(preact2, 1, dim)  # u2 => [n_samples * dim]

            preactx2 = tensor.dot(h1, Ux_nl) + bx_nl  # Ux_nl:[dim * dim]; preactx2=> [n_samples * dim]
            preactx2 *= r2  # [n_samples * dim]
            preactx2 += tensor.dot(ctx_, Wcx)  # Wcx: [(dimctx=2dim) * dim] preactx2 => [n_samples * dim]

            h2 = tensor.tanh(preactx2)

            h2 = u2 * h1 + (1. - u2) * h2
            h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1  # h2=>[n_samples * dim]

            return h2, ctx_, alpha.T, coverage_new  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    # seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'U_cov')],
                   tparams[_p(prefix, 'Wa_cov')],
                   tparams[_p(prefix, 'W_ac')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'W_cov_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Uc_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'U_ncov')],
                   tparams[_p(prefix, 'U_hc')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bc_nl')],
                   tparams[_p(prefix, 'bx_nl')],
                   tparams[_p(prefix, 'b_ncov')]]

    temp = [init_state, None, None, init_cover, pctx_, pctx_cov, pctx_hc, context]
    temp1 =[pctx_, pctx_cov, pctx_hc, context]
    if options['use_tree_cover']:
        _step = _step_tree_slice
        shared_vars += [tparams[_p(prefix, 'U_left')],
                        tparams[_p(prefix, 'U_right')],
                        tparams[_p(prefix, 'Wa_l')],
                        tparams[_p(prefix, 'Wa_r')],
                        tparams[_p(prefix, 'U_nleft')],
                        tparams[_p(prefix, 'U_nright')],
                        tparams[_p(prefix, 'Wac_l')],
                        tparams[_p(prefix, 'Wac_r')]]
        temp += [mask_left, tree_mask, mask_right, write_mask]
        temp1 += [mask_left, tree_mask, mask_right, write_mask]

    if one_step:
        rval = _step(*(seqs + temp +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),  # ctx_: [n_samples * (dimctx=2dim)]
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0]),  # alpha.T:[n_samples * n_timestep]
                                                  tensor.alloc(0., context.shape[0], n_samples,
                                                               dim_cov)],    # added by chenhd
                                    non_sequences=temp1 + shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: bidirectional RNN
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    ctxdim = 2 * options['dim']

    params = get_layer(options['tree_encoder'])[0](options, params,
                                                   prefix='encoder_tree',
                                                   nin=options['dim_word'],
                                                   dim=options['dim'],
                                                   dimctx=ctxdim)

    params = get_layer(options['td_encoder'])[0](options, params,
                                                   prefix='encoder_td',
                                                   nin=options['dim_word'],
                                                   dim=options['dim'],
                                                   dimctx=ctxdim)

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=2*ctxdim, nout=options['dim'])
    # decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=2*ctxdim)
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=2*ctxdim, nout=options['dim_word'],       #concatenate sequential and tree attention
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    return params


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    # added by chenhd: for source-tree
    mask_l = tensor.tensor3('mask_l', dtype='float32')
    mask_r = tensor.tensor3('mask_r', dtype='float32')
    tree_mask = tensor.matrix('tree_mask', dtype='float32')
    wrt_mask = tensor.matrix('wrt_mask', dtype='float32')
    
    # for bi-tree
    eot = tensor.matrix('eot', dtype='float32')
    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    emb = tparams['Wemb'][x.flatten()]   # x.flatten() Matrix to vector and get corresponding embedding
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']]) # Matrix: n_timesteps * n_samples * dim_word
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)   # n_timesetp * n_samples * dim
    # word embedding for backward rnn (source)
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask)   # n_timesetp * n_samples * dim

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)  # n_timesetp * n_samples * 2dim

    # print ctx.ndim
    # added by chenhd: build tree-encoder
    #com_ctx
    ctx_all, root = get_layer(options['tree_encoder'])[1](tparams, mask_l, options,
                                                    prefix='encoder_tree',
                                                    mask=tree_mask, mask_right=mask_r, write_mask=wrt_mask, context=ctx, 
                                                    stateblew=emb)
    
    mask_l_r = mask_l[::-1]
    mask_r_r = mask_r[::-1]
    wrt_mask_r = wrt_mask[::-1]     
    ctx_td = get_layer(options['td_encoder'])[1](tparams, mask_l_r, options,
                                                 prefix='encoder_td',
                                                 mask_right=mask_r_r, write_mask=wrt_mask_r,
                                                 stateblew=ctx_all, root=root, eot=eot)
    
    ctx_all = concatenate([ctx_all, ctx_td], axis=2)
    # print ctx_all.ndim
    #ctx_all = com_ctx[0][-1]
    ctx_mask = concatenate([x_mask, tree_mask], axis=0)
    # mean of the context (across time) will be used to initialize decoder rnn
    # ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]  #n_samples * 2dim

    # added by chenhd
    ctx_mean = (ctx_all * ctx_mask[:, :, None]).sum(0) / ctx_mask.sum(0)[:, None]  #n_samples * 2dim

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state: ctx_mean(n_samples * 2dim) => init_state(n_samples * dim)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right.                            ==>  tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    # This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.        ==> emb[:-1]
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    if options['use_tree_cover']:
        proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                                prefix='decoder',
                                                mask=y_mask, context=ctx_all,
                                                # maxlen=n_timesteps,
                                                context_mask=ctx_mask,  # change by chenhd
                                                mask_left=mask_l, tree_mask=tree_mask, mask_right=mask_r,
                                                write_mask=wrt_mask,
                                                one_step=False,
                                                init_state=init_state)
    else:
        proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                                prefix='decoder',
                                                mask=y_mask, context=ctx_all,
                                                maxlen=n_timesteps,
                                                context_mask=ctx_mask,  # change by chenhd
                                                one_step=False,
                                                init_state=init_state)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)           # [n_timesteps_trg * n_samples * dim_word]
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')  # [n_timesteps_trg * n_samples * n_words]
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, mask_l, mask_r, tree_mask, wrt_mask, eot, opt_ret, cost


# build a sampler
def build_sampler(tparams, options, trng, use_noise):
    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    mask_l = tensor.tensor3('mask_l', dtype='float32')
    mask_r = tensor.tensor3('mask_r', dtype='float32')
    wrt_mask = tensor.matrix('wrt_mask', dtype='float32')
    
    eot = tensor.matrix('eot', dtype='float32') 
    # word embedding (source), forward and backward
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder')
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r')

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    ctx_all, root = get_layer(options['tree_encoder'])[1](tparams, mask_l, options,
                                             prefix='encoder_tree',
                                             mask=None, mask_right=mask_r, write_mask=wrt_mask, context=ctx, stateblew=emb)
    
    mask_l_r = mask_l[::-1]
    mask_r_r = mask_r[::-1]
    wrt_mask_r = wrt_mask[::-1]
    ctx_td = get_layer(options['td_encoder'])[1](tparams, mask_l_r, options,
                                                 prefix='encoder_td',
                                                 mask_right=mask_r_r, write_mask=wrt_mask_r,
                                                 stateblew=ctx_all, root=root, eot=eot)

    ctx_all = concatenate([ctx_all, ctx_td], axis=2)
    #ctx_all = com_ctx[0][-1]

    # get the input for decoder rnn initializer mlp
    # ctx_mean = ctx.mean(0)                          #average value of all raws

    # added by chenhd
    ctx_mean = ctx_all.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # for coverage decoder
    init_cover = None 
    if options['decoder'] == 'gru_cover':
        init_cover = tensor.alloc(0., ctx_all.shape[0], n_samples, options['dim_cov'])

    print 'Building f_init...',
    if options['decoder'] == 'gru_cover':
        outs = [init_state, ctx_all, init_cover]
    else:
        outs = [init_state, ctx_all]

    f_init = theano.function([x, mask_l, mask_r, wrt_mask, eot], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')


    init_cover = tensor.tensor3('init_cover', dtype='float32')

    # maxlen = tensor.scalar('maxlen', dtype='int64')
    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    if options['use_tree_cover']:
        proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                                prefix='decoder',
                                                mask=None, context=ctx_all,
                                                mask_left=mask_l, tree_mask=None, mask_right=mask_r,
                                                write_mask=wrt_mask,
                                                # maxlen=maxlen,
                                                one_step=True,
                                                init_state=init_state, init_cover=init_cover)
    else:
        proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                                prefix='decoder',
                                                mask=None, context=ctx_all,
                                                # maxlen=maxlen,
                                                one_step=True,
                                                init_state=init_state, init_cover=init_cover)
    # get the next hidden state
    next_state = proj[0]

    next_cover = None
    if options['decoder'] == 'gru_cover':
        next_cover = proj[3]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)  # [1 * dim_word]
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')  # [1 * n_words]

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next..',
    #inps = [y, ctx_all, init_state]  #, maxlen]
    #outs = [next_probs, next_sample, next_state]
    inps = [y, ctx_all]
    if options['use_tree_cover']:
        inps += [mask_l, mask_r, wrt_mask]
    inps += [init_state, init_cover]
    outs = [next_probs, next_sample, next_state, next_cover]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, mask_l, mask_r, wrt_mask, eot, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x, mask_l, mask_r, wrt_mask, eot)
   
    next_cover = None
    if options['decoder'] == 'gru_cover':
        next_cover = ret[2]


    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator
    timestep = ctx0.shape[0]
    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        # ctx0: [timesteps , n_samples, dimctx]; => ctx: [timesteps, (n_samples * live_k ), (dimctx * 1)]

        inps = [next_w, ctx]   #, x.shape[0]]
        mask_lx = None
        mask_rx = None
        if options['use_tree_cover']:
            mask_lx = numpy.tile(mask_l, live_k)
            mask_rx = numpy.tile(mask_r, live_k)
            inps += [mask_lx, mask_rx, wrt_mask]
        inps.append(next_state)

        #ret = f_next(*inps)
        if options['decoder'] == 'gru_cover':
            inps.append(next_cover)
        ret = f_next(*inps)

        if options['decoder'] == 'gru_cover':
            next_cover = ret[3]

        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0, nw]
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []

            new_hyp_covers = [[] for ll in xrange(timestep)]              # coverage  

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))
                if options['decoder'] == 'gru_cover':
                    for ci in xrange(timestep):
                        new_hyp_covers[ci].append(copy.copy(next_cover[ci][ti]))    # coverage

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            hyp_covers = [[] for ll in xrange(timestep)]                 # coverage
  
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    if options['decoder'] == 'gru_cover':
                        for ci in xrange(timestep):
                            hyp_covers[ci].append(new_hyp_covers[ci][idx])    # coverage

            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)
            if options['decoder'] == 'gru_cover':
                next_cover = numpy.array(hyp_covers[0])[None, :, :]    # coverage
                for ci in xrange(len(hyp_covers) - 1):
                    next_cover = numpy.concatenate([next_cover, numpy.array(hyp_covers[ci + 1])[None, :, :]], axis=0)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_treedata, options, iterator, verbose=True):
    probs = []

    n_done = 0

    for x, y, tree in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask, mask_l, mask_r, tree_mask, wrt_mask, eot = prepare_treedata(x, y, tree,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask, y, y_mask, mask_l, mask_r, tree_mask, wrt_mask, eot)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    # updates (iterable over pairs (shared_variable, new_expression). List, tuple or dict.)
    # expressions for new SharedVariable values
    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update


def train(dim_word=512,  # word vector dimensionality
          dim=512,  # the number of LSTM units
          dim_cov=50,  # the number of coverage vector
          encoder='gru',
          decoder='gru_cover',
          tree_encoder='tree_gru',
          td_encoder='td_tgru',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=100000,  # source vocabulary size
          n_words=100000,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          initpara='model_init.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],
          dictionaries=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
          treeset=['/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.tree',
                   '/data/lisatmp3/chokyun/valid.tree'],
          use_dropout=False,
          reload_=False,
          finetune_=True,
          overwrite=False,
          use_tree_cover=False,
          shuffle_each_epoch=False):

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'Reloading model options'
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data'
    train = TreeTextIterator(datasets[0], datasets[1],
                         dictionaries[0], dictionaries[1], treeset[0],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen,
                         shuffle_each_epoch=shuffle_each_epoch)
    valid = TreeTextIterator(valid_datasets[0], valid_datasets[1],
                         dictionaries[0], dictionaries[1], treeset[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)

    if finetune_ and os.path.exists(initpara):
        print 'Finetuning model parameters'
        params = load_ftparams(initpara, params)

    # reload parameters
    if reload_ and os.path.exists(saveto):
        print 'Reloading model parameters'
        params = load_params(saveto, params)

    # put params into theano shared variable
    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        mask_l, mask_r, tree_mask, wrt_mask, eot,  \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask, mask_l, mask_r, tree_mask, wrt_mask, eot]

    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng, use_noise)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:                       # hyper parameter for L2 reg
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights: alignment regularization
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        # // represent for float divided (round)
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here   gradients' square exceed clip_c, norm it
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    for eidx in xrange(max_epochs):
        n_samples = 0
        for x, y, tree in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            # filter by maxlen
            x, x_mask, y, y_mask, mask_l, mask_r, tree_mask, wrt_mask, eot = prepare_treedata(x, y, tree, maxlen=maxlen,
                                                                                              n_words_src=n_words_src,
                                                                                              n_words=n_words)
            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            #print x.shape, x_mask.shape, y.shape, y_mask.shape, mask_l.shape, mask_r.shape, tree_mask.shape, wrt_mask.shape
            cost = f_grad_shared(x, x_mask, y, y_mask, mask_l, mask_r, tree_mask, wrt_mask, eot)
            
            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                uidx -= 1
                continue
                #return 1., 1., 1.

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            #if numpy.isnan(cost) or numpy.isinf(cost):
            #    print 'NaN detected'
                # continue
            #    return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, history_errs=history_errs,
                                uidx=uidx, **unzip(tparams))
                    print 'Done'

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = True
                    # jj-th sentence in n_samples x=>[n_timesteps * n_samples]
                    # x[:, jj][:, None]=>[n_timesteps * 1]
                    sample, score = gen_sample(tparams, f_init, f_next,
                                               x[:, jj][:, None],
                                               mask_l[:, :, jj][:, :, None],
                                               mask_r[:, :, jj][:, :, None],
                                               wrt_mask,
                                               eot[:, jj][:, None],
                                               model_options, trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=False)
                    print 'Source ', jj, ': ',
                    for vv in x[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[0]:
                            print worddicts_r[0][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Truth ', jj, ' : ',
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print worddicts_r[1][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print worddicts_r[1][vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_treedata,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)
                # if it is the first update or valid_err smaller than min history error best_p record current tparams
                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                # the length of history errors bigger than patient and valid_error bigger than min history error
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print 'Valid ', valid_err

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_treedata,
                           model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err


if __name__ == '__main__':
    pass
