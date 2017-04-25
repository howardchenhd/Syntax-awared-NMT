# Syntax-awared-NMT
Improved Neural Machine Translation with a Syntax-Aware Encoder and Decoder

This package is developed by Huadong Chen, which is based on the <a href="https://github.com/nyu-dl/dl4mt-tutorial">dl4mt-tutorial</a>(Kyunghyun Cho et al., 2014 and Bahdanau et al., 2015).

If you use the code, please cite our papers:
<pre>
<code>
@InProceedings{Chen:2017:ACL,
      author    = {Huadong Chen, Shujian Huang, David Chiang and Jiajun Chen},
      title     = {Improved Neural Machine Translation with a Syntax-Aware Encoder and Decoder},
      booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
      year      = {2017},
}
</code>
</pre>

Data
------------
source sentences: ./Data/examples.ch

target sentences: ./Data/examples.en

syntactic trees of source sentences: ./Data/examples.tree

Note: We use binarized trees in our paper. The syntactic tree was decomposed into a sequence of triples, i.e., [left-child, right-child, parent]. 
