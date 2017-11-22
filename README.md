# Syntax-awared-NMT
Improved Neural Machine Translation with a Syntax-Aware Encoder and Decoder

This package is developed by Huadong Chen, which is based on the <a href="https://github.com/nyu-dl/dl4mt-tutorial">dl4mt-tutorial</a>(Kyunghyun Cho et al., 2014 and Bahdanau et al., 2015).

If you use the code, please cite our papers:
<pre>
<code>
@InProceedings{chen-EtAl:2017:Long6,
  author    = {Chen, Huadong  and  Huang, Shujian  and  Chiang, David  and  Chen, Jiajun},
  title     = {Improved Neural Machine Translation with a Syntax-Aware Encoder and Decoder},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {1936--1945},
  url       = {http://aclweb.org/anthology/P17-1177}
}
</code>
</pre>

Data
------------
Source sentences: ./Data/examples.ch

Target sentences: ./Data/examples.en

Syntactic trees of source sentences: ./Data/examples.tree

Note: We use binarized trees in our paper. The syntactic tree was decomposed into a sequence of triples, i.e., [left-child, right-child, parent]. 

script
------------
Berkeley parser tree to triple

Source: Berkeley pareser tree(./script/nput_example.txt)

Output: sequences of triples
