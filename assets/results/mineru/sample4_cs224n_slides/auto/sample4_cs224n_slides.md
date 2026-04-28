# Natural Language Processing with Deep Learning CS224N/Ling284

![](images/52a5d75b20db6d591779c53d5d6ecf1f9c3c23ce74af992f007ab9ca94826066.jpg)

Tatsunori Hashimoto Lecture 8: Self-Attention and Transformers

## The starting point: mean-pooling for RNNs

![](images/32a54edb062e315f7e5fbf2451ec54d15725ff623db2fa9975cf6a77b75092db.jpg)

• Starting point: a very basic way of ‘passing information from the encoder’ is to average

## Attention is weighted averaging, which lets you do lookups!

Attention is just a weighted average – this is very powerful if the weights are learned!

In attention, the query matches all keys softly, to a weight between 0 and 1. The keys’ values are multiplied by the weights and summed.

![](images/aad01a4a2913b6d708690296ce9f05e74cfd74de4772c0eaa8adbc988d4d04ad.jpg)

In a lookup table, we have a table of keys that map to values. The query matches one of the keys, returning its value.

![](images/1d03b8f30f6a7ac6b20b009f85e1286821d7ad76903d47de3bf53a00aeb95a97.jpg)

## Sequence-to-sequence with attention

Core idea: on each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence

![](images/7aef101ca807b892dc5c43010713fdb0bf0a889c3e1e0de9ca496f6ef0d80522.jpg)

## Lecture Plan

1. From recurrence (RNN) to attention-based NLP models

2. The Transformer model

3. Great results with Transformers

4. Drawbacks and variants of Transformers

Reminders:

See the 2023 lecture notes for some bonus material

Assignment 4 due a week from today! Use Colab for the final training if you don’t have a GPU.

Final project proposal out tonight, due Tuesday, Feb 14 at 4:30PM PST!

Please try to hand in the project proposal on time; we want to get you feedback quickly!

## Last last lecture: Multi-layer RNN for machine translation

[Sutskever et al. 2014; Luong et al. 2015]

The hidden states from RNN layer i are the inputs to RNN layer i + 1

![](images/edf41f4d372705bf31aedcb3a64a3e65f3cc1b34c2145c48e7e6fae5c39f5c91.jpg)  
Conditioning = Bottleneck

## NMT: the first big success story of NLP Deep Learning

Neural Machine Translation went from a fringe research attempt in 2014 to the leading standard method in 2016

• 2014: First seq2seq paper published [Sutskever et al. 2014]

• 2016: Google Translate switches from SMT to NMT – and by 2018 everyone has

![](images/f7c891dde9883891d21bce0daef11856bfe85ede53156c096648c46f3708ad13.jpg)

![](images/3c412f20d82d21a13d0bd7f44f3ae0d1b85973227bdf3619fa256e2e61d33e8f.jpg)

![](images/454c1fdde8057cbb52e5450f7592f553792157eb7466eb61a0e2c2fb3a216d53.jpg)

![](images/c32fec294ef9ed45c20c4c733166ac816f53edeeb287695da7ae92c44f02783e.jpg)

![](images/413fb9c747525e5d56c12aa35dc47050c5a43a4b4e2fbc346798efb8433d47af.jpg)

![](images/c2c5c2159a669225ae07395d549c46b64b30a4f6fefb8c69eb8f1fc348d449d7.jpg)

![](images/69fa1c91c33bf7acc17e89c19c866bf197f179657f1471198ad242ecbeffeb1b.jpg)

![](images/a3f0c175f3416cf85698664b82c3ca8c71031006a119255a50f12ef4b3c43df4.jpg)

## C This is amazing!

• SMT systems, built by hundreds of engineers over many years, outperformed by NMT systems trained by small groups of engineers in a few months

## The final piece: the bottleneck problem in RNNs

![](images/9773f31f4736d3ddc3bfeaae3e7c366e6e2c1dbf97726140edea188ddedf66cf.jpg)

## 1. Why attention? Sequence-to-sequence: the bottleneck problem

![](images/686c8361f9489ec6ee784b6e2fdcd85b06da839d8419f59bbb7a3eed2467d1c2.jpg)

## Issues with recurrent models: Linear interaction distance

• O(sequence length) steps for distant word pairs to interact means:

Hard to learn long-distance dependencies (because gradient problems!)

• Linear order of words is “baked in”; we already know linear order isn’t the right way to think about sentences…

## ••• —

The chef who …

Info of chef has gone through O(sequence length) many layers!

## Issues with recurrent models: Lack of parallelizability

• Forward and backward passes have O(sequence length) unparallelizable operations

C GPUs can perform a bunch of independent computations at once!

• But future RNN hidden states can’t be computed in full before past RNN hidden states have been computed

• Inhibits training on very large datasets!

![](images/aaf191a653a8a5cf356ce25c63d74f79d3de778efa01dd717d0667cf4b9f6913.jpg)

Numbers indicate min # of steps before a state can be computed

## Attention

• Attention provides a solution to the bottleneck problem.

• Core idea: on each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence

![](images/4fde29ad04d9e525b5094ff1a091d55788b9ffd9436f3f25b9e485f3a0141f4c.jpg)

• First, we will show via diagram (no equations), then we will show with equations