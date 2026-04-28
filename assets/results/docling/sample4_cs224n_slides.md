## Natural Language Processing with Deep Learning CS224N/Ling284

<!-- image -->

Tatsunori Hashimoto

Lecture 8: Self-Attention and Transformers

## The starting point: mean-pooling for RNNs

<!-- image -->

- Starting point: a very basic way of 'passing information from the encoder' is to average

keys values

keys values Weighted

## Attention is weighted averaging, which lets you do lookups!

Attention is just a weighted average - this is very powerful if the weights are learned!

output

In attention , the query matches all keys softly , to a weight between 0 and 1. The keys' values are multiplied by the weights and summed.

k5

v5

<!-- image -->

In a lookup table , we have a table of keys that map to values . The query matches one of the keys, returning its value.

<!-- image -->

## Sequence-to-sequence with attention

Core idea : on each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence

<!-- image -->

## Lecture Plan

1. From recurrence (RNN) to attention-based NLP models
2. The Transformer model
3. Great results with Transformers
4. Drawbacks and variants of Transformers

## Reminders:

See the 2023 lecture notes for some bonus material

Assignment 4 due a week from today! Use Colab for the final training if you don't have a GPU.

Final project proposal out tonight, due Tuesday, Feb 14 at 4:30PM PST!

Please try to hand in the project proposal on time; we want to get you feedback quickly!

## Last last lecture: Multi-layer RNN for machine translation

[Sutskever et al. 2014; Luong et al. 2015]

The hidden states from RNN layer i are the inputs to RNN layer i + 1

<!-- image -->

Conditioning = Bottleneck SYSTRAN

## NMT: the first big success story of NLP Deep Learning

Bai du NETEASE

www.163.com

Neural Machine Translation went from a fringe research attempt in 2014 to the leading standard method in 2016

- 2014 : First seq2seq paper published [Sutskever et al. 2014]
- 2016 : Google Translate switches from SMT to NMT - and by 2018 everyone has
- This is amazing!
- SMT systems, built by hundreds of engineers over many years, outperformed by NMT systems trained by small groups of engineers in a few months

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

## The final piece: the bottleneck problem in RNNs

Encoding of the

source sentence.

<!-- image -->

Problems with this architecture?

## 1. Why attention? Sequence-to-sequence: the bottleneck problem

<!-- image -->

## Issues with recurrent models: Linear interaction distance

- O(sequence length) steps for distant word pairs to interact means:
- Hard to learn long-distance dependencies (because gradient problems!)
- Linear order of words is 'baked in'; we already know linear order isn't the right way to think about sentences…

<!-- image -->

Info of chef has gone through O(sequence length) many layers!

## Issues with recurrent models: Lack of parallelizability

- Forward and backward passes have O(sequence length) unparallelizable operations
- GPUs can perform a bunch of independent computations at once!
- But future RNN hidden states can't be computed in full before past RNN hidden states have been computed
- Inhibits training on very large datasets!

<!-- image -->

Numbers indicate min # of steps before a state can be computed

## Attention

- Attention provides a solution to the bottleneck problem.
- Core idea : on each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence
- First, we will show via diagram (no equations), then we will show with equations

<!-- image -->