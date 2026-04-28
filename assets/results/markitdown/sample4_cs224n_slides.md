Natural Language Processing
with Deep Learning
CS224N/Ling284

Tatsunori Hashimoto

Lecture 8: Self-Attention and Transformers

The starting point: mean-pooling for RNNs

positive

How to compute
sentence encoding?

Sentence
encoding

Usually better:
Take element-wise
max or mean of all
hidden states

overall

I

enjoyed

the

movie

a

lot

• Starting point: a very basic way of ‘passing information from the encoder’ is to average

10

Attention is weighted averaging, which lets you do lookups!

Attention is just a weighted average – this is very powerful if the weights are learned!

In attention, the query matches all keys softly,
to a weight between 0 and 1. The keys’ values
are multiplied by the weights and summed.

In a lookup table, we have a table of keys
that map to values. The query matches
one of the keys, returning its value.

11

Sequence-to-sequence with attention
Core idea: on each step of the decoder, use direct connection to the encoder to focus on a
particular part of the source sequence

dot product

n
o
i
t
n
e
t
t
A

r
e
d
o
c
n
E

s
e
r
o
c
s

N
N
R

il           a         m’      entarté

<START>

12

Source sentence (input)

D
e
c
o
d
e
r
R
N
N

Lecture Plan

1. From recurrence (RNN) to attention-based NLP models
2. The Transformer model
3. Great results with Transformers
4. Drawbacks and variants of Transformers
Reminders:

See the 2023 lecture notes for some bonus material
Assignment 4 due a week from today! Use Colab for the final training if you don’t have
a GPU.
Final project proposal out tonight, due Tuesday, Feb 14 at 4:30PM PST!
Please try to hand in the project proposal on time; we want to get you feedback
quickly!

2

Last last lecture: Multi-layer RNN for machine translation
[Sutskever et al. 2014; Luong et al. 2015]

The hidden states from RNN layer i
are the inputs to RNN layer i + 1

The      protests  escalated    over         the      weekend   <EOS>

Translation
generated

0.1
0.3
0.1
-0.4
0.2

0.2
-0.2
-0.1
0.1
0.1

0.2
0.6
-0.1
-0.7
0.1

0.2
0.6
-0.1
-0.7
0.1

0.2
0.6
-0.1
-0.7
0.1

0.4
-0.6
0.2
-0.3
0.4

0.4
0.4
0.3
-0.2
-0.3

0.1
0.3
-0.1
-0.7
0.1

0.2
-0.3
-0.1
-0.4
0.2

0.5
0.5
0.9
-0.3
-0.2

0.2
0.6
-0.1
-0.4
0.1

0.2
0.4
0.1
-0.5
-0.2

0.2
0.6
-0.1
-0.5
0.1

0.2
-0.8
-0.1
-0.5
0.1

0.4
-0.2
-0.3
-0.4
-0.2

-0.1
0.6
-0.1
-0.7
0.1

0.2
0.6
-0.1
-0.7
0.1

0.2
0.6
-0.1
-0.7
0.1

0.2
0.6
-0.1
-0.7
0.1

0.2
-0.1
-0.1
-0.7
0.1

0.2
0.6
-0.1
-0.7
0.1

0.2
0.6
-0.1
-0.7
0.1

-0.4
0.6
-0.1
-0.7
0.1

0.2
0.6
-0.1
-0.7
0.1

0.3
0.6
-0.1
-0.7
0.1

0.2
0.6
-0.1
0.3
0.1

0.2
0.6
-0.1
-0.7
0.1

0.4
0.4
-0.1
-0.7
0.1

-0.1
0.6
-0.1
0.3
0.1

-0.1
0.3
-0.1
-0.7
0.1

-0.2
0.6
-0.1
-0.7
0.1

0.2
0.4
-0.1
0.2
0.1

-0.2
0.6
0.1
0.3
0.1

-0.4
0.6
-0.1
-0.7
0.1

0.3
0.6
-0.1
-0.5
0.1

-0.4
0.5
-0.5
0.4
0.1

-0.3
0.5
-0.1
-0.7
0.1

0.2
0.6
-0.1
-0.7
0.1

0.2
0.6
-0.1
-0.7
0.1

Die       Proteste    waren    am  Wochenende eskaliert <EOS>      The       protests    escalated    over          the     weekend

Decoder

Feeding in
last word

Conditioning =
Bottleneck

Encoder:
Builds up
sentence
meaning

Source
sentence

3

NMT: the first big success story of NLP Deep Learning

Neural Machine Translation went from a fringe research attempt in 2014 to the leading
standard method in 2016

• 2014: First seq2seq paper published [Sutskever et al. 2014]

• 2016: Google Translate switches from SMT to NMT – and by 2018 everyone has

• This is amazing!

• SMT systems, built by hundreds of engineers over many years, outperformed by

NMT systems trained by small groups of engineers in a few months

4

The final piece: the bottleneck problem in RNNs

Encoding of the
source sentence.

Target sentence (output)

he        hit        me       with        a          pie    <END>

N
N
R
r
e
d
o
c
n
E

5

D
e
c
o
d
e
r
R
N
N

il           a         m’      entarté

<START>    he        hit        me       with        a         pie

Source sentence (input)

Problems with this architecture?

1. Why attention? Sequence-to-sequence: the bottleneck problem

Encoding of the
source sentence.
This needs to capture all
information about the
source sentence.
Information bottleneck!

N
N
R
r
e
d
o
c
n
E

Target sentence (output)

he        hit        me       with        a          pie    <END>

D
e
c
o
d
e
r
R
N
N

il           a         m’      entarté

<START>    he        hit        me       with        a         pie

Source sentence (input)

6

Issues with recurrent models: Linear interaction distance

• O(sequence length) steps for distant word pairs to interact means:
• Hard to learn long-distance dependencies (because gradient problems!)
• Linear order of words is “baked in”; we already know linear order isn’t the

right way to think about sentences…

The

chef who  …

was

Info of chef has gone through
O(sequence length) many layers!

7

Issues with recurrent models: Lack of parallelizability

• Forward and backward passes have O(sequence length)

unparallelizable operations
• GPUs can perform a bunch of independent computations at once!
• But future RNN hidden states can’t be computed in full before past RNN

hidden states have been computed
• Inhibits training on very large datasets!

3

2

1

0

2

1

h1

h2

n

hT

8

Numbers indicate min # of steps before a state can be computed

Attention

• Attention provides a solution to the bottleneck problem.

• Core idea: on each step of the decoder, use direct connection to the encoder to focus

on a particular part of the source sequence

• First, we will show via diagram (no equations), then we will show with equations

9

