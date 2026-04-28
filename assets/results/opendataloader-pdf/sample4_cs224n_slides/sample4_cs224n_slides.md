# Natural Language Processing with Deep Learning CS224N/Ling284

![image 1](sample4_cs224n_slides_images/imageFile1.png)

Tatsunori Hashimoto Lecture 8: Self-Attention and Transformers

### The starting point: mean-pooling for RNNs

How to compute sentence encoding?

###### positive

Usually better: Take element-wise max or mean of all hidden states

Sentence encoding

overall I enjoyed the movie a lot

• Starting point: a very basic way of ‘passing information from the encoder’ is to average

### Attention is weighted averaging, which lets you do lookups!

Attention is just a weighted average – this is very powerful if the weights are learned!

In a lookup table, we have a table of keys that map to values. The query matches one of the keys, returning its value.

In attention, the query matches all keys softly, to a weight between 0 and 1. The keys’ values are multiplied by the weights and summed.

![image 2](sample4_cs224n_slides_images/imageFile2.png)

![image 3](sample4_cs224n_slides_images/imageFile3.png)

![image 4](sample4_cs224n_slides_images/imageFile4.png)

### Sequence-to-sequence with attention

##### Core idea: on each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence

dot product

Attention

scores

DecoderRNN

Encoder

RNN

il a m’ entarté <START>

Source sentence (input)

### Lecture Plan

- 1. From recurrence (RNN) to attention-based NLP models
- 2. The Transformer model
- 3. Great results with Transformers
- 4. Drawbacks and variants of Transformers Reminders:


See the 2023 lecture notes for some bonus material Assignment 4 due a week from today! Use Colab for the final training if you don’t have a GPU. Final project proposal out tonight, due Tuesday, Feb 14 at 4:30PM PST! Please try to hand in the project proposal on time; we want to get you feedback quickly!

### Last last lecture: Multi-layer RNN for machine translation

[Sutskever et al. 2014; Luong et al. 2015]

|The hidden states from RNN layer i are the inputs to RNN layer i + 1|
|---|


Translation generated

The protests escalated over the weekend <EOS>

| | |
|---|---|
|0.2 0.6<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.2 0.6<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.3 0.6<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|-0.4 0.6<br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.4 0.4<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|-0.2 0.6<br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|-0.3 0.5<br>-0.1<br>-0.7 0.1<br>| |


![image 5](sample4_cs224n_slides_images/imageFile5.png)

![image 6](sample4_cs224n_slides_images/imageFile6.png)

|0.2 0.6<br><br>-0.1<br>-0.5 0.1<br>|
|---|


|-0.1 0.6<br>-0.1<br>-0.7 0.1<br>|
|---|


|0.4 0.4 0.3<br><br>-0.2<br>-0.3<br>|
|---|


|0.5 0.5 0.9<br><br>-0.3<br>-0.2<br>|
|---|


|0.1 0.3 0.1<br><br>-0.4 0.2|
|---|


|0.2 0.6<br><br>-0.1<br>-0.7 0.1<br>|
|---|


Encoder: Builds up sentence meaning

| | |
|---|---|
|-0.4 0.6<br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.2<br><br>-0.1<br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.2 0.6<br><br>-0.1 0.3 0.1| |


| | |
|---|---|
|0.2<br><br>-0.8<br>-0.1<br>-0.5 0.1<br>| |


| | |
|---|---|
|0.2 0.6<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.2 0.6<br><br>-0.1<br>-0.4 0.1<br>| |


| | |
|---|---|
|0.3 0.6<br><br>-0.1<br>-0.5 0.1<br>| |


| | |
|---|---|
|0.1 0.3<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|-0.1 0.6<br>-0.1 0.3 0.1<br>| |


| | |
|---|---|
|0.2 0.4<br><br>-0.1 0.2 0.1| |


| | |
|---|---|
|0.2 0.6<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.2<br><br>-0.2<br>-0.1 0.1 0.1<br>| |


| | |
|---|---|
|0.2 0.6<br><br>-0.1<br>-0.7 0.1<br>| |


Decoder

| | |
|---|---|
|0.2 0.6<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.2 0.6<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.2 0.6<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.4<br><br>-0.2<br>-0.3<br>-0.4<br>-0.2<br>| |


| | |
|---|---|
|0.2 0.6<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.2 0.4 0.1<br><br>-0.5<br>-0.2<br>| |


| | |
|---|---|
|-0.1 0.3<br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|-0.2 0.6 0.1 0.3 0.1| |


| | |
|---|---|
|-0.4 0.5<br>-0.5 0.4 0.1<br>| |


| | |
|---|---|
|0.2<br><br>-0.3<br>-0.1<br>-0.4 0.2<br>| |


| | |
|---|---|
|0.2 0.6<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.2 0.6<br><br>-0.1<br>-0.7 0.1<br>| |


| | |
|---|---|
|0.4<br><br>-0.6 0.2<br>-0.3 0.4<br>| |


Source sentence

Feeding in last word

Die Proteste waren am Wochenende eskaliert <EOS> The protests escalated over the weekend

Conditioning = Bottleneck

### NMT: the first big success story of NLP Deep Learning

Neural Machine Translation went from a fringe research attempt in 2014 to the leading standard method in 2016

- • 2014: First seq2seq paper published [Sutskever et al. 2014]
- • 2016: Google Translate switches from SMT to NMT – and by 2018 everyone has
- • This is amazing!


![image 7](sample4_cs224n_slides_images/imageFile7.png)

![image 8](sample4_cs224n_slides_images/imageFile8.png)

![image 9](sample4_cs224n_slides_images/imageFile9.png)

![image 10](sample4_cs224n_slides_images/imageFile10.png)

![image 11](sample4_cs224n_slides_images/imageFile11.png)

![image 12](sample4_cs224n_slides_images/imageFile12.png)

![image 13](sample4_cs224n_slides_images/imageFile13.png)

![image 14](sample4_cs224n_slides_images/imageFile14.png)

• SMT systems, built by hundreds of engineers over many years, outperformed by NMT systems trained by small groups of engineers in a few months

### The final piece: the bottleneck problem in RNNs

Encoding of the source sentence.

Target sentence (output)

he hit me with a pie <END>

EncoderRNN

DecoderRNN

il a m’ entarté <START> he hit me with a pie

Source sentence (input)

|Problems with this architecture?|
|---|


### 1. Why attention? Sequence-to-sequence: the bottleneck problem

Encoding of the source sentence. This needs to capture all information about the source sentence. Information bottleneck!

Target sentence (output)

he hit me with a pie <END>

EncoderRNN

DecoderRNN

il a m’ entarté <START> he hit me with a pie

Source sentence (input)

## Issues with recurrent models: Linear interaction distance

#### • O(sequence length) steps for distant word pairs to interact means:

- • Hard to learn long-distance dependencies (because gradient problems!)
- • Linear order of words is “baked in”; we already know linear order isn’t the right way to think about sentences…


The chef who … was

|Info of chef has gone through O(sequence length) many layers!|
|---|


## Issues with recurrent models: Lack of parallelizability

#### • Forward and backward passes have O(sequence length) unparallelizable operations

- • GPUs can perform a bunch of independent computations at once!
- • But future RNN hidden states can’t be computed in full before past RNN hidden states have been computed
- • Inhibits training on very large datasets!


- 0
- 1 n


- 1
- 2


- 2
- 3


h2 hT

h1

|Numbers indicate min # of steps before a state can be computed|
|---|


### Attention

- • Attention provides a solution to the bottleneck problem.
- • Core idea: on each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence
- • First, we will show via diagram (no equations), then we will show with equations


![image 15](sample4_cs224n_slides_images/imageFile15.png)

