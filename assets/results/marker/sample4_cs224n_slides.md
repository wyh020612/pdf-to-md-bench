# **Natural Language Processing with Deep Learning CS224N/Ling284**

![](_page_0_Picture_1.jpeg)

Tatsunori Hashimoto

Lecture 8: Self-Attention and Transformers

### **The starting point: mean-pooling for RNNs**

![](_page_1_Picture_1.jpeg)

• Starting point: a *very* basic way of 'passing information from the encoder' is to *average*

### **Attention is** *weighted* **averaging, which lets you do lookups!**

Attention is just a **weighted** average – this is very powerful if the weights are learned!

In **attention**, the **query** matches all **keys** *softly*, to a weight between 0 and 1. The keys' **values** are multiplied by the weights and summed.

In a **lookup table**, we have a table of **keys** that map to **values**. The **query** matches one of the keys, returning its value.

![](_page_2_Figure_5.jpeg)

### **Sequence-to-sequence with attention**

**Core idea**: on each step of the decoder, use *direct connection to the encoder* to *focus on a particular part* of the source sequence

![](_page_3_Picture_2.jpeg)

![](_page_3_Picture_3.jpeg)

### **Lecture Plan**

- 1. From recurrence (RNN) to attention-based NLP models
- 2. The Transformer model
- 3. Great results with Transformers
- 4. Drawbacks and variants of Transformers

#### Reminders:

See the [2023 lecture notes](http://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf) for some bonus material

Assignment 4 due a week from today! Use Colab for the final training if you don't have a GPU.

Final project proposal out tonight, due Tuesday, Feb 14 at 4:30PM PST!

Please try to hand in the project proposal on time; we want to get you feedback quickly!

### **Last last lecture: Multi-layer RNN for machine translation**

[Sutskever et al. 2014; Luong et al. 2015] The hidden states from RNN layer *i* are the inputs to RNN layer *i* + 1

![](_page_5_Figure_3.jpeg)

Conditioning = Bottleneck

### **NMT: the first big success story of NLP Deep Learning**

Neural Machine Translation went from a fringe research attempt in **2014** to the leading standard method in **2016**

- **2014**: First seq2seq paper published [Sutskever et al. 2014]
- **2016**: Google Translate switches from SMT to NMT and by 2018 everyone has

![](_page_6_Picture_4.jpeg)

![](_page_6_Picture_5.jpeg)

![](_page_6_Picture_6.jpeg)

![](_page_6_Picture_7.jpeg)

![](_page_6_Picture_8.jpeg)

![](_page_6_Picture_9.jpeg)

![](_page_6_Picture_10.jpeg)

![](_page_6_Picture_11.jpeg)

- This is amazing!
  - **SMT** systems, built by hundreds of engineers over many years, outperformed by NMT systems trained by small groups of engineers in a few months

### **The final piece: the bottleneck problem in RNNs**

![](_page_7_Figure_1.jpeg)

**Problems with this architecture?**

### **1. Why attention? Sequence-to-sequence: the bottleneck problem**

![](_page_8_Figure_1.jpeg)

## Issues with recurrent models: **Linear interaction distance**

- **O(sequence length)** steps for distant word pairs to interact means:
  - Hard to learn long-distance dependencies (because gradient problems!)
  - Linear order of words is "baked in"; we already know linear order isn't the right way to think about sentences…

![](_page_9_Figure_4.jpeg)

Info of *chef* has gone through O(sequence length) many layers!

#### Issues with recurrent models: Lack of parallelizability

- Forward and backward passes have O(sequence length)\nunparallelizable operations
  - GPUs can perform a bunch of independent computations at once!
  - But future RNN hidden states can't be computed in full before past RNN hidden states have been computed
  - Inhibits training on very large datasets!

![](_page_10_Figure_5.jpeg)

Numbers indicate min # of steps before a state can be computed

### **Attention**

• **Attention** provides a solution to the bottleneck problem.

• **Core idea**: on each step of the decoder, use *direct connection to the encoder* to *focus on a particular part* of the source sequence

![](_page_11_Picture_3.jpeg)

• First, we will show via diagram (no equations), then we will show with equations