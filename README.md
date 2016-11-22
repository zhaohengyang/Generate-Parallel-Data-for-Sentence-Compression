# Generate-Parallel-Data-for-Sentence-Compression

## Movtivation

There is a lack of paralled data set resource for sentence reduction. One could extract healine and the first sentece of each article since they are usually semantically similar too each other. However, healines syntactically quite different from normal sentence. For example, they may have no main verb, omit determiners and appear incomplete, making it hard for a supervised deletion-based system to learn useful rules. We implement the approach created by Katja Filippova's research group to create a reduction of sentence with same syntactic of original sentence using sentence / healine pairs as input. 

## Description

This is a python project that generate original sentence / reduction sentence pair. 

## Output

Usage demostration and visualization are in demo.ipynb file

## Example

### Input sentence / headline pair

-------- Sentence -------------------

Pakistan said on Thursday it has decided to resume cross-border trade in Kashmir after weeks of suspension over the arrest of a Pakistani driver by the Indian authorities on drugs smuggling charges.

-------- Headline -------------------

Pakistan to resume cross-border trade in Kashmir

### Output sentence / reduced_sentence pair

-------- Sentence -------------------

Pakistan said on Thursday it has decided to resume cross-border trade in Kashmir after weeks of suspension over the arrest of a Pakistani driver by the Indian authorities on drugs smuggling charges.

-------- Reduced_sentence -------------------

Pakistan has decided to resume cross - border trade in Kashmir

![Alt text](/reduction_tree.png?raw=true "Optional Title")
