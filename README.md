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

The Bombay high court on Tuesday directed the BEST staff, including drivers and conductors, to call off their strike and report to work immediately.

-------- Headline -------------------

Bombay high court directs BEST staff to call off strike & report to work

### Output sentence / reduced_sentence pair

-------- Sentence -------------------

The Bombay high court on Tuesday directed the BEST staff, including drivers and conductors, to call off their strike and report to work immediately.

-------- reduced_sentence -------------------

The Bombay high court directed the BEST staff , to call off their strike and report to work immediately
