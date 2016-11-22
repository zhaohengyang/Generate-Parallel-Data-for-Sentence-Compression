# Generate-Parallel-Data-for-Sentence-Compression

## Movtivation

There is a lack of paralled data set resource for sentence reduction. One could extract healine and the first sentece of each article since they are usually semantically similar too each other. However, healines syntactically quite different from normal sentence. For example, they may have no main verb, omit determiners and appear incomplete, making it hard for a supervised deletion-based system to learn useful rules. We implement the approach created by Katja Filippova's research group to create a reduction of sentence with same syntactic of original sentence using sentence / healine pairs as input. 

## Description

This is a python project that generate original sentence / reduction sentence pair. 
