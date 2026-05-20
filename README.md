# Frog
Frog makes QEC.

How to use:
+ You can play with examples in examples folder. It uses all main routines and objects from code, evaluate thresholds of decoders and so on.
+ Structure of code is following:
  + Decoding is subject of code, noise channel and decoder. Each of this objects defined as class with abstract interface (abc class).
    Than each class follows this interface. It worth to keep it.
  + Experiment, modules, net_modules, loss, trainer contain code related to training/learnable decoder.

Paper:
+ [The END: An Equivariant Neural Decoder for Quantum Error Correction](https://arxiv.org/abs/2304.07362)
Many thanks to Roberto Bondesan and Max Welling
