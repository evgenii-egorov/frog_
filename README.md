# Frog
Frog makes QEC.



# How to use:
+ You can explore the examples in the `examples_notebook/` folder. These notebooks use the main routines and objects from the codebase, including code construction, symmetry actions, coset decoding, and threshold evaluation. 
+ Structure of code is following:
  + Decoding is subject of code, noise channel and decoder. Each of this objects defined as class with abstract interface (abc class).
    Than each class follows this interface. It worth to keep it.
  + Experiment, modules, net_modules, loss, trainer contain code related to training/learnable decoder.

##

## Example notebooks

The `examples_notebook/` folder contains small Jupyter notebooks demonstrating the main components of Frog and the experiments used for equivariant neural decoding of quantum error-correcting codes.

- [`_draw_code.ipynb`](examples_notebook/_draw_code.ipynb)  
  Visualizes the lattice/code structure and the objects used by the decoder. Useful as a first notebook for understanding the geometry of the problem.

- [`_symmetries_action.ipynb`](examples_notebook/_symmetries_action.ipynb)  
  Demonstrates how code symmetries act on syndromes, errors, and decoder inputs. This is the main notebook for understanding the equivariance structure used in END.

- [`_test_translation.ipynb`](examples_notebook/_test_translation.ipynb)  
  Checks translation actions and verifies that the implemented symmetry transformations behave consistently.

- [`_coset_decoder.ipynb`](examples_notebook/_coset_decoder.ipynb)  
  Shows how the coset decoder is constructed and tested. Useful for comparing neural decoding against a structured/reference decoding procedure.

- [`threshold_evaluation.ipynb`](examples_notebook/threshold_evaluation.ipynb)  
  Runs threshold-style evaluation experiments for decoders across different physical error rates and code sizes.

# Paper:
+ [The END: An Equivariant Neural Decoder for Quantum Error Correction](https://arxiv.org/abs/2304.07362)
Many thanks to Roberto Bondesan and Max Welling
