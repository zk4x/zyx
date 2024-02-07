# Execution Model

PyTorch executes most of the ops immediatelly. This straightforward, but it means that in order to be able to backpropagate, it needs to know which tensors must be stored in memory. Zyx uses lazy execution. It does not evaluate anything until user explicitly requests the data. This would not work for training/inference loops, so zyx uses caching mechanism that detects repetition of parts of graph and once any part of graph is repeated more than once, the whole graph get evaluated in order to remove no longer needed nodes (node is internal representation of unrealized tensor, takes only few bytes).

This is cool by itself, because it means that zyx is as dynamic as pytorch while allowing for optimizations only possible in static graphs, but it also allows zyx to be more dynamic than PyTorch, because there is no longer need to specify which tensors require gradient, as you can see in [autograd](autograd.md) chapter.
