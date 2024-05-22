
# 2024-05-21

## Runtime realize function

Realizes all tensors given as parameters.
This function creates graph of nodes that are needed for evaluation of given tensors.
This graph is copied and cached so that it can be compared and reused.
Program is compiled from this graph and the program is cached.

Then we need to write tiled representation, which needs to always create 3d tiles.

Then we need to write ir/looped representation

Then we need to write ir to opencl compiler

Then we need to optimize looped representation

# 2024-05-22

Move tiled from zyx-compiler to zyx.
