
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

Higher level abstraction, basically with tiles, views, bound dimensions and loops
On this level we can do ops reordering, reshapes, permutes and pads can be moved even
with binary ops, while expands can be only reordered between unary ops.

So we need to add loops,
mark movement of leafs between global, local and register tiles
Add loops for expands bigger than work size,
those must loops end with reduce ops, since initial work size (global and local loops)
is calculated with output size in mind.
Apply movement ops directly on tiles,
Leave unary and binary ops as they are.
Here we can do all the local and register tile optimizations and have special instructions
for things like 4x4x4 matmuls, where we can just use tensor cores, wmma cores or like strassen

For more low level (this needs to be rewritten once the more higher level approach is finalized
Nodes like this (don't worry to make some assumptions)

If values is in global scope
  -> create new register mem
  -> copy data from global into register (TODO use wide loads if it increases performance)
Exp, Tanh, Neg, Sin, Cos
  -> create new register mem
  -> add instruction for unary op
  -> decrease reference count from mems
ReLU
  -> same as unary, just rewrite as binary max with const
Add, Sub, Mul, Div, Pow, Cmplt
  -> same as unary
Sum, Max
  -> mark reduce start before first movement op on any of the leafs
  -> create reduce loop
  -> apply


As opencl kernel

let x = dev.randn([1024, 2048], DType::F32)?;
let z = (&x + x.exp()).sum(-1);

AST (simple version, no optimizations):
 0 loop global 1024
 1 loop global 2048
 2 move global to register from id 0, view contiguous
 3 exp 2
 4 add 2, 3
 5 loop
this seems very complicated
 4
 3 move global to register from id 0, view contiguous, bind existing ids
 2 add 4, 3
 1 sum (end loop) mark bind idx1 to 1, redefine id 1
 0 move register to global, view contiguous (mark bind idx0 to dimension 0, idx1 to 1)

```c
float rmem0 = data0[];
float rmem1 = exp(rmem0);
// x has the same view on both sides of binary, so no need for second load
float rmem2 = rmem0 + rmem1;
```

Perhaps just do standard ASTs as in 0.12.1 and just join them together?

## Conclusion

We are using tiled kernels. These begin with reduce, binary or movement op. All consecutive unary ops are joined
into single tile. Tiles with the same work size are joined together unless some tile is used more than once.
Tiles with results used more than once are evaluated separately, so that their results can be reused.
This should be a good compromise. Unary ops are usually cheap enough to be worth recalculating, but other ops
usually are not worth recalculating. Tiled version does not contain loops.

# 2024-05-30

We need to write from IR to OpenCL compiler.
Then finish IR compiler without optimizations.
Then we need to write extensive tests to check that it works correctly.
Only after everything works fine can we start optimizing with local memory and register tiling and such.
