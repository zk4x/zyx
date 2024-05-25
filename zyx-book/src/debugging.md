# Debugging

Zyx removes a number of PyTorch errors. Zyx tensors are immutable, so there is no:
- RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: ... , which is output 0 of TBackward, is at version 2; expected version 1 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!

Zyx technically allows mutability of tensors using set method, setting values of tensor A to tensor B, but tensors are just pointers, so this means merely that tensor B will now point to values previously pointed to be tensor A and tensor A will not exist anymore.

Another error that cannot occur:
- RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed. Specify retain_graph=True when calling backward the first time.

Zyx does not store intermediate tensors, so they cannot be freed :)

# Visualization

One aspect of debugging which is often overlooked is visual representation of graph. Programmers often like reading code more than looking at visualizatinos, but in particular if you are using complex modules defined somewhere outside of your code, it may be beneficial to be able to look at any part of the graph visually.

Zyx asks you to give it any number of tensors and then plots all relations between them into picture. Let x, y and z be tensors.
```rust
let dot_graph = Tensor::plot_graph([&x, &y, &z]);
fs::write("graph.dot", dot_graph).unwrap();
```

If you want to see just forward part of graph, you can do for example this:
```rust
let dot_graph = Tensor::plot_graph(model.into_iter().chain([&x, &loss]));
```
Where model is your model, x is your input and loss is your loss/error.

If you want to only look at the backward part of graph, that is also simple:
```rust
let dot_graph = Tensor::plot_graph(grads.chain([&loss]));
```

Zyx will order nodes automatically, so there is no difference in the order in which tensors are stored in the iterator.

