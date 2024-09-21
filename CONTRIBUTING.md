
- The simplest contribution is writing more integration tests. These are always very appretiated.

- More documentation, cleaner documentation typo fixes are great.

- Adding new functions to Tensor. If you want to add new modules, add those to zyx-nn. If you want to add pure (stateless)
functions, add them directly to Tensor.

- Adding new backends. To add a new backend, look at existing backends. The code is pretty straightforward
and only requires adding a single file into runtime/backend forlder, but you have to make sure that your backend does not do any compile
time linking. All backends in zyx search for available .so files during runtime.

- Work on optimizations. This is most involved work, by far the hardest and it is very easy to introduce bugs. Even though
zyx has pretty comprehensive integration test suite, no test suite can catch all possible bugs. Thus unless your code
is very easy to understand, it will probably get rejected. However if you can produce extremely good code that significantly
increases the performance of at least some devices (>10% perf improvement), then this is the best thing you can do to
help zyx grow.

- Find bugs - finding a bug is amazing news, because correctness is no. ! goal

- Fix bugs - this is good too, but finding new bugs is even better
