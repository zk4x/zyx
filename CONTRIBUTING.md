# Contributing

Check out [good first issues](https://github.com/zk4x/zyx/labels/good%20first%20issue) for a place to start.

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

- Find bugs - finding a bug is amazing news, because correctness is no. 1 goal

- Fix bugs - this is good too, but finding new bugs is even better

## License

zyx is licensed under `LGPL-3.0-only WITH Classpath-exception-2.0` (see [LICENSE.md](LICENSE.md)).

By contributing code, you agree that your contribution is licensed under the same license — `LGPL-3.0-only WITH Classpath-exception-2.0` — and you retain copyright on your contribution. A `Signed-off-by` line in your commit (Developer Certificate of Origin, `git commit -s`) certifying that you have the right to submit the contribution is required. No CLA is needed.
