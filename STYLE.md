# Style Guide

## Contents

- [Philosophy](#philosophy)
- [State Management](#state-management)
- [Code Organization](#code-organization)
- [Data Structures](#data-structures)
- [API Design](#api-design)
- [Hardware](#hardware)
- [Conventions](#conventions)
- [Metrics](#metrics)

## Philosophy

### Simplicity

Simplicity is the ultimate goal. The best metric is usefulness divided by lines of code. Lines of code matter only if the code is readable — short unreadable code is no good.

### Explicit over implicit

Use explicit return to make code more readable. The rule of thumb: if you are not sure what is going on, make the code more explicit until you know without thinking. If the code makes perfect sense and feels too verbose, make it more implicit.

### Debuggable over clean

Don't write "clean code." Write debuggable code — understandable code.

### Duplicate first, abstract later

Always duplicate. Design the user API and write a list of requirements. Then write the simplest code that makes that API work. If you need similar code in different places, copy it. Once the program passes integration tests, remove the duplication by adding abstraction. The resulting abstraction will be much less likely to need a refactor.

### Additive features (orthogonality)

The goal of good abstractions is **orthogonality** — features compose without interfering. Complexity scales linearly while capabilities scale exponentially.

Most libraries get this wrong. In PyTorch, features are **multiplicative**: adding a dtype means writing new kernels for every op on every backend. That is `num_dtypes × num_ops × num_backends` — each new feature multiplies the work.

In zyx, features are **orthogonal**: adding a dtype adds it to all backends and all ops automatically. Adding a backend implements all ops for it. Adding an op adds it to all backends and all dtypes. That is `num_dtype + num_ops + num_backends`.

This is the goal — but don't force the abstraction upfront. Duplicate first, then let the right abstraction emerge.

### Cut the trash out

Don't get slowed down by the language. Rust doesn't implement Hash, Eq, or Ord for floats — use ordered-float and accept the bad syntax. Lifetimes are rarely needed — use globals or Arc. The orphan rule makes traits painful — use procedural style like you are writing C.

If the language does not support your approach, fake it till you make it.

### Functional programming is borderline useless for this kind of work

There is a consensus that software complexity comes from state. Functional programming only allows mutation by passing variables to functions and returning new ones. This is theoretically nice, but functional languages are inherently slow and borderline useless for systems work because of this. All software can be written in FP, but without persistent state and mutations it gets annoying to write. Most software is shorter and clearer when written procedurally.

What matters is making state transparent, not eliminating it.

## State Management

Single mutable global state is fine when there are many handles pointing to it. In zyx, tensors share a single mutable graph. This works well with destructors for reference counting.

Avoid `Rc<RefCell<T>>` and `&RefCell<T>` in structs. Variables should be mutated from one place, not from a dozen structs. This prevents logic bugs.

All state should be stored in as few places as possible, ideally one, and easily accessible for debugging. In zyx, everything is in the runtime — the whole state can be inspected at any point. Functions operate on this state and have no side effects beyond changing it.

OOP pushes many small structs instead of one big one. This makes debugging and state management harder — memory leaks become difficult to spot because state is scattered across tiny classes everywhere. In zyx, memory leaks are found by simply logging the size of each field in the runtime struct: size of the graph, number of compiled programs, and such.

A small number of stateful structs with well-defined APIs is the goal. Store collections of objects inside those structs (SOA over AOS) for performance and debuggability. Structs should be written with debugging in mind — their representation should be human readable.

There is no point drawing a UML diagram of a single struct — just read its declaration and fields. Since programmers do not like drawing UML and there is no way to guarantee a diagram matches the code, write code so that UML diagrams are not needed to understand the program.

A related language design problem: many functions take `self` which grants access to the whole struct even if they only need part of it. Languages could let functions limit their access to specific fields. In the meantime, don't over-split your structs to work around this — that causes more problems than it solves.

## Code Organization

Keep modules around 1000 LOC. More is hard to track, less means code is split across too many files. Do not split the project into too many short files — short files grow, large files get refactored. Add files only when truly necessary.

Functions should only exist if called in two or more places. Otherwise put everything in one function, no matter how long. Use code blocks to make long functions understandable.

Define requirements first, then data structures, then application logic. Core data structures should map inputs to outputs as directly as possible — minimal intermediate values. Programs can only scale in preplanned directions (zyx is a tensor library, it will never be a GUI library). Primary requirements must be set in stone from the start.

## Data Structures

- **Vec over Box\<\[\]\>**: Vec is flexible, can be created from parallel iterators, and the extra 8 bytes is not worth the pain.
- **Mutex over RefCell**: Mutex works with multithreading. Use RefCell only when you are 100% certain you won't ever make it multithreaded. If you truly need the performance, use UnsafeCell.
- **Enums over dyn**: Don't use virtual tables. Use enums instead. If you need function pointers, pass them directly. Dyn can be used carefully to avoid macro hell.
- **Avoid Arc/Rc**: Not needed when using one big singleton struct. Zyx only uses Arc in the wgpu backend out of necessity.
- **Arenas**: Use arenas to group allocations for hot paths. For cold paths use whatever is easiest to type.

## API Design

The default way to do things should be the shortest to write, with many parameters implicit. Later add options for more explicitness. The underlying code should be written so that a high-performance API can be easily added on top. This is top-down API design: first the succinct high-level API, then the detailed API for power users.

## Hardware

Interface with hardware in very specific places. In zyx, each backend is fully interfaced in a single file. Putting FFI calls across multiple files is messy and makes refactoring slow. This rule is usually upheld in the first draft but carelessly broken later — which is why many projects are hard to port.

## Conventions

- **Asserts**: Use debug asserts everywhere. Any time an invariant should hold, assert it. This includes maximum sizes, memory limits, and any program constraints.
- **Documentation**: Document everything, even obvious things. Documentation is easier to remove than to add. Aim for ~20% comments (50% in complex code). Update comments as code gets more concise.
- **Inheritance**: Don't use it.
- **Returns**: Use explicit return keyword.

## Metrics

Good measurements for code:

1. **Readability** — how long it takes to understand 100 lines of code.
2. **Lines of code** — usefulness divided by LOC.
3. **Fragility** — how many other lines need to change in response to a change in one line.
4. **Call stack depth** — max 20 functions deep per module, 5 is average.
