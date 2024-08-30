# These are notes on style used in zyx and possibly general coding practices

## On virtual tables and other junk

Don't use dyn, or virtual tables, just use enums instead. If there is necessity to work with function pointers,
then pass those function pointers around directly.

Single mutable global state variable is OK, if there is lot of handles that point to it. In zyx tensors
share single mutable graph, for example in ECS this could be Entities sharing single world struct. This works
well with destructors for reference counting and such.

Avoid references in structs as much as possible. variables should be mutated from single place, not from dozen
structs. This helps prevent logic bugs.

Simplicity is the ultimate goal, probably the best metric is usefulness of program divided by number of lines
of code. Lines of code however matter only if the code is readable. Short unreadable code is no good.

Inheritance is also bad. Do not use it.

## Asserts

Use debug asserts everywhere to check the code. Anytime some invariant should hold, just put in an assert.
This goes for maximum size of numbers, maximum memory usage and any limits that are put on the program.
Debug asserts in rust sometimes do not work. So either create new cargo feature for it or just use normal asserts.

## Code should make sense

Always use documentation, even if documenting obvious things. Documentation is much easier to remove than to add.
Code that seems obvious to the person writing it is not obvious to the person reading it. In general most functions
longer than five lines should have comments. In general it is a good ratio of 1 line of comments per 5 lines of code,
that is ~20% percent should be comments, but more like 50% in complex code and algorithms.

## Explicit over implicit

Always use explicit return keyword, just to make code more readable. Some operations can be implicit, like casting.
The rule of thumb is: If I am not sure what is going on, I need to make the code more explicit until I know what is
going on without thinking about it. The code should intuitivelly make sence. If it does not, get more explicit.
If the code makes perfect sence and feels too verbose, do not be afraid to make it more implicit.

## Cut the trash out

Try to not get slowed down by the inferiority of your language. For example Rust does not implement Hash or Eq, Ord
for floats. Just use ordered float crate and accept everything bad about it (horrible syntax, slower performance).
Another example is lifetimes. Try to use lifetimes very rarely, do not go into complex lifetime annotations, just
use global variables or Rc/Arc. Another example is orphan rule, just do to not use traits if possible. Use procedural
programming like if you are writing c.


## Prefer functionality over performance when picking datastructres

Use Vec instead of Box<[]> even on immutable data structures. Box<[]> simply isn't flexible enough. Box<[]> can not
be created from parallel iterator. Many other libraries will return Vec<> and converting that to Box<[]> is expensive.
Just use Vec and accept that it takes 24 bytes instead of 16 bytes (on 64 bit machines).

## Functions should exist only if called in two or more places

Creating new functions should be done when there is code that is duplicated in two places. Otherwise put everything into one function,
no matter how long it is. In other words, function should not exist if it is called only in one place.

## Memory

Just use arenas to group allocations for high performance code. For low performance code just use whatever is easiest to type.

## API design

The default way to do things should be the way that is shortest to write and many parameters can be implicit. Later options for more explicitness should be added. The underlying code should be however written in such a way, that at last API for high performance usage can be easily added.
This is top down approach in API design. First write high level API that is succint and later add more detailed API for high performance users.

## Interfacing with hardware

Interfaces to hardware should be done in very specific places. In zyx each backend is fully interfaced with in single file. Putting function calls through FFI
to different files would make it very messy. This is important rule and makes refactoring fast (keeps the code flexible under changing requirements).
This rule seems obvious and intuitive. It is usually uphold in the first draft of the file structure of the project, but in later stages it is often
carelessly broken. This is the reason why many projects are hard to get working with different hardware even though the hardware fullfills virtually
the same role as the hardware the software was written for in the first place.

## On state

There is a consensus that the complexity of software comes from existence of state. Functional programming only allows for mutation by passing variables to functions and returning new variables.
This is a nice property, but functional languages are inherently slow and borderline useless due to this. Highly complex optimizations can be applied to make functional languages just as fast
as procedural/oop languages. All software in the world can also be written in functional programming languages, but without persistent state and mutations it gets annoying to write. Most software
is shorter (as in LOC) when it is written in procedural/oop way.
What I advocate for is transparent stateful programming. That is all state should be stored in very few places, ideally in one place and should be easily accessible for debugging. Functions
operate on this state and they do not have side effects except for changing the state. In zyx, everything is stored in runtime, so the whole state of zyx can be debugged at any point and all
functions do local contained calculations. Nodes are added to graph by calling function on runtime. Backpropagation adds necessary nodes to graph and nothing else. Realization evaluates required
tensors and nothing else. Realization has multiple steps where it creates kernels, schedules them to devices and compiles kernels first to ir and then to assembly, but in the end there is only
program stored on device and all intermediate state (such as IRKernel and VOps) is deleted.
One improvement for programming languages could be a simpler way to document which state is changed by each function. For example many functions take self as argument, which gives them access
to the whole struct even if they only need to access part of it. It would be nice if each function could limit it's access only to certain fields of struct.
OOP tries to push many small structs instead of one big one. This makes debugging and state management difficult. Instead it makes memory leaks hard to spot. Memory leaks in zyx can be simply found
by logging the size of each field in runtime struct. That is size of the graph, number of compiled programs and such (FFI can cause memory leaks which are hardest to spot, so we limit FFI to single file
per each backend).

So the approach should be small number of statefull structs with well defined APIs. Collections of objects should be preferrably stored in those structs (preffer SOA instead of AOS) both for performance,
but mainly for ease of debugging.

Structs themeselves should be written with debugging in mind. The representation of state should be human readable.

Basically there is no point in drawing UML diagram of single struct. Just read the declaration of the struct and it's fields. Since programmers do not like drawing UML diagrams
and there is almost no way to guarantee that UML diagram is accurate representation of the code (other than automatically generating UML diagrams from code), it is better
to write code in such a way that UML diagrams are not needed to understand the whole program.

If all programs stored their whole state in single struct, they would be very transparent. As a side effect this makes maintaining software and dealing with changing requirements surprisingly easy.

## Clean code

Don't write clean code, rather write debuggable and understandable code.
