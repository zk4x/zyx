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

Just use arenas to group allocations for high performance code.

## API design

The default way to do things should be the way that is shortest to write and many parameters can be implicit. Later options for more explicitness should be added. The underlying code should be however written in such a way, that at last API for high performance usage can be easily added.
This is top down approach in API design. First write high level API that is succint and later add more detailed API for high performance users.
