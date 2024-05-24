# These are notes on style used in zyx and possibly general coding practices

Don't use dyn, or virtual tables, just use enums instead. If there is necessity to work with function pointers,
then pass those function pointers around directly.

Single mutable global state variable is OK, if there is lot of handles that point to it. In zyx tensors
share single mutable graph, for example in ECS this could be Entities sharing single world struct. This works
well with destructors for reference counting and such.

Avoid references in structs as much as possible. variables should be mutated from single place, not from dozen
structs. This helps prevent logic bugs.

Simplicity is the ultimate goal, probably the best metric is usefulness of program divided by number of lines
of code. Lines of code however matter only if the code is readable. Short unreadable code is no good.

Use debug asserts everywhere to check the code. Anytime some invariant should hold, just put in an assert.
This goes for maximum size of numbers, maximum memory usage and any limits that are put on the program.

Always use documentation, even if documenting obvious things. Documentation is much easier to remove than to add.
Code that seems obvious to the person writing it is not obvious to the person reading it. In general most functions
longer than five lines should have comments. In general it is a good ratio of 1 line of comments per 5 lines of code,
that is ~20% percent should be comments, but more like 50% in complex code and algorithms.

Always use explicit return keyword, just to make code more readable.

Try to not get slowed down by the inferiority of your language. For example Rust does not implement Hash or Eq, Ord
for floats. Just use ordered float crate and accept everything bad about it (horrible syntax, slower performance).
Another example is lifetimes. Try to use lifetimes very rarely, do not go into complex lifetime annotations, just
use global variables or Rc/Arc. Another example is orphan rule, just do to not use traits if possible. Use procedural
programming like if you are writing c.
