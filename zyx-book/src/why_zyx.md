# Why zyx?

Zyx was created as a learning exercise to understand machine learning from low level, close to the metal perspective.
As the time went on, I saw the lack of good ML libraries in Rust ecosystem which meant that if zyx got completed,
other people could use it too.

When researching architecture of other ML libraries, in particular the most popular ones - pytorch and tensorflow,
I found that their creators made certain compromises in order to simplify the development and reach the widest
possible audience.

These days we have a pretty good perspective on what a good ML library should look like. It should run on all hardware
with solid performance, it should not take too much disk space or memory and it should be easy to use.
Crucially it does not need to support many operations, just a few unary, binary, movement and reduce operations
are plenty, as shown by tinygrad.

In zyx, core operations are rarely added, but adding backends is very simple. Zyx automatically optimizes
for backpropagation and uses as little memory as possible during both training and inference.

And as for the ease of use? I want you to be the judge of that.

