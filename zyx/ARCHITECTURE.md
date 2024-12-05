# Architecture

This document describes limitations and design choices behind zyx.

## Hardware support

Zyx supports many hardware backends through singular intermediate representation, which is very close to assembly
and simply uses enum as an op.

## Performance

Zyx creates graph of nodes at runtime. Kernel are generated from the graph and immediatelly asynchronously launched
during realization. Optionally these kernels can be optimized before launching.

## Error handling

Zyx uses simple method to differentiate when to panic and when to return a result. If a user provides incorrect
input, function should return result. Everything else should be panic. That is if zyx panics, it is a bug.
All panics and assertions (there are many and we will add more) check for consistency of invariants that cannot
be broken. Breaking any of these invariants puts zyx into irrecoverable state, thus the only right thing to do
is to immediately stop execution. One other option when panic happens is in case of hardware failure.
Zyx already detects hardware devices at runtime and disallows explicit programming for cpu or gpu only.
However zyx currently assumes that hardware configuration stays constant as long as at least one tensor exists.
