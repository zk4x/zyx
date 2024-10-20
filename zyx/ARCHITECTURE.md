# Architecture

This document describes limitations and design choices behind zyx.

## Hardware support

Zyx supports many hardware backends through singular intermediate representation, which is very close to assembly
and simply uses enum as an op.

## Performance

Zyx creates graph of nodes at runtime. It takes about a second to compile graph with 10k nodes. Typical
models have fewer nodes than that, but zyx should be able to scale to millions of nodes if it is needed.
Both compilation of the whole graph and hardware kernels is cached, therefore this cost is payed only once
per model launch. As for kernel optimizations, these are done automatically and results are cached to disk,
so price is payed once per model per hardware configuration. It takes about 30 minutes to finish all optimizations
on typical LLM on modern GPU. We will work on further optimizations to cut this time down.

## Error handling

Zyx uses simple method to differentiate when to panic and when to return a result. If a user provides incorrect
input, function should return result. Everything else should be panic. That is if zyx panics, it is a bug.
All panics and assertions (there are many and we will add more) check for consistency of invariants that cannot
be broken. Breaking any of these invariants puts zyx into irrecoverable state, thus the only right thing to do
is to immediately stop execution. One other option when panic happens is in case of hardware failure.
Zyx already detects hardware devices at runtime and disallows explicit programming for cpu or gpu only.
However zyx currently assumes that hardware configuration stays constant as long as at least one tensor exists.
