Environment variables can be set to enable debugging.

Zyx uses one variable for debugging:

ZYX_DEBUG
It is a bitmask with following options:

0000 0001
Zyx prints information about used hardware devices and hardware configuration.
1 - dev

0000 0010
Zyx prints graph execution characteristics and performance
2 - perf

0000 0100
Zyx prints kernels created by scheduler.
4 - sched

0000 1000
Zyx printgs kernels in intermediate representation.
8 - ir

0001 0000
Zyx prints kernels in native assembly or other native code (i.e. opencl kernel source code).
16 - asm


For kernel search zyx uses ZYX_SEARCH variable.
For example ZYX_SEARCH=1000 will search over 1000 variations of each kernel before caching
them to disk and continuing with the next kernel.
