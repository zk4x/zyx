# Codegen and the Physical IR

The kernel IR's two jobs are both value-shaped: lower movement ops into index arithmetic, and be the space the autotuner searches over — tens of thousands of variants, measured time decides. Everything decision-shaped happens before codegen. What remains for the backend is purely machine-shaped: registers, sync, placement, then bytes.

This chapter describes how codegen is organized, and the Tenstorrent physical IR (TTIR) that exists because that hardware needs more than a straight lowering.

## The Two Boundary Tests

Two tests compose into the full boundary between kernel IR and backend code:

### The search test — which IR holds an op

> An op belongs in the kernel IR iff the autotuner should ever enumerate over it — iff there exists a variant of the program where it is placed or structured differently, and the hardware decides between them.

Movement lowering, loop structure, split/coarsen/register blocking, MAD fusion: search dimensions, kernel IR. Locks, syncs, reconfigs: their placement is invariant under restructuring, nothing to search — they cost the search nothing and belong to the physical IR, placed by derivation from loop structure and checked by verify.

Consequences:

- **The search space is closed.** The autotuner never sees a physical IR, and no backend can smuggle a decision out of the search. If a backend-specific dimension is ever worth searching, it enters as a kernel IR config + `MakeOpt`, keeping the cost model measured-time-only and backend-agnostic.
- **The physical IR's no-restructuring rule is a definition, not a discipline.** Restructuring is a search dimension; search lives in the kernel IR; a physical-IR op that wants restructuring failed the search test and is in the wrong IR.
- **Criterion for future disputes:** "would the autotuner enumerate over this?" Yes → kernel IR + `MakeOpt`. No → physical IR, derived placement, structural verify.

### The subject test — where a pass lives

> A pass belongs in an IR only if its subject exists in that IR.

If a pass reasons about values, it belongs in the kernel IR (or above). SSA machinery — rewrites, solvers, def-use analysis — is never duplicated in a physical IR. When a physical-IR pass needs a dataflow fact (e.g. "is this operand's last use?"), it queries the kernel IR through the origin op recorded at lowering time.

The search test decides where an *op* lives; the subject test decides where a *pass* lives.

## Straight Codegen Is the Default

For most backends (C, CUDA, OpenCL, Vulkan, HIP, WGPU), codegen is trivial by design: deSSA, then one linear pass over the kernel IR emitting target code. No further optimizations, no backend-specific lowering. The hard work happened in the optimization passes; the IR was deliberately shaped so backends stay thin.
