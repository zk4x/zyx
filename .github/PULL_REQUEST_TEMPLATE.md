## What

<!-- What changed, in one or two sentences. -->

## Why

<!-- The problem or goal. Link the issue: Fixes #123 -->

## How it was tested

<!-- Say what actually ran, on which backends/hardware. -->

- [ ] `./ci.sh`
- [ ] Targeted test: `cd zyx && AGENT=1 ZYX_DEV=cpu cargo test --test <file> <name> -- --exact`
- [ ] Backend(s) exercised: <!-- C / CUDA / OpenCL / Vulkan / WGPU / Tenstorrent -->
- [ ] Docs/book updated if behavior or architecture changed

## Checklist

- [ ] Commits are signed off (`git commit -s`, DCO)
- [ ] No unrelated formatting churn
- [ ] New ops/backends have a correct arm in every match (or a loud `todo!()`)
- [ ] Failures and performance regressions are reported, not hidden
- [ ] Breaking changes are called out below

## Notes for reviewers

<!-- Known failures, hardware quirks, follow-ups. -->
