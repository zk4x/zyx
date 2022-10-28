# 0.6.0
- Updated documentation
- accel::cpu::Buffer now uses Arc to mitigate the cost of cloning (lowers RAM usage and improves performance)
- Removed tensor::B placeholder, as this was never an intuitive thing from user perspective. This is a change in the public API, so we have to make it a major version bump.

# 0.5.0

- Created Parameters trait
- Some performance improvements for Buffer
- Started adding support for different input-output types in tensor operations

# 0.4.2

- Added more documentation