# Device config

First create folder zyx in some of your $XDG_CONFIG folders.
Usually just create folder $HOME/.config/zyx
Zyx will also save optimized kernels into this folder if you enable disk_cache feature.

Then create file config.json in that folder: $HOME/.config/zyx/config.json

Copy this into your config.json:
```json
{
  "dummy": {
    "enabled": false
  },
  "autotune": {
    "save_to_disk": true,
    "n_launches": 10,
    "n_seeds": 100,
    "n_added_per_step": 10,
    "n_removed_per_step": 5,
    "n_total_opts": 1000
  },
  "cuda": {
    "device_ids": [0]
  },
  "hip": {
    "device_ids": []
  },
  "opencl": {
    "platform_ids": []
  },
  "wgpu": {
    "enabled": true
  },
  "vulkan": {}
}
```
Then put numbers starting at zero into hip, cuda and or opencl configuration ids. In the above example, zyx will utilize cuda device with id 0.

WGPU (via wgpu crate) can be disabled or enabled. Set "enabled" to false to disable, or disable the wgpu cargo feature to do the same.

Vulkan backend is not yet written, so this is only placeholder in the config. Please ignore it.

CUDA, HIP and OpenCL backends cannot be disabled using cargo features. They are always compiled into zyx. Each of those backends takes about 1000 loc, so it does not significantly increase compile times. These backends automatically at runtime search for required .so files.
