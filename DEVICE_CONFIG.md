# Device config

First create folder zyx in some of your $XDG_CONFIG folders.
Usually just create folder $HOME/.config/zyx
Zyx will also save optimized kernels into this folder if you enable disk_cache feature.

The create file device_config.json in that folder. $HOME/.config/zyx/device_config.json

Copy this into your device_config.json:
```json
{
  "cuda": {
    "device_ids": [0]
  },
  "hip": {
    "device_ids": []
  },
  "opencl": {
    "platform_ids": []
  },
  "wgsl": {
    "use_wgsl": true
  },
  "vulkan": {
  }
}
```
Then put numbers beginning at zero into hip, cuda and or opencl configuration ids. In the above example, zyx will utilize cuda device with id 0.

WGSL currently can only be disabled or enabled and it runs only on one device. You can change use_wgsl to false in order to disable wgsl backend, or you can disable feature wgsl to do the same.

Vulkan backend is not yet written, so this is only placeholder in the config. Please ignore it.

CUDA, HIP and OpenCL backends cannot be disabled using cargo features. They are always compiled into zyx. Each of those backends takes about 1000 loc, so it does not significantly increase compile times. These backends automatically at runtime search for required .so files.

