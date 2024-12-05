use std::{collections::BTreeMap, time::Duration};

use crate::{backend::{Device, DeviceInfo, MemoryPool}, kernel::Kernel};

pub(super) struct Optimizer {
    cache: BTreeMap<(Kernel, DeviceInfo), OptimizerProgress>,
}

enum OptimizerProgress {
    Finished {
        optimization: Optimization,
        //time: Duration,
    },
    Optimizing {
        best: Optimization,
        done: BTreeMap<Optimization, Duration>,
    },
}

struct Optimization {}

impl Optimizer {
    pub(super) const fn new() -> Optimizer {
        Optimizer {
            cache: BTreeMap::new(),
        }
    }

    pub(super) fn get(&self, key: &(Kernel, DeviceInfo)) -> Option<&OptimizerProgress> {
        self.cache.get(key)
    }

    pub(super) fn get_optimization(&mut self, kernel: &Kernel, device: &mut Device, memory_pool: &mut MemoryPool, search_iters: usize) -> &Optimization {
        match self.get(&(kernel.clone(), device.info().clone())) {
            Some(OptimizerProgress::Finished { optimization }) => {
                optimization
            }
            Some(OptimizerProgress::Optimizing { best, done }) => {
                if search_iters == 0 {
                    best
                } else {
                    //self.optimize_kernel(kernel.clone(), device, memory_pool, search_iters)
                    todo!()
                }
            }
            None => {
                if search_iters == 0 {
                    self.default_optimizations(kernel, device.info())
                } else {
                    //self.optimize_kernel(kernel.clone(), device, memory_pool, search_iters);
                    todo!()
                }
            }
        }
    }

    fn default_optimizations(&self, kernel: &Kernel, device_info: &DeviceInfo) -> &Optimization {
        let _ = kernel;
        let _ = device_info;
        todo!()
    }
    //fn optimize_kernel(&mut self, kernel: Kernel, device: &mut Device, memory_pool: &mut MemoryPool, search_iters: usize) { todo!() }
}
