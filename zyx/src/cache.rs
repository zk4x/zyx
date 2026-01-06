use crate::{
    Map,
    backend::{Device, DeviceInfo, ProgramId},
    kernel::Kernel,
    optimizer::{self, Optimizer},
};
use nanoserde::{DeBin, SerBin};
use std::hash::BuildHasherDefault;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, DeBin, SerBin)]
pub struct DeviceInfoId(u32);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, DeBin, SerBin)]
pub struct DeviceId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, DeBin, SerBin)]
pub struct KernelId(u32);

#[derive(Debug)]
pub struct Cache {
    pub device_infos: Map<DeviceInfo, DeviceInfoId>,
    pub kernels: Map<Kernel, KernelId>,
    // Finished optimizations of kernels for given devices
    pub optimizations: Map<(KernelId, DeviceInfoId), optimizer::Optimizer>,
    // This last one is not stored to disk
    pub programs: Map<(KernelId, DeviceId), ProgramId>,
}

impl SerBin for Cache {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        self.device_infos.len().ser_bin(output);
        for (key, value) in &self.device_infos {
            key.ser_bin(output);
            value.ser_bin(output);
        }
        self.kernels.len().ser_bin(output);
        for (key, value) in &self.kernels {
            key.ser_bin(output);
            value.ser_bin(output);
        }
        self.optimizations.len().ser_bin(output);
        for (key, value) in &self.optimizations {
            key.ser_bin(output);
            value.ser_bin(output);
        }
    }
}

impl DeBin for Cache {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let len = usize::de_bin(offset, bytes)?;
        if len > bytes.len() - *offset {
            return Err(nanoserde::DeBinErr::new(*offset, len, bytes.len() - *offset));
        }
        let mut device_infos = Map::with_capacity_and_hasher(len, BuildHasherDefault::new());
        for _ in 0..len {
            let key = DeviceInfo::de_bin(offset, bytes)?;
            let value = DeviceInfoId::de_bin(offset, bytes)?;
            device_infos.insert(key, value);
        }

        let len = usize::de_bin(offset, bytes)?;
        if len > bytes.len() - *offset {
            return Err(nanoserde::DeBinErr::new(*offset, len, bytes.len() - *offset));
        }
        let mut kernels = Map::with_capacity_and_hasher(len, BuildHasherDefault::new());
        for _ in 0..len {
            let key = Kernel::de_bin(offset, bytes)?;
            let value = KernelId::de_bin(offset, bytes)?;
            kernels.insert(key, value);
        }

        let len = usize::de_bin(offset, bytes)?;
        if len > bytes.len() - *offset {
            return Err(nanoserde::DeBinErr::new(*offset, len, bytes.len() - *offset));
        }
        let mut optimizations = Map::with_capacity_and_hasher(len, BuildHasherDefault::new());
        for _ in 0..len {
            let k1 = KernelId::de_bin(offset, bytes)?;
            let k2 = DeviceInfoId::de_bin(offset, bytes)?;
            let key = (k1, k2);
            let value = Optimizer::de_bin(offset, bytes)?;
            optimizations.insert(key, value);
        }

        let programs = Map::with_hasher(BuildHasherDefault::new());
        Ok(Cache { device_infos, kernels, optimizations, programs })
    }
}

impl Cache {
    pub const fn new() -> Cache {
        Cache {
            device_infos: Map::with_hasher(BuildHasherDefault::new()),
            kernels: Map::with_hasher(BuildHasherDefault::new()),
            optimizations: Map::with_hasher(BuildHasherDefault::new()),
            programs: Map::with_hasher(BuildHasherDefault::new()),
        }
    }

    pub fn deinitialize(&mut self, devices: &mut [Device]) {
        for (&(_, dev_id), &program_id) in &self.programs {
            devices[dev_id.0 as usize].release(program_id);
        }
        self.device_infos = Default::default();
        self.kernels = Default::default();
        self.optimizations = Default::default();
        self.programs = Default::default();
    }

    pub fn get_or_add_dev_info(&mut self, device_info: &DeviceInfo) -> DeviceInfoId {
        if let Some(&dev_info_id) = self.device_infos.get(device_info) {
            dev_info_id
        } else {
            self.insert_device_info(device_info.clone())
        }
    }

    pub fn insert_device_info(&mut self, device_info: DeviceInfo) -> DeviceInfoId {
        let dev_info_id = DeviceInfoId(self.device_infos.values().max().map_or(0, |id| id.0.checked_add(1).unwrap()));
        let newly_inserted = self.device_infos.insert(device_info, dev_info_id).is_none();
        assert!(newly_inserted);
        dev_info_id
    }

    pub fn insert_kernel(&mut self, kernel: Kernel) -> KernelId {
        let kernel_id = KernelId(self.kernels.values().copied().max().map_or(0, |id| id.0.checked_add(1).unwrap()));
        let newly_inserted = self.kernels.insert(kernel, kernel_id).is_none();
        assert!(newly_inserted);
        kernel_id
    }
}

#[allow(clippy::similar_names)]
pub fn get_perf(flop: u64, bytes_read: u64, bytes_written: u64, nanos: u64) -> String {
    if nanos == u64::MAX {
        return format!("INF time taken");
    }
    const fn value_unit(x: u64) -> (u64, &'static str) {
        match x {
            0..1000 => (x * 100, ""),
            1_000..1_000_000 => (x / 10, "k"),
            1_000_000..1_000_000_000 => (x / 10_000, "M"),
            1_000_000_000..1_000_000_000_000 => (x / 10_000_000, "G"),
            1_000_000_000_000..1_000_000_000_000_000 => (x / 10_000_000_000, "T"),
            1_000_000_000_000_000..1_000_000_000_000_000_000 => (x / 10_000_000_000_000, "P"),
            1_000_000_000_000_000_000.. => (x / 10_000_000_000_000_000, "E"),
        }
    }

    //let (f, f_u) = value_unit(flop);
    //let (br, br_u) = value_unit(bytes_read);
    //let (bw, bw_u) = value_unit(bytes_written);
    let (t, t_u) = match nanos {
        0..1_000 => (nanos * 10, "ns"),
        1_000..1_000_000 => (nanos / 100, "μs"),
        1_000_000..1_000_000_000 => (nanos / 100_000, "ms"),
        1_000_000_000..1_000_000_000_000 => (nanos / 100_000_000, "s"),
        1_000_000_000_000.. => (nanos / 6_000_000_000, "min"),
    };

    let (fs, f_us) = value_unit(flop / nanos * 1_000_000_000);
    let (brs, br_us) = value_unit(bytes_read / nanos * 1_000_000_000);
    let (bws, bw_us) = value_unit(bytes_written / nanos * 1_000_000_000);

    /*format!(
        "{}.{} {t_u} ~ {}.{:02} {f_us}FLOP/s, {}.{:02} {br_us}B/s r, {}.{:02} {bw_us}B/s w, {}.{:02} {f_u}FLOP, {}.{:02} {br_u}B r, {}.{:02} {bw_u}B w",
        t / 10,
        t % 10,
        fs / 100,
        fs % 100,
        brs / 100,
        brs % 100,
        bws / 100,
        bws % 100,
        f / 100,
        f % 100,
        br / 100,
        br % 100,
        bw / 100,
        bw % 100,
    )*/

    format!(
        "{}.{} {t_u} ~ {}.{:02} {f_us}FLOP/s, {}.{:02} {br_us}B/s r, {}.{:02} {bw_us}B/s w",
        t / 10,
        t % 10,
        fs / 100,
        fs % 100,
        brs / 100,
        brs % 100,
        bws / 100,
        bws % 100,
    )
}
