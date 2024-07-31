#![allow(non_camel_case_types)]

#![allow(dead_code)]

use super::{Compiler, HWInfo};
use alloc::vec;
use alloc::{string::String, vec::Vec};
use core::ffi::{c_char, c_int, c_void};
use core::ptr;

#[cfg(feature = "debug1")]
use std::println;

#[derive(Debug)]
pub struct HSAError {
    status: HSAStatus,
    info: String,
}

pub(crate) struct HSABuffer {
    memory: *mut c_void,
}

pub(crate) struct HSAProgram {
    name: String,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    args_read_only: Vec<bool>,
    program: *mut c_void,
}

pub(crate) struct HSARuntime {
    agent: Agent,
    mem_region: Region,
}

// These pointers are on device and do not get invalidated when accessing
// the device from different thread
unsafe impl Send for HSABuffer {}
unsafe impl Send for HSAProgram {}
unsafe impl Send for HSARuntime {}

fn check(status: HSAStatus, info: &str) -> Result<(), HSAError> {
    if status != HSAStatus::HSA_STATUS_SUCCESS {
        return Err(HSAError {
            status,
            info: info.into(),
        });
    } else {

    return Ok(());
    }
}

impl Drop for HSARuntime {
    fn drop(&mut self) {
        check(unsafe { hsa_shut_down() }, "hsa_shut_down").unwrap();
    }
}

impl Compiler for HSARuntime {
    type Buffer = HSABuffer;
    type Program = HSAProgram;
    type Error = HSAError;

    fn initialize() -> Result<Self, HSAError> {
        check(unsafe { hsa_init() }, "hsa_init")?;

        extern "C" fn agent_list_callback(agent: Agent, data: *mut c_void) -> HSAStatus {
            unsafe { (*(data as *mut Vec<Agent>)).push(agent) };
            return HSAStatus::HSA_STATUS_SUCCESS;
        }

        let mut agents: Vec<Agent> = Vec::new();
        check(unsafe { hsa_iterate_agents(agent_list_callback, &mut agents as *mut _ as *mut c_void) }, "hsa_iterate_agents").unwrap();
        println!("Agents:\n{agents:?}");
        let agent = agents[1];

        let mut device: DeviceType = unsafe { core::mem::zeroed() };
        check(unsafe { hsa_agent_get_info(agent, AgentInfo::Device, &mut device as *mut _ as *mut c_void) }, "hsa_agent_get_info").unwrap();
        println!("Device: {device:?}");

        extern "C" fn region_list_callback(region: Region, data: *mut c_void) -> HSAStatus {
            unsafe { (*(data as *mut Vec<Region>)).push(region) };
            return HSAStatus::HSA_STATUS_SUCCESS;
        }

        let mut regions: Vec<Region> = Vec::new();
        let p: *mut c_void = &mut regions as *mut _ as *mut c_void;
        check(unsafe { hsa_agent_iterate_regions(agent, region_list_callback, p) }, "hsa_agent_iterate_regions").unwrap();

        let mem_region = regions[0];

        Ok(Self { agent, mem_region })
    }

    fn hardware_information(&mut self) -> Result<HWInfo, HSAError> {
        Ok(HWInfo {
            max_work_item_sizes: vec![1024, 1024, 1024],
            max_work_group_size: 256,
            preferred_vector_size: 4,
            f16_support: true,
            f64_support: true,
            fmadd: true,
            global_mem_size: 2 * 1024 * 1024 * 1024,
            max_mem_alloc: 512 * 1024 * 1024,
            mem_align: 1024,
            page_size: 1024,
            local_mem_size: 1024 * 1024,
            num_registers: 96,
            native_mm16x16_support: false,
        })
    }

    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, HSAError> {
        println!("HSA Allocating {byte_size} B");
        let memory = ptr::null_mut();
        check(unsafe { hsa_memory_allocate(self.mem_region, byte_size, &memory) }, "hsa_memory_allocate").unwrap();
        return Ok(HSABuffer { memory });
    }

    fn store_memory<T: crate::Scalar>(
        &mut self,
        buffer: &mut Self::Buffer,
        data: Vec<T>,
    ) -> Result<(), HSAError> {
        println!("HSA Storing {} B", data.len() * T::dtype().byte_size());
        check(unsafe { hsa_memory_copy(&mut buffer.memory as *mut _ as *mut c_void, data.as_ptr().cast(), data.len() * T::dtype().byte_size()) }, "hsa memory copy store").unwrap();
        return Ok(());
    }

    fn load_memory<T: crate::Scalar>(
        &mut self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<alloc::vec::Vec<T>, HSAError> {
        let mut data: Vec<T> = Vec::with_capacity(length);
        check(unsafe { hsa_memory_copy(data.as_mut_ptr() as *mut c_void, &buffer.memory as *const _ as *const c_void, data.len() * T::dtype().byte_size()) }, "hsa_memory_copy load").unwrap();
        unsafe { data.set_len(length) };
        return Ok(data);
    }

    fn deallocate_memory(&mut self, mut buffer: Self::Buffer) -> Result<(), HSAError> {
        // TODO
        check(unsafe { hsa_memory_free(&mut buffer.memory as *mut _ as *mut c_void ) }, "hsa_memory_free").unwrap();
        return Ok(());
    }

    fn compile_program(
        &mut self,
        kernel: &super::IRKernel,
    ) -> Result<Self::Program, HSAError> {
        todo!()
    }

    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), HSAError> {
        todo!()
    }

    fn release_program(&mut self, program: Self::Program) -> Result<(), HSAError> {
        // TODO
        return Ok(());
    }
}








#[link(name = "hsa-runtime64")]
extern "C" {
    // 2.1 Initialization and shut down
    pub fn hsa_init() -> HSAStatus;
    pub fn hsa_shut_down() -> HSAStatus;

    // 2.2 Runtime notifications
    pub fn hsa_status_string(status: HSAStatus, status_string: &*const c_char) -> HSAStatus;

    // 2.3 System and agent information
    pub fn hsa_system_get_info(attribute: SystemInfo, value: *mut c_void) -> HSAStatus;
    pub fn hsa_extension_get_name(extension: Extension, name: &*const c_char) -> HSAStatus;
    pub fn hsa_system_extension_supported(
        extension: Extension,
        version_major: u16,
        version_minor: u16,
        result: *mut bool,
    ) -> HSAStatus;
    pub fn hsa_system_major_extension_supported(
        extension: Extension,
        version_major: u16,
        version_minor: *mut u16,
        result: *mut bool,
    ) -> HSAStatus;
    pub fn hsa_system_get_major_extension_table(
        extension: Extension,
        version_major: u16,
        table_length: usize,
        table: *mut c_void,
    ) -> HSAStatus;
    pub fn hsa_agent_get_info(agent: Agent, attribute: AgentInfo, value: *mut c_void) -> HSAStatus;
    pub fn hsa_iterate_agents(
        callback: extern "C" fn(Agent, *mut c_void) -> HSAStatus,
        data: *mut c_void,
    ) -> HSAStatus;
    pub fn hsa_cache_get_info(cache: Cache, attribute: CacheInfo, value: *mut c_void) -> HSAStatus;
    pub fn hsa_agent_iterate_caches(
        agent: Agent,
        callback: extern "C" fn(Cache, *mut c_void) -> HSAStatus,
        data: *mut c_void,
    ) -> HSAStatus;
    //#[deprecated]
    pub fn hsa_agent_extension_supported(
        extension: Extension,
        agent: Agent,
        version_major: u16,
        version_minor: u16,
        result: *mut bool,
    ) -> HSAStatus;
    pub fn hsa_agent_major_extension_supported(
        extension: Extension,
        agent: Agent,
        version_major: u16,
        version_minor: *mut u16,
        result: *mut bool,
    ) -> HSAStatus;

    // 2.4 Signals
    pub fn hsa_signal_create(
        initial_value: SignalValue,
        num_consumers: u32,
        consumers: *const Agent,
        signal: &SignalHandle,
    ) -> HSAStatus;
    pub fn hsa_signal_destroy(signal: SignalHandle) -> HSAStatus;
    pub fn hsa_signal_load_scacquire(signal: SignalHandle) -> SignalValue;
    pub fn hsa_signal_load_relaxed(signal: SignalHandle) -> SignalValue;
    //#[deprecated]
    pub fn hsa_signal_load_acquire(signal: SignalHandle) -> SignalValue;
    pub fn hsa_signal_store_relaxed(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_store_screlease(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_store_release(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_silent_store_relaxed(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_silent_store_screlease(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_exchange_scacq_screl(signal: SignalHandle, value: SignalValue)
        -> SignalValue;
    pub fn hsa_signal_exchange_scacquire(signal: SignalHandle, value: SignalValue) -> SignalValue;
    pub fn hsa_signal_exchange_relaxed(signal: SignalHandle, value: SignalValue) -> SignalValue;
    pub fn hsa_signal_exchange_screlease(signal: SignalHandle, value: SignalValue) -> SignalValue;
    //#[deprecated]
    pub fn hsa_signal_exchange_acq_rel(signal: SignalHandle, value: SignalValue) -> SignalValue;
    //#[deprecated]
    pub fn hsa_signal_exchange_acquire(signal: SignalHandle, value: SignalValue) -> SignalValue;
    //#[deprecated]
    pub fn hsa_signal_exchange_release(signal: SignalHandle, value: SignalValue) -> SignalValue;
    pub fn hsa_signal_cas_scacq_screl(
        signal: SignalHandle,
        expected: SignalValue,
        value: SignalValue,
    ) -> SignalValue;
    pub fn hsa_signal_cas_scacquire(
        signal: SignalHandle,
        expected: SignalValue,
        value: SignalValue,
    ) -> SignalValue;
    pub fn hsa_signal_cas_relaxed(
        signal: SignalHandle,
        expected: SignalValue,
        value: SignalValue,
    ) -> SignalValue;
    pub fn hsa_signal_cas_screlease(
        signal: SignalHandle,
        expected: SignalValue,
        value: SignalValue,
    ) -> SignalValue;
    //#[deprecated]
    pub fn hsa_signal_cas_acq_rel(
        signal: SignalHandle,
        expected: SignalValue,
        value: SignalValue,
    ) -> SignalValue;
    //#[deprecated]
    pub fn hsa_signal_cas_acquire(
        signal: SignalHandle,
        expected: SignalValue,
        value: SignalValue,
    ) -> SignalValue;
    //#[deprecated]
    pub fn hsa_signal_cas_release(
        signal: SignalHandle,
        expected: SignalValue,
        value: SignalValue,
    ) -> SignalValue;
    pub fn hsa_signal_add_scacq_screl(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_add_scacquire(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_add_relaxed(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_add_screlease(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_add_acq_rel(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_add_acquire(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_add_release(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_subtract_scacq_screl(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_subtract_scacquire(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_subtract_relaxed(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_subtract_screlease(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_subtract_acq_rel(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_subtract_acquire(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_subtract_release(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_and_scacq_screl(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_and_scacquire(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_and_relaxed(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_and_screlease(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_and_acq_rel(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_and_acquire(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_and_release(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_or_scacq_screl(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_or_scacquire(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_or_relaxed(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_or_screlease(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_or_acq_rel(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_or_acquire(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_or_release(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_xor_scacq_screl(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_xor_scacquire(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_xor_relaxed(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_xor_screlease(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_xor_acq_rel(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_xor_acquire(signal: SignalHandle, value: SignalValue);
    //#[deprecated]
    pub fn hsa_signal_xor_release(signal: SignalHandle, value: SignalValue);
    pub fn hsa_signal_wait_scacquire(
        signal: SignalHandle,
        condition: SignalCondition,
        compare_value: SignalValue,
        timeout_hint: u64,
        wait_state_hint: WaitState,
    ) -> SignalValue;
    pub fn hsa_signal_wait_relaxed(
        signal: SignalHandle,
        condition: SignalCondition,
        compare_value: SignalValue,
        timeout_hint: u64,
        wait_state_hint: WaitState,
    ) -> SignalValue;
    //#[deprecated]
    pub fn hsa_signal_wait_acquire(
        signal: SignalHandle,
        condition: SignalCondition,
        compare_value: SignalValue,
        timeout_hint: u64,
        wait_state_hint: WaitState,
    ) -> SignalValue;
    pub fn hsa_signal_group_create(
        num_signals: u32,
        signals: *const SignalHandle,
        num_consumers: u32,
        consumers: *const Agent,
        signal_group: &SignalGroupHandle,
    ) -> HSAStatus;
    pub fn hsa_signal_group_destroy(signal_group: SignalGroupHandle) -> HSAStatus;
    pub fn hsa_signal_group_wait_any_scacquire(
        signal_group: SignalGroupHandle,
        conditions: *const SignalCondition,
        compare_values: *const SignalValue,
        wait_state_hint: WaitState,
        signal: &SignalHandle,
        value: &SignalValue,
    ) -> HSAStatus;
    pub fn hsa_signal_group_wait_any_relaxed(
        signal_group: SignalGroupHandle,
        conditions: *const SignalCondition,
        compare_values: *const SignalValue,
        wait_state_hint: WaitState,
        signal: &SignalHandle,
        value: &SignalValue,
    ) -> HSAStatus;

    // 2.5 Queues
    pub fn hsa_queue_create(
        agent: Agent,
        size: u32,
        typ: QueueType,
        callback: *const c_void, //extern "C" fn(HSAStatus, *const QueueHandle, *const c_void),
        data: *const c_void,
        private_segment_size: u32,
        group_segment_size: u32,
        queue: &*const QueueHandle,
    ) -> HSAStatus;
    pub fn hsa_soft_queue_create(
        region: Region,
        size: u32,
        typ: QueueType,
        features: u32,
        doorbell_signal: SignalHandle,
        queue: &*const QueueHandle,
    ) -> HSAStatus;
    pub fn hsa_queue_destroy(queue: *const QueueHandle) -> HSAStatus;
    pub fn hsa_queue_inactivate(queue: *const QueueHandle) -> HSAStatus;
    pub fn hsa_queue_load_read_index_scacquire(queue: *const QueueHandle) -> u64;
    pub fn hsa_queue_load_read_index_relaxed(queue: *const QueueHandle) -> u64;
    //#[deprecated]
    pub fn hsa_queue_load_read_index_acquire(queue: *const QueueHandle) -> u64;
    pub fn hsa_queue_load_write_index_scacquire(queue: *const QueueHandle) -> u64;
    pub fn hsa_queue_load_write_index_relaxed(queue: *const QueueHandle) -> u64;
    //#[deprecated]
    pub fn hsa_queue_load_write_index_acquire(queue: *const QueueHandle) -> u64;
    pub fn hsa_queue_store_write_index_relaxed(queue: *const QueueHandle, value: u64);
    pub fn hsa_queue_store_write_index_screlease(queue: *const QueueHandle, value: u64);
    //#[deprecated]
    pub fn hsa_queue_store_write_index_release(queue: *const QueueHandle, value: u64);
    pub fn hsa_queue_cas_write_index_scacq_screl(
        queue: *const QueueHandle,
        expected: u64,
        value: u64,
    ) -> u64;
    pub fn hsa_queue_cas_write_index_scacquire(
        queue: *const QueueHandle,
        expected: u64,
        value: u64,
    ) -> u64;
    pub fn hsa_queue_cas_write_index_relaxed(
        queue: *const QueueHandle,
        expected: u64,
        value: u64,
    ) -> u64;
    pub fn hsa_queue_cas_write_index_screlease(
        queue: *const QueueHandle,
        expected: u64,
        value: u64,
    ) -> u64;
    //#[deprecated]
    pub fn hsa_queue_cas_write_index_acq_rel(
        queue: *const QueueHandle,
        expected: u64,
        value: u64,
    ) -> u64;
    //#[deprecated]
    pub fn hsa_queue_cas_write_index_acquire(
        queue: *const QueueHandle,
        expected: u64,
        value: u64,
    ) -> u64;
    //#[deprecated]
    pub fn hsa_queue_cas_write_index_release(
        queue: *const QueueHandle,
        expected: u64,
        value: u64,
    ) -> u64;
    pub fn hsa_queue_add_write_index_scacq_screl(queue: *const QueueHandle, value: u64) -> u64;
    pub fn hsa_queue_add_write_index_scacquire(queue: *const QueueHandle, value: u64) -> u64;
    pub fn hsa_queue_add_write_index_relaxed(queue: *const QueueHandle, value: u64) -> u64;
    pub fn hsa_queue_add_write_index_screlease(queue: *const QueueHandle, value: u64) -> u64;
    //#[deprecated]
    pub fn hsa_queue_add_write_index_acq_rel(queue: *const QueueHandle, value: u64) -> u64;
    //#[deprecated]
    pub fn hsa_queue_add_write_index_acquire(queue: *const QueueHandle, value: u64) -> u64;
    //#[deprecated]
    pub fn hsa_queue_add_write_index_release(queue: *const QueueHandle, value: u64) -> u64;
    pub fn hsa_queue_store_read_index_relaxed(queue: *const QueueHandle, value: u64);
    pub fn hsa_queue_store_read_index_screlease(queue: *const QueueHandle, value: u64);
    //#[deprecated]
    pub fn hsa_queue_store_read_index_release(queue: *const QueueHandle, value: u64);

    // 2.7 Memory
    pub fn hsa_region_get_info(
        region: Region,
        attribute: RegionInfo,
        value: *mut c_void,
    ) -> HSAStatus;
    pub fn hsa_agent_iterate_regions(
        agent: Agent,
        callback: extern "C" fn(Region, *mut c_void) -> HSAStatus,
        data: *mut c_void,
    ) -> HSAStatus;
    pub fn hsa_memory_allocate(region: Region, size: usize, ptr: &*mut c_void) -> HSAStatus;
    pub fn hsa_memory_free(ptr: *mut c_void) -> HSAStatus;
    pub fn hsa_memory_copy(dst: *mut c_void, src: *const c_void, size: usize) -> HSAStatus;
    pub fn hsa_memory_assign_agent(
        ptr: *mut c_void,
        agent: Agent,
        access: AccessPermission,
    ) -> HSAStatus;
    pub fn hsa_memory_register(ptr: *mut c_void, size: usize) -> HSAStatus;
    pub fn hsa_memory_deregister(ptr: *mut c_void, size: usize) -> HSAStatus;

    // 2.8 Code object loading
    pub fn hsa_isa_from_name(name: *const c_char, isa: &ISA) -> HSAStatus;
    pub fn hsa_agent_iterate_isas(
        agent: Agent,
        callback: extern "C" fn(ISA, *mut c_void) -> HSAStatus,
        data: *mut c_void,
    ) -> HSAStatus;
    /*#[deprecated]
    pub fn hsa_isa_get_info(
        isa: ISA,
        attribute: ISAInfo,
        index: u32,
        value: *mut c_void,
    ) -> HSAStatus;*/
    pub fn hsa_isa_get_info_alt(isa: ISA, attribute: ISAInfo, value: *mut c_void) -> HSAStatus;
    pub fn hsa_isa_get_exception_policies(isa: ISA, profile: Profile, mask: &mut u16) -> HSAStatus;
    pub fn hsa_isa_get_round_method(
        isa: ISA,
        fp_type: FpType,
        flush_mode: FlushMode,
        round_method: &RoundMethod,
    ) -> HSAStatus;
    pub fn hsa_wavefront_get_info(
        wavefront: Wavefront,
        attribute: WavefrontInfo,
        value: *mut c_void,
    ) -> HSAStatus;
    pub fn hsa_isa_iterate_wavefronts(
        isa: ISA,
        callback: extern "C" fn(Wavefront, *mut c_void) -> HSAStatus,
        data: *mut c_void,
    ) -> HSAStatus;
    /*#[deprecated]
    pub fn hsa_isa_compatible(code_object_isa: ISA, agent_isa: ISA, result: &bool) -> HSAStatus;*/
    /*pub fn hsa_code_object_reader_create_from_file(
        file: HSAFile,
        code_object_reader: &CodeObjectReader,
    ) -> HSAStatus;
    pub fn hsa_code_object_reader_create_from_memory(
        code_object: *const c_void,
        size: usize,
        code_object_reader: &CodeObjectReader,
    ) -> HSAStatus;
    pub fn hsa_code_object_reader_destroy(code_object_reader: CodeObjectReader) -> HSAStatus;
    #[deprecated]
    pub fn hsa_executable_create(
        profile: Profile,
        executable_state: ExecutableState,
        options: *const c_char,
        executable: &Executable,
    ) -> HSAStatus;*/
    pub fn hsa_executable_create_alt(
        profile: Profile,
        default_float_rouding_mode: DefaultFloatRoundingMode,
        options: *const c_char,
        executable: &Executable,
    ) -> HSAStatus;
    pub fn hsa_executable_destroy(executable: Executable) -> HSAStatus;
    /*pub fn hsa_executable_load_program_code_object(
        executable: Executable,
        code_object_reader: CodeObjectReader,
        options: *const c_char,
        loaded_code_object: &LoadedCodeObject,
    ) -> HSAStatus;
    pub fn hsa_executable_load_agent_code_object(
        executable: Executable,
        agent: Agent,
        code_object_reader: CodeObjectReader,
        options: *const c_void,
        loaded_code_object: &LoadedCodeObject,
    ) -> HSAStatus;*/
    pub fn hsa_executable_freeze(executable: Executable, options: *const c_char) -> HSAStatus;
    pub fn hsa_executable_get_info(
        executable: Executable,
        attribute: ExecutableInfo,
        value: *mut c_void,
    ) -> HSAStatus;
    /*pub fn hsa_executable_global_variable_define(
        executable: Executable,
        variable_name: *const c_char,
        address: *mut c_void,
    ) -> HSAStatus;
    pub fn hsa_executable_agent_global_variable_define(
        executable: Executable,
        agent: Agent,
        variable_name: *const c_char,
        address: *mut c_void,
    ) -> HSAStatus;
    pub fn hsa_executable_readonly_variable_define(
        executable: Executable,
        agent: Agent,
        variable_name: *const c_char,
        address: *mut c_void,
    ) -> HSAStatus;
    pub fn hsa_executable_validate(executable: Executable, result: *mut u32) -> HSAStatus;
    pub fn hsa_executable_validate_alt(
        executable: Executable,
        options: *const c_char,
        result: *mut u32,
    ) -> HSAStatus;*/
    //#[deprecated]
    pub fn hsa_executable_get_symbol(
        executable: Executable,
        module_name: *const c_char,
        symbol_name: *const c_char,
        agent: Agent,
        call_convention: i32,
        symbol: &ExecutableSymbol,
    ) -> HSAStatus;
    /*#[deprecated]
    pub fn hsa_executable_get_symbol_by_name(
        executable: Executable,
        symbol_name: *const c_char,
        agent: &Agent,
        symbol: &ExecutableSymbol,
    ) -> HSAStatus;
    pub fn hsa_executable_get_symbol_by_linker_name(
        executable: Executable,
        linker_name: *const c_void,
        agent: &Agent,
        symbol: &ExecutableSymbol,
    ) -> HSAStatus;*/
    pub fn hsa_executable_symbol_get_info(
        executable_symbol: ExecutableSymbol,
        attribute: ExecutableSymbolInfo,
        value: *mut c_void,
    ) -> HSAStatus;
    pub fn hsa_executable_iterate_agent_symbols(
        executable: Executable,
        agent: Agent,
        callback: extern "C" fn(Executable, Agent, ExecutableSymbol, *mut c_void) -> HSAStatus,
        data: *mut c_void,
    ) -> HSAStatus;
    pub fn hsa_executable_iterate_program_symbols(
        executable: Executable,
        callback: extern "C" fn(Executable, ExecutableSymbol, *mut c_void) -> HSAStatus,
        data: *mut c_void,
    ) -> HSAStatus;
    //#[deprecated]
    pub fn hsa_executable_iterate_symbols(
        executable: Executable,
        callback: extern "C" fn(Executable, ExecutableSymbol, *mut c_void) -> HSAStatus,
        data: *mut c_void,
    ) -> HSAStatus;
    /*#[deprecated]
    pub fn hsa_code_object_serialize(
        code_object_reader: CodeObject,
        alloc_callback: extern "C" fn(size: usize, data: CallbackData, address: *mut *mut c_void)
                                      -> HSAStatus,
        callback_data: CallbackData,
        options: *const c_char,
        serialized_code_object: *mut *mut c_void,
        serialized_code_object_size: *mut usize,
    ) -> HSAStatus;
    #[deprecated]
    pub fn hsa_code_object_deserialize(
        serialized_code_object: *mut c_void,
        serialized_code_object_size: usize,
        options: *const c_char,
        code_object: CodeObject,
    ) -> HSAStatus;*/
    //#[deprecated]
    pub fn hsa_code_object_destroy(code_object: CodeObject) -> HSAStatus;
    //#[deprecated]
    pub fn hsa_code_object_get_info(
        code_object: CodeObject,
        attribute: CodeObjectInfo,
        value: *mut c_void,
    ) -> HSAStatus;
    //#[deprecated]
    pub fn hsa_executable_load_code_object(
        executable: Executable,
        agent: Agent,
        code_object: CodeObject,
        options: *const c_char,
    ) -> HSAStatus;
    //#[deprecated]
    /*pub fn hsa_code_object_get_symbol(
        code_object: CodeObject,
        symbol_name: *const c_char,
        symbol: &CodeSymbol,
    ) -> HSAStatus;
    //#[deprecated]
    pub fn hsa_code_object_get_symbol_from_name(
        code_object: CodeObject,
        module_name: *const c_char,
        symbol_name: *const c_char,
        symbol: &CodeSymbol,
    ) -> HSAStatus;
    //#[deprecated]
    pub fn hsa_code_symbol_get_info(
        code_symbol: CodeSymbol,
        attribute: CodeSymbolInfo,
        value: *mut c_void,
    ) -> HSAStatus;
    //#[deprecated]
    pub fn hsa_code_object_iterate_symbols(
        code_object: CodeObject,
        callback: extern "C" fn(CodeObject, CodeSymbol, *mut c_void) -> HSAStatus,
        data: *mut c_void,
    ) -> HSAStatus;*/

    // 3.2 HSAIL finalization (Extension)
    /*pub fn hsa_ext_finalizer_iterate_isa(
        callback: extern "C" fn(ISA, *mut c_void) -> HSAStatus,
        data: *mut c_void,
    ) -> HSAStatus;
    pub fn hsa_ext_isa_from_name(name: *const c_char, isa: &mut ISA) -> HSAStatus;
    //#[deprecated]
    pub fn hsa_ext_isa_get_info(
        isa: ISA,
        attribute: ISAInfo,
        index: u32,
        value: *mut c_void,
    ) -> HSAStatus;
    pub fn hsa_ext_code_object_writer_create_from_file(
        file: HSAFile,
        code_object_writer: &ExtCodeObjectWriterHandle,
    ) -> HSAStatus;*/
    pub fn hsa_ext_code_object_writer_create_from_memory(
        memory_allocate: extern "C" fn(usize, usize, *mut *mut c_void, *mut c_void) -> HSAStatus,
        data: *mut c_void,
        code_object_writer: &ExtCodeObjectWriterHandle,
    ) -> HSAStatus;
    pub fn hsa_ext_code_object_writer_destroy(
        code_object_writer: ExtCodeObjectWriterHandle,
    ) -> HSAStatus;
    pub fn hsa_ext_program_create(
        machine_model: MachineModel,
        profile: Profile,
        default_float_rouding_mode: DefaultFloatRoundingMode,
        options: *const c_char,
        program: &ExtProgramHandle,
    ) -> HSAStatus;
    pub fn hsa_ext_program_destroy(program: ExtProgramHandle) -> HSAStatus;
    pub fn hsa_ext_program_add_module(program: ExtProgramHandle, module: ExtModule) -> HSAStatus;
    /*pub fn hsa_ext_program_iterate_modules(
        program: ExtProgramHandle,
        callback: extern "C" fn(ExtProgramHandle, ExtModule, *mut c_void) -> HSAStatus,
        data: *mut c_void,
    ) -> HSAStatus;*/
    pub fn hsa_ext_program_get_info(
        program: ExtProgramHandle,
        attribute: ExtProgramInfo,
        value: *mut c_void,
    ) -> HSAStatus;
    /*pub fn hsa_ext_program_code_object_finalize(
        program: ExtProgramHandle,
        options: *const c_char,
        code_object_writer: &ExtCodeObjectWriterHandle,
    ) -> HSAStatus;*/
    pub fn hsa_ext_agent_code_object_finalize(
        program: ExtProgramHandle,
        isa: ISA,
        options: *const c_char,
        code_object_writer: &ExtCodeObjectWriterHandle,
    ) -> HSAStatus;
    //#[deprecated]
    pub fn hsa_ext_program_finalize(
        program: ExtProgramHandle,
        isa: ISA,
        call_convention: i32,
        control_directives: ExtControlDirectives,
        options: *const c_char,
        code_object_type: CodeObjectType,
        code_object: &CodeObject,
    ) -> HSAStatus;
}

// 2.3 System and agent information

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum Endianness {
    Little = 0,
    Big = 1,
}

#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
#[repr(C)]
pub enum MachineModel {
    Small = 0,
    Large = 1,
}

#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
#[repr(C)]
pub enum Profile {
    Base = 0,
    Full = 1,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum SystemInfo {
    VersionMajor = 0,
    VersionMinor = 1,
    Timestamp = 2,
    TimestampFrequency = 3,
    SignalMaxWait = 4,
    Endianness = 5,
    MachineModel = 6,
    Extensions = 7,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(u16)]
pub enum Extension {
    Finalizer = 0,
    Images = 1,
    PerformanceCounters = 2,
    ProfilingEvents = 3,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Agent {
    handle: u64,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum AgentFeature {
    KernelDispatch = 1,
    AgentDispatch = 2,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum DeviceType {
    CPU = 0,
    GPU = 1,
    DSP = 2,
}

#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
#[repr(C)]
pub enum DefaultFloatRoundingMode {
    Default = 0,
    Zero = 1,
    Near = 2,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum AgentInfo {
    Name = 0,
    VendorName = 1,
    Feature = 2,
    MachineModel = 3,
    Profile = 4,
    DefaultFloatRoundingMode = 5,
    BaseProfileDefaultFloatRoundingModes = 23,
    FastF16Operation = 24,
    WavefrontSize = 6,
    WorkgroupMaxDim = 7,
    WorkgroupMaxSize = 8,
    GridMaxDim = 9,
    GridMaxSize = 10,
    FbarrierMaxSize = 11,
    QueuesMax = 12,
    QueueMinSize = 13,
    QueueMaxSize = 14,
    QueueType = 15,
    Node = 16,
    Device = 17,
    CacheSize = 18,
    ISA = 19,
    Extensions = 20,
    VersionMajor = 21,
    VersionMinor = 22,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
#[repr(C)]
pub enum ExceptionPolicy {
    Break = 1,
    Detect = 2,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Cache {
    handle: u64,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum CacheInfo {
    // NameLength = 0, // not unsed
    Name = 1,
    Level = 2,
    Size = 3,
}

// 2.4 Signals

#[cfg(target_pointer_width = "32")]
pub type SignalValue = i32;

#[cfg(target_pointer_width = "64")]
pub type SignalValue = i64;

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct SignalHandle {
    pub handle: u64,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum SignalCondition {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Gte = 3,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum WaitState {
    Blocked = 0,
    Active = 1,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct SignalGroupHandle {
    pub handle: u64,
}

// 2.5 Queues

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum QueueType {
    Multi = 0,
    Single = 1,
}

pub type QueueType32 = u32;

#[derive(Clone, Debug)]
#[repr(C)]
pub struct QueueHandle {
    typ: QueueType32,
    features: QueueFeature,
    pub base_address: *const c_void,

    #[cfg(target_pointer_width = "32")]
    reserved0: u32,

    pub doorbell_signal: SignalHandle,
    pub size: u32,
    reserved1: u32,
    id: u64,
}

#[allow(dead_code)]
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(u32)]
pub enum QueueFeature {
    KernelDispatch = 1,
    AgentDispatch = 2,
}

// 2.6 Architected Queuing Language packets

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum PacketType {
    VendorSpecific = 0,
    Invalid = 1,
    KernelDispatch = 2,
    BarrierAnd = 3,
    AgentDispatch = 4,
    BarrierOr = 5,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum FenceScope {
    None = 0,
    Agent = 1,
    System = 2,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum PacketHeader {
    Type = 0,
    Barrier = 8,
    ScacquireFenceScope = 9,
    ScreleaseFenceScope = 11,
}
#[allow(non_upper_case_globals)]
impl PacketHeader {
    pub const AcquireFenceScope: PacketHeader = PacketHeader::ScacquireFenceScope;
    pub const ReleaseFenceScope: PacketHeader = PacketHeader::ScreleaseFenceScope;
}

/*
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum PacketHeaderWidth {
    Type = 8,
    Barrier = 1,
    ScacquireFenceScope = 2,
}

#[allow(non_upper_case_globals)]
impl PacketHeaderWidth {
    pub const AcquireFenceScope: PacketHeaderWidth = PacketHeaderWidth::ScacquireFenceScope;
    pub const ScreleaseFenceScope: PacketHeaderWidth = PacketHeaderWidth::ScacquireFenceScope;
    pub const ReleaseFenceScope: PacketHeaderWidth = PacketHeaderWidth::ScacquireFenceScope;
}
*/

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum KernelDispatchPacketSetup {
    Dimensions = 0,
}

/*
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum KernelDispatchPacketSetupWidth {
    Dimensions = 0,
}
*/

#[derive(Clone, PartialEq, Debug)]
#[repr(C)]
pub struct KernelDispatchPacket {
    pub header: u16,
    pub setup: u16,
    pub workgroup_size_x: u16,
    pub workgroup_size_y: u16,
    pub workgroup_size_z: u16,
    reserved0: u16,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: u64,
    pub kernarg_address: *const c_void,
    #[cfg(target_pointer_width = "32")]
    reserved1: u32,
    reserved2: u64,
    pub completion_signal: SignalHandle,
}

#[derive(Clone, PartialEq, Debug)]
#[repr(C)]
pub struct AgentDispatchPacket {
    header: u16,
    typ: u16,
    reserved0: u32,
    return_address: *const c_void,
    #[cfg(target_pointer_width = "32")]
    reserved1: u32,
    args: [u64; 4],
    reserved2: u64,
    completion_signal: SignalHandle,
}

#[derive(Clone, PartialEq, Debug)]
#[repr(C)]
pub struct BarrierAndPacket {
    header: u16,
    reserved0: u16,
    reserved1: u32,
    dep_signal: [SignalHandle; 5],
    reserved2: u64,
    completion_signal: SignalHandle,
}

#[derive(Clone, PartialEq, Debug)]
#[repr(C)]
pub struct BarrierOrPacket {
    header: u16,
    reserved0: u16,
    reserved1: u32,
    dep_signal: [SignalHandle; 5],
    reserved2: u64,
    completion_signal: SignalHandle,
}

// 2.7 Memory
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Region {
    pub handle: u64,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum RegionSegment {
    Global = 0,
    ReadOnly = 1,
    Private = 2,
    Group = 3,
    KernArg = 4,
}

#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
#[repr(C)]
pub enum RegionGlobalFlag {
    KernArg = 1,
    FineGrained = 2,
    CoarseGrained = 4,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum RegionInfo {
    Segment = 0,
    GlobalFlags = 1,
    Size = 2,
    AllocMaxSize = 4,
    AllocMaxPrivateWorkgroupSize = 8,
    RuntimeAllocAllowed = 5,
    RuntimeAllocGranule = 6,
    RuntimeAllocAlignment = 7,
}

// 2.8 Code object loading

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct ISA {
    handle: u64,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum ISAInfo {
    NameLength = 0,
    Name = 1,
    CallConvertionCount = 2,
    CallConvertionInfoWavefrontSize = 3,
    CallConvertionInfoWavefrontsPerComputeUnit = 4,
    MachineModels = 5,
    Profiles = 6,
    DefaultFloatRoundingModes = 7,
    BaseProfileDefaultFloatRoundingModes = 8,
    FastF16Operation = 9,
    WorkgroupMaxDim = 12,
    WorkgroupMaxSize = 13,
    GridMaxDim = 14,
    GridMaxSize = 16,
    FbarrierMaxSize = 17,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum FpType {
    Fp16 = 1,
    Fp32 = 2,
    Fp64 = 4,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum FlushMode {
    Ftz = 1,
    NonFtz = 2,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum RoundMethod {
    Single = 1,
    Double = 2,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Wavefront {
    handle: u64,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum WavefrontInfo {
    Size = 0,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct CodeObjectReader {
    handle: u64,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Executable {
    pub handle: u64,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum ExecutableState {
    Unfrozen = 0,
    Frozen = 1,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct LoadedCodeObject {
    handle: u64,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum ExecutableInfo {
    Profile = 1,
    State = 2,
    DefaultFloatRoundingMode = 3,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct ExecutableSymbol {
    handle: u64,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum SymbolKind {
    Variable = 0,
    Kernel = 1,
    IndirectFunction = 2,
}

#[deprecated]
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum SymbolKindLinkage {
    Module = 0,
    Program = 1,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum VariableAllocation {
    Agent = 0,
    Program = 1,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum VariableSegment {
    Global = 0,
    ReadOnly = 1,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum ExecutableSymbolInfo {
    Type = 0,
    NameLength = 1,
    Name = 2,
    ModuleNameLength = 3,
    ModuleName = 4,
    LinkerNameLength = 24,
    LinkerName = 25,
    Agent = 20,
    VariableAddress = 21,
    Linkage = 5,
    IsDefinition = 17,
    VariableAllocation = 6,
    VariableSegment = 7,
    VariableAlignment = 8,
    VariableSize = 9,
    VariableIsConst = 10,
    KernelObject = 22,
    KernelKernArgSegmentSize = 11,
    KernelKernArgSegmentAlignment = 12,
    KernelGroupSegmentSize = 13,
    KernelPrivateSegmentSize = 14,
    KernelDynamicCallstack = 15,
    KernelCallConvertion = 18,
    IndirectFunctionObject = 23,
    IndirectFunctionCallConvertion = 16,
}

#[deprecated]
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct CodeObject {
    handle: u64,
}

#[deprecated]
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct CallbackData {
    handle: u64,
}

#[deprecated]
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum CodeObjectType {
    Program = 0,
}

#[deprecated]
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum CodeObjectInfo {
    Version = 0,
    Type = 1,
    ISA = 2,
    MachineModel = 3,
    Profile = 4,
    DefaultFloatRoundingMode = 5,
}

#[deprecated]
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct CodeSymbol {
    handle: u64,
}

/*
#[deprecated]
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum CodeSymbolInfo {
    Type = 0,
    NameLength = 1,
    Name = 2,
    ModuleNameLength = 3,
    ModuleName = 4,
    Linkage = 5,
    IsDefinition = 17,
    VariableAllocation = 6,
    VariableSegment = 7,
    VariableAlignment = 8,
    VariableSize = 9,
    VariableIsConst = 10,
    KernelKernArgSegmentSize = 11,
    KernelKernArgSegmentAlignment = 12,
    KernelGroupSegmentSize = 13,
    KernelPrivateSegmentSize = 14,
    KernelDynamicCallstack = 15,
    KernelCallConvertion = 18,
    IndirectFunctionCallConvertion = 16,
}
*/

// 2.9 Common definitions

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum AccessPermission {
    RO = 1,
    WO = 2,
    RW = 3,
}

//pub type HSAFile = c_int;

// 3.2 HSAIL finalization (Extension)

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct ExtCodeObjectWriterHandle {
    pub handle: u64,
}

pub type ExtModule = *const c_void; //BrigModule

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct ExtProgramHandle {
    pub handle: u64,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum ExtProgramInfo {
    MachineModel = 0,
    Profile = 1,
    DefaultFloatRoundingMode = 2,
}

/*
#[deprecated]
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum ExtFinalizerCallConvention {
    Auto = -1,
}
*/

#[deprecated]
#[repr(C)]
pub struct ExtControlDirectives {
    control_directives_mask: u64,
    break_exceptions_mask: u16,
    detect_exceptions_mask: u16,
    max_dynamic_group_size: u32,
    max_flat_grid_size: u64,
    max_flat_workgroup_size: u32,
    reserved1: u32,
    required_grid_size: [u64; 3],
    required_workgroup_size: Dim3,
    required_dim: u8,
    reserved2: [u8; 75],
}

#[repr(C)]
pub struct ExtFinalizer1 {
    pub create: fn(MachineModel,
                   Profile,
                   DefaultFloatRoundingMode,
                   *const c_char,
                   &ExtProgramHandle)
                   -> HSAStatus,
    pub destroy: fn(ExtProgramHandle) -> HSAStatus,
    _0: fn() -> HSAStatus,
    _1: fn() -> HSAStatus,
    _2: fn() -> HSAStatus,
    _3: fn() -> HSAStatus,
    _4: fn() -> HSAStatus,
    _5: fn() -> HSAStatus,
    _6: fn() -> HSAStatus,
    _7: fn() -> HSAStatus,
    _8: fn() -> HSAStatus,
    _9: fn() -> HSAStatus,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
enum HSAStatus {
  /**
   * The function has been executed successfully.
   */
  HSA_STATUS_SUCCESS = 0x0,
  /**
   * A traversal over a list of elements has been interrupted by the
   * application before completing.
   */
  HSA_STATUS_INFO_BREAK = 0x1,
  /**
   * A generic error has occurred.
   */
  HSA_STATUS_ERROR = 0x1000,
  /**
   * One of the actual arguments does not meet a precondition stated in the
   * documentation of the corresponding formal argument.
   */
  HSA_STATUS_ERROR_INVALID_ARGUMENT = 0x1001,
  /**
   * The requested queue creation is not valid.
   */
  HSA_STATUS_ERROR_INVALID_QUEUE_CREATION = 0x1002,
  /**
   * The requested allocation is not valid.
   */
  HSA_STATUS_ERROR_INVALID_ALLOCATION = 0x1003,
  /**
   * The agent is invalid.
   */
  HSA_STATUS_ERROR_INVALID_AGENT = 0x1004,
  /**
   * The memory region is invalid.
   */
  HSA_STATUS_ERROR_INVALID_REGION = 0x1005,
  /**
   * The signal is invalid.
   */
  HSA_STATUS_ERROR_INVALID_SIGNAL = 0x1006,
  /**
   * The queue is invalid.
   */
  HSA_STATUS_ERROR_INVALID_QUEUE = 0x1007,
  /**
   * The HSA runtime failed to allocate the necessary resources. This error
   * may also occur when the HSA runtime needs to spawn threads or create
   * internal OS-specific events.
   */
  HSA_STATUS_ERROR_OUT_OF_RESOURCES = 0x1008,
  /**
   * The AQL packet is malformed.
   */
  HSA_STATUS_ERROR_INVALID_PACKET_FORMAT = 0x1009,
  /**
   * An error has been detected while releasing a resource.
   */
  HSA_STATUS_ERROR_RESOURCE_FREE = 0x100A,
  /**
   * An API other than ::hsa_init has been invoked while the reference count
   * of the HSA runtime is 0.
   */
  HSA_STATUS_ERROR_NOT_INITIALIZED = 0x100B,
  /**
   * The maximum reference count for the object has been reached.
   */
  HSA_STATUS_ERROR_REFCOUNT_OVERFLOW = 0x100C,
  /**
   * The arguments passed to a functions are not compatible.
   */
  HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS = 0x100D,
  /**
   * The index is invalid.
   */
  HSA_STATUS_ERROR_INVALID_INDEX = 0x100E,
  /**
   * The instruction set architecture is invalid.
   */
  HSA_STATUS_ERROR_INVALID_ISA = 0x100F,
  /**
   * The instruction set architecture name is invalid.
   */
  HSA_STATUS_ERROR_INVALID_ISA_NAME = 0x1017,
  /**
   * The code object is invalid.
   */
  HSA_STATUS_ERROR_INVALID_CODE_OBJECT = 0x1010,
  /**
   * The executable is invalid.
   */
  HSA_STATUS_ERROR_INVALID_EXECUTABLE = 0x1011,
  /**
   * The executable is frozen.
   */
  HSA_STATUS_ERROR_FROZEN_EXECUTABLE = 0x1012,
  /**
   * There is no symbol with the given name.
   */
  HSA_STATUS_ERROR_INVALID_SYMBOL_NAME = 0x1013,
  /**
   * The variable is already defined.
   */
  HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED = 0x1014,
  /**
   * The variable is undefined.
   */
  HSA_STATUS_ERROR_VARIABLE_UNDEFINED = 0x1015,
  /**
   * An HSAIL operation resulted in a hardware exception.
   */
  HSA_STATUS_ERROR_EXCEPTION = 0x1016,
  /**
   * The code object symbol is invalid.
   */
  HSA_STATUS_ERROR_INVALID_CODE_SYMBOL = 0x1018,
  /**
   * The executable symbol is invalid.
   */
  HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL = 0x1019,
  /**
   * The file descriptor is invalid.
   */
  HSA_STATUS_ERROR_INVALID_FILE = 0x1020,
  /**
   * The code object reader is invalid.
   */
  HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER = 0x1021,
  /**
   * The cache is invalid.
   */
  HSA_STATUS_ERROR_INVALID_CACHE = 0x1022,
  /**
   * The wavefront is invalid.
   */
  HSA_STATUS_ERROR_INVALID_WAVEFRONT = 0x1023,
  /**
   * The signal group is invalid.
   */
  HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP = 0x1024,
  /**
   * The HSA runtime is not in the configuration state.
   */
  HSA_STATUS_ERROR_INVALID_RUNTIME_STATE = 0x1025,
  /**
  * The queue received an error that may require process termination.
  */
  HSA_STATUS_ERROR_FATAL = 0x1026
}
