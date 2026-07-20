// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

// DEBUG: Write pattern, read it back, compare, output comparison result.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t n_srcs = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_low = get_arg_val<uint32_t>(1 + n_srcs * 2);
    uint32_t dst_noc_high = get_arg_val<uint32_t>(1 + n_srcs * 2 + 1);
    uint64_t dst_noc_addr = (uint64_t)dst_noc_high << 32 | dst_noc_low;

    constexpr uint32_t cb_id = tt::CBIndex::c_16;
    constexpr uint32_t scratch_cb = tt::CBIndex::c_17;
    uint32_t tile_bytes = get_tile_size(cb_id);
    uint32_t n_elems = tile_bytes / 4;

    // We need two CB slots: one for the pattern, one for readback
    // But we only have c_16. Let's use the CB L1 as scratch and
    // manually manage two areas within it: offset 0 for write src, offset tile_bytes for readback
    cb_reserve_back(cb_id, 2);
    uint32_t write_l1 = get_write_ptr(cb_id);
    uint32_t readback_l1 = write_l1 + tile_bytes;

    // Fill write area with 1.0
    volatile uint32_t* write_buf = (volatile uint32_t*)write_l1;
    for (uint32_t i = 0; i < n_elems; i++) {
        write_buf[i] = 0x3F800000; // 1.0f
    }

    // Write pattern to dst noc
    noc_async_write(write_l1, dst_noc_addr, tile_bytes);
    noc_async_write_barrier();

    // Read back from dst noc into readback area
    noc_async_read(dst_noc_addr, readback_l1, tile_bytes);
    noc_async_read_barrier();

    // Compare
    volatile uint32_t* read_buf = (volatile uint32_t*)readback_l1;
    uint32_t match = 1;
    for (uint32_t i = 0; i < n_elems; i++) {
        if (write_buf[i] != read_buf[i]) {
            match = 0;
            break;
        }
    }

    // Write result via CB_L1
    cb_push_back(cb_id, 2);

    // Now write match status to first word of new CB slot
    // First pop both, then write to dedi offset
    // hmm this is getting complex

    // Instead, just write the match flag directly to dst_noc_addr[0]
    // overwrite first word with match value
    volatile uint32_t* result = (volatile uint32_t*)write_l1;
    result[0] = match ? 0x3F800000 : 0x00000000;
    // also put the first readback value
    result[1] = read_buf[0];
    // and first write value
    result[2] = write_buf[0];

    noc_async_write(write_l1, dst_noc_addr, tile_bytes);
    noc_async_write_barrier();

    for (volatile uint32_t i = 0; i < 10000000; i++);
}
