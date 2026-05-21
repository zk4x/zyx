// SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, in0_addr);

    Noc noc;
    CircularBuffer cb_in0_buf(cb_in0);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_in0_buf.reserve_back(1);
        noc.async_read(in0, cb_in0_buf, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0_buf.push_back(1);
    }
}
