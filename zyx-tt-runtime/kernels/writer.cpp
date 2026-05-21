// SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    constexpr auto out0_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out0_args, c_addr);

    Noc noc;
    CircularBuffer cb_out(cb_out0);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_out.wait_front(1);
        noc.async_write(cb_out, out0, tile_size_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        cb_out.pop_front(1);
    }
}
