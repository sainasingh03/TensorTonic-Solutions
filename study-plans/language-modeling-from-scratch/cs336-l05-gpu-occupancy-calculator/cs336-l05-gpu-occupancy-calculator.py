def gpu_occupancy(threads_per_block, registers_per_thread,
                 shared_mem_per_block, max_threads_per_sm,
                 max_warps_per_sm, max_blocks_per_sm,
                 max_registers_per_sm, max_shared_mem_per_sm,
                 warp_size=32):

    # Calculate warps per block (ceiling division)
    warps = (threads_per_block + warp_size - 1) // warp_size

    # Effective threads rounded up to full warps
    effective_threads = warps * warp_size

    # Resource constraints
    thread_limit = max_threads_per_sm // effective_threads
    warp_limit = max_warps_per_sm // warps
    block_limit = max_blocks_per_sm

    # Register limit
    if registers_per_thread == 0:
        register_limit = max_blocks_per_sm
    else:
        registers_per_block = registers_per_thread * effective_threads
        register_limit = max_registers_per_sm // registers_per_block

    # Shared memory limit
    if shared_mem_per_block == 0:
        shared_limit = max_blocks_per_sm
    else:
        shared_limit = max_shared_mem_per_sm // shared_mem_per_block

    # Maximum resident blocks
    blocks = min(
        thread_limit,
        warp_limit,
        block_limit,
        register_limit,
        shared_limit
    )

    resident_warps = blocks * warps
    occupancy = resident_warps / max_warps_per_sm

    return {
        "blocks_per_sm": int(blocks),
        "resident_warps": int(resident_warps),
        "occupancy": float(occupancy)
    }