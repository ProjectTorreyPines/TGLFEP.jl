# Thread placement for the threaded CPU scan.
#
# The dense eigensolves of the :grid scan are limited by cache and memory, so where the
# Julia threads sit matters: left to the OS, the threads of a worker now and then end up
# doubled on one core or crowded onto a few cache groups, and that worker runs ~1.4x slower
# for the whole scan. Fixing thread t on its own physical core removes those slow workers.

"""
    pin_threads_to_cores!() -> String

Pin Julia thread `t` to the `t`-th physical core of the CPU set this process is allowed to
run on (one hardware thread per core, in CPU-number order), and return a one-line summary.

Nothing is pinned, and the summary says why, when the system is not Linux or the process
owns fewer physical cores than it has Julia threads. Call it once per process, before the
scan, and only when each process has its own CPU set (for example one Slurm task per
worker); processes that share a CPU set would all pin to the same cores.
"""
function pin_threads_to_cores!()
    Sys.islinux() || return "threads not pinned: not Linux"
    nbytes = 1024                                   # room for 8192 CPUs
    mask = zeros(UInt8, nbytes)
    rc = ccall(:sched_getaffinity, Cint, (Cint, Csize_t, Ptr{UInt8}), 0, nbytes, mask)
    rc == 0 || return "threads not pinned: sched_getaffinity failed"
    cores = Int[]                                   # first allowed CPU of each physical core
    seen = Set{String}()
    for cpu in 0:(8 * nbytes - 1)
        (mask[cpu >> 3 + 1] >> (cpu & 7)) & 0x01 == 0x01 || continue
        f = "/sys/devices/system/cpu/cpu$(cpu)/topology/thread_siblings_list"
        key = isfile(f) ? String(strip(read(f, String))) : string(cpu)
        key in seen && continue
        push!(seen, key)
        push!(cores, cpu)
    end
    n = Threads.nthreads()
    length(cores) >= n || return "threads not pinned: $(length(cores)) physical cores for $n threads"
    failed = Threads.Atomic{Int}(0)
    Threads.@threads :static for t in 1:n
        m = zeros(UInt8, nbytes)
        cpu = cores[t]
        m[cpu >> 3 + 1] = 0x01 << (cpu & 7)
        ccall(:sched_setaffinity, Cint, (Cint, Csize_t, Ptr{UInt8}), 0, nbytes, m) == 0 || Threads.atomic_add!(failed, 1)
    end
    failed[] == 0 || return "thread pinning incomplete: sched_setaffinity failed for $(failed[]) of $n threads"
    return "pinned $n threads to physical cores, CPUs $(cores[1])-$(cores[n])"
end
