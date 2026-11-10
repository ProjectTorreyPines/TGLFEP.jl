# pin_threads_to_cores!: one Julia thread per physical core. The test restores every
# thread's original CPU set afterwards so the rest of the suite runs unpinned.
@testset "pin_threads_to_cores!" begin
    if Sys.islinux()
        nbytes = 1024
        n = Threads.nthreads()
        getmask() = (m = zeros(UInt8, nbytes);
                     ccall(:sched_getaffinity, Cint, (Cint, Csize_t, Ptr{UInt8}), 0, nbytes, m); m)
        before = Vector{Vector{UInt8}}(undef, n)
        Threads.@threads :static for t in 1:n
            before[t] = getmask()
        end
        msg = pin_threads_to_cores!()
        @test msg isa String
        after = Vector{Vector{UInt8}}(undef, n)
        Threads.@threads :static for t in 1:n
            after[t] = getmask()
        end
        if startswith(msg, "pinned")
            @test all(m -> sum(count_ones, m) == 1, after)   # each thread on exactly one CPU
            @test allunique(after)                            # and no two threads share one
        else
            @test occursin("not pinned", msg)
            @test after == before
        end
        Threads.@threads :static for t in 1:n
            ccall(:sched_setaffinity, Cint, (Cint, Csize_t, Ptr{UInt8}), 0, nbytes, before[t])
        end
    else
        @test occursin("not pinned", pin_threads_to_cores!())
    end
end
