# compilation cache

using Core.Compiler: retrieve_code_info, CodeInfo, MethodInstance, SSAValue, SlotNumber, ReturnNode
using Base: _methods_by_ftype

# generated function that crafts a custom code info to call the actual compiler.
# this gives us the flexibility to insert manual back edges for automatic recompilation.
#
# we also increment a global specialization counter and pass it along to index the cache.

const specialization_counter = Ref{UInt}(0)
@generated function specialization_id(job::CompilerJob{<:Any,<:Any,FunctionSpec{f,tt}}) where {f,tt}
    # get a hold of the method and code info of the kernel function
    sig = Tuple{f, tt.parameters...}
    # XXX: instead of typemax(UInt) we should use the world-age of the fspec
    mthds = _methods_by_ftype(sig, -1, typemax(UInt))
    Base.isdispatchtuple(tt) || return(:(error("$tt is not a dispatch tuple")))
    length(mthds) == 1 || return (:(throw(MethodError(job.source.f,job.source.tt))))
    mtypes, msp, m = mthds[1]
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance}, (Any, Any, Any), m, mtypes, msp)
    ci = retrieve_code_info(mi)::CodeInfo

    # generate a unique id to represent this specialization
    # TODO: just use the lower world age bound in which this code info is valid.
    #       (the method instance doesn't change when called functions are changed).
    #       but how to get that? the ci here always has min/max world 1/-1.
    # XXX: don't use `objectid(ci)` here, apparently it can alias (or the CI doesn't change?)
    id = (specialization_counter[] += 1)

    # prepare a new code info
    new_ci = copy(ci)
    empty!(new_ci.code)
    empty!(new_ci.codelocs)
    resize!(new_ci.linetable, 1)                # see note below
    empty!(new_ci.ssaflags)
    new_ci.ssavaluetypes = 0
    new_ci.edges = MethodInstance[mi]
    # XXX: setting this edge does not give us proper method invalidation, see
    #      JuliaLang/julia#34962 which demonstrates we also need to "call" the kernel.
    #      invoking `code_llvm` also does the necessary codegen, as does calling the
    #      underlying C methods -- which GPUCompiler does, so everything Just Works.

    # prepare the slots
    new_ci.slotnames = Symbol[Symbol("#self#"), :cache, :job, :compiler, :linker]
    new_ci.slotflags = UInt8[0x00 for i = 1:5]
    cache = SlotNumber(2)
    job = SlotNumber(3)
    compiler = SlotNumber(4)
    linker = SlotNumber(5)

    # call the compiler
    push!(new_ci.code, ReturnNode(id))
    push!(new_ci.ssaflags, 0x00)   # Julia's native compilation pipeline (and its verifier) expects `ssaflags` to be the same length as `code`
    push!(new_ci.codelocs, 1)   # see note below
    new_ci.ssavaluetypes += 1

    # NOTE: we keep the first entry of the original linetable, and use it for location info
    #       on the call to check_cache. we can't not have a codeloc (using 0 causes
    #       corruption of the back trace), and reusing the target function's info
    #       has as advantage that we see the name of the kernel in the backtraces.

    return new_ci
end

const disk_cache = parse(Bool, @load_preference("disk_cache", "false"))
const cache_key = @load_preference("cache_key", "")

"""
    enable_cache!(state=true)

Activate the GPUCompiler disk cache in the current environment.
You will need to restart your Julia environment for it to take effect.

!!! warning
    The disk cache is not automatically invalidated. It is sharded upon
    `cache_key` (see [`set_cache_key``](@ref)), the GPUCompiler version
    and your Julia version.
"""
function enable_cache!(state=true)
    @set_preferences!("disk_cache"=>state)
end

"""
    set_cache_key(key)

If you are deploying an application it is recommended that you use your
application name and version as a cache key. To minimize the risk of
encountering spurios cache hits.
"""
function set_cache_key(key)
    @set_preferences!("cache_key"=>key)
end

key(ver::VersionNumber) = "$(ver.major)_$(ver.minor)_$(ver.patch)"
cache_path() = @get_scratch!(cache_key * "-kernels-" * key(VERSION) * "-" * key(pkg_version))
clear_disk_cache!() = rm(cache_path(); recursive=true, force=true)


const cache_lock = ReentrantLock()
function cached_compilation(cache::AbstractDict,
                            @nospecialize(job::CompilerJob),
                            compiler::Function, linker::Function)
    # XXX: CompilerJob contains a world age, so can't be respecialized.
    #      have specialization_id take a f/tt and return a world to construct a CompilerJob?
    key = hash(job, specialization_id(job))
    force_compilation = compile_hook[] !== nothing

    # XXX: by taking the hash, we index the compilation cache directly with the world age.
    #      that's wrong; we should perform an intersection with the entry its bounds.

    # NOTE: no use of lock(::Function)/@lock/get! to keep stack traces clean
    lock(cache_lock)
    try
        obj = get(cache, key, nothing)
        if obj === nothing || force_compilation
            asm = nothing

            # can we load from the disk cache?
            if disk_cache && !force_compilation
                path = joinpath(cache_path(), "$key.jls")
                if isfile(path)
                    try
                        asm = deserialize(path)
                        @debug "Loading compiled kernel for $spec from $path"
                    catch ex
                        @warn "Failed to load compiled kernel at $path" exception=(ex, catch_backtrace())
                    end
                end
            end

            # compile
            if asm === nothing
                if compile_hook[] !== nothing
                    compile_hook[](job)
                end

                asm = compiler(job)

                if disk_cache && !isfile(path)
                    serialize(path, asm)
                end
            end

            # link (but not if we got here because of forced compilation)
            if obj === nothing
                obj = linker(job, asm)
                cache[key] = obj
            end
        end
        obj
    finally
        unlock(cache_lock)
    end
end
