module WGPUCore

using CEnum
using Infiltrator
using Tracy

include("utils.jl")
include("log.jl")
include("backendType.jl")
include("backend.jl")
include("adapter.jl")
include("instance.jl")
include("queue.jl")
include("device.jl")



include("droppable.jl")
include("buffer.jl")

include("shader.jl")


function requestAdapter(::WGPUAbstractBackend, canvas, powerPreference)
    @error "Backend is not defined yet"
end




mutable struct GPUTexture <: Droppable
    label::Any
    internal::Any
    device::Any
    texExtent::Any
    texInfo::Any
    desc::Any
end

mutable struct BufferDimensions
    height::UInt32
    width::UInt32
    padded_bytes_per_row::UInt32
    unpadded_bytes_per_row::UInt32
    function BufferDimensions(width, height)
        bytes_per_pixel = sizeof(UInt32)
        unpadded_bytes_per_row = width * bytes_per_pixel
        align = 256
        padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align
        padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding
        return new(height, width, padded_bytes_per_row, unpadded_bytes_per_row)
    end
end

function createTexture(
    gpuDevice,
    label,
    size,
    mipLevelCount,
    sampleCount,
    dimension,
    format,
    usage,
)
    textureExtent =
        cStruct(
            WGPUExtent3D;
            width = size[1],
            height = size[2],
            depthOrArrayLayers = size[3],
        )
    
    textureDesc = cStruct(
            WGPUTextureDescriptor;
            label = toCString(label),
            size = textureExtent |> concrete,
            mipLevelCount = mipLevelCount,
            sampleCount = sampleCount,
            dimension = dimension,
            format = format,
            usage = usage,
        ) 
    
    texture = GC.@preserve label wgpuDeviceCreateTexture(
        gpuDevice.internal[],
		textureDesc |> ptr,
    )

    texInfo = Dict(
        "size" => size,
        "mipLevelCount" => mipLevelCount,
        "sampleCount" => sampleCount,
        "dimension" => dimension,
        "format" => format,
        "usage" => usage,
    )

    GPUTexture(label, texture |> Ref, gpuDevice, textureExtent, texInfo, textureDesc)
end

mutable struct GPUTextureView <: Droppable
    label::Any
    internal::Any
    device::Any
    texture::Any
    size::Any
    desc::Any
end


function createView(gpuTexture::GPUTexture; dimension = nothing)
    gpuTextureInternal = gpuTexture.internal[]
    dimension = split(string(gpuTexture.texInfo["dimension"]), "_")[end]
    T = WGPUTextureViewDimension
    pairs = CEnum.name_value_pairs(T)
    for (key, value) in pairs
        pattern = split(string(key), "_")[end] # TODO MallocInfo
        if pattern == dimension # TODO partial matching will be good but tie break will happen
            dimension = T(value)
        end
    end
    texSize = gpuTexture.texInfo["size"]
    viewDescriptor =
        cStruct(
            WGPUTextureViewDescriptor;
            label = toCString(gpuTexture.label),
            format = gpuTexture.texInfo["format"],
            dimension = dimension,
            aspect = WGPUTextureAspect_All,
            baseMipLevel = 0, # TODO
            mipLevelCount = 1, # TODO
            baseArrayLayer = 0, # TODO
            arrayLayerCount = last(texSize),  # TODO
        )
    view = GC.@preserve gpuTextureInternal wgpuTextureCreateView(
        gpuTexture.internal[],
        viewDescriptor |> ptr,
    ) |> Ref
    return GPUTextureView(
        gpuTexture.label,
        view,
        gpuTexture.device,
        gpuTexture |> Ref,
        gpuTexture.texInfo["size"],
        viewDescriptor,
    )
end

mutable struct GPUSampler <: Droppable
    label::Any
    internal::Any
    device::Any
    desc::Any
end

function createSampler(
    gpuDevice;
    label = " SAMPLER DESCRIPTOR",
    addressModeU = WGPUAddressMode_ClampToEdge,
    addressModeV = WGPUAddressMode_ClampToEdge,
    addressModeW = WGPUAddressMode_ClampToEdge,
    magFilter = WGPUFilterMode_Nearest,
    minFilter = WGPUFilterMode_Nearest,
    mipmapFilter = WGPUMipmapFilterMode_Nearest,
    lodMinClamp = 0,
    lodMaxClamp = 32,
    compare = WGPUCompareFunction_Undefined,
    maxAnisotropy = 1,
)
	desc = cStruct(
        WGPUSamplerDescriptor;
        label = toCString(label),
        addressModeU = addressModeU,
        addressModeV = addressModeV,
        addressModeW = addressModeW,
        magFilter = magFilter,
        minFilter = minFilter,
        mipmapFilter = mipmapFilter,
        lodMinClamp = lodMinClamp,
        lodMaxClamp = lodMaxClamp,
        compare = compare == nothing ? 0 : compare,
        maxAnisotropy = maxAnisotropy,
    )
    sampler = wgpuDeviceCreateSampler(
            gpuDevice.internal[],
            desc |> ptr,
        ) |> Ref
    return GPUSampler(label, sampler, gpuDevice, desc)
end

mutable struct GPUBindGroupLayout <: Droppable
    label::Any
    internal::Any
    device::Any
    bindings::Any
    desc::Any
end

abstract type WGPUEntryType end

struct WGPUBufferEntry <: WGPUEntryType end

struct GPUBufferEntry
	entry
	layout
end

struct WGPUSamplerEntry <: WGPUEntryType end

struct GPUSamplerEntry
	entry
	layout
end

struct WGPUTextureEntry <: WGPUEntryType end

struct GPUTextureEntry
	entry
	layout
end

struct WGPUStorageTextureEntry <: WGPUEntryType end

struct GPUStorageTextureEntry
	entry
	layout
end


function createLayoutEntry(::Type{WGPUBufferEntry}; args...)
    # binding::Int,
    # visibility::Int,
    # buffertype::WGPUBufferBindingType)
    bufferBindingLayout = cStruct(
        WGPUBufferBindingLayout;
        type = getEnum(WGPUBufferBindingType, args[:type]),
    )
    entry =cStruct(
        WGPUBindGroupLayoutEntry;
        binding = args[:binding],
        visibility = getEnum(WGPUShaderStage, args[:visibility]),
        buffer = bufferBindingLayout |> concrete,
    )
    return GPUBufferEntry(entry, bufferBindingLayout)
end

function createLayoutEntry(::Type{WGPUSamplerEntry}; args...)
    # binding::Int,
    # visibility::Int,
    # sampertype::WGPUBufferBindingType
    samplerBindingLayout = cStruct(
        WGPUSamplerBindingLayout;
        type = getEnum(WGPUSamplerBindingType, args[:type]),
    )
    entry = cStruct(
        WGPUBindGroupLayoutEntry;
        binding = args[:binding],
        visibility = getEnum(WGPUShaderStage, args[:visibility]),
        sampler = samplerBindingLayout |> concrete,
    )
    return GPUSamplerEntry(entry, samplerBindingLayout)
end

function createLayoutEntry(::Type{WGPUTextureEntry}; args...)
    # binding::UInt32 = 0,
    # visibility::UInt32 = 0,
    # type::WGPUTextureSampleType = WGPUTextureSampleType_Float,
    # viewDimension::WGPUTextureViewDimension = WGPUTextureViewDimension_2D,
    # multisampled::Bool=false
	textureBindingLayout = cStruct(
        WGPUTextureBindingLayout;
        sampleType = getEnum(WGPUTextureSampleType, args[:sampleType]),
        viewDimension = getEnum(WGPUTextureViewDimension, args[:viewDimension]),
        multisampled = args[:multisampled],
    )
    entry = cStruct(
        WGPUBindGroupLayoutEntry;
        binding = args[:binding],
        visibility = getEnum(WGPUShaderStage, args[:visibility]),
        texture = textureBindingLayout |> concrete,
    )
    return GPUTextureEntry(entry, textureBindingLayout)
end

function createLayoutEntry(::Type{WGPUStorageTextureEntry}; args...)
    # binding,
    # visibility,
    # access::WGPUStorageTextureAccess,
    # format::WGPUTextureFormat;
    # viewDimension::WGPUTextureViewDimension=WGPUTextureViewDimension_2D
    storageTextureBindingLayout = cStruct(
        WGPUStorageTextureBindingLayout;
        access = getEnum(WGPUStorageTextureAccess, args[:access]),
        viewDimension = getEnum(WGPUTextureViewDimension, args[:viewDimension]),
        format = getEnum(WGPUTextureFormat, args[:format]),
    )

    entry = cStruct(
        WGPUBindGroupLayoutEntry;
        binding = args[:binding],
        visibility = getEnum(WGPUShaderStage, args[:visibility]),
        storageTexture = storageTextureBindingLayout |> concrete
    )
    return GPUStorageTexture(entry, storageTextureBindingLayout)
end

function createBindGroupEntry(::Type{GPUBuffer}; args...)
    cStruct(
        WGPUBindGroupEntry;
        binding = args[:binding],
        buffer = args[:buffer].internal[],
        offset = args[:offset],
        size = args[:size],
        sampler = C_NULL,
        textureView = C_NULL,
    )
end

function createBindGroupEntry(::Type{GPUTextureView}; args...)
    cStruct(
        WGPUBindGroupEntry;
        binding = args[:binding],
        textureView = args[:textureView].internal[],
    )
end

function createBindGroupEntry(::Type{GPUSampler}; args...)
    cStruct(
        WGPUBindGroupEntry;
        binding = args[:binding],
        sampler = args[:sampler].internal[],
    )
end

struct BindGroupLayoutEntryList
	cEntries
	layoutEntries
end

function makeLayoutEntryList(entries)
    @assert typeof(entries) <: Array "Entries should be an array"
    entryLen = length(entries)
    layoutEntries = []
    cEntries = convert(
    	Ptr{WGPUBindGroupLayoutEntry},
    	Libc.malloc(sizeof(WGPUBindGroupLayoutEntry)*entryLen)
   	)
    if entryLen > 0
        for (idx, entry) in enumerate(entries)
        	entryCntxt = createLayoutEntry(entry.first; entry.second...)
        	push!(layoutEntries, entryCntxt)
            unsafe_store!(cEntries, entryCntxt.entry |> concrete, idx)
        end
    end
    return BindGroupLayoutEntryList(
    		unsafe_wrap(Array, cEntries, entryLen; own=false),
    		layoutEntries
    	)
    		
end

function createBindGroupLayout(gpuDevice, label, entries)
    @assert typeof(entries.cEntries) <: Array "Entries should be an array"
    count = length(entries.cEntries)
    bindGroupLayout = C_NULL
    bindGroupLayoutDesc = nothing
    if count > 0
		bindGroupLayoutDesc = cStruct(
	       WGPUBindGroupLayoutDescriptor;
	       label = toCString(label),
	       entries = count == 0 ? C_NULL : entries.cEntries |> pointer, # assuming array of entries
	       entryCount = count,
	   	)

        bindGroupLayout = GC.@preserve label wgpuDeviceCreateBindGroupLayout(
            gpuDevice.internal[],
		 	bindGroupLayoutDesc |> ptr,
        )
    end
    GPUBindGroupLayout(label, Ref(bindGroupLayout), gpuDevice, entries, bindGroupLayoutDesc)
end

mutable struct GPUBindGroup <: Droppable
    label::Any
    internal::Any
    layout::Any
    device::Any
    bindings::Any
    desc::Any
end

struct BindGroupEntryList
	cEntries
	bgEntries
end

function makeBindGroupEntryList(entries)
    @assert typeof(entries) <: Array "Entries should be an array"
    entriesLen = length(entries)
    bgEntries = []
    cEntries = convert(
    	Ptr{WGPUBindGroupEntry},
    	Libc.malloc(sizeof(WGPUBindGroupEntry)*entriesLen)
    )
    if entriesLen > 0
        for (idx, entry) in enumerate(entries)
        	entryCStruct = createBindGroupEntry(entry.first; entry.second...)
        	push!(bgEntries, entryCStruct)
            unsafe_store!(cEntries, entryCStruct |> concrete, idx)
        end
    end
    return BindGroupEntryList(
    			unsafe_wrap(Array, cEntries, entriesLen; own=false),
    			bgEntries
			)
end

function createBindGroup(label, gpuDevice, bindingLayout, entries)
    @assert typeof(entries.cEntries) <: Array "Entries should be an array"
    count = length(entries.bgEntries)
    bindGroup = C_NULL
    bindGroupDesc = nothing
    if bindingLayout.internal[] != C_NULL && count > 0
        bindGroupDesc = GC.@preserve label cStruct(
	        WGPUBindGroupDescriptor;
	        label = toCString(label),
	        layout = bindingLayout.internal[],
	        entries = count == 0 ? C_NULL : entries.cEntries |> pointer,
	        entryCount = count,
	    )
        bindGroup = wgpuDeviceCreateBindGroup(
            gpuDevice.internal[],
			bindGroupDesc |> ptr
        )
    end
    GPUBindGroup(label, Ref(bindGroup), bindingLayout, gpuDevice, entries, bindGroupDesc)
end

# deprecated function
function makeBindGroupAndLayout(gpuDevice, bindingLayouts, bindings)
    @assert length(bindings) == length(bindingLayouts)
    cBindingLayoutsCntxtList = makeLayoutEntryList(bindingLayouts)
    cBindingsCntxtList = makeBindGroupEntryList(bindings)
    bindGroupLayout =
        createBindGroupLayout(gpuDevice, "Bind Group Layout", cBindingLayoutsCntxtList)
    bindGroup = createBindGroup("BindGroup", gpuDevice, bindGroupLayout, cBindingsCntxtList)
    return (bindGroupLayout, bindGroup)
end

mutable struct GPUPipelineLayout <: Droppable
    label::Any
    internal::Any
    device::Any
    descriptor::Any
    bindGroup::Any
    bindGroupLayout::Any
    cBindingsLayoutsList
    cBindingsList
end

function createPipelineLayout(gpuDevice, label, bindingLayouts, bindings)
    # bindGroupLayoutArray = Ptr{WGPUBindGroupLayoutImpl}()
    # if bindGroupLayoutObj.internal[] != C_NULL
        # bindGroupLayoutArray = bindGroupLayoutObj.internal[]
        # layoutCount = length(bindGroupLayoutArray)
    # else
    	# layoutCount = 0
    # end
    @assert length(bindings) == length(bindingLayouts)
    cBindingLayoutsList = makeLayoutEntryList(bindingLayouts)
    cBindingList = makeBindGroupEntryList(bindings)
    bindGroupLayout =
        createBindGroupLayout(gpuDevice, "Bind Group Layout", cBindingLayoutsList)
    bindGroup = createBindGroup("BindGroup", gpuDevice, bindGroupLayout, cBindingList)
    bindGroupLayoutArray = []
    if bindGroupLayout.internal[] != C_NULL
        bindGroupLayoutArray = map((x) -> x.internal[], [bindGroupLayout])
        layoutCount = length(bindGroupLayoutArray) # will always be one
    else
    	layoutCount = 0
    end 
    pipelineDescriptor = GC.@preserve bindGroupLayoutArray label cStruct(
        WGPUPipelineLayoutDescriptor;
        label = toCString(label),
        bindGroupLayouts = layoutCount == 0 ? C_NULL : bindGroupLayoutArray |> pointer,
        bindGroupLayoutCount = layoutCount,
    )
    pipelineLayout =
        wgpuDeviceCreatePipelineLayout(gpuDevice.internal[], pipelineDescriptor |> ptr)
    GPUPipelineLayout(
        label,
        pipelineLayout |> Ref,
        gpuDevice,
        pipelineDescriptor,
        bindGroup,
        bindGroupLayout,
        cBindingLayoutsList,
        cBindingList
    )
end


mutable struct GPUComputePipeline <: Droppable
    label::Any
    internal::Any
    device::Any
    layout::Any
    desc::Any
end

mutable struct ComputeStage <: Droppable
    internal::Any
    entryPoint::Any
end

function createComputeStage(shaderModule, entryPoint::String)
    computeStage = cStruct(
        WGPUProgrammableStageDescriptor;
        _module = shaderModule.internal[],
        entryPoint = toCString(entryPoint),
    )
    return ComputeStage(computeStage, entryPoint)
end

function createComputePipeline(gpuDevice, label, pipelinelayout, computeStage)
	desc =	cStruct(
        WGPUComputePipelineDescriptor;
        label = toCString(label),
        layout = pipelinelayout.internal[],
        compute = computeStage.internal |> concrete,
    )
    computepipeline = GC.@preserve label wgpuDeviceCreateComputePipeline(
        gpuDevice.internal[],
        desc |> ptr,
    ) |> Ref
    GPUComputePipeline(label, computepipeline, gpuDevice, pipelinelayout, desc)
end

mutable struct GPUVertexAttribute
    internal::Any
    strongRefs::Any
end

function createEntry(::Type{GPUVertexAttribute}; args...)
    GPUVertexAttribute(
        cStruct(
            WGPUVertexAttribute;
            format = getEnum(WGPUVertexFormat, args[:format]),
            offset = args[:offset],
            shaderLocation = args[:shaderLocation],
        ),
        args,
    )
end

mutable struct GPUVertexBufferLayout
    internal::Any
    attributeArgs
    args
    attributeArrayPtr
    attributeArray
end

function createEntry(::Type{GPUVertexBufferLayout}; args...)
    attributeArgs = args[:attributes]
	numAttrs = length(attributeArgs)
    attributeArrayPtr = convert(
    	Ptr{WGPUVertexAttribute},
    	Libc.malloc(sizeof(WGPUVertexAttribute)*numAttrs)
    )
    attributeObjs = GPUVertexAttribute[]

    for (idx, attribute) in enumerate(attributeArgs)
        obj = createEntry(GPUVertexAttribute; attribute.second...)
        push!(attributeObjs, obj)
        unsafe_store!(attributeArrayPtr, obj.internal |> concrete, idx)
    end

    aref = GC.@preserve attributeArrayPtr cStruct(
        WGPUVertexBufferLayout;
        arrayStride = args[:arrayStride],
        stepMode = getEnum(WGPUVertexStepMode, args[:stepMode]),
        attributes = attributeArrayPtr,
        attributeCount = numAttrs,
    )
    return GPUVertexBufferLayout(aref, attributeArgs, args, attributeArrayPtr, attributeObjs)
end

mutable struct GPUVertexState
    internal::Any
    shader
    buffers
    bufferDescriptor
    bufferArray
    entryPoint
    args
end

function createEntry(::Type{GPUVertexState}; args...)
    buffers = args[:buffers]
	bufferLen = length(buffers)
	
    bufferDescArrayPtr = convert(
    	Ptr{WGPUVertexBufferLayout},
    	Libc.malloc(sizeof(WGPUVertexBufferLayout)*bufferLen)
    )
    
    buffersArrayObjs = GPUVertexBufferLayout[]
    entryPointArg = args[:entryPoint]

    for (idx, buffer) in enumerate(buffers)
        obj = createEntry(buffer.first; buffer.second...)
        push!(buffersArrayObjs, obj)
        unsafe_store!(bufferDescArrayPtr, obj.internal |> concrete, idx)
    end

    shader = args[:_module]
    if shader != C_NULL
        shaderInternal = shader.internal
    else
        shaderInternal = C_NULL |> Ref
    end

    aRef = GC.@preserve entryPointArg bufferDescArrayPtr cStruct(
        WGPUVertexState;
        _module = shaderInternal[],
        entryPoint = toCString(entryPointArg),
        buffers = length(buffers) == 0 ? C_NULL : bufferDescArrayPtr,
        bufferCount = length(buffers),
    )
    
    GPUVertexState(
    	aRef |> Ref, 
    	shader, 
    	buffers, 
    	bufferDescArrayPtr, 
    	buffersArrayObjs, 
    	entryPointArg, 
    	args
   	)
end

mutable struct GPUPrimitiveState
    internal::Any
    strongRefs::Any
end

function createEntry(::Type{GPUPrimitiveState}; args...)
    a = cStruct(
        WGPUPrimitiveState;
        topology = getEnum(WGPUPrimitiveTopology, args[:topology]),
        stripIndexFormat = getEnum(WGPUIndexFormat, args[:stripIndexFormat]),
        frontFace = getEnum(WGPUFrontFace, args[:frontFace]), # TODO 
        cullMode = getEnum(WGPUCullMode, args[:cullMode]),
    ) 
    return GPUPrimitiveState(a |> Ref, args)
end

mutable struct GPUStencilFaceState
    internal::Any
    strongRefs::Any
end


defaultInit(::Type{WGPUStencilFaceState}) = begin
    cStruct(
    	WGPUStencilFaceState; 
    	compare = WGPUCompareFunction_Always,
    	failOp = WGPUStencilOperation_Keep,
    	depthFailOp = WGPUStencilOperation_Keep,
    	passOp = WGPUStencilOperation_Keep
   	)
end

mutable struct GPUDepthStencilState
    internal::Any
    strongRefs::Any
end

function createEntry(::Type{GPUDepthStencilState}; args...)
	a = CStruct(WGPUDepthStencilState)
    if length(args) > 0 && args != C_NULL
	    a.format = args[:format]
	    a.depthWriteEnabled = args[:depthWriteEnabled]
	    a.depthCompare = args[:depthCompare]
	    a.stencilReadMask = get(args, :stencilReadMask, 0xffffffff)
	    a.stencilWriteMask = get(args, :stencilWriteMask, 0xffffffff)
	    a.stencilFront = defaultInit(WGPUStencilFaceState) |> concrete
	    a.stencilBack = defaultInit(WGPUStencilFaceState) |> concrete
	    a.depthBias = 0 # TODO
	    a.depthBiasSlopeScale = 0 # TODO
	    a.depthBiasClamp = 0 # TODO
    else
        setfield!(a, :ptr, Ptr{WGPUDepthStencilState}(0))
    end
    return GPUDepthStencilState(a |> Ref, args)
end

mutable struct GPUMultiSampleState
    internal::Any
    strongRefs::Any
end

function createEntry(::Type{GPUMultiSampleState}; args...)
    a =
        cStruct(
            WGPUMultisampleState;
            count = args[:count],
            mask = args[:mask],
            alphaToCoverageEnabled = args[:alphaToCoverageEnabled],
        )
    return GPUMultiSampleState(a |> Ref, args)
end

mutable struct GPUBlendComponent
    internal::Any
    strongRefs::Any
end

function createEntry(::Type{GPUBlendComponent}; args...)
    a = cStruct(
        WGPUBlendComponent;
        srcFactor = getEnum(WGPUBlendFactor, args[:srcFactor]),
        dstFactor = getEnum(WGPUBlendFactor, args[:dstFactor]),
        operation = getEnum(WGPUBlendOperation, args[:operation]),
    ) 
    return GPUBlendComponent(a |> Ref, args)
end

mutable struct GPUBlendState
    internal::Any
    strongRefs::Any
end

function createEntry(::Type{GPUBlendState}; args...)
    a = cStruct(WGPUBlendState; color = args[:color], alpha = args[:alpha])
    return GPUBlendState(a |> Ref, args)
end

mutable struct GPUColorTargetState
    internal::Any
    strongRefs::Any
end

function createEntry(::Type{GPUColorTargetState}; args...)
    kargs = Dict(args)
    colorEntry = createEntry(GPUBlendComponent; args[:color]...)
    alphaEntry = createEntry(GPUBlendComponent; args[:alpha]...)
    # blendArgs = [:color => colorEntry |> concrete, :alpha => alphaEntry |> concrete]
    blend = CStruct(WGPUBlendState)
    blend.color = colorEntry.internal[] |> concrete
    blend.alpha = alphaEntry.internal[] |> concrete
    kargs[:writeMask] = get(kargs, :writeMask, WGPUColorWriteMask_All)
    aref = GC.@preserve args blend cStruct(
        WGPUColorTargetState;
        format = kargs[:format],
        blend = blend |> ptr,
        writeMask = kargs[:writeMask],
    )
    return GPUColorTargetState(
        aref |> Ref,
        (blend |> Ref, colorEntry, alphaEntry, args, kargs),
    )
end

mutable struct GPUFragmentState
    internal::Any
    strongRefs::Any
end

function createEntry(::Type{GPUFragmentState}; args...)
    targetsArg = args[:targets]
    targetsLen = length(targetsArg)
    ctargets = convert(
    	Ptr{WGPUColorTargetState},
    	Libc.malloc(sizeof(WGPUColorTargetState)*targetsLen)
    )
    targetObjs = GPUColorTargetState[]

    for (idx, target) in enumerate(targetsArg)
        obj = createEntry(target.first; target.second...)
        push!(targetObjs, obj)
        unsafe_store!(ctargets, obj.internal[] |> concrete, idx)
    end

    entryPointArg = args[:entryPoint]
    shader = args[:_module]
    aref = GC.@preserve entryPointArg ctargets shader cStruct(
        WGPUFragmentState;
        _module = shader.internal[],
        entryPoint = toCString(entryPointArg),
        targets = ctargets,
        targetCount = targetsLen,
    ) 
    
    return GPUFragmentState(
        aref |> Ref,
        (
            aref,
            args,
            shader,
            entryPointArg |> Ref,
            targetsArg,
            ctargets |> Ref,
            targetObjs |> Ref,
            entryPointArg,
            shader,
        ),
    )
end

mutable struct GPURenderPipeline <: Droppable
    label::Any
    internal::Any
    descriptor::Any
    renderArgs::Any
    device::Any
    layout::Any
    vertexState::Any
    primitiveState::Any
    depthStencilState::Any
    MultiSampleState::Any
    FragmentState::Any
end

function createRenderPipeline(
    gpuDevice,
    pipelinelayout,
    renderpipeline;
    label = "RenderPipeLine",
)
    renderArgs = Dict()
    for state in renderpipeline
        obj = createEntry(state.first; state.second...)
        renderArgs[state.first] = obj
    end

    vertexState = renderArgs[GPUVertexState].internal
    primitiveState = renderArgs[GPUPrimitiveState].internal
    depthStencilState = renderArgs[GPUDepthStencilState].internal
    multiSampleState = renderArgs[GPUMultiSampleState].internal
    fragmentState = renderArgs[GPUFragmentState].internal

    pipelineDesc = GC.@preserve vertexState primitiveState depthStencilState multiSampleState fragmentState label  cStruct(
        WGPURenderPipelineDescriptor;
        label = toCString(label),
        layout = pipelinelayout.internal[],
        vertex = vertexState[] |> concrete,
        primitive = primitiveState[] |> concrete,
        depthStencil = let ds = depthStencilState[]
         	 ds == C_NULL ? C_NULL : ds |> ptr 
       	end,
        multisample = multiSampleState[] |> concrete,
        fragment = fragmentState[] |> ptr,
    )

    renderpipeline = GC.@preserve pipelineDesc vertexState primitiveState depthStencilState multiSampleState fragmentState wgpuDeviceCreateRenderPipeline(
        gpuDevice.internal[],
        pipelineDesc |> ptr,
    ) |> Ref

    return GPURenderPipeline(
        label,
        renderpipeline,
        pipelineDesc,
        renderArgs,
        gpuDevice,
        pipelinelayout,
        vertexState,
        primitiveState,
        depthStencilState,
        multiSampleState,
        fragmentState,
    )
end

mutable struct GPUColorAttachments
    internal::Any
    attachmentObjs
    args
end

mutable struct GPUColorAttachment
    internal::Any
    textureView::Any
    args
end


mutable struct GPUDepthStencilAttachments
    internal::Any
    attachmentObjs::Any
    args
end

mutable struct GPUDepthStencilAttachment
    internal::Any
    args
    depthView
end

mutable struct GPURenderPassEncoder
    label::Any
    internal::Any
    pipeline::Any
    cmdEncoder::Any
    desc::Any
end

function createEntry(::Type{GPUColorAttachment}; args...)
    textureView = args[:view]
    a = cStruct(
        WGPURenderPassColorAttachment;
        view = textureView.internal[],
        resolveTarget = args[:resolveTarget],
        clearValue = WGPUColor(args[:clearValue]...),
        loadOp = args[:loadOp],
        storeOp = args[:storeOp],
    )
    return GPUColorAttachment(a |> Ref, textureView, args)
end


function createEntry(::Type{GPUColorAttachments}; args...)
	attachmentsArg = get(args, :attachments, [])
    attachments = convert(
    	Ptr{WGPURenderPassColorAttachment},
    	Libc.malloc(sizeof(WGPURenderPassColorAttachment)*length(attachmentsArg))
    )
    attachmentObjs = GPUColorAttachment[]
    for (idx, attachment) in enumerate(attachmentsArg)
        obj = createEntry(attachment.first; attachment.second...) # TODO MallocInfo
        push!(attachmentObjs, obj)
        unsafe_store!(attachments, obj.internal[] |> concrete, idx)
    end
    return GPUColorAttachments(attachments |> Ref, attachmentObjs, args)
end


function createEntry(::Type{GPUDepthStencilAttachment}; args...)
    depthview = args[:view]
    a = GC.@preserve depthview cStruct(
        WGPURenderPassDepthStencilAttachment;
        view = depthview.internal[],
        depthClearValue = args[:depthClearValue],
        depthLoadOp = args[:depthLoadOp],
        depthStoreOp = args[:depthStoreOp],
        stencilLoadOp = get(args, :stencilLoadOp, WGPULoadOp_Clear),
        stencilStoreOp = get(args, :stencilStoreOp, WGPUStoreOp_Store),
    )
    return GPUDepthStencilAttachment(a, args, depthview)
end


function createEntry(::Type{GPUDepthStencilAttachments}; args...)
	attachmentArgs = get(args, :attachments, [])
	attachmentLen = length(attachmentArgs)
	attachmentsPtr = convert(
		Ptr{WGPURenderPassDepthStencilAttachment},
		Libc.malloc(sizeof(WGPURenderPassDepthStencilAttachment)*attachmentLen)
	)
    attachmentObjs = GPUDepthStencilAttachment[]
    for (idx, attachment) in enumerate(attachmentArgs)
        obj = createEntry(attachment.first; attachment.second...) # TODO MallocInfo
        push!(attachmentObjs, obj)
        unsafe_store!(attachmentsPtr, obj.internal |> concrete, idx)
    end
    return GPUDepthStencilAttachments(
        attachmentsPtr |> Ref,
        attachmentObjs,
        args
    )
end

mutable struct GPUCommandBuffer <: Droppable
    label::Any
    internal::Any
    device::Any
    desc::Any
end


function createCommandBuffer()

end


mutable struct GPUCommandEncoder <: Droppable
    label::Any
    internal::Any
    device::Any
    desc::Any
end


mutable struct GPUComputePassEncoder
    label::Any
    internal::Any
    cmdEncoder::Any
    desc::Any
end


function createCommandEncoder(gpuDevice, label)
    cmdEncDesc = GC.@preserve label cStruct(
        WGPUCommandEncoderDescriptor;
        label = toCString(label),
    )
    commandEncoder =
        wgpuDeviceCreateCommandEncoder(
            gpuDevice.internal[],
            cmdEncDesc |> ptr,
        ) |> Ref
    return GPUCommandEncoder(label, commandEncoder, gpuDevice, cmdEncDesc)
end

function beginComputePass(
    cmdEncoder::GPUCommandEncoder;
    label = " COMPUTE PASS DESCRIPTOR ",
    timestampWrites = [],
)
    desc =
        GC.@preserve label cStruct(
        	WGPUComputePassDescriptor; label = toCString(label)
       	)
    computePass = wgpuCommandEncoderBeginComputePass(cmdEncoder.internal[], desc |> ptr) |> Ref
    GPUComputePassEncoder(label, computePass, cmdEncoder, desc)
end

function beginRenderPass(
    cmdEncoder::GPUCommandEncoder,
    renderPipelinePairs;
    label = " BEGIN RENDER PASS ",
)
	attachment = nothing
	colorAttachmentsIn = nothing
	depthStencilAttachmentIn = nothing
    for config in renderPipelinePairs[]
    	attachment = createEntry(config.first; config.second...)
    	if config.first == GPUColorAttachments
    		colorAttachmentsIn = attachment
    	elseif config.first == GPUDepthStencilAttachments
    		depthStencilAttachmentIn = attachment
    	end
    end
    # Both color and depth attachments requires pointer
    desc = GC.@preserve label cStruct(
        WGPURenderPassDescriptor;
        label = toCString(label),
        colorAttachments = let ca = colorAttachmentsIn
            length(ca.internal[]) > 0 ? ca.internal[] : C_NULL
        end,
        colorAttachmentCount = length(colorAttachmentsIn.internal[]),
        depthStencilAttachment = let da = depthStencilAttachmentIn
                    length(da.attachmentObjs) > 0 ? da.internal[] : C_NULL
                end,
    )
    renderPass = wgpuCommandEncoderBeginRenderPass(cmdEncoder.internal[], desc |> ptr)
    GPURenderPassEncoder(
        label,
        renderPass |> Ref,
        renderPipelinePairs,
        cmdEncoder,
        desc,
    )
end

function copyBufferToBuffer(
    cmdEncoder::GPUCommandEncoder,
    source::GPUBuffer,
    sourceOffset::Int,
    destination::GPUBuffer,
    destinationOffset::Int,
    size::Int,
)
    @assert sourceOffset % 4 == 0 "Source offset must be multiple of 4"
    @assert destinationOffset % 4 == 0 "Destination offset must be a multiple of 4"
    @assert size % 4 == 0 "Size must be a multiple of 4"

    wgpuCommandEncoderCopyBufferToBuffer(
        cmdEncoder.internal[],
        source.internal[],
        sourceOffset,
        destination.internal[],
        destinationOffset,
        size,
    )
end

function copyBufferToTexture(
    cmdEncoder::GPUCommandEncoder,
    source::Dict{Symbol,Any},
    destination::Dict{Symbol,Any},
    copySize::Dict{Symbol,Int64},
)
    rowAlignment = 256
    bytesPerRow = source[:layout][:bytesPerRow]
    @assert bytesPerRow % rowAlignment == 0 "BytesPerRow must be multiple of $rowAlignment"
    origin = get(source, :origin, [:x => 0, :y => 0, :z => 0] |> Dict)
    cOrigin = cStruct(WGPUOrigin3D; origin...)
    cDestination =
        cStruct(
            WGPUImageCopyTexture;
            texture = source[:texture].internal[],
            mipLevel = get(source, :mipLevel, 0),
            origin = cOrigin |> concrete,
            aspect = getEnum(WGPUTextureAspect, "All"),
        )
    texLayout = cStruct(WGPUTextureDataLayout; destination[:layout]...)
    cSource =
        cStruct(
            WGPUImageCopyBuffer;
            buffer = destination[:buffer].internal[],
            layout = texLayout |> concrete,
        )
    
    cCopySize = cStruct(WGPUExtent3D; copySize...)

    wgpuCommandEncoderCopyBufferToTexture(
        cmdEncoder.internal[],
        cSource  |> ptr,
        cDestination  |> ptr,
        cCopySize |> ptr,
    )
end

function copyTextureToBuffer(
    cmdEncoder::GPUCommandEncoder,
    source::Dict{Symbol,Any},
    destination::Dict{Symbol,Any},
    copySize::Dict{Symbol,Int64},
)
    rowAlignment = 256
    dest = Dict(destination)
    bytesPerRow = dest[:layout][:bytesPerRow]
    @assert bytesPerRow % rowAlignment == 0 "BytesPerRow must be multiple of $rowAlignment"
    origin = get(source, :origin, [:x => 0, :y => 0, :z => 0] |> Dict)
    cOrigin = cStruct(WGPUOrigin3D; origin...)
    cSource =
        cStruct(
            WGPUImageCopyTexture;
            texture = source[:texture].internal[],
            mipLevel = get(source, :mipLevel, 0),
            origin = cOrigin |> concrete,
            aspect = getEnum(WGPUTextureAspect, "All"),
        )
   	textureLayout =	cStruct(
            WGPUTextureDataLayout;
            destination[:layout]..., # should document these obscure
        )
    cDestination =
        cStruct(
            WGPUImageCopyBuffer;
            buffer = destination[:buffer].internal[],
            layout = textureLayout  |> concrete,
        )
    cCopySize = cStruct(WGPUExtent3D; copySize...)

    wgpuCommandEncoderCopyTextureToBuffer(
        cmdEncoder.internal[],
        cSource  |> ptr,
        cDestination  |> ptr,
        cCopySize |> ptr,
    )
end

function copyTextureToTexture(
    cmdEncoder::GPUCommandEncoder,
    source::Dict{Symbol,Any},
    destination::Dict{Symbol,Any},
    copySize::Dict{Symbol,Int64},
)
    origin1 = get(source, :origin, [:x => 0, :y => 0, :z => 0])
    cOrigin1 = cStruct(WGPUOrigin3D; origin1...)

    cSource =
        cStruct(
            WGPUImageCopyTexture;
            texture = source[:texture].internal[],
            mipLevel = get(source, :mipLevel, 0),
            origin = cOrigin1  |> concrete,
        )

    origin2 = get(destination, :origin, [:x => 0, :y => 0, :z => 0])

    cOrigin2 = cStruct(WGPUOrigin3D; origin2...)

    cDestination =
        cStruct(
            WGPUImageCopyTexture;
            texture = destination[:texture].internal[],
            mipLevel = get(destination, :mipLevel, 0),
            origin = cOrigin2  |> concrete,
        )

    cCopySize = cStruct(WGPUExtent3D; copySize...)

    wgpuCommandEncoderCopyTextureToTexture(
        cmdEncoder.internal[],
        cSource  |> ptr,
        cDestination  |> ptr,
        cCopySize  |> ptr,
    )

end

function finish(cmdEncoder::GPUCommandEncoder; label = " CMD ENCODER COMMAND BUFFER ")
	desc = cStruct(WGPUCommandBufferDescriptor; label = toCString(label))
    cmdEncoderFinish = wgpuCommandEncoderFinish(
        cmdEncoder.internal[],
        desc |> ptr,
    )
    cmdEncoder.internal[] = C_NULL # Just to avoid 'Cannot remove a vacant resource'
    return GPUCommandBuffer(label, Ref(cmdEncoderFinish), cmdEncoder, desc)
end


function createRenderBundleEncoder()

end


function createComputePassEncoder()

end


function createRenderPassEncoder()

end

function setPipeline(computePass::GPUComputePassEncoder, pipeline)
    wgpuComputePassEncoderSetPipeline(computePass.internal[], pipeline.internal[])
end

function setBindGroup(
    computePass::GPUComputePassEncoder,
    index::Int,
    bindGroup::GPUBindGroup,
    dynamicOffsetsData::Vector{UInt32},
    start::Int,
    dataLength::Int,
)
    offsetcount = length(dynamicOffsetsData)
    setbindgroup = wgpuComputePassEncoderSetBindGroup(
        computePass.internal[],
        index,
        bindGroup.internal[],
        offsetcount,
        (offsetcount == 0) ? C_NULL : pointer(dynamicOffsetsData), # TODO GCCHECK
    )
    return nothing
end

function setBindGroup(
    renderPass::GPURenderPassEncoder,
    index::Int,
    bindGroup::GPUBindGroup,
    dynamicOffsetsData::Vector{UInt32},
    start::Int,
    dataLength::Int,
)
    offsetcount = length(dynamicOffsetsData)
    setbindgroup = wgpuRenderPassEncoderSetBindGroup(
        renderPass.internal[],
        index,
        bindGroup.internal[],
        offsetcount,
        offsetcount == 0 ? C_NULL : pointer(dynamicOffsetsData), # TODO GCCHECK
    )
    return nothing
end

function dispatchWorkGroups(
    computePass::GPUComputePassEncoder,
    countX,
    countY = 1,
    countZ = 1,
)
    wgpuComputePassEncoderDispatchWorkgroups(computePass.internal[], countX, countY, countZ)
end

function dispatchWorkGroupsIndirect(
    computePass::GPUComputePassEncoder,
    indirectBuffer,
    indirectOffset,
)
    bufferId = indirectBuffer.internal[]
    wgpuComputePassEncoderDispatchIndirect(computePass.internal[], bufferId, indirectOffset)
end


function endComputePass(computePass::GPUComputePassEncoder)
    wgpuComputePassEncoderEnd(computePass.internal[])
end

function setViewport(
    renderPass::GPURenderPassEncoder,
    x,
    y,
    width,
    height,
    minDepth,
    maxDepth,
)
    wgpuRenderPassEncoderSetViewport(
        renderPass.internal[],
        float(x),
        float(y),
        float(width),
        float(height),
        float(minDepth),
        float(maxDepth),
    )
end

function setScissorRect(renderPass::GPURenderPassEncoder, x, y, width, height)
    wgpuRenderPassEncoderSetScissorRect(
        renderPass.internal[],
        Int.([x, y, width, height])...,
    )
end

function setPipeline(
    renderPassEncoder::GPURenderPassEncoder,
    renderpipeline::GPURenderPipeline,
)
    wgpuRenderPassEncoderSetPipeline(
        renderPassEncoder.internal[],
        renderpipeline.internal[],
    )
end

function setIndexBuffer(
    rpe::GPURenderPassEncoder,
    buffer,
    indexFormat;
    offset = 0,
    size = nothing,
)
    if size == nothing
        size = buffer.size - offset
    end
    cIndexFormat = getEnum(WGPUIndexFormat, indexFormat)
    wgpuRenderPassEncoderSetIndexBuffer(
        rpe.internal[],
        buffer.internal[],
        cIndexFormat,
        offset,
        size,
    )
end

function setVertexBuffer(
    rpe::GPURenderPassEncoder,
    slot,
    buffer,
    offset = 0,
    size = nothing,
)
    if size == nothing
        size = buffer.size - offset
    end
    wgpuRenderPassEncoderSetVertexBuffer(
        rpe.internal[],
        slot,
        buffer.internal[],
        offset,
        size,
    )
end

function draw(
    renderPassEncoder::GPURenderPassEncoder,
    vertexCount;
    instanceCount = 1,
    firstVertex = 0,
    firstInstance = 0,
)
    wgpuRenderPassEncoderDraw(
        renderPassEncoder.internal[],
        vertexCount,
        instanceCount,
        firstVertex,
        firstInstance,
    )
end

function drawIndexed(
    renderPassEncoder::GPURenderPassEncoder,
    indexCount;
    instanceCount = 1,
    firstIndex = 0,
    baseVertex = 0,
    firstInstance = 0,
)
    wgpuRenderPassEncoderDrawIndexed(
        renderPassEncoder.internal[],
        indexCount,
        instanceCount,
        firstIndex,
        baseVertex,
        firstInstance,
    )
end

function endEncoder(renderPass::GPURenderPassEncoder)
    wgpuRenderPassEncoderEnd(renderPass.internal[])
end

function submit(queue::GPUQueue, commandBuffers)
    commandBufferList = map((cmdbuf) -> cmdbuf.internal[], commandBuffers)
    GC.@preserve commandBufferList wgpuQueueSubmit(
        queue.internal[],
        length(commandBuffers),
        commandBufferList |> pointer,
    )
    for cmdbuf in commandBuffers
        cmdbuf.internal[] = C_NULL
    end
end

function writeTexture(queue::GPUQueue; args...)
    args = Dict(args)
    dst = args[:dst]
    for i in dst
        @eval $(i.first) = $(i.second)
    end

    cOrigin = WGPUOrigin3D(origin...) |> Ref

    destination =
        cStruct(
            WGPUImageCopyTexture;
            texture = texture.internal[],
            mipLevel = mipLevel,
            origin = cOrigin[],
        ) 

    layout = args[:layout]
    for i in layout
        @eval $(i.first) = $(i.second)
    end
    cDataLayout =
        cStruct(
            WGPUTextureDataLayout;
            offset = offset,
            bytesPerRow = bytesPerRow,
            rowsPerImage = rowsPerImage,
        )
    texSize = args[:textureSize]
    size =
        cStruct(
            WGPUExtent3D;
            width = texSize[1],
            height = texSize[2],
            depthOrArrayLayers = texSize[3],
        ) 
    texData = args[:textureData]
    dataLength = length(texData)
    texDataPtr = convert(Ptr{eltype(texData)}, Libc.malloc(sizeof(texData)))
    unsafe_copyto!(texDataPtr, pointer(texData), dataLength)
    GC.@preserve texDataPtr wgpuQueueWriteTexture(
        queue.internal[],
        destination |> ptr,
        texDataPtr,
        dataLength,
        cDataLayout |> ptr,
        size |> ptr,
    )
end


function readTexture(gpuDevice::GPUDevice, cntxt)
    usage = ["CopyDst", "MapRead"]
    cntxtSize = determineSize(cntxt)
    bufferDims = BufferDimensions(cntxtSize...)
    bufferSize = bufferDims.padded_bytes_per_row*bufferDims.height
    outputBuffer = createBuffer(
        "OUTPUT BUFFER",
        gpuDevice,
        bufferSize,
        usage,
        false,
    )

    nextTexture = getCurrentTexture(cntxt)
    
    cmdEncoder = createCommandEncoder(gpuDevice, "readTexture command")
    
    copyTextureToBuffer(
        cmdEncoder,
        [
            :texture => nextTexture,
            :mipLevel => 0,
            :origin => [
                :x => 0,
                :y => 0,
                :z => 0,
            ] |> Dict,
        ] |> Dict,
        [
            :buffer => outputBuffer,
            :layout => [
                :offset => 0,
                :bytesPerRow => bufferDims.padded_bytes_per_row,
                :rowsPerImage => WGPU_COPY_STRIDE_UNDEFINED,
            ] |> Dict,
        ] |> Dict,
        [
            :width => bufferDims.width,
            :height => bufferDims.height,
            :depthOrArrayLayers => 1,
        ] |> Dict
    )
    
    submit(gpuDevice.queue, [finish(cmdEncoder)])

    data = mapRead(outputBuffer)

    destroy(outputBuffer)
    
    return data
end

function saveTexture(gpuDevice, cntxt, name::String)
    data = readTexture(gpuDevice, cntxt)
    datareshaped = reshape(data, (4, (cntxt.size |> reverse)...) .|> Int)
    img = reinterpret(RGBA{N0f8}, datareshaped) |> (x) -> reshape(x, cntxt.size)
    save(name, img |> adjoint)
end

function readBuffer(gpuDevice, buffer, bufferOffset, size)
    # TODO more implementation is required
    # Took shortcut
    usage = ["CopyDst", "MapRead"]
    tmpBuffer = WGPUCore.createBuffer(" READ BUFFER TEMP ", gpuDevice, size, usage, false)
    commandEncoder = createCommandEncoder(gpuDevice, " READ BUFFER COMMAND ENCODER ")
    copyBufferToBuffer(commandEncoder, buffer, bufferOffset, tmpBuffer, 0, size)
    submit(gpuDevice.queue, [finish(commandEncoder)])
    data = mapRead(tmpBuffer)
    destroy(tmpBuffer)
    return data
end

function writeBuffer(queue::GPUQueue, buffer, data; dataOffset = 0, size = nothing)
    # TODO checks
    wgpuQueueWriteBuffer(queue.internal[], buffer.internal[], 0, data, sizeof(data))
end

include("canvas.jl")

function destroy(texView::GPUTextureView)
    if texView.internal[] != C_NULL
        tmpTex = texView.internal[]
        texView.internal[] = C_NULL
        wgpuTextureViewRelease(tmpTex)
    end
end

function Base.setproperty!(texview::GPUTextureView, s::Symbol, value)
    if s == :internal && texview.internal[] != C_NULL
        if value == nothing || value == C_NULL
        	@info "Destroying GPUTextureView"
            destroy(texview)
        end
    end
end

function destroy(tex::GPUTexture)
    if tex.internal[] != C_NULL
        tmpTex = tex.internal[]
        tex.internal[] = C_NULL
        wgpuTextureRelease(tmpTex)
    end
end

function Base.setproperty!(tex::GPUTexture, s::Symbol, value)
    if s == :internal && tex.internal[] != C_NULL
        if value == nothing || value == C_NULL
        	@info "Destroying GPUTexture"
            destroy(tex)
        end
    end
end

function destroy(sampler::GPUSampler)
    if sampler.internal[] != C_NULL
        tmpSampler = sampler.internal[]
        sampler.internal[] = C_NULL
        wgpuSamplerRelease(tmpSampler)
    end
end

function Base.setproperty!(sampler::GPUSampler, s::Symbol, value)
    if s == :internal && sampler.internal[] != C_NULL
        if value == nothing || value == C_NULL
        	@info "Destroying GPUSampler"
            destroy(sampler)
        end
    end
end

function destroy(layout::GPUBindGroupLayout)
    if layout.internal[] != C_NULL
        tmpLayout = layout.internal[]
        layout.internal[] = C_NULL
        wgpuBindGroupLayoutRelease(tmpLayout)
    end
end

function Base.setproperty!(layout::GPUBindGroupLayout, s::Symbol, value)
    if s == :internal && layout.internal[] != C_NULL
        if value == nothing || value == C_NULL
        	@info "Destroying GPUBindGroupLayout"
            destroy(layout)
        end
    end
end

function destroy(bindGroup::GPUBindGroup)
    if bindGroup.internal[] != C_NULL
        tmpBindGroup = bindGroup.internal[]
        bindGroup.internal[] = C_NULL
        wgpuBindGroupRelease(tmpBindGroup)
    end
end

function Base.setproperty!(bindGroup::GPUBindGroup, s::Symbol, value)
    if s == :internal && bindGroup.internal[] != C_NULL
        if value == nothing || value == C_NULL
        	@info "Destroying GPUBindGroup"
            destroy(bindGroup)
        end
    end
end

function destroy(layout::GPUPipelineLayout)
    if layout.internal[] != C_NULL
        tmpLayout = layout.internal[]
        layout.internal[] = C_NULL
        wgpuPipelineLayoutRelease(tmpLayout)
    end
end

function Base.setproperty!(pipeline::GPUPipelineLayout, s::Symbol, value)
    if s == :internal && pipeline.internal[] != C_NULL
        if value == nothing || value == C_NULL
        	@info "Destroying GPUPipelineLayout"
            destroy(pipeline)
        end
    end
end

function destroy(shader::GPUShaderModule)
    if shader.internal[] != C_NULL
        tmpShader = shader.internal[]
        shader.internal[] = C_NULL
        wgpuShaderModuleRelease(tmpShader)
    end
end

function Base.setproperty!(shader::GPUShaderModule, s::Symbol, value)
    if s == :internal && shader.internal[] != C_NULL
        if value == nothing || value == C_NULL
        	@info "Destroying GPUShaderModule"
            destroy(shader)
        end
    end
end

function destroy(pipeline::GPUComputePipeline)
    if pipeline.internal[] != C_NULL
        tmpPipeline = pipeline.internal[]
        pipeline.internal[] = C_NULL
        wgpuComputePipelineRelease(tmpPipeline)
    end
end

function Base.setproperty!(pipeline::GPUComputePipeline, s::Symbol, value)
    if s == :internal && pipeline.internal[] != C_NULL
        if value == nothing || value == C_NULL
            destroy(pipeline)
            @info "Destroying compute pipeline"
        end
    end
end

function destroy(pipeline::GPURenderPipeline)
    if pipeline.internal[] != C_NULL
        tmpPipeline = pipeline.internal[]
        pipeline.internal[] = C_NULL
        wgpuRenderPipelineRelease(tmpPipeline)
    end
end

function Base.setproperty!(pipeline::GPURenderPipeline, s::Symbol, value)
    if s == :internal && pipeline.internal[] != C_NULL
        if value == nothing || value == C_NULL
        	@info "Destroying render pipeline"
            destroy(pipeline)
        end
    end
end

function destroy(gpuBuffer::GPUBuffer)
    if gpuBuffer.internal[] != C_NULL
        tmpBufferPtr = gpuBuffer.internal[]
        gpuBuffer.internal[] = C_NULL
        wgpuBufferRelease(tmpBufferPtr)
    end
end

function Base.setproperty!(buf::GPUBuffer, s::Symbol, value)
    if s == :internal && buf.internal[] != C_NULL
        if value == nothing || value == C_NULL
        	@info "Destroying GPUBuffer"
            destroy(buf)
        end
    end
end

function destroy(cmdBuffer::GPUCommandBuffer)
    if cmdBuffer.internal[] != C_NULL
        tmpCmdBufferPtr = cmdBuffer.internal[]
        cmdBuffer.internal[] = C_NULL
        wgpuCommandBufferRelease(tmpCmdBufferPtr)
    end
end

function Base.setproperty!(buf::GPUCommandBuffer, s::Symbol, value)
    if s == :internal && buf.internal[] != C_NULL
        if value == nothing || value == C_NULL
        	@info "Destroying GPUCommandBuffer"
            destroy(buf)
        end
    end
end

function destroy(cmdEncoder::GPUCommandEncoder)
    if cmdEncoder.internal[] != C_NULL
        tmpCmdEncoderPtr = cmdEncoder.internal[]
        cmdEncoder.internal[] = C_NULL
        wgpuCommandEncoderRelease(tmpCmdEncoderPtr)
    end
end

function Base.setproperty!(enc::GPUCommandEncoder, s::Symbol, value)
    if s == :internal && enc.internal[] != C_NULL
        if value == nothing || value == C_NULL
        	@info "Destroying GPUCommandEncoder"
            destroy(enc)
        end
    end
end

function destroy(adapter::GPUAdapter)
    if adapter.internal[] != C_NULL
        tmpAdapterPtr = adapter.internal[]
        wgpuAdapterRelease(tmpAdapterPtr)
        adapter.internal = C_NULL
    end
end

function destroy(adapterImpl::Ptr{WGPUAdapterImpl})
    if adapterImpl != C_NULL
        tmpAdapterPtr = adapterImpl
        adapterImpl = C_NULL
    end
end

function Base.setproperty!(adapter::GPUAdapter, s::Symbol, value)
    if s == :internal && adapter.internal[] != C_NULL
        if value == nothing || value == C_NULL
        	@info "Destroying GPUAdapter"
            destroy(adapter)
        end
    end
end

function destroy(device::GPUDevice)
    if device.internal[] != C_NULL
        tmpDevicePtr = device.internal[]
        device.internal[] = C_NULL
        wgpuDeviceRelease(tmpDevicePtr)
    end
end

function Base.setproperty!(device::GPUDevice, s::Symbol, value)
    if s == :internal && device.internal[] != C_NULL
        if value == nothing || value == C_NULL
            sleep(1)
            destroy(device)
        end
    end
end

function isDestroyable(obj)
	typeof(obj) <: Droppable
end

end
