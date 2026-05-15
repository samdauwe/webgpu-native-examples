/*
  WAjic - WebAssembly JavaScript Interface Creator
  WebGPU bindings for WAjic

  Based on the WebGPU C API standard (webgpu-native/webgpu.h).
  All handle types are uint32_t indices into JS-side object tables.
  Struct layouts match the wasm32 ABI (pointers = 4 bytes, size_t = 4 bytes).

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <wajic.h>

// ============================================================================
// Handle types — opaque integer IDs referencing JS-side object tables
// ============================================================================
typedef uint32_t WGPUAdapter;
typedef uint32_t WGPUBindGroup;
typedef uint32_t WGPUBindGroupLayout;
typedef uint32_t WGPUBuffer;
typedef uint32_t WGPUCommandBuffer;
typedef uint32_t WGPUCommandEncoder;
typedef uint32_t WGPUComputePassEncoder;
typedef uint32_t WGPUComputePipeline;
typedef uint32_t WGPUDevice;
typedef uint32_t WGPUInstance;
typedef uint32_t WGPUPipelineLayout;
typedef uint32_t WGPUQuerySet;
typedef uint32_t WGPUQueue;
typedef uint32_t WGPURenderBundle;
typedef uint32_t WGPURenderBundleEncoder;
typedef uint32_t WGPURenderPassEncoder;
typedef uint32_t WGPURenderPipeline;
typedef uint32_t WGPUSampler;
typedef uint32_t WGPUShaderModule;
typedef uint32_t WGPUSurface;
typedef uint32_t WGPUTexture;
typedef uint32_t WGPUTextureView;

typedef uint32_t WGPUBool;
typedef uint32_t WGPUFlags;

// WGPUOptionalBool: used for fields that can be true/false/undefined.
// False=0 and True=1 map directly to WGPUBool, so structs using either type
// are layout-compatible in the wasm32 ABI.
typedef enum WGPUOptionalBool {
    WGPUOptionalBool_False     = 0,
    WGPUOptionalBool_True      = 1,
    WGPUOptionalBool_Undefined = 2,
    WGPUOptionalBool_Force32   = 0x7FFFFFFF
} WGPUOptionalBool;
typedef WGPUFlags WGPUBufferUsageFlags;
typedef WGPUFlags WGPUColorWriteMaskFlags;
typedef WGPUFlags WGPUShaderStageFlags;
typedef WGPUFlags WGPUTextureUsageFlags;
typedef WGPUFlags WGPUMapModeFlags;

// ============================================================================
// Constants
// ============================================================================
#define WGPU_WHOLE_SIZE          (~(size_t)0)
#define WGPU_STRLEN              (~(size_t)0)
#define WGPU_DEPTH_SLICE_UNDEFINED 0xFFFFFFFF
#define WGPU_WHOLE_MAP_SIZE      (~(size_t)0)
#define WGPU_COPY_STRIDE_UNDEFINED 0xFFFFFFFF
#define WGPU_LIMIT_U32_UNDEFINED 0xFFFFFFFF

// ============================================================================
// Enums
// ============================================================================

typedef enum WGPUSType {
    WGPUSType_ShaderSourceSPIRV                           = 0x00000001,
    WGPUSType_ShaderSourceWGSL                            = 0x00000002,
    WGPUSType_RenderPassMaxDrawCount                      = 0x00000003,
    WGPUSType_SurfaceSourceMetalLayer                     = 0x00000004,
    WGPUSType_SurfaceSourceWindowsHWND                    = 0x00000005,
    WGPUSType_SurfaceSourceXlibWindow                     = 0x00000006,
    WGPUSType_SurfaceSourceWaylandSurface                 = 0x00000007,
    WGPUSType_SurfaceSourceAndroidNativeWindow            = 0x00000008,
    WGPUSType_SurfaceSourceXCBWindow                      = 0x00000009,
    // Emscripten-specific extension (matches emdawnwebgpu)
    WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector    = 0x00030001,
    WGPUSType_Force32                                     = 0x7FFFFFFF
} WGPUSType;

typedef enum WGPUTextureFormat {
    WGPUTextureFormat_Undefined        = 0x00000000,
    WGPUTextureFormat_R8Unorm          = 0x00000001,
    WGPUTextureFormat_R8Snorm          = 0x00000002,
    WGPUTextureFormat_R8Uint           = 0x00000003,
    WGPUTextureFormat_R8Sint           = 0x00000004,
    WGPUTextureFormat_R16Uint          = 0x00000005,
    WGPUTextureFormat_R16Sint          = 0x00000006,
    WGPUTextureFormat_R16Float         = 0x00000007,
    WGPUTextureFormat_RG8Unorm         = 0x00000008,
    WGPUTextureFormat_RG8Snorm         = 0x00000009,
    WGPUTextureFormat_RG8Uint          = 0x0000000A,
    WGPUTextureFormat_RG8Sint          = 0x0000000B,
    WGPUTextureFormat_R32Float         = 0x0000000C,
    WGPUTextureFormat_R32Uint          = 0x0000000D,
    WGPUTextureFormat_R32Sint          = 0x0000000E,
    WGPUTextureFormat_RG16Uint         = 0x0000000F,
    WGPUTextureFormat_RG16Sint         = 0x00000010,
    WGPUTextureFormat_RG16Float        = 0x00000011,
    WGPUTextureFormat_RGBA8Unorm       = 0x00000012,
    WGPUTextureFormat_RGBA8UnormSrgb   = 0x00000013,
    WGPUTextureFormat_RGBA8Snorm       = 0x00000014,
    WGPUTextureFormat_RGBA8Uint        = 0x00000015,
    WGPUTextureFormat_RGBA8Sint        = 0x00000016,
    WGPUTextureFormat_BGRA8Unorm       = 0x00000017,
    WGPUTextureFormat_BGRA8UnormSrgb   = 0x00000018,
    WGPUTextureFormat_RGB10A2Uint      = 0x00000019,
    WGPUTextureFormat_RGB10A2Unorm     = 0x0000001A,
    WGPUTextureFormat_RG11B10Ufloat    = 0x0000001B,
    WGPUTextureFormat_RGB9E5Ufloat     = 0x0000001C,
    WGPUTextureFormat_RG32Float        = 0x0000001D,
    WGPUTextureFormat_RG32Uint         = 0x0000001E,
    WGPUTextureFormat_RG32Sint         = 0x0000001F,
    WGPUTextureFormat_RGBA16Uint       = 0x00000020,
    WGPUTextureFormat_RGBA16Sint       = 0x00000021,
    WGPUTextureFormat_RGBA16Float      = 0x00000022,
    WGPUTextureFormat_RGBA32Float      = 0x00000023,
    WGPUTextureFormat_RGBA32Uint       = 0x00000024,
    WGPUTextureFormat_RGBA32Sint       = 0x00000025,
    WGPUTextureFormat_Stencil8         = 0x00000026,
    WGPUTextureFormat_Depth16Unorm     = 0x00000027,
    WGPUTextureFormat_Depth24Plus      = 0x00000028,
    WGPUTextureFormat_Depth24PlusStencil8 = 0x00000029,
    WGPUTextureFormat_Depth32Float     = 0x0000002A,
    WGPUTextureFormat_Depth32FloatStencil8 = 0x0000002B,
    WGPUTextureFormat_Force32          = 0x7FFFFFFF
} WGPUTextureFormat;

typedef enum WGPUTextureUsage {
    WGPUTextureUsage_None              = 0x00000000,
    WGPUTextureUsage_CopySrc           = 0x00000001,
    WGPUTextureUsage_CopyDst           = 0x00000002,
    WGPUTextureUsage_TextureBinding    = 0x00000004,
    WGPUTextureUsage_StorageBinding    = 0x00000008,
    WGPUTextureUsage_RenderAttachment  = 0x00000010,
    WGPUTextureUsage_Force32           = 0x7FFFFFFF
} WGPUTextureUsage;

typedef enum WGPUBufferUsage {
    WGPUBufferUsage_None               = 0x00000000,
    WGPUBufferUsage_MapRead            = 0x00000001,
    WGPUBufferUsage_MapWrite           = 0x00000002,
    WGPUBufferUsage_CopySrc            = 0x00000004,
    WGPUBufferUsage_CopyDst            = 0x00000008,
    WGPUBufferUsage_Index              = 0x00000010,
    WGPUBufferUsage_Vertex             = 0x00000020,
    WGPUBufferUsage_Uniform            = 0x00000040,
    WGPUBufferUsage_Storage            = 0x00000080,
    WGPUBufferUsage_Indirect           = 0x00000100,
    WGPUBufferUsage_QueryResolve       = 0x00000200,
    WGPUBufferUsage_Force32            = 0x7FFFFFFF
} WGPUBufferUsage;

typedef enum WGPUShaderStage {
    WGPUShaderStage_None               = 0x00000000,
    WGPUShaderStage_Vertex             = 0x00000001,
    WGPUShaderStage_Fragment           = 0x00000002,
    WGPUShaderStage_Compute            = 0x00000004,
    WGPUShaderStage_Force32            = 0x7FFFFFFF
} WGPUShaderStage;

typedef enum WGPUPrimitiveTopology {
    WGPUPrimitiveTopology_Undefined    = 0x00000000,
    WGPUPrimitiveTopology_PointList    = 0x00000001,
    WGPUPrimitiveTopology_LineList     = 0x00000002,
    WGPUPrimitiveTopology_LineStrip    = 0x00000003,
    WGPUPrimitiveTopology_TriangleList = 0x00000004,
    WGPUPrimitiveTopology_TriangleStrip= 0x00000005,
    WGPUPrimitiveTopology_Force32      = 0x7FFFFFFF
} WGPUPrimitiveTopology;

typedef enum WGPUFrontFace {
    WGPUFrontFace_Undefined            = 0x00000000,
    WGPUFrontFace_CCW                  = 0x00000001,
    WGPUFrontFace_CW                   = 0x00000002,
    WGPUFrontFace_Force32              = 0x7FFFFFFF
} WGPUFrontFace;

typedef enum WGPUCullMode {
    WGPUCullMode_Undefined             = 0x00000000,
    WGPUCullMode_None                  = 0x00000001,
    WGPUCullMode_Front                 = 0x00000002,
    WGPUCullMode_Back                  = 0x00000003,
    WGPUCullMode_Force32               = 0x7FFFFFFF
} WGPUCullMode;

typedef enum WGPUIndexFormat {
    WGPUIndexFormat_Undefined          = 0x00000000,
    WGPUIndexFormat_Uint16             = 0x00000001,
    WGPUIndexFormat_Uint32             = 0x00000002,
    WGPUIndexFormat_Force32            = 0x7FFFFFFF
} WGPUIndexFormat;

typedef enum WGPUVertexFormat {
    WGPUVertexFormat_Uint8x2           = 0x00000001,
    WGPUVertexFormat_Uint8x4           = 0x00000002,
    WGPUVertexFormat_Sint8x2           = 0x00000003,
    WGPUVertexFormat_Sint8x4           = 0x00000004,
    WGPUVertexFormat_Unorm8x2          = 0x00000005,
    WGPUVertexFormat_Unorm8x4          = 0x00000006,
    WGPUVertexFormat_Snorm8x2          = 0x00000007,
    WGPUVertexFormat_Snorm8x4          = 0x00000008,
    WGPUVertexFormat_Uint16x2          = 0x00000009,
    WGPUVertexFormat_Uint16x4          = 0x0000000A,
    WGPUVertexFormat_Sint16x2          = 0x0000000B,
    WGPUVertexFormat_Sint16x4          = 0x0000000C,
    WGPUVertexFormat_Unorm16x2         = 0x0000000D,
    WGPUVertexFormat_Unorm16x4         = 0x0000000E,
    WGPUVertexFormat_Snorm16x2         = 0x0000000F,
    WGPUVertexFormat_Snorm16x4         = 0x00000010,
    WGPUVertexFormat_Float16x2         = 0x00000011,
    WGPUVertexFormat_Float16x4         = 0x00000012,
    WGPUVertexFormat_Float32           = 0x00000013,
    WGPUVertexFormat_Float32x2         = 0x00000014,
    WGPUVertexFormat_Float32x3         = 0x00000015,
    WGPUVertexFormat_Float32x4         = 0x00000016,
    WGPUVertexFormat_Uint32            = 0x00000017,
    WGPUVertexFormat_Uint32x2          = 0x00000018,
    WGPUVertexFormat_Uint32x3          = 0x00000019,
    WGPUVertexFormat_Uint32x4          = 0x0000001A,
    WGPUVertexFormat_Sint32            = 0x0000001B,
    WGPUVertexFormat_Sint32x2          = 0x0000001C,
    WGPUVertexFormat_Sint32x3          = 0x0000001D,
    WGPUVertexFormat_Sint32x4          = 0x0000001E,
    WGPUVertexFormat_Unorm10_10_10_2   = 0x0000001F,
    WGPUVertexFormat_Force32           = 0x7FFFFFFF
} WGPUVertexFormat;

typedef enum WGPUVertexStepMode {
    WGPUVertexStepMode_Undefined              = 0x00000000,
    WGPUVertexStepMode_VertexBufferNotUsed     = 0x00000001,
    WGPUVertexStepMode_Vertex                  = 0x00000002,
    WGPUVertexStepMode_Instance                = 0x00000003,
    WGPUVertexStepMode_Force32                 = 0x7FFFFFFF
} WGPUVertexStepMode;

typedef enum WGPULoadOp {
    WGPULoadOp_Undefined               = 0x00000000,
    WGPULoadOp_Clear                   = 0x00000001,
    WGPULoadOp_Load                    = 0x00000002,
    WGPULoadOp_Force32                 = 0x7FFFFFFF
} WGPULoadOp;

typedef enum WGPUStoreOp {
    WGPUStoreOp_Undefined              = 0x00000000,
    WGPUStoreOp_Store                  = 0x00000001,
    WGPUStoreOp_Discard                = 0x00000002,
    WGPUStoreOp_Force32                = 0x7FFFFFFF
} WGPUStoreOp;

typedef enum WGPUBlendOperation {
    WGPUBlendOperation_Undefined       = 0x00000000,
    WGPUBlendOperation_Add             = 0x00000001,
    WGPUBlendOperation_Subtract        = 0x00000002,
    WGPUBlendOperation_ReverseSubtract = 0x00000003,
    WGPUBlendOperation_Min             = 0x00000004,
    WGPUBlendOperation_Max             = 0x00000005,
    WGPUBlendOperation_Force32         = 0x7FFFFFFF
} WGPUBlendOperation;

typedef enum WGPUBlendFactor {
    WGPUBlendFactor_Undefined          = 0x00000000,
    WGPUBlendFactor_Zero               = 0x00000001,
    WGPUBlendFactor_One                = 0x00000002,
    WGPUBlendFactor_Src                = 0x00000003,
    WGPUBlendFactor_OneMinusSrc        = 0x00000004,
    WGPUBlendFactor_SrcAlpha           = 0x00000005,
    WGPUBlendFactor_OneMinusSrcAlpha   = 0x00000006,
    WGPUBlendFactor_Dst                = 0x00000007,
    WGPUBlendFactor_OneMinusDst        = 0x00000008,
    WGPUBlendFactor_DstAlpha           = 0x00000009,
    WGPUBlendFactor_OneMinusDstAlpha   = 0x0000000A,
    WGPUBlendFactor_SrcAlphaSaturated  = 0x0000000B,
    WGPUBlendFactor_Constant           = 0x0000000C,
    WGPUBlendFactor_OneMinusConstant   = 0x0000000D,
    WGPUBlendFactor_Src1               = 0x0000000E,
    WGPUBlendFactor_OneMinusSrc1       = 0x0000000F,
    WGPUBlendFactor_Src1Alpha          = 0x00000010,
    WGPUBlendFactor_OneMinusSrc1Alpha  = 0x00000011,
    WGPUBlendFactor_Force32            = 0x7FFFFFFF
} WGPUBlendFactor;

typedef enum WGPUColorWriteMask {
    WGPUColorWriteMask_None            = 0x00000000,
    WGPUColorWriteMask_Red             = 0x00000001,
    WGPUColorWriteMask_Green           = 0x00000002,
    WGPUColorWriteMask_Blue            = 0x00000004,
    WGPUColorWriteMask_Alpha           = 0x00000008,
    WGPUColorWriteMask_All             = 0x0000000F,
    WGPUColorWriteMask_Force32         = 0x7FFFFFFF
} WGPUColorWriteMask;

typedef enum WGPUBufferBindingType {
    WGPUBufferBindingType_Undefined      = 0x00000000,
    WGPUBufferBindingType_Uniform        = 0x00000001,
    WGPUBufferBindingType_Storage        = 0x00000002,
    WGPUBufferBindingType_ReadOnlyStorage= 0x00000003,
    WGPUBufferBindingType_Force32        = 0x7FFFFFFF
} WGPUBufferBindingType;

typedef enum WGPUSamplerBindingType {
    WGPUSamplerBindingType_Undefined     = 0x00000000,
    WGPUSamplerBindingType_Filtering     = 0x00000001,
    WGPUSamplerBindingType_NonFiltering  = 0x00000002,
    WGPUSamplerBindingType_Comparison    = 0x00000003,
    WGPUSamplerBindingType_Force32       = 0x7FFFFFFF
} WGPUSamplerBindingType;

typedef enum WGPUTextureSampleType {
    WGPUTextureSampleType_Undefined      = 0x00000000,
    WGPUTextureSampleType_Float          = 0x00000001,
    WGPUTextureSampleType_UnfilterableFloat = 0x00000002,
    WGPUTextureSampleType_Depth          = 0x00000003,
    WGPUTextureSampleType_Sint           = 0x00000004,
    WGPUTextureSampleType_Uint           = 0x00000005,
    WGPUTextureSampleType_Force32        = 0x7FFFFFFF
} WGPUTextureSampleType;

typedef enum WGPUTextureViewDimension {
    WGPUTextureViewDimension_Undefined   = 0x00000000,
    WGPUTextureViewDimension_1D          = 0x00000001,
    WGPUTextureViewDimension_2D          = 0x00000002,
    WGPUTextureViewDimension_2DArray     = 0x00000003,
    WGPUTextureViewDimension_Cube        = 0x00000004,
    WGPUTextureViewDimension_CubeArray   = 0x00000005,
    WGPUTextureViewDimension_3D          = 0x00000006,
    WGPUTextureViewDimension_Force32     = 0x7FFFFFFF
} WGPUTextureViewDimension;

typedef enum WGPUStorageTextureAccess {
    WGPUStorageTextureAccess_Undefined   = 0x00000000,
    WGPUStorageTextureAccess_WriteOnly   = 0x00000001,
    WGPUStorageTextureAccess_ReadOnly    = 0x00000002,
    WGPUStorageTextureAccess_ReadWrite   = 0x00000003,
    WGPUStorageTextureAccess_Force32     = 0x7FFFFFFF
} WGPUStorageTextureAccess;

typedef enum WGPUPresentMode {
    WGPUPresentMode_Fifo               = 0x00000001,
    WGPUPresentMode_FifoRelaxed        = 0x00000002,
    WGPUPresentMode_Immediate          = 0x00000003,
    WGPUPresentMode_Mailbox            = 0x00000004,
    WGPUPresentMode_Force32            = 0x7FFFFFFF
} WGPUPresentMode;

typedef enum WGPUCompositeAlphaMode {
    WGPUCompositeAlphaMode_Auto        = 0x00000000,
    WGPUCompositeAlphaMode_Opaque      = 0x00000001,
    WGPUCompositeAlphaMode_Premultiplied= 0x00000002,
    WGPUCompositeAlphaMode_Unpremultiplied = 0x00000003,
    WGPUCompositeAlphaMode_Inherit     = 0x00000004,
    WGPUCompositeAlphaMode_Force32     = 0x7FFFFFFF
} WGPUCompositeAlphaMode;

typedef enum WGPUSurfaceGetCurrentTextureStatus {
    WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal    = 0x00000001,
    WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal = 0x00000002,
    WGPUSurfaceGetCurrentTextureStatus_Timeout           = 0x00000003,
    WGPUSurfaceGetCurrentTextureStatus_Outdated          = 0x00000004,
    WGPUSurfaceGetCurrentTextureStatus_Lost              = 0x00000005,
    WGPUSurfaceGetCurrentTextureStatus_Force32           = 0x7FFFFFFF
} WGPUSurfaceGetCurrentTextureStatus;

typedef enum WGPUTextureDimension {
    WGPUTextureDimension_Undefined     = 0x00000000,
    WGPUTextureDimension_1D            = 0x00000001,
    WGPUTextureDimension_2D            = 0x00000002,
    WGPUTextureDimension_3D            = 0x00000003,
    WGPUTextureDimension_Force32       = 0x7FFFFFFF
} WGPUTextureDimension;

typedef enum WGPUTextureAspect {
    WGPUTextureAspect_Undefined        = 0x00000000,
    WGPUTextureAspect_All              = 0x00000001,
    WGPUTextureAspect_StencilOnly      = 0x00000002,
    WGPUTextureAspect_DepthOnly        = 0x00000003,
    WGPUTextureAspect_Force32          = 0x7FFFFFFF
} WGPUTextureAspect;

typedef enum WGPUCompareFunction {
    WGPUCompareFunction_Undefined      = 0x00000000,
    WGPUCompareFunction_Never          = 0x00000001,
    WGPUCompareFunction_Less           = 0x00000002,
    WGPUCompareFunction_Equal          = 0x00000003,
    WGPUCompareFunction_LessEqual      = 0x00000004,
    WGPUCompareFunction_Greater        = 0x00000005,
    WGPUCompareFunction_NotEqual       = 0x00000006,
    WGPUCompareFunction_GreaterEqual   = 0x00000007,
    WGPUCompareFunction_Always         = 0x00000008,
    WGPUCompareFunction_Force32        = 0x7FFFFFFF
} WGPUCompareFunction;

typedef enum WGPUAddressMode {
    WGPUAddressMode_Undefined      = 0x00000000,
    WGPUAddressMode_ClampToEdge    = 0x00000001,
    WGPUAddressMode_Repeat         = 0x00000002,
    WGPUAddressMode_MirrorRepeat   = 0x00000003,
    WGPUAddressMode_Force32        = 0x7FFFFFFF
} WGPUAddressMode;

typedef enum WGPUFilterMode {
    WGPUFilterMode_Undefined = 0x00000000,
    WGPUFilterMode_Nearest   = 0x00000001,
    WGPUFilterMode_Linear    = 0x00000002,
    WGPUFilterMode_Force32   = 0x7FFFFFFF
} WGPUFilterMode;

typedef enum WGPUMipmapFilterMode {
    WGPUMipmapFilterMode_Undefined = 0x00000000,
    WGPUMipmapFilterMode_Nearest   = 0x00000001,
    WGPUMipmapFilterMode_Linear    = 0x00000002,
    WGPUMipmapFilterMode_Force32   = 0x7FFFFFFF
} WGPUMipmapFilterMode;

typedef enum WGPUStencilOperation {
    WGPUStencilOperation_Undefined     = 0x00000000,
    WGPUStencilOperation_Keep          = 0x00000001,
    WGPUStencilOperation_Zero          = 0x00000002,
    WGPUStencilOperation_Replace       = 0x00000003,
    WGPUStencilOperation_Invert        = 0x00000004,
    WGPUStencilOperation_IncrementClamp= 0x00000005,
    WGPUStencilOperation_DecrementClamp= 0x00000006,
    WGPUStencilOperation_IncrementWrap = 0x00000007,
    WGPUStencilOperation_DecrementWrap = 0x00000008,
    WGPUStencilOperation_Force32       = 0x7FFFFFFF
} WGPUStencilOperation;

// ============================================================================
// Struct definitions (wasm32 ABI: ptr=4, size_t=4, uint64_t=8 aligned 8)
// ============================================================================

typedef struct WGPUChainedStruct {
    struct WGPUChainedStruct const * next;
    WGPUSType sType;
} WGPUChainedStruct;

typedef struct WGPUChainedStructOut {
    struct WGPUChainedStructOut * next;
    WGPUSType sType;
} WGPUChainedStructOut;

typedef struct WGPUStringView {
    char const * data;
    size_t length;
} WGPUStringView;

#define WGPU_STRING_VIEW_INIT { NULL, WGPU_STRLEN }

typedef struct WGPUColor {
    double r;
    double g;
    double b;
    double a;
} WGPUColor;

// --- Sampler descriptor ---

typedef struct WGPUSamplerDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
    WGPUAddressMode addressModeU;
    WGPUAddressMode addressModeV;
    WGPUAddressMode addressModeW;
    WGPUFilterMode magFilter;
    WGPUFilterMode minFilter;
    WGPUMipmapFilterMode mipmapFilter;
    float lodMinClamp;
    float lodMaxClamp;
    WGPUCompareFunction compare;
    uint16_t maxAnisotropy;
} WGPUSamplerDescriptor;

// --- Instance / Surface / Device ---

typedef struct WGPUInstanceDescriptor {
    WGPUChainedStruct const * nextInChain;
} WGPUInstanceDescriptor;

typedef struct WGPUSurfaceDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
} WGPUSurfaceDescriptor;

typedef struct WGPUEmscriptenSurfaceSourceCanvasHTMLSelector {
    WGPUChainedStruct chain;
    WGPUStringView selector;
} WGPUEmscriptenSurfaceSourceCanvasHTMLSelector;

typedef struct WGPUSurfaceConfiguration {
    WGPUChainedStruct const * nextInChain;
    WGPUDevice device;
    WGPUTextureFormat format;
    WGPUTextureUsageFlags usage;
    size_t viewFormatCount;
    WGPUTextureFormat const * viewFormats;
    WGPUCompositeAlphaMode alphaMode;
    uint32_t width;
    uint32_t height;
    WGPUPresentMode presentMode;
} WGPUSurfaceConfiguration;

typedef struct WGPUSurfaceTexture {
    WGPUTexture texture;
    WGPUBool suboptimal;
    WGPUSurfaceGetCurrentTextureStatus status;
} WGPUSurfaceTexture;

// --- Shader ---

typedef struct WGPUShaderModuleDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
} WGPUShaderModuleDescriptor;

typedef struct WGPUShaderSourceWGSL {
    WGPUChainedStruct chain;
    WGPUStringView code;
} WGPUShaderSourceWGSL;

// --- Buffer ---

typedef struct WGPUBufferDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
    WGPUBufferUsageFlags usage;
    uint64_t size;
    WGPUBool mappedAtCreation;
} WGPUBufferDescriptor;

// --- Vertex layout ---

typedef struct WGPUVertexAttribute {
    WGPUVertexFormat format;
    uint64_t offset;
    uint32_t shaderLocation;
} WGPUVertexAttribute;

typedef struct WGPUVertexBufferLayout {
    uint64_t arrayStride;
    WGPUVertexStepMode stepMode;
    size_t attributeCount;
    WGPUVertexAttribute const * attributes;
} WGPUVertexBufferLayout;

// --- Constant entry ---

typedef struct WGPUConstantEntry {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView key;
    double value;
} WGPUConstantEntry;

// --- Pipeline states ---

typedef struct WGPUVertexState {
    WGPUChainedStruct const * nextInChain;
    WGPUShaderModule module;
    WGPUStringView entryPoint;
    size_t constantCount;
    WGPUConstantEntry const * constants;
    size_t bufferCount;
    WGPUVertexBufferLayout const * buffers;
} WGPUVertexState;

typedef struct WGPUPrimitiveState {
    WGPUChainedStruct const * nextInChain;
    WGPUPrimitiveTopology topology;
    WGPUIndexFormat stripIndexFormat;
    WGPUFrontFace frontFace;
    WGPUCullMode cullMode;
    WGPUBool unclippedDepth;
} WGPUPrimitiveState;

typedef struct WGPUBlendComponent {
    WGPUBlendOperation operation;
    WGPUBlendFactor srcFactor;
    WGPUBlendFactor dstFactor;
} WGPUBlendComponent;

typedef struct WGPUBlendState {
    WGPUBlendComponent color;
    WGPUBlendComponent alpha;
} WGPUBlendState;

typedef struct WGPUColorTargetState {
    WGPUChainedStruct const * nextInChain;
    WGPUTextureFormat format;
    WGPUBlendState const * blend;
    WGPUColorWriteMaskFlags writeMask;
} WGPUColorTargetState;

typedef struct WGPUFragmentState {
    WGPUChainedStruct const * nextInChain;
    WGPUShaderModule module;
    WGPUStringView entryPoint;
    size_t constantCount;
    WGPUConstantEntry const * constants;
    size_t targetCount;
    WGPUColorTargetState const * targets;
} WGPUFragmentState;

typedef struct WGPUMultisampleState {
    WGPUChainedStruct const * nextInChain;
    uint32_t count;
    uint32_t mask;
    WGPUBool alphaToCoverageEnabled;
} WGPUMultisampleState;

typedef struct WGPUStencilFaceState {
    WGPUCompareFunction compare;
    WGPUStencilOperation failOp;
    WGPUStencilOperation depthFailOp;
    WGPUStencilOperation passOp;
} WGPUStencilFaceState;

typedef struct WGPUDepthStencilState {
    WGPUChainedStruct const * nextInChain;
    WGPUTextureFormat format;
    WGPUBool depthWriteEnabled;
    WGPUCompareFunction depthCompare;
    WGPUStencilFaceState stencilFront;
    WGPUStencilFaceState stencilBack;
    uint32_t stencilReadMask;
    uint32_t stencilWriteMask;
    int32_t depthBias;
    float depthBiasSlopeScale;
    float depthBiasClamp;
} WGPUDepthStencilState;

typedef struct WGPURenderPipelineDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
    WGPUPipelineLayout layout;
    WGPUVertexState vertex;
    WGPUPrimitiveState primitive;
    WGPUDepthStencilState const * depthStencil;
    WGPUMultisampleState multisample;
    WGPUFragmentState const * fragment;
} WGPURenderPipelineDescriptor;

typedef struct WGPUProgrammableStageDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUShaderModule module;
    WGPUStringView entryPoint;
    size_t constantCount;
    WGPUConstantEntry const * constants;
} WGPUProgrammableStageDescriptor;

typedef struct WGPUComputePipelineDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
    WGPUPipelineLayout layout;
    WGPUProgrammableStageDescriptor compute;
} WGPUComputePipelineDescriptor;

// --- Bind group layout ---

typedef struct WGPUBufferBindingLayout {
    WGPUChainedStruct const * nextInChain;
    WGPUBufferBindingType type;
    WGPUBool hasDynamicOffset;
    uint64_t minBindingSize;
} WGPUBufferBindingLayout;

typedef struct WGPUSamplerBindingLayout {
    WGPUChainedStruct const * nextInChain;
    WGPUSamplerBindingType type;
} WGPUSamplerBindingLayout;

typedef struct WGPUTextureBindingLayout {
    WGPUChainedStruct const * nextInChain;
    WGPUTextureSampleType sampleType;
    WGPUTextureViewDimension viewDimension;
    WGPUBool multisampled;
} WGPUTextureBindingLayout;

typedef struct WGPUStorageTextureBindingLayout {
    WGPUChainedStruct const * nextInChain;
    WGPUStorageTextureAccess access;
    WGPUTextureFormat format;
    WGPUTextureViewDimension viewDimension;
} WGPUStorageTextureBindingLayout;

typedef struct WGPUBindGroupLayoutEntry {
    WGPUChainedStruct const * nextInChain;
    uint32_t binding;
    WGPUShaderStageFlags visibility;
    WGPUBufferBindingLayout buffer;
    WGPUSamplerBindingLayout sampler;
    WGPUTextureBindingLayout texture;
    WGPUStorageTextureBindingLayout storageTexture;
} WGPUBindGroupLayoutEntry;

typedef struct WGPUBindGroupLayoutDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
    size_t entryCount;
    WGPUBindGroupLayoutEntry const * entries;
} WGPUBindGroupLayoutDescriptor;

// --- Pipeline layout ---

typedef struct WGPUPipelineLayoutDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
    size_t bindGroupLayoutCount;
    WGPUBindGroupLayout const * bindGroupLayouts;
} WGPUPipelineLayoutDescriptor;

// --- Bind group ---

typedef struct WGPUBindGroupEntry {
    WGPUChainedStruct const * nextInChain;
    uint32_t binding;
    WGPUBuffer buffer;
    uint64_t offset;
    uint64_t size;
    WGPUSampler sampler;
    WGPUTextureView textureView;
} WGPUBindGroupEntry;

typedef struct WGPUBindGroupDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
    WGPUBindGroupLayout layout;
    size_t entryCount;
    WGPUBindGroupEntry const * entries;
} WGPUBindGroupDescriptor;

// --- Render pass ---

typedef struct WGPURenderPassColorAttachment {
    WGPUChainedStruct const * nextInChain;
    WGPUTextureView view;
    uint32_t depthSlice;
    WGPUTextureView resolveTarget;
    WGPULoadOp loadOp;
    WGPUStoreOp storeOp;
    WGPUColor clearValue;
} WGPURenderPassColorAttachment;

typedef struct WGPURenderPassDepthStencilAttachment {
    WGPUTextureView view;
    WGPULoadOp depthLoadOp;
    WGPUStoreOp depthStoreOp;
    float depthClearValue;
    WGPUBool depthReadOnly;
    WGPULoadOp stencilLoadOp;
    WGPUStoreOp stencilStoreOp;
    uint32_t stencilClearValue;
    WGPUBool stencilReadOnly;
} WGPURenderPassDepthStencilAttachment;

typedef struct WGPURenderPassDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
    size_t colorAttachmentCount;
    WGPURenderPassColorAttachment const * colorAttachments;
    WGPURenderPassDepthStencilAttachment const * depthStencilAttachment;
    WGPUQuerySet occlusionQuerySet;
} WGPURenderPassDescriptor;

// --- Command encoder ---

typedef struct WGPUCommandEncoderDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
} WGPUCommandEncoderDescriptor;

typedef struct WGPUCommandBufferDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
} WGPUCommandBufferDescriptor;

// --- Texture ---

typedef struct WGPUExtent3D {
    uint32_t width;
    uint32_t height;
    uint32_t depthOrArrayLayers;
} WGPUExtent3D;

typedef struct WGPUTextureDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
    WGPUTextureUsageFlags usage;
    WGPUTextureDimension dimension;
    WGPUExtent3D size;
    WGPUTextureFormat format;
    uint32_t mipLevelCount;
    uint32_t sampleCount;
    size_t viewFormatCount;
    WGPUTextureFormat const * viewFormats;
} WGPUTextureDescriptor;

typedef struct WGPUTextureViewDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
    WGPUTextureFormat format;
    WGPUTextureViewDimension dimension;
    uint32_t baseMipLevel;
    uint32_t mipLevelCount;
    uint32_t baseArrayLayer;
    uint32_t arrayLayerCount;
    WGPUTextureAspect aspect;
} WGPUTextureViewDescriptor;

// --- Texel copy (for wgpuQueueWriteTexture) ---

typedef struct WGPUOrigin3D {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} WGPUOrigin3D;

typedef struct WGPUTexelCopyTextureInfo {
    WGPUChainedStruct const * nextInChain;
    WGPUTexture texture;
    uint32_t mipLevel;
    WGPUOrigin3D origin;
    WGPUTextureAspect aspect;
} WGPUTexelCopyTextureInfo;

typedef struct WGPUTexelCopyBufferLayout {
    WGPUChainedStruct const * nextInChain;
    uint64_t offset;
    uint32_t bytesPerRow;
    uint32_t rowsPerImage;
} WGPUTexelCopyBufferLayout;

// WGPUTexelCopyBufferInfo: used as source in wgpuCommandEncoderCopyBufferToTexture.
// wasm32 layout: +0:nextInChain(4) +4:layout(WGPUTexelCopyBufferLayout=24) +28:buffer(4) => sizeof=32
typedef struct WGPUTexelCopyBufferInfo {
    WGPUChainedStruct const * nextInChain;
    WGPUTexelCopyBufferLayout layout;
    WGPUBuffer buffer;
} WGPUTexelCopyBufferInfo;

// --- Render bundle encoder ---

typedef struct WGPURenderBundleEncoderDescriptor {
    WGPUChainedStruct const * nextInChain;
    WGPUStringView label;
    size_t colorFormatCount;
    WGPUTextureFormat const * colorFormats;
    WGPUTextureFormat depthStencilFormat;
    uint32_t sampleCount;
    WGPUBool depthReadOnly;
    WGPUBool stencilReadOnly;
} WGPURenderBundleEncoderDescriptor;

// ============================================================================
// JavaScript implementation — WAJIC_LIB_WITH_INIT block
// ============================================================================

WAJIC_LIB_WITH_INIT(WEBGPU,
(
    var WGPUcnt = 1;
    var WI = []; // instances
    var WA_ = []; // adapters
    var WD = []; // devices
    var WQ = []; // queues
    var WS = []; // surfaces
    var WSM = []; // shader modules
    var WB = []; // buffers
    var WT = []; // textures
    var WTV = []; // texture views
    var WSa = []; // samplers
    var WBGL = []; // bind group layouts
    var WPL = []; // pipeline layouts
    var WRP = []; // render pipelines
    var WCP = []; // compute pipelines
    var WBG = []; // bind groups
    var WCE = []; // command encoders
    var WRPE = []; // render pass encoders
    var WCPE = []; // compute pass encoders
    var WCB = []; // command buffers
    var WQS = []; // query sets
    var WRB = []; // render bundles
    var WRBE = []; // render bundle encoders

    var MF64;
    var WFree = []; // freed handle IDs for reuse
    var WMBUF = {}; // mapped buffer tracking: handle -> {ptr, size, offset}
    var WSurfTexIds = {}; // texture handle -> surface handle (surface texture tracking)
    var WSurfVwIds = {}; // view handle -> surface handle (surface view tracking)

    function Wnew(table, obj) {
        var id = WFree.length ? WFree.pop() : WGPUcnt++;
        for (var i = table.length; i <= id; i++) table[i] = null;
        table[id] = obj;
        return id;
    }
    function Wdel(table, id) { if (id && table[id]) { table[id] = null; WFree.push(id); } }

    // Handle validation: returns the JS object or aborts with a clear diagnostic
    function Wget(table, id, tname, fname) {
        var obj = id ? table[id] : null;
        if (!obj) abort('WEBGPU', fname + ': invalid ' + tname + ' handle ' + id);
        return obj;
    }

    // Debug logging helper — set WA.webgpuDebug = true before WASM load to enable
    function Wlog() {
        if (WA.webgpuDebug) console.log.apply(console, ['[WEBGPU]'].concat(Array.prototype.slice.call(arguments)));
    }

    // Safe Float64Array accessor — recreates view if WASM memory has grown
    function GF64() {
        if (!MF64 || MF64.buffer !== MEM.buffer) MF64 = new Float64Array(MEM.buffer);
        return MF64;
    }

    // Read WGPUStringView from ptr (offset 0: data ptr, offset 4: length)
    function Wsv(p) {
        if (!p) return undefined;
        var d = MU32[p>>2], l = MU32[(p+4)>>2];
        return d ? MStrGet(d, (l>>>0) === 0xFFFFFFFF ? 0 : l) : undefined;
    }

    // Enum-to-string lookup tables
    var EFmt = ',r8unorm,r8snorm,r8uint,r8sint,r16uint,r16sint,r16float,rg8unorm,rg8snorm,rg8uint,rg8sint,r32float,r32uint,r32sint,rg16uint,rg16sint,rg16float,rgba8unorm,rgba8unorm-srgb,rgba8snorm,rgba8uint,rgba8sint,bgra8unorm,bgra8unorm-srgb,rgb10a2uint,rgb10a2unorm,rg11b10ufloat,rgb9e5ufloat,rg32float,rg32uint,rg32sint,rgba16uint,rgba16sint,rgba16float,rgba32float,rgba32uint,rgba32sint,stencil8,depth16unorm,depth24plus,depth24plus-stencil8,depth32float,depth32float-stencil8'.split(',');
    var EVFmt = ',uint8x2,uint8x4,sint8x2,sint8x4,unorm8x2,unorm8x4,snorm8x2,snorm8x4,uint16x2,uint16x4,sint16x2,sint16x4,unorm16x2,unorm16x4,snorm16x2,snorm16x4,float16x2,float16x4,float32,float32x2,float32x3,float32x4,uint32,uint32x2,uint32x3,uint32x4,sint32,sint32x2,sint32x3,sint32x4,unorm10-10-10-2'.split(',');
    var ETopo = ',point-list,line-list,line-strip,triangle-list,triangle-strip'.split(',');
    var EFace = ',ccw,cw'.split(',');
    var ECull = ',none,front,back'.split(',');
    var ELdOp = ',clear,load'.split(',');
    var EStOp = ',store,discard'.split(',');
    var EBOp = ',add,subtract,reverse-subtract,min,max'.split(',');
    var EBFact = ',zero,one,src,one-minus-src,src-alpha,one-minus-src-alpha,dst,one-minus-dst,dst-alpha,one-minus-dst-alpha,src-alpha-saturated,constant,one-minus-constant,src1,one-minus-src1,src1-alpha,one-minus-src1-alpha'.split(',');
    var EIFmt = ',uint16,uint32'.split(',');
    var EBBType = ',uniform,storage,read-only-storage'.split(',');
    var ESStep = ',,vertex,instance'.split(',');
    var EAlpha = 'auto,opaque,premultiplied,unpremultiplied,inherit'.split(',');

    // Read WGPUBlendComponent from ptr (3 x uint32: operation, srcFactor, dstFactor)
    function RdBlend(p) {
        var op = MU32[p>>2], sf = MU32[(p+4)>>2], df = MU32[(p+8)>>2];
        return { operation: EBOp[op]||'add', srcFactor: EBFact[sf]||'one', dstFactor: EBFact[df]||'zero' };
    }

    // Async WebGPU device acquisition — runs before main() via WA.preMain hook
    WA.preMain = async function() {
        if (!navigator.gpu) {
            var loc = (typeof location !== 'undefined') ? location : null;
            var isSecure = !loc || loc.protocol === 'https:' || loc.hostname === 'localhost' || loc.hostname === '127.0.0.1';
            var msg = isSecure
                ? 'navigator.gpu is unavailable. Your browser may not support WebGPU. Try Chrome 113+ or Firefox with dom.webgpu.enabled=true in about:config.'
                : 'WebGPU requires HTTPS (or http://localhost). Current origin is not a secure context: ' + (loc ? loc.origin : '?');
            abort('WEBGPU', msg);
            return;
        }
        Wlog('Requesting adapter...');
        var adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) { abort('WEBGPU', 'No suitable GPU adapter found. Ensure your GPU supports Vulkan (Linux/Windows) or Metal (macOS).'); return; }
        var info = adapter.info || {};
        Wlog('Adapter:', info.vendor || '?', info.architecture || '?', info.device || '?', 'driver:', info.description || '?');
        Wlog('Requesting device...');
        var device;
        try {
            device = await adapter.requestDevice();
        } catch(err) {
            abort('WEBGPU', 'requestDevice failed: ' + err.message);
            return;
        }
        device.lost.then(function(info) {
            // reason === 'destroyed' means we called device.destroy() intentionally (beforeunload, release)
            if (info.reason !== 'destroyed') {
                // Stop the render loop immediately then report the loss
                try { WA.abort('WEBGPU', 'GPU device lost (' + info.reason + '): ' + info.message); } catch(e) {}
            }
        });
        device.onuncapturederror = function(ev) {
            console.error('[WEBGPU] Uncaptured device error:', ev.error.message);
        };
        WA.webgpuAdapter = adapter;
        WA.webgpuDevice = device;
        // Destroy the device immediately when the page navigates away or reloads.
        // Without this, the old GPUDevice holds all Vulkan VRAM until GC runs — which
        // may not happen before the next page load, causing vkAllocateMemory to fail.
        window.addEventListener('beforeunload', function() {
            try { if (WA.webgpuDevice) { WA.webgpuDevice.destroy(); WA.webgpuDevice = null; } } catch(e) {}
        }, { once: true });
        Wlog('Device ready, preferred format:', navigator.gpu.getPreferredCanvasFormat());
        // Create Float64Array view for reading WGPUColor (double) fields
        MF64 = new Float64Array(MEM.buffer);
    };
),
// wgpuCreateInstance: wraps navigator.gpu as the browser WebGPU "instance"
WGPUInstance, wgpuCreateInstance, (const void* descriptor),
{
    return Wnew(WI, navigator.gpu);
})

// ---- Device / Queue --------------------------------------------------------

WAJIC_LIB(WEBGPU, WGPUDevice, wgpuWajicGetDevice, (),
{
    if (!WA.webgpuDevice) abort('WEBGPU', 'No pre-initialized WebGPU device');
    return Wnew(WD, WA.webgpuDevice);
})

WAJIC_LIB(WEBGPU, WGPUQueue, wgpuDeviceGetQueue, (WGPUDevice device),
{
    var dev = Wget(WD, device, 'device', 'wgpuDeviceGetQueue');
    return Wnew(WQ, dev.queue);
})

// ---- Surface ---------------------------------------------------------------

WAJIC_LIB(WEBGPU, WGPUSurface, wgpuInstanceCreateSurface,
    (WGPUInstance instance, const void* descriptor),
{
    var canvas = WA.canvas;
    if (descriptor) {
        var next = MU32[descriptor>>2];
        if (next) {
            var sType = MU32[(next+4)>>2];
            if ((sType>>>0) === 0x00030001) {
                var sel = Wsv(next+8);
                if (sel) { var c = document.querySelector(sel); if (c) canvas = c; }
            }
        }
    }
    return Wnew(WS, { canvas: canvas, ctx: null, fmt: null, renderW: 0, renderH: 0, padW: 0, padH: 0 });
})

WAJIC_LIB(WEBGPU, void, wgpuSurfaceConfigure,
    (WGPUSurface surface, const void* config),
{
    // WGPUSurfaceConfiguration wasm32 layout:
    // +0: nextInChain(4) +4: device(4) +8: format(4) +12: usage(4)
    // +16: viewFormatCount(4) +20: viewFormats(4) +24: alphaMode(4)
    // +28: width(4) +32: height(4) +36: presentMode(4)
    var s = WS[surface];
    var devId = MU32[(config+4)>>2];
    var dev = Wget(WD, devId, 'device', 'wgpuSurfaceConfigure');
    var fmtIdx = MU32[(config+8)>>2];
    var fmt = EFmt[fmtIdx] || navigator.gpu.getPreferredCanvasFormat();
    var usage = MU32[(config+12)>>2] || 0x10; // default RENDER_ATTACHMENT
    var am = EAlpha[MU32[(config+24)>>2]] || 'auto';
    var w = MU32[(config+28)>>2];
    var h = MU32[(config+32)>>2];
    // Browser WebGPU handles surface texture alignment internally.
    // Store the app dimensions for viewport/scissor auto-application.
    s.renderW = w; s.renderH = h;
    s.padW = w; s.padH = h;
    Wlog('SurfaceConfigure: ' + w + 'x' + h + ' fmt=' + fmt + ' usage=0x' + usage.toString(16) + ' alpha=' + am);
    s.canvas.width = w;
    s.canvas.height = h;
    var ctx = s.canvas.getContext('webgpu');
    if (!ctx) { abort('WEBGPU', 'wgpuSurfaceConfigure: canvas.getContext("webgpu") returned null'); return; }
    try {
        ctx.configure({ device: dev, format: fmt, usage: usage, alphaMode: am });
    } catch(err) {
        abort('WEBGPU', 'wgpuSurfaceConfigure: ctx.configure failed: ' + err.message);
        return;
    }
    s.ctx = ctx;
    s.fmt = fmt;
    Wlog('Surface configured OK (' + w + 'x' + h + ')');
})

WAJIC_LIB(WEBGPU, void, wgpuSurfaceGetCurrentTexture,
    (WGPUSurface surface, void* surfaceTexture),
{
    // WGPUSurfaceTexture: +0: texture(4) +4: suboptimal(4) +8: status(4)
    var s = WS[surface];
    var tex, status = 1; // SuccessOptimal
    if (!s || !s.ctx) {
        Wlog('GetCurrentTexture: surface not configured (surface=' + surface + ')');
        MU32[surfaceTexture>>2] = 0;
        MU32[(surfaceTexture+4)>>2] = 0;
        MU32[(surfaceTexture+8)>>2] = 5; // Lost
        return;
    }
    try {
        tex = s.ctx.getCurrentTexture();
        // Detect invalid textures (zero-sized or destroyed)
        if (!tex || tex.width === 0 || tex.height === 0) {
            status = 5; // Lost
        }
    } catch(e) {
        status = 5; // Lost
        tex = null;
    }
    var texId = tex ? Wnew(WT, tex) : 0;
    if (tex) WSurfTexIds[texId] = surface; // track surface origin
    MU32[surfaceTexture>>2] = texId;
    MU32[(surfaceTexture+4)>>2] = (s.padW !== s.renderW || s.padH !== s.renderH) ? 1 : 0;
    MU32[(surfaceTexture+8)>>2] = status;
})

WAJIC_LIB(WEBGPU, void, wgpuSurfaceUnconfigure, (WGPUSurface surface),
{
    var s = WS[surface];
    if (s.ctx) { s.ctx.unconfigure(); s.ctx = null; }
})

// ---- Shader ----------------------------------------------------------------

WAJIC_LIB(WEBGPU, WGPUShaderModule, wgpuDeviceCreateShaderModule,
    (WGPUDevice device, const void* descriptor),
{
    // +0: nextInChain(4) +4: label(WGPUStringView,8)
    var dev = Wget(WD, device, 'device', 'wgpuDeviceCreateShaderModule');
    var next = MU32[descriptor>>2];
    var code = '';
    if (next) {
        var sType = MU32[(next+4)>>2];
        if (sType == 2) { // WGPUSType_ShaderSourceWGSL
            // WGPUShaderSourceWGSL: +0: chain(8) +8: code(WGPUStringView,8)
            code = Wsv(next+8) || '';
        }
    }
    try { return Wnew(WSM, dev.createShaderModule({ code: code })); }
    catch(err) { abort('WEBGPU', 'wgpuDeviceCreateShaderModule failed: ' + err.message); }
})

// ---- Buffer ----------------------------------------------------------------

WAJIC_LIB(WEBGPU, WGPUBuffer, wgpuDeviceCreateBuffer,
    (WGPUDevice device, const void* descriptor),
{
    // WGPUBufferDescriptor wasm32:
    // +0: nextInChain(4) +4: label(8) +12: usage(4)
    // +16: size(uint64,8) +24: mappedAtCreation(4)
    var dev = Wget(WD, device, 'device', 'wgpuDeviceCreateBuffer');
    var usage = MU32[(descriptor+12)>>2];
    var szLo = MU32[(descriptor+16)>>2], szHi = MU32[(descriptor+20)>>2];
    var sz = szHi * 0x100000000 + (szLo>>>0);
    var mapped = MU32[(descriptor+24)>>2];
    Wlog('CreateBuffer: size=' + sz + ' usage=0x' + usage.toString(16) + ' mapped=' + !!mapped);
    try { return Wnew(WB, dev.createBuffer({ usage: usage, size: sz, mappedAtCreation: !!mapped })); }
    catch(err) { abort('WEBGPU', 'wgpuDeviceCreateBuffer failed (usage=0x' + usage.toString(16) + ' size=' + sz + '): ' + err.message); }
})

WAJIC_LIB(WEBGPU, void, wgpuQueueWriteBuffer,
    (WGPUQueue queue, WGPUBuffer buffer, unsigned int bufferOffset,
     const void* data, unsigned int size),
{
    var q = Wget(WQ, queue, 'queue', 'wgpuQueueWriteBuffer');
    var b = Wget(WB, buffer, 'buffer', 'wgpuQueueWriteBuffer');
    var off = bufferOffset >>> 0, sz = size >>> 0;
    q.writeBuffer(b, off, MU8.subarray(data, data + sz));
})

// ---- Texture / TextureView -------------------------------------------------

WAJIC_LIB(WEBGPU, WGPUTextureView, wgpuTextureCreateView,
    (WGPUTexture texture, const void* descriptor),
{
    var tex = Wget(WT, texture, 'texture', 'wgpuTextureCreateView');
    try {
    if (!descriptor) {
        var vid = Wnew(WTV, tex.createView());
        if (WSurfTexIds[texture]) WSurfVwIds[vid] = WSurfTexIds[texture];
        return vid;
    }
    // WGPUTextureViewDescriptor: +0: nextInChain(4) +4: label(8) +12: format(4)
    // +16: dimension(4) +20: baseMipLevel(4) +24: mipLevelCount(4)
    // +28: baseArrayLayer(4) +32: arrayLayerCount(4) +36: aspect(4)
    var desc = {};
    var fmt = MU32[(descriptor+12)>>2]; if (fmt) desc.format = EFmt[fmt];
    var dim = MU32[(descriptor+16)>>2]; if (dim) desc.dimension = ['1d','2d','2d-array','cube','cube-array','3d'][dim-1];
    desc.baseMipLevel = MU32[(descriptor+20)>>2];
    desc.mipLevelCount = MU32[(descriptor+24)>>2];
    desc.baseArrayLayer = MU32[(descriptor+28)>>2];
    desc.arrayLayerCount = MU32[(descriptor+32)>>2];
    var vid = Wnew(WTV, tex.createView(desc));
    if (WSurfTexIds[texture]) WSurfVwIds[vid] = WSurfTexIds[texture];
    return vid;
    } catch(err) { Wlog('wgpuTextureCreateView failed: ' + err.message); return 0; }
})

WAJIC_LIB(WEBGPU, WGPUTexture, wgpuDeviceCreateTexture,
    (WGPUDevice device, const void* descriptor),
{
    var dev = Wget(WD, device, 'device', 'wgpuDeviceCreateTexture');
    // WGPUTextureDescriptor: +0:nextInChain(4) +4:label(8) +12:usage(4)
    // +16:dimension(4) +20:width(4) +24:height(4) +28:depthOrArrayLayers(4)
    // +32:format(4) +36:mipLevelCount(4) +40:sampleCount(4)
    var desc = {};
    desc.usage = MU32[(descriptor+12)>>2];
    var dim = MU32[(descriptor+16)>>2];
    desc.dimension = dim ? ['1d','2d','3d'][dim-1] : '2d';
    desc.size = { width: MU32[(descriptor+20)>>2], height: MU32[(descriptor+24)>>2], depthOrArrayLayers: MU32[(descriptor+28)>>2] };
    desc.format = EFmt[MU32[(descriptor+32)>>2]] || 'rgba8unorm';
    desc.mipLevelCount = MU32[(descriptor+36)>>2] || 1;
    desc.sampleCount = MU32[(descriptor+40)>>2] || 1;
    try { return Wnew(WT, dev.createTexture(desc)); }
    catch(err) { abort('WEBGPU', 'wgpuDeviceCreateTexture failed: ' + err.message); }
})

// ---- Bind group layout -----------------------------------------------------

WAJIC_LIB(WEBGPU, WGPUBindGroupLayout, wgpuDeviceCreateBindGroupLayout,
    (WGPUDevice device, const void* descriptor),
{
    // WGPUBindGroupLayoutDescriptor: +0:nextInChain(4) +4:label(8) +12:entryCount(4) +16:entries(4)
    var dev = Wget(WD, device, 'device', 'wgpuDeviceCreateBindGroupLayout');
    var count = MU32[(descriptor+12)>>2];
    var eptr = MU32[(descriptor+16)>>2];
    var entries = [];
    // WGPUBindGroupLayoutEntry has complex layout due to alignment.
    // We compute sizeof by examining the struct:
    // +0:nextInChain(4) +4:binding(4) +8:visibility(4)
    // [pad 4 for buffer.minBindingSize uint64 alignment]
    // +16: buffer(WGPUBufferBindingLayout,24)
    // +40: sampler(WGPUSamplerBindingLayout,8)
    // +48: texture(WGPUTextureBindingLayout,16)
    // +64: storageTexture(WGPUStorageTextureBindingLayout,16)
    // sizeof = 80
    for (var i = 0; i < count; i++) {
        var p = eptr + i * 80;
        var e = { binding: MU32[(p+4)>>2], visibility: MU32[(p+8)>>2] };
        // buffer: +16: nextInChain(4) +20: type(4) +24: hasDynamicOffset(4) [pad4] +32: minBindingSize(uint64,8)
        var bt = MU32[(p+20)>>2];
        if (bt) e.buffer = { type: EBBType[bt], hasDynamicOffset: !!MU32[(p+24)>>2] };
        // sampler: +40: nextInChain(4) +44: type(4)
        var st = MU32[(p+44)>>2];
        if (st) e.sampler = { type: ['filtering','non-filtering','comparison'][st-1] };
        // texture: +48: nextInChain(4) +52: sampleType(4) +56: viewDimension(4) +60: multisampled(4)
        var tt = MU32[(p+52)>>2];
        if (tt) {
            e.texture = {
                sampleType: [,'float','unfilterable-float','depth','sint','uint'][tt],
                viewDimension: [,'1d','2d','2d-array','cube','cube-array','3d'][MU32[(p+56)>>2]] || '2d',
                multisampled: !!MU32[(p+60)>>2]
            };
        }
        // storageTexture: +64: nextInChain(4) +68: access(4) +72: format(4) +76: viewDimension(4)
        var sa = MU32[(p+68)>>2];
        if (sa) {
            e.storageTexture = {
                access: [,'write-only','read-only','read-write'][sa],
                format: EFmt[MU32[(p+72)>>2]],
                viewDimension: [,'1d','2d','2d-array','cube','cube-array','3d'][MU32[(p+76)>>2]] || '2d'
            };
        }
        entries.push(e);
    }
    try { return Wnew(WBGL, dev.createBindGroupLayout({ entries: entries })); }
    catch(err) { abort('WEBGPU', 'wgpuDeviceCreateBindGroupLayout failed: ' + err.message); }
})

// ---- Pipeline layout -------------------------------------------------------

WAJIC_LIB(WEBGPU, WGPUPipelineLayout, wgpuDeviceCreatePipelineLayout,
    (WGPUDevice device, const void* descriptor),
{
    // +0:nextInChain(4) +4:label(8) +12:bindGroupLayoutCount(4) +16:bindGroupLayouts(4)
    var dev = Wget(WD, device, 'device', 'wgpuDeviceCreatePipelineLayout');
    var count = MU32[(descriptor+12)>>2];
    var ptr = MU32[(descriptor+16)>>2];
    var layouts = [];
    for (var i = 0; i < count; i++) layouts.push(WBGL[MU32[(ptr>>2)+i]]);
    try { return Wnew(WPL, dev.createPipelineLayout({ bindGroupLayouts: layouts })); }
    catch(err) { abort('WEBGPU', 'wgpuDeviceCreatePipelineLayout failed: ' + err.message); }
})

// ---- Render pipeline -------------------------------------------------------

WAJIC_LIB(WEBGPU, WGPURenderPipeline, wgpuDeviceCreateRenderPipeline,
    (WGPUDevice device, const void* d),
{
    var dev = Wget(WD, device, 'device', 'wgpuDeviceCreateRenderPipeline');
    // WGPURenderPipelineDescriptor wasm32 layout:
    // +0: nextInChain(4) +4: label(8) +12: layout(4)
    // +16: vertex(WGPUVertexState) +48+: primitive, depthStencil, multisample, fragment
    // Need to compute vertex state offset and size.

    // WGPUVertexState: +0:nextInChain(4) +4:module(4) +8:entryPoint(8)
    //                  +16:constantCount(4) +20:constants(4) +24:bufferCount(4) +28:buffers(4)
    // sizeof = 32, starts at d+16

    var vp = d + 16;
    var vmod = WSM[MU32[(vp+4)>>2]];
    var vep = Wsv(vp+8);
    var vbufCount = MU32[(vp+24)>>2];
    var vbufPtr = MU32[(vp+28)>>2];
    var vbufs = [];
    // WGPUVertexBufferLayout: +0:arrayStride(uint64,8) +8:stepMode(4) +12:attributeCount(4) +16:attributes(4)
    // sizeof=24 (padded to align 8)
    for (var i = 0; i < vbufCount; i++) {
        var bp = vbufPtr + i * 24;
        var stride = MU32[bp>>2]; // low 32 bits of arrayStride
        var sm = MU32[(bp+8)>>2];
        var ac = MU32[(bp+12)>>2];
        var ap = MU32[(bp+16)>>2];
        var attrs = [];
        // WGPUVertexAttribute: +0:format(4) [pad4] +8:offset(uint64,8) +16:shaderLocation(4) [pad4]
        // sizeof=24
        for (var j = 0; j < ac; j++) {
            var a = ap + j * 24;
            attrs.push({
                format: EVFmt[MU32[a>>2]],
                offset: MU32[(a+8)>>2], // low 32 bits
                shaderLocation: MU32[(a+16)>>2]
            });
        }
        var bl = { arrayStride: stride, attributes: attrs };
        if (sm >= 2) bl.stepMode = ESStep[sm];
        vbufs.push(bl);
    }

    // WGPUPrimitiveState starts at d+48, sizeof=24
    // +0:nextInChain(4) +4:topology(4) +8:stripIndexFormat(4) +12:frontFace(4) +16:cullMode(4) +20:unclippedDepth(4)
    var pp = d + 48;
    var prim = {};
    var topo = MU32[(pp+4)>>2]; if (topo) prim.topology = ETopo[topo];
    var sif = MU32[(pp+8)>>2]; if (sif) prim.stripIndexFormat = EIFmt[sif];
    var ff = MU32[(pp+12)>>2]; if (ff) prim.frontFace = EFace[ff];
    var cm = MU32[(pp+16)>>2]; if (cm) prim.cullMode = ECull[cm];
    if (MU32[(pp+20)>>2]) prim.unclippedDepth = true;

    // depthStencil ptr at d+72 (4 bytes)
    var dsp = MU32[(d+72)>>2];
    var ds = undefined;
    if (dsp) {
        // WGPUDepthStencilState: +0:nextInChain(4) +4:format(4) +8:depthWriteEnabled(4) +12:depthCompare(4)
        // +16:stencilFront(16) +32:stencilBack(16) +48:stencilReadMask(4) +52:stencilWriteMask(4)
        // +56:depthBias(4) +60:depthBiasSlopeScale(f32,4) +64:depthBiasClamp(f32,4)
        var ECmp = [,'never','less','equal','less-equal','greater','not-equal','greater-equal','always'];
        var ESOp = [,'keep','zero','replace','invert','increment-clamp','decrement-clamp','increment-wrap','decrement-wrap'];
        ds = {
            format: EFmt[MU32[(dsp+4)>>2]],
            depthWriteEnabled: !!MU32[(dsp+8)>>2],
            depthCompare: ECmp[MU32[(dsp+12)>>2]] || 'always'
        };
        // stencilFront: +16: compare(4) failOp(4) depthFailOp(4) passOp(4)
        var sf = dsp+16;
        ds.stencilFront = { compare: ECmp[MU32[sf>>2]]||'always', failOp: ESOp[MU32[(sf+4)>>2]]||'keep', depthFailOp: ESOp[MU32[(sf+8)>>2]]||'keep', passOp: ESOp[MU32[(sf+12)>>2]]||'keep' };
        var sb = dsp+32;
        ds.stencilBack = { compare: ECmp[MU32[sb>>2]]||'always', failOp: ESOp[MU32[(sb+4)>>2]]||'keep', depthFailOp: ESOp[MU32[(sb+8)>>2]]||'keep', passOp: ESOp[MU32[(sb+12)>>2]]||'keep' };
        ds.stencilReadMask = MU32[(dsp+48)>>2];
        ds.stencilWriteMask = MU32[(dsp+52)>>2];
        ds.depthBias = MI32[(dsp+56)>>2];
        ds.depthBiasSlopeScale = MF32[(dsp+60)>>2];
        ds.depthBiasClamp = MF32[(dsp+64)>>2];
    }

    // WGPUMultisampleState at d+76, sizeof=16
    // +0:nextInChain(4) +4:count(4) +8:mask(4) +12:alphaToCoverageEnabled(4)
    var mp = d + 76;
    var ms = {
        count: MU32[(mp+4)>>2] || 1,
        mask: MU32[(mp+8)>>2] || 0xFFFFFFFF,
        alphaToCoverageEnabled: !!MU32[(mp+12)>>2]
    };

    // fragment ptr at d+92
    var fp = MU32[(d+92)>>2];
    var frag = undefined;
    if (fp) {
        // WGPUFragmentState: +0:nextInChain(4) +4:module(4) +8:entryPoint(8)
        //                    +16:constantCount(4) +20:constants(4) +24:targetCount(4) +28:targets(4)
        var fmod = WSM[MU32[(fp+4)>>2]];
        var fep = Wsv(fp+8);
        var tc = MU32[(fp+24)>>2];
        var tp = MU32[(fp+28)>>2];
        var targets = [];
        // WGPUColorTargetState: +0:nextInChain(4) +4:format(4) +8:blend(ptr,4) +12:writeMask(4) = sizeof 16
        for (var i = 0; i < tc; i++) {
            var cp = tp + i * 16;
            var ct = { format: EFmt[MU32[(cp+4)>>2]] };
            var wm = MU32[(cp+12)>>2];
            ct.writeMask = wm;
            var bptr = MU32[(cp+8)>>2];
            if (bptr) {
                // WGPUBlendState: color(12) + alpha(12) = 24 bytes
                ct.blend = { color: RdBlend(bptr), alpha: RdBlend(bptr+12) };
            }
            targets.push(ct);
        }
        frag = { module: fmod, targets: targets };
        if (fep) frag.entryPoint = fep;
    }

    var desc = {
        vertex: { module: vmod, buffers: vbufs },
        primitive: prim,
        multisample: ms
    };
    if (vep) desc.vertex.entryPoint = vep;
    if (ds) desc.depthStencil = ds;
    if (frag) desc.fragment = frag;
    var layout = MU32[(d+12)>>2];
    if (layout) desc.layout = WPL[layout]; else desc.layout = 'auto';
    Wlog('CreateRenderPipeline: topology=' + (prim.topology||'?') + ' targets=' + (frag ? frag.targets.length : 0) + ' vbufs=' + vbufs.length);
    try { return Wnew(WRP, dev.createRenderPipeline(desc)); }
    catch(err) { abort('WEBGPU', 'wgpuDeviceCreateRenderPipeline failed: ' + err.message + (err.stack ? '\n' + err.stack : '')); }
})

WAJIC_LIB(WEBGPU, WGPUBindGroupLayout, wgpuRenderPipelineGetBindGroupLayout,
    (WGPURenderPipeline pipeline, unsigned int groupIndex),
{
    var rp = Wget(WRP, pipeline, 'pipeline', 'wgpuRenderPipelineGetBindGroupLayout');
    return Wnew(WBGL, rp.getBindGroupLayout(groupIndex));
})

// ---- Bind group ------------------------------------------------------------

WAJIC_LIB(WEBGPU, WGPUBindGroup, wgpuDeviceCreateBindGroup,
    (WGPUDevice device, const void* descriptor),
{
    // WGPUBindGroupDescriptor: +0:nextInChain(4) +4:label(8) +12:layout(4) +16:entryCount(4) +20:entries(4)
    var dev = Wget(WD, device, 'device', 'wgpuDeviceCreateBindGroup');
    var layout = WBGL[MU32[(descriptor+12)>>2]];
    var count = MU32[(descriptor+16)>>2];
    var eptr = MU32[(descriptor+20)>>2];
    var entries = [];
    // WGPUBindGroupEntry wasm32:
    // +0:nextInChain(4) +4:binding(4) +8:buffer(4) [pad4]
    // +16:offset(uint64,8) +24:size(uint64,8)
    // +32:sampler(4) +36:textureView(4)
    // sizeof=40
    for (var i = 0; i < count; i++) {
        var p = eptr + i * 40;
        var e = { binding: MU32[(p+4)>>2] };
        var buf = MU32[(p+8)>>2];
        if (buf) {
            e.resource = {
                buffer: WB[buf],
                offset: MU32[(p+16)>>2], // low 32 bits
                size: (MU32[(p+24)>>2]>>>0) === 0xFFFFFFFF ? undefined : MU32[(p+24)>>2]
            };
        }
        var sam = MU32[(p+32)>>2];
        if (sam) e.resource = WSa[sam];
        var tv = MU32[(p+36)>>2];
        if (tv) e.resource = WTV[tv];
        entries.push(e);
    }
    try { return Wnew(WBG, dev.createBindGroup({ layout: layout, entries: entries })); }
    catch(err) { abort('WEBGPU', 'wgpuDeviceCreateBindGroup failed: ' + err.message); }
})

// ---- Command encoder -------------------------------------------------------

WAJIC_LIB(WEBGPU, WGPUCommandEncoder, wgpuDeviceCreateCommandEncoder,
    (WGPUDevice device, const void* descriptor),
{
    var dev = Wget(WD, device, 'device', 'wgpuDeviceCreateCommandEncoder');
    return Wnew(WCE, dev.createCommandEncoder());
})

WAJIC_LIB(WEBGPU, WGPURenderPassEncoder, wgpuCommandEncoderBeginRenderPass,
    (WGPUCommandEncoder encoder, const void* descriptor),
{
    // WGPURenderPassDescriptor:
    // +0:nextInChain(4) +4:label(8) +12:colorAttachmentCount(4) +16:colorAttachments(4)
    // +20:depthStencilAttachment(4) +24:occlusionQuerySet(4)
    var caCount = MU32[(descriptor+12)>>2];
    var caPtr = MU32[(descriptor+16)>>2];
    var colorAttachments = [];
    // Refresh MF64 via GF64() in case memory grew
    var f64 = GF64();
    // WGPURenderPassColorAttachment wasm32:
    // +0:nextInChain(4) +4:view(4) +8:depthSlice(4) +12:resolveTarget(4)
    // +16:loadOp(4) +20:storeOp(4)
    // +24:clearValue(WGPUColor, 4xf64=32, aligned 8) → starts at 24 (24%8==0)
    // sizeof=56
    var surfId = 0; // track if any color attachment targets a surface
    for (var i = 0; i < caCount; i++) {
        var p = caPtr + i * 56;
        var view = MU32[(p+4)>>2];
        if (i === 0 && WSurfVwIds[view]) surfId = WSurfVwIds[view];
        var ca = {
            view: WTV[view],
            loadOp: ELdOp[MU32[(p+16)>>2]] || 'clear',
            storeOp: EStOp[MU32[(p+20)>>2]] || 'store',
            clearValue: { r: f64[(p+24)>>3], g: f64[(p+32)>>3], b: f64[(p+40)>>3], a: f64[(p+48)>>3] }
        };
        var ds = MU32[(p+8)>>2];
        if (ds !== 0xFFFFFFFF) ca.depthSlice = ds;
        var rt = MU32[(p+12)>>2];
        if (rt) ca.resolveTarget = WTV[rt];
        colorAttachments.push(ca);
    }
    var desc = { colorAttachments: colorAttachments };
    var dsaPtr = MU32[(descriptor+20)>>2];
    if (dsaPtr) {
        // WGPURenderPassDepthStencilAttachment:
        // +0:view(4) +4:depthLoadOp(4) +8:depthStoreOp(4) +12:depthClearValue(f32,4)
        // +16:depthReadOnly(4) +20:stencilLoadOp(4) +24:stencilStoreOp(4) +28:stencilClearValue(4)
        // +32:stencilReadOnly(4)
        desc.depthStencilAttachment = {
            view: WTV[MU32[dsaPtr>>2]],
            depthLoadOp: ELdOp[MU32[(dsaPtr+4)>>2]],
            depthStoreOp: EStOp[MU32[(dsaPtr+8)>>2]],
            depthClearValue: MF32[(dsaPtr+12)>>2],
            depthReadOnly: !!MU32[(dsaPtr+16)>>2],
            stencilLoadOp: ELdOp[MU32[(dsaPtr+20)>>2]],
            stencilStoreOp: EStOp[MU32[(dsaPtr+24)>>2]],
            stencilClearValue: MU32[(dsaPtr+28)>>2],
            stencilReadOnly: !!MU32[(dsaPtr+32)>>2]
        };
    }
    var rpe = Wget(WCE, encoder, 'encoder', 'wgpuCommandEncoderBeginRenderPass').beginRenderPass(desc);
    // Auto-apply viewport/scissor when rendering to a padded surface canvas
    if (surfId) {
        var surf = WS[surfId];
        if (surf && (surf.padW !== surf.renderW || surf.padH !== surf.renderH)) {
            rpe.setViewport(0, 0, surf.renderW, surf.renderH, 0, 1);
            rpe.setScissorRect(0, 0, surf.renderW, surf.renderH);
        }
    }
    return Wnew(WRPE, rpe);
})

WAJIC_LIB(WEBGPU, WGPUCommandBuffer, wgpuCommandEncoderFinish,
    (WGPUCommandEncoder encoder, const void* descriptor),
{
    return Wnew(WCB, Wget(WCE, encoder, 'encoder', 'wgpuCommandEncoderFinish').finish());
})

WAJIC_LIB(WEBGPU, void, wgpuCommandEncoderCopyBufferToTexture,
    (WGPUCommandEncoder encoder, const void* source, const void* destination, const void* copySize),
{
    // WGPUTexelCopyBufferInfo (source): +0:nextInChain(4)
    // +4:layout(WGPUTexelCopyBufferLayout): +4:nextInChain(4) +8:pad(4)[uint64 align] +12:pad(4) +16:bytesPerRow(4) +20:rowsPerImage(4)
    // (WGPUTexelCopyBufferLayout.offset is uint64 at +8, low word at +12, but always 0 in practice)
    // +28:buffer(4) => sizeof=32
    // WGPUTexelCopyTextureInfo (destination): +0:nextInChain(4) +4:texture(4) +8:mipLevel(4) +12:origin.x(4) +16:origin.y(4) +20:origin.z(4) +24:aspect(4)
    var enc = Wget(WCE, encoder, 'encoder', 'wgpuCommandEncoderCopyBufferToTexture');
    var buf = Wget(WB, MU32[(source+28)>>2], 'srcBuf', 'copyB2T');
    var layout = { offset: MU32[(source+12)>>2], bytesPerRow: MU32[(source+16)>>2], rowsPerImage: MU32[(source+20)>>2] };
    var dstTex = Wget(WT, MU32[(destination+4)>>2], 'dstTex', 'copyB2T');
    var dst = { texture: dstTex, mipLevel: MU32[(destination+8)>>2], origin: { x: MU32[(destination+12)>>2], y: MU32[(destination+16)>>2], z: MU32[(destination+20)>>2] } };
    var sz = { width: MU32[copySize>>2], height: MU32[(copySize+4)>>2], depthOrArrayLayers: MU32[(copySize+8)>>2] };
    enc.copyBufferToTexture({ buffer: buf, bytesPerRow: layout.bytesPerRow, rowsPerImage: layout.rowsPerImage, offset: layout.offset }, dst, sz);
})

WAJIC_LIB(WEBGPU, void, wgpuCommandEncoderCopyTextureToTexture,
    (WGPUCommandEncoder encoder, const void* source, const void* destination, const void* copySize),
{
    // WGPUTexelCopyTextureInfo: +0:nextInChain(4) +4:texture(4) +8:mipLevel(4) +12:origin.x(4) +16:origin.y(4) +20:origin.z(4) +24:aspect(4)
    var enc = Wget(WCE, encoder, 'encoder', 'wgpuCommandEncoderCopyTextureToTexture');
    var srcTex = Wget(WT, MU32[(source+4)>>2], 'srcTex', 'copyT2T');
    var dstTex = Wget(WT, MU32[(destination+4)>>2], 'dstTex', 'copyT2T');
    var src = { texture: srcTex, mipLevel: MU32[(source+8)>>2], origin: { x: MU32[(source+12)>>2], y: MU32[(source+16)>>2], z: MU32[(source+20)>>2] } };
    var dst = { texture: dstTex, mipLevel: MU32[(destination+8)>>2], origin: { x: MU32[(destination+12)>>2], y: MU32[(destination+16)>>2], z: MU32[(destination+20)>>2] } };
    var sz = { width: MU32[copySize>>2], height: MU32[(copySize+4)>>2], depthOrArrayLayers: MU32[(copySize+8)>>2] };
    enc.copyTextureToTexture(src, dst, sz);
})

// ---- Compute pipeline ------------------------------------------------------

WAJIC_LIB(WEBGPU, WGPUComputePipeline, wgpuDeviceCreateComputePipeline,
    (WGPUDevice device, const void* descriptor),
{
    // WGPUComputePipelineDescriptor wasm32:
    // +0:nextInChain(4) +4:label(8) +12:layout(4)
    // +16:compute(WGPUProgrammableStageDescriptor)
    //   +16:nextInChain(4) +20:module(4) +24:entryPoint(8) +32:constantCount(4) +36:constants(4)
    var dev = Wget(WD, device, 'device', 'wgpuDeviceCreateComputePipeline');
    var layout = MU32[(descriptor+12)>>2];
    var module = MU32[(descriptor+20)>>2];
    var desc = {
        layout: layout ? Wget(WPL, layout, 'layout', 'createComputePipeline') : 'auto',
        compute: { module: Wget(WSM, module, 'shaderModule', 'createComputePipeline') }
    };
    var ep = Wsv(descriptor+24);
    if (ep) desc.compute.entryPoint = ep;
    try { return Wnew(WCP, dev.createComputePipeline(desc)); }
    catch(err) { abort('WEBGPU', 'wgpuDeviceCreateComputePipeline failed: ' + err.message); }
})

// ---- Compute pass encoder --------------------------------------------------

WAJIC_LIB(WEBGPU, WGPUComputePassEncoder, wgpuCommandEncoderBeginComputePass,
    (WGPUCommandEncoder encoder, const void* descriptor),
{
    return Wnew(WCPE, Wget(WCE, encoder, 'encoder', 'beginComputePass').beginComputePass());
})

WAJIC_LIB(WEBGPU, void, wgpuComputePassEncoderSetPipeline,
    (WGPUComputePassEncoder pass, WGPUComputePipeline pipeline),
{
    Wget(WCPE, pass, 'pass', 'setComputePipeline').setPipeline(Wget(WCP, pipeline, 'pipeline', 'setComputePipeline'));
})

WAJIC_LIB(WEBGPU, void, wgpuComputePassEncoderSetBindGroup,
    (WGPUComputePassEncoder pass, unsigned int groupIndex,
     WGPUBindGroup group, unsigned int dynamicOffsetCount, const unsigned int* dynamicOffsets),
{
    var offsets;
    if (dynamicOffsetCount > 0) {
        offsets = [];
        for (var i = 0; i < dynamicOffsetCount; i++)
            offsets.push(MU32[(dynamicOffsets>>2)+i]);
    }
    Wget(WCPE, pass, 'pass', 'setComputeBindGroup').setBindGroup(groupIndex, Wget(WBG, group, 'bindGroup', 'setComputeBindGroup'), offsets || []);
})

WAJIC_LIB(WEBGPU, void, wgpuComputePassEncoderDispatchWorkgroups,
    (WGPUComputePassEncoder pass, unsigned int workgroupCountX,
     unsigned int workgroupCountY, unsigned int workgroupCountZ),
{
    Wget(WCPE, pass, 'pass', 'dispatchWorkgroups').dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
})

WAJIC_LIB(WEBGPU, void, wgpuComputePassEncoderEnd,
    (WGPUComputePassEncoder pass),
{
    Wget(WCPE, pass, 'pass', 'computePassEnd').end();
})

// ---- Texture getters -------------------------------------------------------

WAJIC_LIB(WEBGPU, unsigned int, wgpuTextureGetMipLevelCount,
    (WGPUTexture texture),
{
    return Wget(WT, texture, 'texture', 'getMipLevelCount').mipLevelCount;
})

WAJIC_LIB(WEBGPU, unsigned int, wgpuTextureGetWidth,
    (WGPUTexture texture),
{
    return Wget(WT, texture, 'texture', 'getWidth').width;
})

WAJIC_LIB(WEBGPU, unsigned int, wgpuTextureGetHeight,
    (WGPUTexture texture),
{
    return Wget(WT, texture, 'texture', 'getHeight').height;
})

WAJIC_LIB(WEBGPU, unsigned int, wgpuTextureGetDepthOrArrayLayers,
    (WGPUTexture texture),
{
    return Wget(WT, texture, 'texture', 'getDepthOrArrayLayers').depthOrArrayLayers;
})

WAJIC_LIB(WEBGPU, unsigned int, wgpuTextureGetFormat,
    (WGPUTexture texture),
{
    var fmt = Wget(WT, texture, 'texture', 'getFormat').format;
    for (var i = 0; i < EFmt.length; i++) if (EFmt[i] === fmt) return i;
    return 0;
})

// ---- Render pass encoder ---------------------------------------------------

WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderSetPipeline,
    (WGPURenderPassEncoder encoder, WGPURenderPipeline pipeline),
{
    Wget(WRPE, encoder, 'encoder', 'wgpuRenderPassEncoderSetPipeline').setPipeline(Wget(WRP, pipeline, 'pipeline', 'wgpuRenderPassEncoderSetPipeline'));
})

WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderSetBindGroup,
    (WGPURenderPassEncoder encoder, unsigned int groupIndex,
     WGPUBindGroup group, unsigned int dynamicOffsetCount, const unsigned int* dynamicOffsets),
{
    var offsets;
    if (dynamicOffsetCount > 0) {
        offsets = [];
        for (var i = 0; i < dynamicOffsetCount; i++)
            offsets.push(MU32[(dynamicOffsets>>2)+i]);
    }
    Wget(WRPE, encoder, 'encoder', 'setBindGroup').setBindGroup(groupIndex, Wget(WBG, group, 'bindGroup', 'setBindGroup'), offsets || []);
})

WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderSetVertexBuffer,
    (WGPURenderPassEncoder encoder, unsigned int slot, WGPUBuffer buffer,
     unsigned int offset, unsigned int size),
{
    var o = offset >>> 0, s = size >>> 0;
    Wget(WRPE, encoder, 'encoder', 'setVertexBuffer').setVertexBuffer(slot, Wget(WB, buffer, 'buffer', 'setVertexBuffer'), o, s === 0xFFFFFFFF ? undefined : s);
})

WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderSetIndexBuffer,
    (WGPURenderPassEncoder encoder, WGPUBuffer buffer,
     unsigned int format, unsigned int offset, unsigned int size),
{
    var o = offset >>> 0, s = size >>> 0;
    Wget(WRPE, encoder, 'encoder', 'setIndexBuffer').setIndexBuffer(Wget(WB, buffer, 'buffer', 'setIndexBuffer'), EIFmt[format], o, s === 0xFFFFFFFF ? undefined : s);
})

WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderDraw,
    (WGPURenderPassEncoder encoder, unsigned int vertexCount,
     unsigned int instanceCount, unsigned int firstVertex, unsigned int firstInstance),
{
    Wget(WRPE, encoder, 'encoder', 'draw').draw(vertexCount, instanceCount, firstVertex, firstInstance);
})

WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderDrawIndexed,
    (WGPURenderPassEncoder encoder, unsigned int indexCount,
     unsigned int instanceCount, unsigned int firstIndex,
     int baseVertex, unsigned int firstInstance),
{
    Wget(WRPE, encoder, 'encoder', 'drawIndexed').drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
})

WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderEnd,
    (WGPURenderPassEncoder encoder),
{
    Wget(WRPE, encoder, 'encoder', 'wgpuRenderPassEncoderEnd').end();
})

WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderSetViewport,
    (WGPURenderPassEncoder encoder, float x, float y, float width, float height,
     float minDepth, float maxDepth),
{
    Wget(WRPE, encoder, 'encoder', 'setViewport').setViewport(x, y, width, height, minDepth, maxDepth);
})

WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderSetScissorRect,
    (WGPURenderPassEncoder encoder, unsigned int x, unsigned int y,
     unsigned int width, unsigned int height),
{
    Wget(WRPE, encoder, 'encoder', 'setScissorRect').setScissorRect(x, y, width, height);
})

WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderSetBlendConstant,
    (WGPURenderPassEncoder encoder, const void* color),
{
    var f64 = GF64();
    Wget(WRPE, encoder, 'encoder', 'setBlendConstant').setBlendConstant({
        r: f64[color>>3], g: f64[(color+8)>>3],
        b: f64[(color+16)>>3], a: f64[(color+24)>>3]
    });
})

WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderSetStencilReference,
    (WGPURenderPassEncoder encoder, unsigned int reference),
{
    Wget(WRPE, encoder, 'encoder', 'setStencilReference').setStencilReference(reference);
})

// ---- Render bundle encoder -------------------------------------------------

WAJIC_LIB(WEBGPU, WGPURenderBundleEncoder, wgpuDeviceCreateRenderBundleEncoder,
    (WGPUDevice device, const void* descriptor),
{
    // WGPURenderBundleEncoderDescriptor:
    // +0:nextInChain(4) +4:label.data(4) +8:label.length(4)
    // +12:colorFormatCount(4) +16:colorFormats(ptr,4)
    // +20:depthStencilFormat(4) +24:sampleCount(4) +28:depthReadOnly(4) +32:stencilReadOnly(4)
    var cfCount = MU32[(descriptor+12)>>2];
    var cfPtr = MU32[(descriptor+16)>>2];
    var colorFormats = [];
    for (var i = 0; i < cfCount; i++)
        colorFormats.push(EFmt[MU32[(cfPtr>>2)+i]]);
    var dsFormat = MU32[(descriptor+20)>>2];
    var desc = {
        colorFormats: colorFormats,
        sampleCount: MU32[(descriptor+24)>>2] || 1
    };
    if (dsFormat) desc.depthStencilFormat = EFmt[dsFormat];
    desc.depthReadOnly = !!MU32[(descriptor+28)>>2];
    desc.stencilReadOnly = !!MU32[(descriptor+32)>>2];
    return Wnew(WRBE, Wget(WD, device, 'device', 'createRenderBundleEncoder').createRenderBundleEncoder(desc));
})

WAJIC_LIB(WEBGPU, void, wgpuRenderBundleEncoderSetPipeline,
    (WGPURenderBundleEncoder encoder, WGPURenderPipeline pipeline),
{
    Wget(WRBE, encoder, 'encoder', 'setPipeline').setPipeline(Wget(WRP, pipeline, 'pipeline', 'setPipeline'));
})

WAJIC_LIB(WEBGPU, void, wgpuRenderBundleEncoderSetVertexBuffer,
    (WGPURenderBundleEncoder encoder, unsigned int slot, WGPUBuffer buffer,
     unsigned int offset, unsigned int size),
{
    var o = offset >>> 0, s = size >>> 0;
    Wget(WRBE, encoder, 'encoder', 'setVertexBuffer').setVertexBuffer(slot, Wget(WB, buffer, 'buffer', 'setVertexBuffer'), o, s === 0xFFFFFFFF ? undefined : s);
})

WAJIC_LIB(WEBGPU, void, wgpuRenderBundleEncoderSetIndexBuffer,
    (WGPURenderBundleEncoder encoder, WGPUBuffer buffer,
     unsigned int indexFormat, unsigned int offset, unsigned int size),
{
    var o = offset >>> 0, s = size >>> 0;
    Wget(WRBE, encoder, 'encoder', 'setIndexBuffer').setIndexBuffer(Wget(WB, buffer, 'buffer', 'setIndexBuffer'), EIFmt[indexFormat], o, s === 0xFFFFFFFF ? undefined : s);
})

WAJIC_LIB(WEBGPU, void, wgpuRenderBundleEncoderSetBindGroup,
    (WGPURenderBundleEncoder encoder, unsigned int groupIndex,
     WGPUBindGroup group, unsigned int dynamicOffsetCount,
     const unsigned int* dynamicOffsets),
{
    var e = Wget(WRBE, encoder, 'encoder', 'setBindGroup');
    var bg = Wget(WBG, group, 'bindGroup', 'setBindGroup');
    if (dynamicOffsetCount > 0) {
        var offsets = [];
        for (var i = 0; i < dynamicOffsetCount; i++)
            offsets.push(MU32[(dynamicOffsets>>2)+i]);
        e.setBindGroup(groupIndex, bg, offsets);
    } else {
        e.setBindGroup(groupIndex, bg);
    }
})

WAJIC_LIB(WEBGPU, void, wgpuRenderBundleEncoderDraw,
    (WGPURenderBundleEncoder encoder, unsigned int vertexCount,
     unsigned int instanceCount, unsigned int firstVertex,
     unsigned int firstInstance),
{
    Wget(WRBE, encoder, 'encoder', 'draw').draw(vertexCount, instanceCount, firstVertex, firstInstance);
})

WAJIC_LIB(WEBGPU, void, wgpuRenderBundleEncoderDrawIndexed,
    (WGPURenderBundleEncoder encoder, unsigned int indexCount,
     unsigned int instanceCount, unsigned int firstIndex,
     int baseVertex, unsigned int firstInstance),
{
    Wget(WRBE, encoder, 'encoder', 'drawIndexed').drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
})

WAJIC_LIB(WEBGPU, WGPURenderBundle, wgpuRenderBundleEncoderFinish,
    (WGPURenderBundleEncoder encoder, const void* descriptor),
{
    var bundle = Wget(WRBE, encoder, 'encoder', 'finish').finish();
    Wdel(WRBE, encoder);
    return Wnew(WRB, bundle);
})

WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderExecuteBundles,
    (WGPURenderPassEncoder encoder, unsigned int bundleCount,
     const void* bundles),
{
    var e = Wget(WRPE, encoder, 'encoder', 'executeBundles');
    var b = [];
    for (var i = 0; i < bundleCount; i++)
        b.push(Wget(WRB, MU32[(bundles>>2)+i], 'renderBundle', 'executeBundles'));
    e.executeBundles(b);
})

// ---- Queue -----------------------------------------------------------------

WAJIC_LIB(WEBGPU, void, wgpuQueueSubmit,
    (WGPUQueue queue, unsigned int commandCount, const void* commands),
{
    var q = Wget(WQ, queue, 'queue', 'wgpuQueueSubmit');
    var cmds = [];
    for (var i = 0; i < commandCount; i++)
        cmds.push(Wget(WCB, MU32[(commands>>2)+i], 'commandBuffer', 'wgpuQueueSubmit'));
    q.submit(cmds);
})

// ---- Sampler ---------------------------------------------------------------

WAJIC_LIB(WEBGPU, WGPUSampler, wgpuDeviceCreateSampler,
    (WGPUDevice device, const void* descriptor),
{
    var dev = Wget(WD, device, 'device', 'wgpuDeviceCreateSampler');
    if (!descriptor) return Wnew(WSa, dev.createSampler());
    // WGPUSamplerDescriptor layout (wasm32):
    // +0:  nextInChain(4)
    // +4:  label(ptr4+len4=8)
    // +12: addressModeU(4) +16: addressModeV(4) +20: addressModeW(4)
    // +24: magFilter(4)    +28: minFilter(4)    +32: mipmapFilter(4)
    // +36: lodMinClamp(f32) +40: lodMaxClamp(f32)
    // +44: compare(4)      +48: maxAnisotropy(u16)
    var EAddr = [,'clamp-to-edge','repeat','mirror-repeat'];
    var EFilt = [,'nearest','linear'];
    var ECmp  = [,'never','less','equal','less-equal','greater','not-equal','greater-equal','always'];
    var desc = {};
    var v;
    v = MU32[(descriptor+12)>>2]; if (v) desc.addressModeU = EAddr[v];
    v = MU32[(descriptor+16)>>2]; if (v) desc.addressModeV = EAddr[v];
    v = MU32[(descriptor+20)>>2]; if (v) desc.addressModeW = EAddr[v];
    v = MU32[(descriptor+24)>>2]; if (v) desc.magFilter    = EFilt[v];
    v = MU32[(descriptor+28)>>2]; if (v) desc.minFilter    = EFilt[v];
    v = MU32[(descriptor+32)>>2]; if (v) desc.mipmapFilter = EFilt[v];
    desc.lodMinClamp = MF32[(descriptor+36)>>2];
    desc.lodMaxClamp = MF32[(descriptor+40)>>2];
    v = MU32[(descriptor+44)>>2]; if (v) desc.compare = ECmp[v];
    v = MU16[(descriptor+48)>>1]; if (v > 1) desc.maxAnisotropy = v;
    return Wnew(WSa, dev.createSampler(desc));
})

// ---- Release functions (clean up JS-side object references) ----------------

WAJIC_LIB(WEBGPU, void, wgpuInstanceRelease, (WGPUInstance h), { Wdel(WI, h); })
// wgpuDeviceRelease: WAjic has no refcounting, so Release == destroy the device immediately.
// Calling device.destroy() is critical — it releases all Vulkan VRAM synchronously so the
// next page load / WebGPU session doesn't inherit exhausted device memory.
WAJIC_LIB(WEBGPU, void, wgpuDeviceRelease, (WGPUDevice h),
{
    if (WD[h]) { try { WD[h].destroy(); } catch(e) {} }
    Wdel(WD, h);
})
WAJIC_LIB(WEBGPU, void, wgpuQueueRelease, (WGPUQueue h), { Wdel(WQ, h); })
WAJIC_LIB(WEBGPU, void, wgpuSurfaceRelease, (WGPUSurface h), { Wdel(WS, h); })
WAJIC_LIB(WEBGPU, void, wgpuShaderModuleRelease, (WGPUShaderModule h), { Wdel(WSM, h); })
// wgpuBufferRelease: call buffer.destroy() to return GPU memory immediately rather than
// waiting for JS GC to collect the buffer object.
WAJIC_LIB(WEBGPU, void*, wgpuBufferGetMappedRange,
    (WGPUBuffer buffer, unsigned int offset, unsigned int size),
{
    // Allocate WASM heap memory to hold the data; track it for wgpuBufferUnmap.
    // The actual GPU buffer was created with mappedAtCreation=true.
    var ptr = ASM.malloc(size);
    if (!ptr) abort('WEBGPU', 'wgpuBufferGetMappedRange: malloc failed');
    WMBUF[buffer] = { ptr: ptr, size: size, offset: offset };
    return ptr;
})

WAJIC_LIB(WEBGPU, void, wgpuBufferUnmap,
    (WGPUBuffer buffer),
{
    var info = WMBUF[buffer];
    if (!info) return;
    var buf = WB[buffer];
    if (buf) {
        // Copy WASM heap data into the GPU buffer's mapped range, then unmap.
        var mapped = buf.getMappedRange(info.offset, info.size);
        new Uint8Array(mapped).set(MU8.subarray(info.ptr, info.ptr + info.size));
        buf.unmap();
    }
    ASM.free(info.ptr);
    delete WMBUF[buffer];
})

WAJIC_LIB(WEBGPU, void, wgpuBufferRelease, (WGPUBuffer h),
{
    if (WB[h]) { try { WB[h].destroy(); } catch(e) {} }
    Wdel(WB, h);
})
// wgpuTextureRelease: destroy non-surface textures eagerly. Surface textures (from
// getCurrentTexture) are owned by the swap chain — only null the handle for those.
WAJIC_LIB(WEBGPU, void, wgpuTextureRelease, (WGPUTexture h),
{
    if (WT[h] && !WSurfTexIds[h]) { try { WT[h].destroy(); } catch(e) {} }
    delete WSurfTexIds[h]; Wdel(WT, h);
})
WAJIC_LIB(WEBGPU, void, wgpuTextureViewRelease, (WGPUTextureView h), { delete WSurfVwIds[h]; Wdel(WTV, h); })
WAJIC_LIB(WEBGPU, void, wgpuSamplerRelease, (WGPUSampler h), { Wdel(WSa, h); })
WAJIC_LIB(WEBGPU, void, wgpuBindGroupLayoutRelease, (WGPUBindGroupLayout h), { Wdel(WBGL, h); })
WAJIC_LIB(WEBGPU, void, wgpuPipelineLayoutRelease, (WGPUPipelineLayout h), { Wdel(WPL, h); })
WAJIC_LIB(WEBGPU, void, wgpuRenderPipelineRelease, (WGPURenderPipeline h), { Wdel(WRP, h); })
WAJIC_LIB(WEBGPU, void, wgpuComputePipelineRelease, (WGPUComputePipeline h), { Wdel(WCP, h); })
WAJIC_LIB(WEBGPU, void, wgpuBindGroupRelease, (WGPUBindGroup h), { Wdel(WBG, h); })
WAJIC_LIB(WEBGPU, void, wgpuCommandEncoderRelease, (WGPUCommandEncoder h), { Wdel(WCE, h); })
WAJIC_LIB(WEBGPU, void, wgpuRenderPassEncoderRelease, (WGPURenderPassEncoder h), { Wdel(WRPE, h); })
WAJIC_LIB(WEBGPU, void, wgpuComputePassEncoderRelease, (WGPUComputePassEncoder h), { Wdel(WCPE, h); })
WAJIC_LIB(WEBGPU, void, wgpuCommandBufferRelease, (WGPUCommandBuffer h), { Wdel(WCB, h); })
WAJIC_LIB(WEBGPU, void, wgpuQuerySetRelease, (WGPUQuerySet h), { Wdel(WQS, h); })
WAJIC_LIB(WEBGPU, void, wgpuAdapterRelease, (WGPUAdapter h), { Wdel(WA_, h); })
WAJIC_LIB(WEBGPU, void, wgpuRenderBundleRelease, (WGPURenderBundle h), { Wdel(WRB, h); })
WAJIC_LIB(WEBGPU, void, wgpuRenderBundleEncoderRelease, (WGPURenderBundleEncoder h), { Wdel(WRBE, h); })

// ---- Destroy functions (explicit GPU resource destruction) ------------------

WAJIC_LIB(WEBGPU, void, wgpuDeviceDestroy, (WGPUDevice h),
{
    if (WD[h]) { try { WD[h].destroy(); } catch(e) {} }
    Wdel(WD, h);
})

WAJIC_LIB(WEBGPU, void, wgpuBufferDestroy, (WGPUBuffer h),
{
    if (WB[h]) { try { WB[h].destroy(); } catch(e) {} }
})

WAJIC_LIB(WEBGPU, void, wgpuTextureDestroy, (WGPUTexture h),
{
    if (WT[h]) { try { WT[h].destroy(); } catch(e) {} }
})

// ---- Queue: texture write --------------------------------------------------

WAJIC_LIB(WEBGPU, void, wgpuQueueWriteTexture,
    (WGPUQueue queue, const void* destination, const void* data,
     unsigned int dataSize, const void* dataLayout, const void* writeSize),
{
    // WGPUTexelCopyTextureInfo (destination): +0:nextInChain(4) +4:texture(4) +8:mipLevel(4)
    // +12:origin.x(4) +16:origin.y(4) +20:origin.z(4) +24:aspect(4) => sizeof=28
    var tex = WT[MU32[(destination+4)>>2]];
    var dst = {
        texture: tex,
        mipLevel: MU32[(destination+8)>>2],
        origin: { x: MU32[(destination+12)>>2], y: MU32[(destination+16)>>2], z: MU32[(destination+20)>>2] }
    };
    // WGPUTexelCopyBufferLayout (dataLayout):
    // +0:nextInChain(4) [pad4 for uint64_t alignment] +8:offset(uint64,8)
    // +16:bytesPerRow(4) +20:rowsPerImage(4)  => sizeof=24
    var layout = {
        offset: MU32[(dataLayout+8)>>2], // low 32 bits (high 32 bits always 0 in practice)
        bytesPerRow: MU32[(dataLayout+16)>>2],
        rowsPerImage: MU32[(dataLayout+20)>>2]
    };
    // WGPUExtent3D (writeSize): +0:width(4) +4:height(4) +8:depthOrArrayLayers(4)
    var sz = {
        width: MU32[writeSize>>2],
        height: MU32[(writeSize+4)>>2],
        depthOrArrayLayers: MU32[(writeSize+8)>>2]
    };
    WQ[queue].writeTexture(dst, MU8.subarray(data, data + dataSize), layout, sz);
})

// ---- Utility: get preferred surface format ---------------------------------

WAJIC_LIB(WEBGPU, WGPUTextureFormat, wgpuSurfaceGetPreferredFormat,
    (WGPUSurface surface, WGPUAdapter adapter),
{
    var fmt = navigator.gpu.getPreferredCanvasFormat();
    for (var i = 0; i < EFmt.length; i++) if (EFmt[i] === fmt) return i;
    return 23; // fallback to BGRA8Unorm
})

// ---- Utility: get surface render dimensions (original app dimensions) ------

WAJIC_LIB(WEBGPU, void, wgpuWajicGetSurfaceRenderSize,
    (WGPUSurface surface, unsigned int* width, unsigned int* height),
{
    var s = WS[surface];
    MU32[width>>2] = s.renderW;
    MU32[height>>2] = s.renderH;
})
