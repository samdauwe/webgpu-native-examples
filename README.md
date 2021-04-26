<img align="right" width="20%" src="https://github.com/samdauwe/webgpu-native-examples/blob/master/doc/images/webgpu_logo.png">

# WebGPU Native Examples and Demos

[WebGPU](https://gpuweb.github.io/gpuweb/) is a new graphics and compute API designed by the [“GPU for the Web”](https://www.w3.org/community/gpu/) W3C community group. It aims to provide modern features such as “GPU compute” as well as lower overhead access to GPU hardware and better, more predictable performance. WebGPU should work with existing platform APIs such as Direct3D 12 from Microsoft, Metal from Apple, and Vulkan from the Khronos Group. 

WebGPU is designed for the Web, used by JavaScript and WASM applications, and driven by the shared principles of Web APIs. However, it doesn’t have to be only for the Web though. Targeting WebGPU on native enables to write extremely portable and fairly performant graphics applications. The WebGPU API is beginner friendly, meaning that the API automates some of the aspects of low-level graphics APIs which have high complexity but low return on investment. It still has the core pieces of the next-gen APIs, such as command buffers, render passes, pipeline states and layouts. Because the complexity is reduced, users will be able to direct more focus towards writing efficiently application code.

From the very beginning, Google had both native and in-browser use of their implementation, which is now called [Dawn](https://dawn.googlesource.com/dawn). Mozilla has a shared interest in allowing developers to target a shared “WebGPU on native” target instead of a concrete “Dawn” or “[wgpu-native](https://github.com/gfx-rs/wgpu-native)”. This is achieved, by a [shared header](https://github.com/webgpu-native/webgpu-headers), and C-compatible libraries implementing it. However, this specification is still a moving target.

<img src="https://github.com/samdauwe/webgpu-native-examples/blob/master/doc/images/WebGPU_API.png"/>

This repository contains a collection of open source C examples for [WebGPU](https://gpuweb.github.io/gpuweb/) using [Dawn](https://dawn.googlesource.com/dawn) the open-source and cross-platform implementation of the work-in-progress [WebGPU](https://gpuweb.github.io/gpuweb/) standard. 

## Supported Platforms

* GNU/Linux

## Get the Sources

This repository contains submodules for external dependencies, so when doing a fresh clone you need to clone recursively:

```
git clone --recursive https://github.com/samdauwe/webgpu-native-examples.git
```

Existing repositories can be updated manually:

```
git submodule init
git submodule update
```

## Building

Coming soon...

## Examples

### Basics

#### [Clear screen](src/examples/clear_screen.c)

This example shows how to set up a swap chain and clearing the screen. The screen clearing animation shows a fade-in and fade-out effect.

#### [Triangle](src/examples/triangle.c)

Basic and verbose example for getting a colored triangle rendered to the screen using WebGPU. This is meant as a starting point for learning WebGPU from the ground up.

#### [Two cubes](src/examples/two_cubes.c)

This example shows some of the alignment requirements involved when updating and binding multiple slices of a uniform buffer.

#### [Dynamic uniform buffers](src/examples/dynamic_uniform_buffer.c)

Dynamic uniform buffers are used for rendering multiple objects with multiple matrices stored in a single uniform buffer object. Individual matrices are dynamically addressed upon bind group binding time, minimizing the number of required bind groups.

#### [Texture mapping](src/examples/textured_quad.c)

Loads a 2D texture from disk (including all mip levels), uses staging to upload it into video memory and samples from it using combined image samplers.

#### [Textured cube](src/examples/textured_cube.c)

This example shows how to bind and sample textures.

#### [Cube map textures](src/examples/skybox.c)

Loads a cube map texture from disk containing six different faces. All faces and mip levels are uploaded into video memory, and the cubemap is displayed on a skybox as a backdrop and on a 3D model as a reflection.

### glTF

These samples show how implement different features of the [glTF 2.0 3D format](https://www.khronos.org/gltf/) 3D transmission file format in detail.

#### [glTF model loading and rendering](src/examples/gltf_loading.c)

Shows how to load a complete scene from a [glTF 2.0](https://github.com/KhronosGroup/glTF) file. The structure of the glTF 2.0 scene is converted into the data structures required to render the scene with Vulkan.

### Advanced

#### [Multi sampling](src/examples/msaa_line.c)

This example shows how to achieve [multisample anti-aliasing](https://en.wikipedia.org/wiki/Multisample_anti-aliasing)(MSAA) in WebGPU. The render pipeline is created with a sample count > 1. A new texture with a sample count > 1 is created and set as the color attachment instead of the swapchain. The swapchain is now specified as a resolve_target.

#### [Cube reflection](src/examples/cube_reflection.c)

This example shows how to create a basic reflection pipeline.

#### [Shadow mapping](src/examples/shadow_mapping.c)

This example shows how to sample from a depth texture to render shadows from a directional light source.

### Performance

#### [Instancing](src/examples/instanced_cube.c)

Uses the instancing feature for rendering (many) instances of the same mesh from a single vertex buffer with variable parameters.

### Compute Shader

#### [Animometer](src/examples/animometer.c)

A WebGPU of port of the Animometer MotionMark benchmark.

#### [Compute boids](src/examples/compute_boids.c)

A GPU compute particle simulation that mimics the flocking behavior of birds. A compute shader updates two ping-pong buffers which store particle data. The data is used to draw instanced particles.

#### [Image blur](src/examples/image_blur.c)

This example shows how to blur an image using a WebGPU compute shader.

#### [GPU particle system](src/examples/compute_particles.c)

Attraction based 2D GPU particle system using compute shaders. Particle data is stored in a shader storage buffer and only modified on the GPU using compute particle updates with graphics pipeline vertex access.

#### [N-body simulation](src/examples/compute_n_body.c)

N-body simulation based particle system with multiple attractors and particle-to-particle interaction using two passes separating particle movement calculation and final integration. Shared compute shader memory is used to speed up compute calculations.

#### [Ray tracing](src/examples/compute_ray_tracing.c)

Simple GPU ray tracer with shadows and reflections using a compute shader. No scene geometry is rendered in the graphics pass.

### User Interface

#### [ImGui overlay](src/examples/imgui_overlay.c)

Generates and renders a complex user interface with multiple windows, controls and user interaction on top of a 3D scene. The UI is generated using [Dear ImGUI](https://github.com/ocornut/imgui) and updated each frame.

### Misc

#### [WebGPU Gears](src/examples/gears.c)

WebGPU interpretation of [glxgears](https://linuxreviews.org/Glxgears). Procedurally generates and animates multiple gears.

#### [Video uploading](src/examples/video_uploading.c)

This example shows how to upload video frames to WebGPU. Uses [FFmpeg](https://www.ffmpeg.org/) for the video decoding.

#### [Shadertoy](src/examples/video_uploading.c)

Minimal "shadertoy launcher" using WebGPU, demonstrating how to load an example Shadertoy shader '[Cube lines](https://www.shadertoy.com/view/NslGRN)'.

## Dependencies

Just like all software, WebGPU Native Examples and Demos are built on the shoulders of incredible people! Here's a list of the used libraries.

### System ###

* [CMake](https://cmake.org) (>= 3.17)
* [FFmpeg](https://www.ffmpeg.org/) used for video decoding (optional)

### Available as git submodules ###

* [cglm](https://github.com/recp/cglm): Highly Optimized Graphics Math (glm) for C.
* [cgltf](https://github.com/jkuhlmann/cgltf): Single-file glTF 2.0 loader and writer written in C99.
* [cimgui](https://github.com/cimgui/cimgui): c-api for [Dear ImGui](https://github.com/ocornut/imgui)
* [ktx](https://github.com/KhronosGroup/KTX-Software): KTX (Khronos Texture) Library and Tools
* [rply](http://w3.impa.br/~diego/software/rply/): ANSI C Library for PLY file format input and output
* [stb](https://github.com/nothings/stb): stb single-file public domain libraries for C/C++

## Credits

A huge thanks to the authors of the folowing repositories who demonstrated the use of the [WebGPU API](https://webgpu.dev/) and how to create a minimal example framework:
* [wgpu-rs](https://github.com/gfx-rs/wgpu-rs)
* [webgpu-samples](https://github.com/austinEng/webgpu-samples)
* [Vulkan C++ examples and demos](https://github.com/SaschaWillems/Vulkan)
* [Awesome WebGPU](https://github.com/mikbry/awesome-webgpu)

## License

Open-source under [Apache 2.0 license](http://www.tldrlegal.com/license/apache-license-2.0-%28apache-2.0%29).