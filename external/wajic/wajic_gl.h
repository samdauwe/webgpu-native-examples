/*
  WAjic - WebAssembly JavaScript Interface Creator

  This code is based on library_webgl.js from the Emscripten project
  https://github.com/emscripten-core/emscripten/blob/master/src/library_webgl.js
  * @license
  * Copyright 2010 The Emscripten Authors
  * SPDX-License-Identifier: MIT

  Copyright (c) 2010-2014 Emscripten authors, see AUTHORS file.
  All rights reserved.

  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the
  "Software"), to deal with the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

      Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimers.

      Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following disclaimers
      in the documentation and/or other materials provided with the
      distribution.

      Neither the names of Mozilla,
      nor the names of its contributors may be used to endorse
      or promote products derived from this Software without specific prior
      written permission. 

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR
  ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
*/

#pragma once

#define GL_GLEXT_PROTOTYPES
#define EGL_EGLEXT_PROTOTYPES
#include <GL/gl.h>
#include <wajic.h>

WAJIC_LIB_WITH_INIT(GL,
(
	const GLMINI_TEMP_BUFFER_SIZE = 256, kUniforms = 'u', kMaxUniformLength = 'm', kMaxAttributeLength = 'a', kMaxUniformBlockNameLength = 'b';
	var GLctx;
	var GLcounter = 1;
	var GLbuffers = [];
	var GLprograms = [];
	var GLframebuffers = [];
	var GLtextures = [];
	var GLrenderbuffers = [];
	var GLuniforms = [];
	var GLshaders = [];
	var GLvaos = [];
	var GLprogramInfos = {};
	var GLstringCache = {};
	var GLpackAlignment = 4;
	var GLunpackAlignment = 4;
	var GLFixedLengthArrays = [];
	var GLminiTempFloatBuffers = [];
	var GLminiTempIntBuffers = [];
	for (let i = 0, fbuf = new Float32Array(GLMINI_TEMP_BUFFER_SIZE), ibuf = new Int32Array(GLMINI_TEMP_BUFFER_SIZE); i < GLMINI_TEMP_BUFFER_SIZE; i++)
	{
		GLminiTempFloatBuffers[i] = fbuf.subarray(0, i+1);
		GLminiTempIntBuffers[i] = ibuf.subarray(0, i+1);
	}

	function GLgetNewId(table)
	{
		for (var ret = GLcounter++, i = table.length; i < ret; i++) table[i] = null;
		return ret;
	}

	function GLrecordError(err)
	{
		if (!GLlastError) GLlastError = err;
	}

	function GLgetTexPixelData(type, format, width, height, pixels, internalFormat)
	{
		var sizePerPixel;
		var numChannels;
		switch(format)
		{
			case 0x1906: case 0x1909: case 0x1902: numChannels = 1; break; //GL_ALPHA, GL_LUMINANCE, GL_DEPTH_COMPONENT
			case 0x190A: numChannels = 2; break; //GL_LUMINANCE_ALPHA
			case 0x1907: case 0x8C40: numChannels = 3; break; //GL_RGB, GL_SRGB_EXT
			case 0x1908: case 0x8C42: numChannels = 4; break; //GL_RGBA, GL_SRGB_ALPHA_EXT
			default: GLrecordError(0x500); return null; //GL_INVALID_ENUM
		}
		switch (type)
		{
			case 0x1401: sizePerPixel = numChannels*1; break; //GL_UNSIGNED_BYTE
			case 0x1403: case 0x8D61: sizePerPixel = numChannels*2; break; //GL_UNSIGNED_SHORT, GL_HALF_FLOAT_OES
			case 0x1405: case 0x1406: sizePerPixel = numChannels*4; break; //GL_UNSIGNED_INT, GL_FLOAT
			case 0x84FA: sizePerPixel = 4; break; //GL_UNSIGNED_INT_24_8_WEBGL/GL_UNSIGNED_INT_24_8
			case 0x8363: case 0x8033: case 0x8034: sizePerPixel = 2; break; //GL_UNSIGNED_SHORT_5_6_5, GL_UNSIGNED_SHORT_4_4_4_4, GL_UNSIGNED_SHORT_5_5_5_1
			default: GLrecordError(0x500); return null; //GL_INVALID_ENUM
		}

		function roundedToNextMultipleOf(x, y) { return Math.floor((x + y - 1) / y) * y; }
		var plainRowSize = width * sizePerPixel;
		var alignedRowSize = roundedToNextMultipleOf(plainRowSize, GLunpackAlignment);
		var bytes = (height <= 0 ? 0 : ((height - 1) * alignedRowSize + plainRowSize));

		switch(type)
		{
			case 0x1401: return MU8.subarray(pixels, pixels+bytes); //GL_UNSIGNED_BYTE
			case 0x1406: return MF32.subarray(pixels>>2, (pixels+bytes)>>2); //GL_FLOAT
			case 0x1405: case 0x84FA: return MU32.subarray(pixels>>2, (pixels+bytes)>>2); //GL_UNSIGNED_INT, GL_UNSIGNED_INT_24_8_WEBGL/GL_UNSIGNED_INT_24_8
			case 0x1403: case 0x8363: case 0x8033: case 0x8034: case 0x8D61: return MU16.subarray(pixels>>1,(pixels+bytes)>>1); //GL_UNSIGNED_SHORT, GL_UNSIGNED_SHORT_5_6_5, GL_UNSIGNED_SHORT_4_4_4_4, GL_UNSIGNED_SHORT_5_5_5_1, GL_HALF_FLOAT_OES
			default: GLrecordError(0x500); return null; //GL_INVALID_ENUM
		}
	}

	function GLget(name, p, type)
	{
		// Guard against user passing a null pointer.
		// Note that GLES2 spec does not say anything about how passing a null pointer should be treated.
		// Testing on desktop core GL 3, the application crashes on glGetIntegerv to a null pointer, but
		// better to report an error instead of doing anything random.
		if (!p) return GLrecordError(0x501); // GL_INVALID_VALUE

		var ret = undefined;
		switch(name)
		{
			// Handle a few trivial GLES values
			case 0x8DFA: ret = 1; break; // GL_SHADER_COMPILER
			case 0x8DF8: // GL_SHADER_BINARY_FORMATS
				if (type !== 0 && type !== 1) GLrecordError(0x500); // GL_INVALID_ENUM
				return; // Do not write anything to the out pointer, since no binary formats are supported.
			case 0x8DF9: ret = 0; break; // GL_NUM_SHADER_BINARY_FORMATS
			case 0x86A2: // GL_NUM_COMPRESSED_TEXTURE_FORMATS
				// WebGL doesn't have GL_NUM_COMPRESSED_TEXTURE_FORMATS (it's obsolete since GL_COMPRESSED_TEXTURE_FORMATS returns a JS array that can be queried for length),
				// so implement it ourselves to allow C++ GLES2 code get the length.
				var formats = GLctx.getParameter(0x86A3); // GL_COMPRESSED_TEXTURE_FORMATS
				ret = formats.length;
				break;
		}

		if (ret === undefined)
		{
			var result = GLctx.getParameter(name);
			switch (typeof(result))
			{
				case 'number':
					ret = result;
					break;
				case 'boolean':
					ret = result ? 1 : 0;
					break;
				case 'string':
					return GLrecordError(0x500); // GL_INVALID_ENUM
				case 'object':
					if (result === null)
					{
						// null is a valid result for some (e.g., which buffer is bound - perhaps nothing is bound), but otherwise
						// can mean an invalid name, which we need to report as an error
						switch(name)
						{
							case 0x8894: // ARRAY_BUFFER_BINDING
							case 0x8B8D: // CURRENT_PROGRAM
							case 0x8895: // ELEMENT_ARRAY_BUFFER_BINDING
							case 0x8CA6: // FRAMEBUFFER_BINDING
							case 0x8CA7: // RENDERBUFFER_BINDING
							case 0x8069: // TEXTURE_BINDING_2D
							case 0x8514: // TEXTURE_BINDING_CUBE_MAP
								ret = 0;
								break;
							default:
								return GLrecordError(0x500); // GL_INVALID_ENUM
						}
					}
					else if (result instanceof Float32Array || result instanceof Uint32Array || result instanceof Int32Array || result instanceof Array)
					{
						for (var i = 0; i < result.length; ++i)
						{
							switch (type)
							{
								case 0: MI32[(p>>2)+i] = result[i]; break;
								case 2: MF32[(p>>2)+i] = result[i]; break;
								case 4: MU8[p+i] = (result[i] ? 1 : 0); break;
								default: abort();
							}
						}
						return;
					}
					else if (result instanceof WebGLBuffer || result instanceof WebGLProgram || result instanceof WebGLFramebuffer || result instanceof WebGLRenderbuffer || result instanceof WebGLTexture)
					{
						ret = result.name | 0;
					}
					else
					{
						return GLrecordError(0x500); // GL_INVALID_ENUM
					}
					break;
				default:
					return GLrecordError(0x500); // GL_INVALID_ENUM
			}
		}

		switch (type)
		{
			case 0: MI32[p>>2] = ret; break;
			case 1: MU32[p>>2] = ret; MU32[(p+4)>>2] = (ret - MU32[p>>2])/4294967296; break;
			case 2: MF32[p>>2] = ret; break;
			case 4: MU8[p] = (ret ? 1 : 0); break;
		}
	}

	function GLwriteNumOrArr(data, params, type)
	{
		if (typeof data == 'number' || typeof data == 'boolean')
			(type ? MF32 : MI32)[params>>2] = data;
		else
			for (var i = 0; i < data.length; i++)
				(type ? MF32 : MI32)[(params>>2)+i] = data[i];
	}

	function GLgetUniform(program, location, params, type)
	{
		GLwriteNumOrArr(GLctx.getUniform(GLprograms[program], GLuniforms[location]), params, type);
	}

	function GLgetVertexAttrib(index, pname, params, type)
	{
		var data = GLctx.getVertexAttrib(index, pname);
		if (pname == 0x889F) //VERTEX_ATTRIB_ARRAY_BUFFER_BINDING
			MI32[params>>2] = (data && data["name"]);
		else
			GLwriteNumOrArr(data, params, type)
	}

	function GLgenObjects(n, buffers, createFunction, objectTable)
	{
		for (var i = 0; i < n; i++)
		{
			var buffer = GLctx[createFunction]();
			var id = (buffer && GLgetNewId(objectTable));
			if (buffer)
			{
				buffer.name = id;
				objectTable[id] = buffer;
			}
			else GLrecordError(0x502); //GL_INVALID_OPERATION
			MI32[(buffers>>2)+(i++)] = id;
		}
	}
),
int, glSetupCanvasContext, (int antialias WA_ARG(1), int depth WA_ARG(0), int stencil WA_ARG(0), int alpha WA_ARG(0)),
{
	var canvas = WA.canvas;
	var attr = { majorVersion: 1, minorVersion: 0, antialias: !!antialias, depth: !!depth, stencil: !!stencil, alpha: !!alpha };
	var msg = "", webgl = escape('webgl'), errorEvent = webgl+'contextcreationerror';
	var onError = function(event) { msg = event.statusMessage || msg; };
	try
	{
		canvas.addEventListener(errorEvent, onError, false);
		try { GLctx = canvas.getContext(webgl, attr) || canvas.getContext('experimental-'+webgl, attr); }
		finally { canvas.removeEventListener(errorEvent, onError, false); }
		if (!GLctx) throw 'Context failed';
	}
	catch (e) { abort('WEBGL', e + (msg ? ' (' + msg + ')' : "")); }

	// Enable all extensions except debugging, async operations and context losing
	for (var exts = GLctx.getSupportedExtensions()||[], i = 0, ext; i != exts.length;i++)
		if (!(ext = exts[i]).match(/debug|lose|parallel|async|moz_|webkit_/i))
			GLctx.getExtension(ext);

	return true;
})

WAJIC_LIB(GL, void, glActiveTexture, (GLenum texture),
{
	GLctx.activeTexture(texture);
})

WAJIC_LIB(GL, void, glAttachShader, (GLuint program, GLuint shader),
{
	GLctx.attachShader(GLprograms[program], GLshaders[shader]);
})

WAJIC_LIB(GL, void, glBindAttribLocation, (GLuint program, GLuint index, const GLchar *name),
{
	GLctx.bindAttribLocation(GLprograms[program], index, MStrGet(name));
})

WAJIC_LIB(GL, void, glBindBuffer, (GLenum target, GLuint buffer),
{
	GLctx.bindBuffer(target, buffer ? GLbuffers[buffer] : null);
})

WAJIC_LIB(GL, void, glBindFramebuffer, (GLenum target, GLuint framebuffer),
{
	GLctx.bindFramebuffer(target, framebuffer ? GLframebuffers[framebuffer] : null);
})

WAJIC_LIB(GL, void, glBindTexture, (GLenum target, GLuint texture),
{
	GLctx.bindTexture(target, texture ? GLtextures[texture] : null);
})

WAJIC_LIB(GL, void, glBlendFunc, (GLenum sfactor, GLenum dfactor),
{
	GLctx.blendFunc(sfactor, dfactor);
})

WAJIC_LIB(GL, void, glBlendFuncSeparate, (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha),
{
	GLctx.blendFuncSeparate(sfactorRGB, dfactorRGB, sfactorAlpha, dfactorAlpha);
})

WAJIC_LIB(GL, void, glBlendColor, (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha),
{
	GLctx.blendColor(red, green, blue, alpha);
})

WAJIC_LIB(GL, void, glBlendEquation, (GLenum mode),
{
	GLctx.blendEquation(mode);
})

WAJIC_LIB(GL, void, glBlendEquationSeparate, (GLenum modeRGB, GLenum modeAlpha),
{
	GLctx.blendEquationSeparate(modeRGB, modeAlpha);
})

WAJIC_LIB(GL, void, glBufferData, (GLenum target, GLsizeiptr size, const void *data, GLenum usage),
{
	if (!data) GLctx.bufferData(target, size, usage);
	else GLctx.bufferData(target, MU8.subarray(data, data+size), usage);
})

WAJIC_LIB(GL, void, glBufferSubData, (GLenum target, GLintptr offset, GLsizeiptr size, const void *data),
{
	GLctx.bufferSubData(target, offset, MU8.subarray(data, data+size));
})

WAJIC_LIB(GL, void, glClear, (GLbitfield mask),
{
	GLctx.clear(mask);
})

WAJIC_LIB(GL, void, glClearColor, (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha),
{
	GLctx.clearColor(red, green, blue, alpha);
})

WAJIC_LIB(GL, void, glColorMask, (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha),
{
	GLctx.colorMask(!!red, !!green, !!blue, !!alpha);
})

WAJIC_LIB(GL, void, glCompileShader, (GLuint shader),
{
	GLctx.compileShader(GLshaders[shader]);
})

WAJIC_LIB(GL, GLuint, glCreateProgram, (),
{
	var id = GLgetNewId(GLprograms);
	var program = GLctx.createProgram();
	program.name = id;
	GLprograms[id] = program;
	return id;
})

WAJIC_LIB(GL, GLuint, glCreateShader, (GLenum type),
{
	var id = GLgetNewId(GLshaders);
	GLshaders[id] = GLctx.createShader(type);
	return id;
})

WAJIC_LIB(GL, void, glDeleteBuffers, (GLsizei n, const GLuint *buffers),
{
	for (var i = 0; i < n; i++)
	{
		var id = MI32[(buffers>>2)+i];
		var buffer = GLbuffers[id];
		if (!buffer) continue; //GL spec: "glDeleteBuffers silently ignores 0's and names that do not correspond to existing buffer objects".
		GLctx.deleteBuffer(buffer);
		buffer.name = 0;
		GLbuffers[id] = null;
	}
})

WAJIC_LIB(GL, void, glDeleteFramebuffers, (GLsizei n, const GLuint *framebuffers),
{
	for (var i = 0; i < n; ++i)
	{
		var id = MI32[(framebuffers>>2)+i];
		var framebuffer = GLframebuffers[id];
		if (!framebuffer) continue; // GL spec: "glDeleteFramebuffers silently ignores 0s and names that do not correspond to existing framebuffer objects".
		GLctx.deleteFramebuffer(framebuffer);
		framebuffer.name = 0;
		GLframebuffers[id] = null;
	}
})

WAJIC_LIB(GL, void, glDeleteProgram, (GLuint program),
{
	if (!program) return;
	var program_obj = GLprograms[program];
	if (!program_obj) 
		// glDeleteProgram actually signals an error when deleting a nonexisting object, unlike some other GL delete functions.
		return GLrecordError(0x501); // GL_INVALID_VALUE

	GLctx.deleteProgram(program_obj);
	program_obj.name = 0;
	GLprograms[program] = null;
	GLprogramInfos[program] = null;
})

WAJIC_LIB(GL, void, glDeleteShader, (GLuint shader),
{
	if (!shader) return;
	var shader_obj = GLshaders[shader];
	if (!shader_obj)
		// glDeleteShader actually signals an error when deleting a nonexisting object, unlike some other GL delete functions.
		return GLrecordError(0x501); // GL_INVALID_VALUE

	GLctx.deleteShader(shader_obj);
	GLshaders[shader] = null;
})

WAJIC_LIB(GL, void, glDeleteTextures, (GLsizei n, const GLuint *textures),
{
	for (var i = 0; i < n; i++)
	{
		var id = MI32[(textures>>2)+i];
		var texture = GLtextures[id];
		if (!texture) continue; // GL spec: "glDeleteTextures silently ignores 0s and names that do not correspond to existing textures".
		GLctx.deleteTexture(texture);
		texture.name = 0;
		GLtextures[id] = null;
	}
})

WAJIC_LIB(GL, void, glDeleteRenderbuffers, (GLsizei n, const GLuint *renderbuffers),
{
	for (var i = 0; i < n; i++)
	{
		var id = MI32[(renderbuffers>>2)+i];
		var renderbuffer = GLrenderbuffers[id];
		if (!renderbuffer) continue; // GL spec: "glDeleteRenderbuffers silently ignores 0s and names that do not correspond to existing renderbuffer objects".
		GLctx.deleteRenderbuffer(renderbuffer);
		renderbuffer.name = 0;
		GLrenderbuffers[id] = null;
	}
})

WAJIC_LIB(GL, void, glDeleteVertexArrays, (GLsizei n, const GLuint *arrays),
{
	for (var i = 0; i < n; i++)
	{
		var id = MI32[(arrays>>2)+i];
		var vao = GLvaos[id];
		if (!vao) continue; // GL spec: "Unused names in arrays are silently ignored, as is the value zero.".
		GLctx.deleteVertexArray(vao);
		vao.name = 0;
		GLvaos[id] = null;
	}
})

WAJIC_LIB(GL, void, glDepthFunc, (GLenum func),
{
	GLctx.depthFunc(func);
})

WAJIC_LIB(GL, void, glDepthMask, (GLboolean flag),
{
	GLctx.depthMask(!!flag);
})

WAJIC_LIB(GL, void, glDetachShader, (GLuint program, GLuint shader),
{
	GLctx.detachShader(GLprograms[program], GLshaders[shader]);
})

WAJIC_LIB(GL, void, glDisable, (GLenum cap),
{
	GLctx.disable(cap);
})

WAJIC_LIB(GL, void, glDisableVertexAttribArray, (GLuint index),
{
	GLctx.disableVertexAttribArray(index);
})

WAJIC_LIB(GL, void, glDrawArrays, (GLenum mode, GLint first, GLsizei count),
{
	GLctx.drawArrays(mode, first, count);
})

WAJIC_LIB(GL, void, glDrawElements, (GLenum mode, GLsizei count, GLenum type, const GLvoid *indices),
{
	GLctx.drawElements(mode, count, type, indices);
})

WAJIC_LIB(GL, void, glEnable, (GLenum cap),
{
	GLctx.enable(cap);
})

WAJIC_LIB(GL, void, glEnableVertexAttribArray, (GLuint index),
{
	GLctx.enableVertexAttribArray(index);
})

WAJIC_LIB(GL, void, glFramebufferTexture2D, (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level),
{
	GLctx.framebufferTexture2D(target, attachment, textarget, GLtextures[texture], level);
})

WAJIC_LIB(GL, void, glGenBuffers, (GLsizei n, GLuint *buffers),
{
	GLgenObjects(n, buffers, 'createBuffer', GLbuffers);
})

WAJIC_LIB(GL, void, glGenFramebuffers, (GLsizei n, GLuint *framebuffers),
{
	GLgenObjects(n, framebuffers, 'createFramebuffer', GLframebuffers);
})

WAJIC_LIB(GL, void, glGenTextures, (GLsizei n, GLuint *textures),
{
	GLgenObjects(n, textures, 'createTexture', GLtextures);
})

WAJIC_LIB(GL, void, glGenRenderbuffers, (GLsizei n, GLuint *renderbuffers),
{
	GLgenObjects(n, renderbuffers, 'createRenderbuffer', GLrenderbuffers);
})

WAJIC_LIB(GL, void, glGenRenderbuffers, (GLsizei n, GLuint *renderbuffers),
{
	GLgenObjects(n, renderbuffers, 'createRenderbuffer', GLrenderbuffers);
})

WAJIC_LIB(GL, void, glGenVertexArrays, (GLsizei n, GLuint *arrays),
{
	GLgenObjects(n, arrays, 'createVertexArray', GLvaos);
})

WAJIC_LIB(GL, void, glGenerateMipmap, (GLenum target),
{
	GLctx.generateMipmap(target);
})

WAJIC_LIB(GL, void, glGetActiveUniform, (GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name),
{
	program = GLprograms[program];
	var info = GLctx.getActiveUniform(program, index);
	if (!info) return; // If an error occurs, nothing will be written to length, size, type and name.

	if (length) MI32[length>>2] = (bufSize > 0 && name ? MStrPut(info.name, name, bufSize) : 0);
	if (size) MI32[size>>2] = info.size;
	if (type) MI32[type>>2] = info.type;
})

WAJIC_LIB(GL, GLint, glGetAttribLocation, (GLuint program, const GLchar *name),
{
	program = GLprograms[program];
	name = MStrGet(name);
	return GLctx.getAttribLocation(program, name);
})

WAJIC_LIB(GL, GLenum, glGetError, (),
{
	if (GLlastError)
	{
		var e = GLlastError;
		GLlastError = 0;
		return e;
	}
	return GLctx.getError();
})

WAJIC_LIB(GL, void, glGetIntegerv, (GLenum pname, GLint *params),
{
	GLget(pname, params, 0);
})

WAJIC_LIB(GL, void, glGetBooleanv, (GLenum pname, GLboolean *params),
{
	GLget(pname, params, 4);
})

WAJIC_LIB(GL, void, glGetFloatv, (GLenum pname, GLfloat *params),
{
	GLget(pname, params, 2);
})

WAJIC_LIB(GL, void, glGetProgramInfoLog, (GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog),
{
	var log = GLctx.getProgramInfoLog(GLprograms[program]);
	if (log === null) log = '(unknown error)';
	if (length) MI32[length>>2] = (bufSize > 0 && infoLog ? MStrPut(log, infoLog, bufSize) : 0);
})

WAJIC_LIB(GL, void, glGetProgramiv, (GLuint program, GLenum pname, GLint *params),
{
	if (program >= GLcounter)
		return GLrecordError(0x501); // GL_INVALID_VALUE

	var ptable = GLprogramInfos[program];
	if (!ptable)
		return GLrecordError(0x502); //GL_INVALID_OPERATION

	var res;
	if (pname == 0x8B84) // GL_INFO_LOG_LENGTH
	{
		var log = GLctx.getProgramInfoLog(GLprograms[program]);
		if (log === null) log = '(unknown error)';
		res = log.length + 1;
	}
	else if (pname == 0x8B87) //GL_ACTIVE_UNIFORM_MAX_LENGTH
	{
		res = ptable[kMaxUniformLength];
	}
	else if (pname == 0x8B8A) //GL_ACTIVE_ATTRIBUTE_MAX_LENGTH
	{
		if (ptable[kMaxAttributeLength] == -1)
		{
			program = GLprograms[program];
			var numAttribs = GLctx.getProgramParameter(program, GLctx.ACTIVE_ATTRIBUTES);
			ptable[kMaxAttributeLength] = 0; // Spec says if there are no active attribs, 0 must be returned.
			for (var i = 0; i < numAttribs; ++i)
			{
				var activeAttrib = GLctx.getActiveAttrib(program, i);
				ptable[kMaxAttributeLength] = Math.max(ptable[kMaxAttributeLength], activeAttrib.name.length+1);
			}
		}
		res = ptable[kMaxAttributeLength];
	}
	else if (pname == 0x8A35) //GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH
	{
		if (ptable[kMaxUniformBlockNameLength] == -1)
		{
			program = GLprograms[program];
			var numBlocks = GLctx.getProgramParameter(program, GLctx.ACTIVE_UNIFORM_BLOCKS);
			ptable[kMaxUniformBlockNameLength] = 0;
			for (var i = 0; i < numBlocks; ++i)
			{
				var activeBlockName = GLctx.getActiveUniformBlockName(program, i);
				ptable[kMaxUniformBlockNameLength] = Math.max(ptable[kMaxUniformBlockNameLength], activeBlockName.length+1);
			}
		}
		res = ptable[kMaxUniformBlockNameLength];
	}
	else
	{
		res = GLctx.getProgramParameter(GLprograms[program], pname);
	}
	MI32[params>>2] = res;
})

WAJIC_LIB(GL, void, glGetShaderInfoLog, (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog),
{
	var log = GLctx.getShaderInfoLog(GLshaders[shader]);
	if (log === null) log = '(unknown error)';
	if (length) MI32[length>>2] = (bufSize > 0 && infoLog ? MStrPut(log, infoLog, bufSize) : 0);
})

WAJIC_LIB(GL, void, glGetShaderiv, (GLuint shader, GLenum pname, GLint *params),
{
	var res;
	if (pname == 0x8B84) // GL_INFO_LOG_LENGTH
	{
		var log = GLctx.getShaderInfoLog(GLshaders[shader]);
		if (log === null) log = '(unknown error)';
		res = log.length + 1;
	}
	else if (pname == 0x8B88) // GL_SHADER_SOURCE_LENGTH
	{
		var source = GLctx.getShaderSource(GLshaders[shader]);
		var sourceLength = (source === null || source.length == 0) ? 0 : source.length + 1;
		res = sourceLength;
	}
	else
	{
		res = GLctx.getShaderParameter(GLshaders[shader], pname);
	}
	MI32[params>>2] = res;
})

WAJIC_LIB(GL, void, glGetUniformfv, (GLuint program, GLint location, GLfloat *params),
{
	GLgetUniform(program, location, params, 2);
})

WAJIC_LIB(GL, void, glGetUniformiv, (GLuint program, GLint location, GLint *params),
{
	GLgetUniform(program, location, params, 0);
})

WAJIC_LIB(GL, GLint, glGetUniformLocation, (GLuint program, const GLchar *name),
{
	name = MStrGet(name);

	var arrayOffset = 0;
	if (name.indexOf(']', name.length-1) !== -1)
	{
		// If user passed an array accessor "[index]", parse the array index off the accessor.
		var ls = name.lastIndexOf('[');
		var arrayIndex = name.slice(ls+1, -1);
		if (arrayIndex.length > 0)
		{
			arrayOffset = parseInt(arrayIndex);
			if (arrayOffset < 0) return -1;
		}
		name = name.slice(0, ls);
	}

	var ptable = GLprogramInfos[program];
	if (!ptable) return -1;
	var utable = ptable[kUniforms];
	var uniformInfo = utable[name]; // returns pair [ dimension_of_uniform_array, uniform_location ]
	if (uniformInfo && arrayOffset < uniformInfo[0])
	{
		// Check if user asked for an out-of-bounds element, i.e. for 'vec4 colors[3];' user could ask for 'colors[10]' which should return -1.
		return uniformInfo[1] + arrayOffset;
	}
	return -1;
})

WAJIC_LIB(GL, void, glLineWidth, (GLfloat width),
{
	GLctx.lineWidth(width);
})

WAJIC_LIB(GL, void, glLinkProgram, (GLuint program),
{
	GLctx.linkProgram(GLprograms[program]);
	GLprogramInfos[program] = null; // uniforms no longer keep the same names after linking

	// Populate uniform table
	var p = GLprograms[program];
	var ptable = GLprogramInfos[program] =
	{
		[kUniforms]: {},
		[kMaxUniformLength]: 0, // This is eagerly computed below, since we already enumerate all uniforms anyway.
		[kMaxAttributeLength]: -1, // This is lazily computed and cached, computed when/if first asked, '-1' meaning not computed yet.
		[kMaxUniformBlockNameLength]: -1 // Lazily computed as well
	};
	var utable = ptable[kUniforms];

	// A program's uniform table maps the string name of an uniform to an integer location of that uniform.
	// The global GLuniforms map maps integer locations to WebGLUniformLocations.
	var numUniforms = GLctx.getProgramParameter(p, GLctx.ACTIVE_UNIFORMS);
	for (var i = 0; i < numUniforms; ++i)
	{
		var u = GLctx.getActiveUniform(p, i);

		var name = u.name;
		ptable[kMaxUniformLength] = Math.max(ptable[kMaxUniformLength], name.length+1);

		// Strip off any trailing array specifier we might have got, e.g. '[0]'.
		if (name.indexOf(']', name.length-1) !== -1)
		{
			var ls = name.lastIndexOf('[');
			name = name.slice(0, ls);
		}

		// Optimize memory usage slightly: If we have an array of uniforms, e.g. 'vec3 colors[3];', then
		// only store the string 'colors' in utable, and 'colors[0]', 'colors[1]' and 'colors[2]' will be parsed as 'colors'+i.
		// Note that for the GLuniforms table, we still need to fetch the all WebGLUniformLocations for all the indices.
		var loc = GLctx.getUniformLocation(p, name);
		if (loc != null)
		{
			var id = GLgetNewId(GLuniforms);
			utable[name] = [u.size, id];
			GLuniforms[id] = loc;

			for (var j = 1; j < u.size; ++j)
			{
				var n = name + '['+j+']';
				loc = GLctx.getUniformLocation(p, n);
				id = GLgetNewId(GLuniforms);

				GLuniforms[id] = loc;
			}
		}
	}
})

WAJIC_LIB(GL, void, glPixelStorei, (GLenum pname, GLint param),
{
	if (pname == 0xD05) GLpackAlignment = param; //GL_PACK_ALIGNMENT
	else if (pname == 0xcf5) GLunpackAlignment = param; //GL_UNPACK_ALIGNMENT
	GLctx.pixelStorei(pname, param);
})

WAJIC_LIB(GL, void, glReadPixels, (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid *pixels),
{
	var pixelData = GLgetTexPixelData(type, format, width, height, pixels, format);
	if (!pixelData) return GLrecordError(0x500); // GL_INVALID_ENUM
	GLctx.readPixels(x, y, width, height, format, type, pixelData);
})

WAJIC_LIB(GL, void, glScissor, (GLint x, GLint y, GLsizei width, GLsizei height),
{
	GLctx.scissor(x, y, width, height);
})

WAJIC_LIB(GL, void, glShaderSource, (GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length),
{
	for (var res = "", i = 0; i < count; ++i)
	{
		var len = (length ? MU32[(length>>2)+i] : -1);
		res += MStrGet(MU32[(string>>2)+i], (len < 0 ? "" : len));
	}
	GLctx.shaderSource(GLshaders[shader], res);
})

WAJIC_LIB(GL, void, glTexImage2D, (GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *pixels),
{
	var pixelData = null;
	if (pixels) pixelData = GLgetTexPixelData(type, format, width, height, pixels, internalFormat);
	GLctx.texImage2D(target, level, internalFormat, width, height, border, format, type, pixelData);
})

WAJIC_LIB(GL, void, glTexParameteri, (GLenum target, GLenum pname, GLint param),
{
	GLctx.texParameteri(target, pname, param);
})

WAJIC_LIB(GL, void, glTexSubImage2D, (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *pixels),
{
	var pixelData = null;
	if (pixels) pixelData = GLgetTexPixelData(type, format, width, height, pixels, 0);
	GLctx.texSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixelData);
})

WAJIC_LIB(GL, void, glUniform1f, (GLint location, GLfloat v0),
{
	GLctx.uniform1f(GLuniforms[location], v0);
})

WAJIC_LIB(GL, void, glUniform1i, (GLint location, GLint v0),
{
	GLctx.uniform1i(GLuniforms[location], v0);
})

WAJIC_LIB(GL, void, glUniform2f, (GLint location, GLfloat v0, GLfloat v1),
{
	GLctx.uniform2f(GLuniforms[location], v0, v1);
})

WAJIC_LIB(GL, void, glUniform2i, (GLint location, GLint v0, GLint v1),
{
	GLctx.glUniform2i(GLuniforms[location], v0, v1);
})

WAJIC_LIB(GL, void, glUniform3f, (GLint location, GLfloat v0, GLfloat v1, GLfloat v2),
{
	GLctx.uniform3f(GLuniforms[location], v0, v1, v2);
})

WAJIC_LIB(GL, void, glUniform3i, (GLint location, GLint v0, GLint v1, GLint v2),
{
	GLctx.glUniform3i(GLuniforms[location], v0, v1, v2);
})

WAJIC_LIB(GL, void, glUniform4f, (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3),
{
	GLctx.uniform4f(GLuniforms[location], v0, v1, v2, v3);
})

WAJIC_LIB(GL, void, glUniform4i, (GLint location, GLint v0, GLint v1, GLint v2, GLint v3),
{
	GLctx.glUniform4i(GLuniforms[location], v0, v1, v2, v3);
})

WAJIC_LIB(GL, void, glUniform1fv, (GLint location, GLsizei count, const GLfloat *value),
{
	value >>= 2;
	var view, heap = MF32;
	if (count <= GLMINI_TEMP_BUFFER_SIZE)
	{
		// avoid allocation when uploading few enough uniforms
		view = GLminiTempFloatBuffers[count-1];
		for (var i = 0; i != count; i++)
			view[i] = heap[value+i];
	}
	else
	{
		view = heap.subarray(value, value + count);
	}
	GLctx.uniform1fv(GLuniforms[location], view);
})

WAJIC_LIB(GL, void, glUniform1iv, (GLint location, GLsizei count, const GLint *value),
{
	value >>= 2;
	var view, heap = MI32;
	if (count <= GLMINI_TEMP_BUFFER_SIZE)
	{
		// avoid allocation when uploading few enough uniforms
		var view = GLminiTempIntBuffers[count-1];
		for (var i = 0; i != count; i++)
			view[i] = heap[value+i];
	}
	else
	{
		view = heap.subarray(value, value + count);
	}
	GLctx.uniform1fv(GLuniforms[location], view);
})

WAJIC_LIB(GL, void, glUniform2fv, (GLint location, GLsizei count, const GLfloat *value),
{
	count *= 2;
	value >>= 2;
	var view, heap = MF32;
	if (count <= GLMINI_TEMP_BUFFER_SIZE)
	{
		// avoid allocation when uploading few enough uniforms
		view = GLminiTempFloatBuffers[count-1];
		for (var i = 0; i != count; i += 2)
		{
			view[i  ] = heap[value+i  ];
			view[i+1] = heap[value+i+1];
		}
	}
	else
	{
		view = heap.subarray(value, value + count);
	}
	GLctx.uniform2fv(GLuniforms[location], view);
})

WAJIC_LIB(GL, void, glUniform2iv, (GLint location, GLsizei count, const GLint *value),
{
	count *= 2;
	value >>= 2;
	var view, heap = MI32;
	if (count <= GLMINI_TEMP_BUFFER_SIZE)
	{
		// avoid allocation when uploading few enough uniforms
		view = GLminiTempIntBuffers[count-1];
		for (var i = 0; i != count; i += 2)
		{
			view[i  ] = heap[value+i  ];
			view[i+1] = heap[value+i+1];
		}
	}
	else
	{
		view = heap.subarray(value, value + count);
	}
	GLctx.uniform2iv(GLuniforms[location], view);
})

WAJIC_LIB(GL, void, glUniform3fv, (GLint location, GLsizei count, const GLfloat *value),
{
	count *= 3;
	value >>= 2;
	var view, heap = MF32;
	if (count <= GLMINI_TEMP_BUFFER_SIZE)
	{
		// avoid allocation when uploading few enough uniforms
		view = GLminiTempFloatBuffers[count-1];
		for (var i = 0; i != count; i += 3)
		{
			view[i  ] = heap[value+i  ];
			view[i+1] = heap[value+i+1];
			view[i+2] = heap[value+i+2];
		}
	}
	else
	{
		view = heap.subarray(value, value + count);
	}
	GLctx.uniform3fv(GLuniforms[location], view);
})

WAJIC_LIB(GL, void, glUniform3iv, (GLint location, GLsizei count, const GLint *value),
{
	count *= 3;
	value >>= 2;
	var view, heap = MI32;
	if (count <= GLMINI_TEMP_BUFFER_SIZE)
	{
		// avoid allocation when uploading few enough uniforms
		view = GLminiTempIntBuffers[count-1];
		for (var i = 0; i != count; i += 3)
		{
			view[i  ] = heap[value+i  ];
			view[i+1] = heap[value+i+1];
			view[i+2] = heap[value+i+2];
		}
	}
	else
	{
		view = heap.subarray(value, value + count);
	}
	GLctx.uniform3iv(GLuniforms[location], view);
})

WAJIC_LIB(GL, void, glUniform4fv, (GLint location, GLsizei count, const GLfloat *value),
{
	count *= 4;
	value >>= 2;
	var view, heap = MF32;
	if (count <= GLMINI_TEMP_BUFFER_SIZE)
	{
		// avoid allocation when uploading few enough uniforms
		view = GLminiTempFloatBuffers[count-1];
		for (var i = 0; i != count; i += 4)
		{
			view[i  ] = heap[value+i  ];
			view[i+1] = heap[value+i+1];
			view[i+2] = heap[value+i+2];
			view[i+3] = heap[value+i+3];
		}
	}
	else
	{
		view = heap.subarray(value, value + count);
	}
	GLctx.uniform4fv(GLuniforms[location], view);
})

WAJIC_LIB(GL, void, glUniform4iv, (GLint location, GLsizei count, const GLint *value),
{
	count *= 4;
	value >>= 2;
	var view, heap = MI32;
	if (count <= GLMINI_TEMP_BUFFER_SIZE)
	{
		// avoid allocation when uploading few enough uniforms
		view = GLminiTempIntBuffers[count-1];
		for (var i = 0; i != count; i += 4)
		{
			view[i  ] = heap[value+i  ];
			view[i+1] = heap[value+i+1];
			view[i+2] = heap[value+i+2];
			view[i+3] = heap[value+i+3];
		}
	}
	else
	{
		view = heap.subarray(value, value + count);
	}
	GLctx.uniform4fv(GLuniforms[location], view);
})

WAJIC_LIB(GL, void, glUniformMatrix2fv, (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value),
{
	count <<= 2;
	value >>= 2;
	var view, heap = MF32;
	if (count <= GLMINI_TEMP_BUFFER_SIZE)
	{
		// avoid allocation when uploading few enough uniforms
		view = GLminiTempFloatBuffers[count-1];
		for (var i = 0; i != count; i += 4)
		{
			view[i  ] = heap[value+i  ];
			view[i+1] = heap[value+i+1];
			view[i+2] = heap[value+i+2];
			view[i+3] = heap[value+i+3];
		}
	}
	else
	{
		view = heap.subarray(value, value + count);
	}
	GLctx.uniformMatrix2fv(GLuniforms[location], !!transpose, view);
})

WAJIC_LIB(GL, void, glUniformMatrix3fv, (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value),
{
	count *= 9;
	value >>= 2;
	var view, heap = MF32;
	if (count <= GLMINI_TEMP_BUFFER_SIZE)
	{
		// avoid allocation when uploading few enough uniforms
		view = GLminiTempFloatBuffers[count-1];
		for (var i = 0; i != count; i += 3)
		{
			view[i  ] = heap[value+i  ];
			view[i+1] = heap[value+i+1];
			view[i+2] = heap[value+i+2];
		}
	}
	else
	{
		view = heap.subarray(value, value + count);
	}
	GLctx.uniformMatrix3fv(GLuniforms[location], !!transpose, view);
})

WAJIC_LIB(GL, void, glUniformMatrix4fv, (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value),
{
	count <<= 4;
	value >>= 2;
	var view, heap = MF32;
	if (count <= GLMINI_TEMP_BUFFER_SIZE)
	{
		// avoid allocation when uploading few enough uniforms
		view = GLminiTempFloatBuffers[count-1];
		for (var i = 0; i != count; i += 4)
		{
			view[i  ] = heap[value+i  ];
			view[i+1] = heap[value+i+1];
			view[i+2] = heap[value+i+2];
			view[i+3] = heap[value+i+3];
		}
	}
	else
	{
		view = heap.subarray(value, value + count);
	}
	GLctx.uniformMatrix4fv(GLuniforms[location], !!transpose, view);
})

WAJIC_LIB(GL, void, glVertexAttrib1f, (GLuint index, GLfloat x),
{
	GLctx.vertexAttrib1f(index, x);
})

WAJIC_LIB(GL, void, glVertexAttrib1fv, (GLuint index, const GLfloat *v),
{
	GLctx.vertexAttrib1f(index, MF32[v>>2]);
})

WAJIC_LIB(GL, void, glVertexAttrib2f, (GLuint index, GLfloat x, GLfloat y),
{
	GLctx.vertexAttrib2f(index, x, y);
})

WAJIC_LIB(GL, void, glVertexAttrib2fv, (GLuint index, const GLfloat *v),
{
	v >>= 2;
	GLctx.vertexAttrib2f(index, MF32[v], MF32[v+1]);
})

WAJIC_LIB(GL, void, glVertexAttrib3f, (GLuint index, GLfloat x, GLfloat y, GLfloat z),
{
	GLctx.vertexAttrib3f(index, x, y, z);
})

WAJIC_LIB(GL, void, glVertexAttrib3fv, (GLuint index, const GLfloat *v),
{
	v >>= 2;
	GLctx.vertexAttrib3f(index, MF32[v], MF32[v+1], MF32[v+2]);
})

WAJIC_LIB(GL, void, glVertexAttrib4f, (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w),
{
	GLctx.vertexAttrib4f(index, x, y, z, w);
})

WAJIC_LIB(GL, void, glVertexAttrib4fv, (GLuint index, const GLfloat *v),
{
	v >>= 2;
	GLctx.vertexAttrib4f(index, MF32[v], MF32[v+1], MF32[v+2], MF32[v+3]);
})

WAJIC_LIB(GL, void, glVertexAttribPointer, (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer),
{
	GLctx.vertexAttribPointer(index, size, type, !!normalized, stride, pointer);
})

WAJIC_LIB(GL, void, glUseProgram, (GLuint program),
{
	GLctx.useProgram(program ? GLprograms[program] : null);
})

WAJIC_LIB(GL, void, glViewport, (GLint x, GLint y, GLsizei width, GLsizei height),
{
	GLctx.viewport(x, y, width, height);
})

WAJIC_LIB(GL, const GLubyte*, glGetString, (GLenum name),
{
	if (GLstringCache[name]) return GLstringCache[name];
	var ret = "";
	switch(name)
	{
		case 0x1F03: //GL_EXTENSIONS
			 // getSupportedExtensions() can return null if context is lost, so coerce to empty array.
			var exts = GLctx.getSupportedExtensions() || [];
			ret = exts.concat(exts.map(e=>"GL_"+e)).join(' ');
			break;
		case 0x1F00: //GL_VENDOR
		case 0x1F01: //GL_RENDERER
		case 0x9245: //UNMASKED_VENDOR_WEBGL
		case 0x9246: //UNMASKED_RENDERER_WEBGL
			ret = GLctx.getParameter(name)||"";
			if (!ret) GLrecordError(0x500); //GL_INVALID_ENUM
			break;
		case 0x1F02: //GL_VERSION
			ret = 'OpenGL ES 2.0 (' + GLctx.getParameter(0x1F02) + ')'; //GL_VERSION
			break;
		case 0x8B8C: //GL_SHADING_LANGUAGE_VERSION
			ret = GLctx.getParameter(0x8B8C); //GL_SHADING_LANGUAGE_VERSION
			// extract the version number 'N.M' from the string 'WebGL GLSL ES N.M ...'
			var ver_num = ret.match("^WebGL GLSL ES ([0-9]\\.[0-9][0-9]?)(?:$| .*)");
			if (ver_num !== null)
			{
				if (ver_num[1].length == 3) ver_num[1] = ver_num[1] + '0'; // ensure minor version has 2 digits
				ret = 'OpenGL ES GLSL ES ' + ver_num[1] + ' (' + glslVersion + ')';
			}
			break;
		default:
			GLrecordError(0x500); //GL_INVALID_ENUM
	}
	return GLstringCache[name] = MStrPut(ret); //will malloc the needed memory
})

WAJIC_LIB(GL, void, glBindRenderbuffer, (GLenum target, GLuint renderbuffer),
{
	GLctx.bindRenderbuffer(target, GLrenderbuffers[renderbuffer]);
})

WAJIC_LIB(GL, void, glRenderbufferStorage, (GLenum target, GLenum internalformat, GLsizei width, GLsizei height),
{
	GLctx.renderbufferStorage(target, internalformat, width, height);
})

WAJIC_LIB(GL, void, glCompressedTexImage2D, (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *data),
{
	GLctx.compressedTexImage2D(target, level, internalformat, width, height, border, (data ? MU8.subarray(data, data + imageSize) : null));
})

WAJIC_LIB(GL, void, glStencilMask, (GLuint mask),
{
	GLctx.stencilMask(mask);
})

WAJIC_LIB(GL, void, glClearDepthf, (GLfloat d),
{
	GLctx.clearDepth(d);
})

WAJIC_LIB(GL, void, glClearStencil, (GLint s),
{
	GLctx.clearStencil(s);
})

WAJIC_LIB(GL, void, glStencilFuncSeparate, (GLenum face, GLenum func, GLint ref, GLuint mask),
{
	GLctx.stencilFuncSeparate(face, func, ref, mask);
})

WAJIC_LIB(GL, void, glStencilOpSeparate, (GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass),
{
	GLctx.stencilOpSeparate(face, sfail, dpfail, dppass);
})

WAJIC_LIB(GL, void, glCullFace, (GLenum mode),
{
	GLctx.cullFace(mode);
})

WAJIC_LIB(GL, void, glFrontFace, (GLenum mode),
{
	GLctx.frontFace(mode);
})

WAJIC_LIB(GL, void, glPolygonOffset, (GLfloat factor, GLfloat units),
{
	GLctx.polygonOffset(factor, units);
})

WAJIC_LIB(GL, void, glStencilFunc, (GLenum func, GLint ref, GLuint mask),
{
	GLctx.stencilFunc(func, ref, mask);
})

WAJIC_LIB(GL, void, glStencilOp, (GLenum fail, GLenum zfail, GLenum zpass),
{
	GLctx.stencilOp(fail, zfail, zpass);
})

WAJIC_LIB(GL, void, glBindVertexArray, (GLuint array),
{
	GLctx.bindVertexArray(GLvaos[array]);
})

WAJIC_LIB(GL, GLenum, glCheckFramebufferStatus, (GLenum target),
{
	return GLctx.checkFramebufferStatus(target);
})

WAJIC_LIB(GL, void, glClearDepth, (GLclampd depth),
{
	GLctx.clearDepth(depth);
})

WAJIC_LIB(GL, void, glCompressedTexSubImage2D, (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const GLvoid *data),
{
	GLctx.compressedTexSubImage2D(target, level, xoffset, yoffset, width, height, format, data ? MU8.subarray((data),(data+imageSize)) : null);
})

WAJIC_LIB(GL, void, glCopyTexImage2D, (GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border),
{
	GLctx.copyTexImage2D(target, level, internalformat, x, y, width, height, border);
})

WAJIC_LIB(GL, void, glCopyTexSubImage2D, (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height),
{
	GLctx.copyTexSubImage2D(target, level, xoffset, yoffset, x, y, width, height);
})

WAJIC_LIB(GL, void, glDepthRange, (GLclampd near_val, GLclampd far_val),
{
	GLctx.depthRange(near_val, far_val);
})

WAJIC_LIB(GL, void, glDepthRangef, (GLfloat n, GLfloat f),
{
	GLctx.depthRange(n, f);
})

WAJIC_LIB(GL, void, glDrawArraysInstanced, (GLenum mode, GLint first, GLsizei count, GLsizei instancecount),
{
	GLctx.drawArraysInstanced(mode, first, count, instancecount);
})

WAJIC_LIB(GL, void, glDrawArraysInstancedARB, (GLenum mode, GLint first, GLsizei count, GLsizei primcount),
{
	GLctx.drawArraysInstanced(mode, first, count, primcount);
})

WAJIC_LIB(GL, void, glDrawArraysInstancedEXT, (GLenum mode, GLint start, GLsizei count, GLsizei primcount),
{
	GLctx.drawArraysInstanced(mode, first, count, primcount);
})

WAJIC_LIB(GL, void, glDrawBuffers, (GLsizei n, const GLenum *bufs),
{
	var arr = GLFixedLengthArrays[n];
	if (!arr) arr = GLFixedLengthArrays[n] = new Array(n);
	for (var i = 0; i < n; i++)
		arr[i] = MI32[(bufs>>2)+i];
	GLctx.drawBuffers(arr);
})

WAJIC_LIB(GL, void, glDrawElementsInstanced, (GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount),
{
	GLctx.drawElementsInstanced(mode, count, type, indices, instancecount);
})

WAJIC_LIB(GL, void, glDrawElementsInstancedARB, (GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei primcount),
{
	GLctx.drawElementsInstanced(mode, count, type, indices, primcount);
})

WAJIC_LIB(GL, void, glDrawElementsInstancedEXT, (GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei primcount),
{
	GLctx.drawElementsInstanced(mode, count, type, indices, primcount);
})

WAJIC_LIB(GL, void, glFinish, (),
{
	GLctx.finish();
})

WAJIC_LIB(GL, void, glFlush, (),
{
	GLctx.flush();
})

WAJIC_LIB(GL, void, glFramebufferRenderbuffer, (GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer),
{
	GLctx.framebufferRenderbuffer(target, attachment, renderbuffertarget, GLrenderbuffers[renderbuffer]);
})

WAJIC_LIB(GL, void, glGetActiveAttrib, (GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name),
{
	program = GLprograms[program];
	var info = GLctx.getActiveAttrib(program, index);
	if (!info) return; // If an error occurs, nothing will be written to length, size and type and name.

	if (length) MI32[length>>2] = (bufSize > 0 && name ? MStrPut(info.name, name, bufSize) : 0);
	if (size) MI32[size>>2] = info.size;
	if (type) MI32[type>>2] = info.type;
})

WAJIC_LIB(GL, void, glGetAttachedShaders, (GLuint program, GLsizei maxCount, GLsizei *count, GLuint *shaders),
{
	var result = GLctx.getAttachedShaders(GLprograms[program]);
	var len = result.length;
	if (len > maxCount) len = maxCount;
	MI32[count>>2] = len;
	for (var i = 0; i < len; ++i)
	{
		var id = GLshaders.indexOf(result[i]);
		MI32[(shaders>>2)+i] = id;
	}
})

WAJIC_LIB(GL, void, glGetBufferParameteriv, (GLenum target, GLenum pname, GLint *params),
{
	MI32[params>>2] = GLctx.getBufferParameter(target, pname);
})

WAJIC_LIB(GL, void, glGetFramebufferAttachmentParameteriv, (GLenum target, GLenum attachment, GLenum pname, GLint *params),
{
	var result = GLctx.getFramebufferAttachmentParameter(target, attachment, pname);
	MI32[params>>2] = ((result instanceof WebGLRenderbuffer || result instanceof WebGLTexture) ? (result.name|0) : result);
})

WAJIC_LIB(GL, void, glGetRenderbufferParameteriv, (GLenum target, GLenum pname, GLint *params),
{
	MI32[params>>2] = GLctx.getRenderbufferParameter(target, pname);
})

WAJIC_LIB(GL, void, glGetShaderPrecisionFormat, (GLenum shadertype, GLenum precisiontype, GLint *range, GLint *precision),
{
	var result = GLctx.getShaderPrecisionFormat(shaderType, precisionType);
	if (!result) return GLrecordError(0x500); // GL_INVALID_ENUM
	MI32[ range   >>2] = result.rangeMin;
	MI32[(range+4)>>2] = result.rangeMax;
	MI32[precision>>2] = result.precision;
})

WAJIC_LIB(GL, void, glGetShaderSource, (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *source),
{
	var result = GLctx.getShaderSource(GLshaders[shader]);
	if (!result) return GLrecordError(0x501); // GL_INVALID_VALUE, if an error occurs, nothing will be written to length or source.
	if (length) MI32[length>>2] = (bufSize > 0 && source ? MStrPut(result, source, bufSize) : 0);
})

WAJIC_LIB(GL, void, glGetTexParameterfv, (GLenum target, GLenum pname, GLfloat *params),
{
	MF32[params>>2] = GLctx.getTexParameter(target, pname);
})

WAJIC_LIB(GL, void, glGetTexParameteriv, (GLenum target, GLenum pname, GLint *params),
{
	MI32[params>>2] = GLctx.getTexParameter(target, pname);
})

WAJIC_LIB(GL, void, glGetVertexAttribPointerv, (GLuint index, GLenum pname, void **pointer),
{
	MI32[pointer>>2] = GLctx.getVertexAttribOffset(index, pname);
})

WAJIC_LIB(GL, void, glGetVertexAttribfv, (GLuint index, GLenum pname, GLfloat *params),
{
	// N.B. This function may only be called if the vertex attribute was specified using thefunction glVertexAttrib*f(),
	// otherwise the results are undefined. (GLES3 spec 6.1.12)
	GLgetVertexAttrib(index, pname, params, 2);
})

WAJIC_LIB(GL, void, glGetVertexAttribiv, (GLuint index, GLenum pname, GLint *params),
{
	// N.B. This function may only be called if the vertex attribute was specified using thefunction glVertexAttrib*f(),
	// otherwise the results are undefined. (GLES3 spec 6.1.12)
	GLgetVertexAttrib(index, pname, params, 0);
})

WAJIC_LIB(GL, void, glHint, (GLenum target, GLenum mode),
{
	GLctx.hint(target, mode);
})

WAJIC_LIB(GL, GLboolean, glIsBuffer, (GLuint buffer),
{
	buffer = GLbuffers[buffer];
	return (buffer ? GLctx.isBuffer(buffer) : 0);
})

WAJIC_LIB(GL, GLboolean, glIsEnabled, (GLenum cap),
{
	return GLctx.isEnabled(cap);
})

WAJIC_LIB(GL, GLboolean, glIsFramebuffer, (GLuint framebuffer),
function glIsFramebuffer(framebuffer)
{
	framebuffer = GLframebuffers[framebuffer];
	return (framebuffer ? GLctx.isFramebuffer(framebuffer) : 0);
})

WAJIC_LIB(GL, GLboolean, glIsProgram, (GLuint program),
{
	program = GLprograms[program];
	return (program ? GLctx.isProgram(program) : 0);
})

WAJIC_LIB(GL, GLboolean, glIsRenderbuffer, (GLuint renderbuffer),
{
	renderbuffer = GLrenderbuffers[renderbuffer];
	return (renderbuffer ? GLctx.isRenderbuffer(renderbuffer) : 0);
})

WAJIC_LIB(GL, GLboolean, glIsShader, (GLuint shader),
{
	shader = GLshaders[shader];
	return (shader ? GLctx.isShader(shader) : 0);
})

WAJIC_LIB(GL, GLboolean, glIsTexture, (GLuint texture),
{
	texture = GLtextures[texture];
	return (texture ? GLctx.isTexture(texture) : 0);
})

WAJIC_LIB(GL, GLboolean, glIsVertexArray, (GLuint array),
{
	array = GLvaos[array];
	return (array ? GLctx.isVertexArray(array) : 0);
})

WAJIC_LIB(GL, void, glReleaseShaderCompiler, (),
{
	// NOP (as allowed by GLES 2.0 spec)
})

WAJIC_LIB(GL, void, glSampleCoverage, (GLfloat value, GLboolean invert),
{
	GLctx.sampleCoverage(value, !!invert);
})

WAJIC_LIB(GL, void, glShaderBinary, (GLsizei count, const GLuint *shaders, GLenum binaryformat, const void *binary, GLsizei length),
{
	GLrecordError(0x500); //GL_INVALID_ENUM
})

WAJIC_LIB(GL, void, glStencilMaskSeparate, (GLenum face, GLuint mask),
{
	GLctx.stencilMaskSeparate(face, mask);
})

WAJIC_LIB(GL, void, glTexParameterf, (GLenum target, GLenum pname, GLfloat param),
{
	GLctx.texParameterf(target, pname, param);
})

WAJIC_LIB(GL, void, glTexParameterfv, (GLenum target, GLenum pname, const GLfloat *params),
{
	GLctx.texParameterf(target, pname, MF32[params>>2]);
})

WAJIC_LIB(GL, void, glTexParameteriv, (GLenum target, GLenum pname, const GLint *params),
{
	GLctx.texParameteri(target, pname, MI32[params>>2]);
})

WAJIC_LIB(GL, void, glValidateProgram, (GLuint program),
{
	GLctx.validateProgram(GLprograms[program]);
})

WAJIC_LIB(GL, void, glVertexAttribDivisor, (GLuint index, GLuint divisor),
{
	GLctx.vertexAttribDivisor(index, divisor);
})

WAJIC_LIB(GL, void, glVertexAttribDivisorARB, (GLuint index, GLuint divisor),
{
	GLctx.vertexAttribDivisor(index, divisor);
})
