/*
  WAjicUp - WebAssembly JavaScript Interface Creator Utility Program
  Copyright (C) 2020-2021 Bernhard Schelling

  Uses Terser JavaScript compressor (https://github.com/terser/terser)
  Terser is based on UglifyJS (https://github.com/mishoo/UglifyJS2)
  UglifyJS Copyright 2012-2018 (c) Mihai Bazon <mihai.bazon@gmail.com>
  UglifyJS parser is based on parse-js (http://marijn.haverbeke.nl/parse-js/).
  License for Terser can be found at the bottom of this file.

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

'use strict';

var terser, verbose = false;

var VERBOSE = function(msg)
{
	if (verbose) console.log(msg);
};

var WARN = function(msg)
{
	console.warn('[WARNING] ' + msg);
};

function ABORT(msg, e, code)
{
	if (e && typeof e == 'string') msg += "\n" + e;
	if (e && typeof e == 'object') msg += "\n" + (e.name||'') + (e.message ? ' - ' + e.message : '') + (e.line ? ' - Line: ' + e.line : '') + (e.col ? ' - Col: ' + e.col : '');
	if (code && e && e.line)
	{
		msg += "\n";
		var lines = code.split(/\n/), errcol = e.col||0;;
		for (var i = (e.line < 4 ? 1 : e.line - 3), iMax = (e.line + 3); i != iMax && i <= lines.length; i++)
		{
			var line = lines[i-1], col = (i == e.line ? Math.max(0, Math.min(line.length - 80, errcol - 40)) : 0);
			msg += "\n" + ('     '+i).slice(-5) + ': ' + (col ? '...' : '') + line.substr(col, 80) + (line.length - col > 80 ? '...' : '');
			if (i == e.line && e.col) msg += "\n" + ' '.repeat(errcol + (col ? 10 - col : 7)) + '^';
		}
	}
	if (typeof process !== 'object') throw msg; //throw if not CLI with node
	console.error('');
	console.error('[ERROR]');
	console.error(msg)
	console.error('');
	console.error('aborting');
	console.error('');
	throw process.exit(1);
}

// Execute CLI if running with node
if (typeof process === 'object') (function()
{
	var args = process.argv.slice(2);

	function ArgErr(err)
	{
		console.error('');
		console.error('WAjicUp - WebAssembly JavaScript Interface Creator Utility Program');
		console.error('');
		console.error('Error:');
		console.error(err);
		console.error('');
		console.error('For help, run: ' + process.argv[0] + ' ' + process.argv[1] + ' -h');
		console.error('');
		throw process.exit(1);
	}

	function ShowHelp()
	{
		console.error('');
		console.error('WAjicUp - WebAssembly JavaScript Interface Creator Utility Program');
		console.error('');
		console.error('Usage wajicup.js [<switches>...] <input_file> [<output_files>...]');
		console.error('');
		console.error('<input_file> must be an unprocessed .wasm file or c source file(s)');
		console.error('');
		console.error('<output_files> can be up to 3 files of different types');
		console.error('  .wasm: Minified/reduced wasm module');
		console.error('  .js:   JavaScript loader with interface embedded');
		console.error('  .html: HTML frontend');
		console.error('');
		console.error('  Possible file combinations:');
		console.error('   [WASM]             Minify functions inside WASM');
		console.error('   [WASM] [JS]        Move functions from WASM to JS');
		console.error('   [WASM] [JS] [HTML] Move functions from WASM to JS and generate HTML');
		console.error('   [WASM] [HTML]      Minify in WASM and create HTML with embedded loader');
		console.error('   [JS]               Embed and merge WASM into JS');
		console.error('   [JS] [HTML]        Embed WASM into JS and generate HTML');
		console.error('   [HTML]             Embed WASM into single file HTML');
		console.error('');
		console.error('<switches>');
		console.error('  -no_minify:   Don\'t minify JavaScript code');
		console.error('  -no_log:      Remove all output logging');
		console.error('  -streaming:   Enable WASM streaming (needs web server support, modern browser)');
		console.error('  -rle:         Use RLE compression when embedding the WASM file');
		console.error('  -loadbar:     Add a loading progress bar to the generated HTML');
		console.error('  -node:        Output JavaScript that runs in Node.js (CLI)');
		console.error('  -embed N P:   Embed data file at path P with name N');
		console.error('  -template F:  Use F as the template for the generated HTML file');
		console.error('  -stacksize N: Set the stack size (defaults to 64kb)');
		console.error('  -gzipreport:  Report the output size after gzip compression');
		console.error('  -v:           Be verbose about processed functions');
		console.error('  -h:           Show this help');
		console.error('');
		throw process.exit(0);
	}

	var fs = require('fs'), saveCount = 0, saveTotal = 0, gzipTotal = 0, gzipReport = false;

	function Load(path)
	{
		if (!path) return ABORT('Missing file path argument');
		try { var buf = fs.readFileSync(path); } catch (e) { return ABORT('Failed to load file: ' + path, e); }
		console.log('  [LOADED] ' + path + ' (' + buf.length + ' bytes)');
		return new Uint8Array(buf);
	}

	function GZipReport(buf)
	{
		var gzip = require('zlib').gzipSync(buf).length;
		gzipTotal += gzip;
		return ' (' + gzip + ' gzipped)';
	}

	function Save(path, buf)
	{
		try { fs.writeFileSync(path, buf); } catch (e) { return ABORT('Failed to save file: ' + path, e); }
		saveCount++;
		saveTotal += buf.length;
		console.log('  [SAVED] ' + path + ' (' + buf.length + ' bytes)' + (gzipReport ? GZipReport(buf) : ''));
	}

	function PathRelatedTo(srcPath, trgPath, isDirectory)
	{
		var path = require('path');
		var dir = path.relative(path.dirname(srcPath + (isDirectory ? '/X' : '')), path.dirname(trgPath + (isDirectory ? '/X' : '')));
		return (dir ? dir.replace(/\\/g, '/') + '/' : '') + (isDirectory ? (dir ? '' : './') : (path.basename(trgPath)));
	}

	function AddEmbed(name, path)
	{
		try { fs.readdirSync(path).forEach(f => p.embeds[name+f] = Load(path+f)); }
		catch (e) { p.embeds[name] = Load(path); }
	}

	var p = { minify: true, log: true, embeds: {} }, inBytes, cfiles = [], cc = '', ld = '', outWasmPath, outJsPath, outHtmlPath, outObjPath, outBcPath;
	for (var i = 0; i != args.length;)
	{
		var arg = args[i++];
		if (arg.match(/^-?\/?(help|h|\?)$/i))  { return ShowHelp(); }
		if (arg.match(/^-?\/?no_?-?minify$/i)) { p.minify    = false; continue; }
		if (arg.match(/^-?\/?no_?-?log$/i))    { p.log       = false; continue; }
		if (arg.match(/^-?\/?streaming$/i))    { p.streaming = true;  continue; }
		if (arg.match(/^-?\/?rle$/i))          { p.rle       = true;  continue; }
		if (arg.match(/^-?\/?loadbar$/i))      { p.loadbar   = true;  continue; }
		if (arg.match(/^-?\/?node$/i))         { p.node      = true;  continue; }
		if (arg.match(/^-?\/?gzipreport$/i))   { gzipReport  = true;  continue; }
		if (arg.match(/^-?\/?(v|verbose)$/i))  { verbose     = true;  continue; }
		if (arg.match(/^-?\/?args$/i))         { p.args      = ['W']; continue; }
		if (arg.match(/^-?\/?arg$/i))          { (p.args||(p.args=[])).push(args[i++]); continue; }
		if (arg.match(/^-?\/?embed$/i))        { AddEmbed(args[i], args[i+1]); i += 2; continue; }
		if (arg.match(/^-?\/?template$/i))     { p.template = Load(args[i++]); continue; }
		if (arg.match(/^-?\/?stacksize$/i))    { p.stacksize = (args[i++]|0); continue; }
		if (arg.match(/^-?\/?cc$/i))           { cc += ' '+args[i++]; continue; }
		if (arg.match(/^-?\/?ld$/i))           { ld += ' '+args[i++]; continue; }
		if (arg.match(/^-/)) return ArgErr('Invalid argument: ' + arg);

		var path = arg.match(/^.*\.(wasm|js|html|c|cpp|cc|cxx|o|bc?)$/i), ext = (path && path[1][0].toUpperCase());
		if (ext == 'C')
		{
			cfiles.push(arg);
		}
		else if (!inBytes && cfiles.length == 0 && !ld)
		{
			if (ext == 'W' || ext == 'J') inBytes = Load(arg);
			else return ArgErr('Invalid input file: ' + arg + "\n" + 'Must be a file ending with .wasm');
		}
		else
		{
			if      (ext == 'W') { if (!outWasmPath) outWasmPath = arg; else return ArgErr('Invalid output file: ' + arg + "\n" + 'Cannot output multiple .wasm files'); }
			else if (ext == 'J') { if (!outJsPath  ) outJsPath   = arg; else return ArgErr('Invalid output file: ' + arg + "\n" + 'Cannot output multiple .js files');   }
			else if (ext == 'H') { if (!outHtmlPath) outHtmlPath = arg; else return ArgErr('Invalid output file: ' + arg + "\n" + 'Cannot output multiple .html files'); }
			else if (ext == 'O') { if (!outObjPath ) outObjPath  = arg; else return ArgErr('Invalid output file: ' + arg + "\n" + 'Cannot output multiple .o files');    }
			else if (ext == 'B') { if (!outBcPath  ) outBcPath   = arg; else return ArgErr('Invalid output file: ' + arg + "\n" + 'Cannot output multiple .bc files');    }
			else return ArgErr('Invalid output file: ' + arg + "\n" + 'Must be a file ending with .wasm/.js/.html');
		}
	}

	// Validate options
	if (!inBytes && !cfiles.length && !ld) return ArgErr('Missing input file and output file(s)');
	if (!outWasmPath && !outJsPath && !outHtmlPath && !outObjPath && !outBcPath) return ArgErr('Missing output file(s)');
	if (outObjPath || outBcPath)
	{
		if (outObjPath && outBcPath) return ArgErr('Unable to output both .o and .bc files');
		if (outObjPath && (inBytes || cfiles.length != 1 || outWasmPath || outJsPath || outHtmlPath)) return ArgErr('When outputting a .o file, there must be exactly one .c file as input');
		if (outBcPath && (inBytes || cfiles.length == 0 || outWasmPath || outJsPath || outHtmlPath)) return ArgErr('When outputting a .bc file, there must be only .c files as input');
	}
	else if (cfiles.length || ld || IsWasmFile(inBytes))
	{
		if ( outWasmPath && p.streaming) return ArgErr('When outputting just a .wasm file, option -streaming is invalid');
		if ( outWasmPath && p.node)      return ArgErr('When outputting just a .wasm file, option -node is invalid');
		if ( outWasmPath && p.rle)       return ArgErr('When outputting a .wasm file, option -rle is invalid');
		if (!outWasmPath && p.streaming) return ArgErr('When embedding the .wasm file, option -streaming is invalid');
		if ( outHtmlPath && p.node)      return ArgErr('When generating the .html file, option -node is invalid');
		if (!outHtmlPath && p.template)  return ArgErr('When not generating the .html file, option -template is invalid');
		if (!outHtmlPath && p.loadbar)   return ArgErr('When not generating the .html file, option -loadbar is invalid');
		if (p.loadbar && !outJsPath && !outWasmPath) return ArgErr('With just a single output file, option -loadbar is invalid');
		if (p.loadbar && p.template) return ArgErr('Options -loadbar and -template can not be used together');
	}
	else
	{
		if (!outJsPath || outWasmPath || outHtmlPath || outObjPath || outBcPath) return ArgErr('When minifying a JS file, only one output file ending with .js is supported');
		if (!p.minify)   return ArgErr('When processing a .js file, minify must be enabled');
		if (p.streaming) return ArgErr('When processing a .js file, option -streaming is invalid');
		if (p.rle)       return ArgErr('When processing a .js file, option -rle is invalid');
		if (p.template)  return ArgErr('When processing a .js file, option -template is invalid');
		if (p.embeds && Object.keys(p.embeds).length) return ArgErr('When processing a .js file, option -embed is invalid');
	}

	// Experimental compile C files to WASM directly
	if (cfiles.length || ld)
	{
		const pathToWajic = PathRelatedTo(process.cwd(), __dirname, true), pathToSystem = pathToWajic + 'system/';
		inBytes = ExperimentalCompile(p, cfiles, cc, ld, pathToWajic, pathToSystem, outWasmPath, outObjPath, outBcPath);
		if (outObjPath || outBcPath) return console.log('  [SAVED] ' + (outObjPath || outBcPath) + ' (' + fs.statSync(outObjPath || outBcPath).size + ' bytes)');
	}

	// Calculate relative paths (HTML -> JS -> WASM)
	p.wasmPath = (outWasmPath ? (outHtmlPath || outJsPath ? PathRelatedTo(outHtmlPath || outJsPath, outWasmPath) : outWasmPath) : undefined);
	p.jsPath   = (outJsPath   ? (outHtmlPath              ? PathRelatedTo(outHtmlPath,                outJsPath) :   outJsPath) : undefined);
	p.htmlPath = outHtmlPath;

	var [wasmOut, jsOut, htmlOut] = ProcessFile(inBytes, p);
	if (wasmOut) Save(outWasmPath, wasmOut);
	if (jsOut)   Save(outJsPath,   jsOut);
	if (htmlOut) Save(outHtmlPath, htmlOut);
	console.log('  [SAVED] ' + saveCount + ' file' + (saveCount != 1 ? 's' : '') + ' (' + saveTotal+ ' bytes)' + (gzipTotal ? ' (' +  gzipTotal + ' gzipped)' : ''));
})();

function ProcessFile(inBytes, p)
{
	var minify_compress = { ecma: 2015, passes: 5, unsafe: true, unsafe_arrows: true, unsafe_math: true, drop_console: !p.log, pure_funcs:['document.getElementById'] };
	var minify_reserved = ['abort', 'MU8', 'MU16', 'MU32', 'MI32', 'MF32', 'WASM_STACK_SIZE', 'STOP', 'TEMP', 'MStrPut', 'MStrGet', 'MArrPut', 'ASM', 'WM', 'J', 'N' ];
	p.terser = require_terser();
	p.terser_options_toplevel = { compress: minify_compress, format: { wrap_func_args: false }, mangle: { eval: 1, reserved: minify_reserved }, toplevel: true };
	p.terser_options_reserve = { compress: minify_compress, format: { wrap_func_args: false }, mangle: { eval: 1, reserved: minify_reserved } };
	p.terser_options_merge = { compress: minify_compress, format: { wrap_func_args: false } };

	if (IsWasmFile(inBytes))
	{
		p.wasm = inBytes;
		if (p.jsPath || p.htmlPath)
		{
			GenerateJsAndWasm(p);
			FinalizeJs(p);
			return [ (p.wasmPath && p.wasm), (p.jsPath && WriteUTF8String(p.js)), (p.htmlPath && WriteUTF8String(GenerateHtml(p))) ];
		}
		else if (p.wasmPath)
		{
			return [ WasmEmbedFiles(GenerateWasm(p), p.embeds), null, null ]
		}
	}
	else
	{
		return [ null, MinifyJs(inBytes, p), null ];
	}
}

function IsWasmFile(inBytes)
{
	return (inBytes && inBytes.length > 4 && inBytes[0] == 0 && inBytes[1] == 0x61 && inBytes[2] == 0x73 && inBytes[3] == 0x6d); //wasm magic header
}

function GenerateHtml(p)
{
	VERBOSE('    [HTML] Generate - Log: ' + p.log + ' - Canvas: ' + p.use_canvas + (p.jsPath ? ' - JS: ' + p.jsPath : '') + (p.wasmPath ? ' - WASM: ' + p.wasmPath : '') + (p.template ? ' - USING TEMPLATE' : ''));
	var both = (p.jsPath && p.wasmPath);
	if (p.template)
	{
		var template = ReadUTF8String(p.template), js = [], indent = (template.match(/\n([\t ]+){{{js}}}/)||[])[1]||'';
		if (p.wasmPath && !p.jsPath) js.push('WA.module = "{{{wasmPath}}}";');
		if (p.args)                  js.push('WA.args = ' + p.args + ';');
		if (!p.jsPath)               js.push(p.js.trimEnd());
		if (p.use_canvas && !template.match(/\bwa_canvas\b/ )) ABORT('Template is missing wa_canvas element');
		if (both         && !template.match(/{{{wasmPath}}}/)) ABORT('Template is missing {{{wasmPath}}} tag to insert path to wasm file - Example:\n<script defer src="{{{jsPath}}}" data-wasm="{{{wasmPath}}}"></'+'script>');
		if (p.jsPath     && !template.match(/{{{jsPath}}}/  )) ABORT('Template is missing {{{jsPath}}} tag to insert path to javascript file - Example:\n<script defer src="{{{jsPath}}}"></'+'script>');
		if (js.length    && !template.match(/{{{js}}}/      )) ABORT('Template is missing {{{js}}} tag to insert generated output - Example:\n<script>"use strict";' + "\n" + (p.meta ? p.meta : '') + '{{{js}}}\n</'+'script>');
		if (!p.jsPath    &&  template.match(/{{{jsPath}}}/  )) ABORT('Template has unused {{{jsPath}}} tag to insert path to javascript file, it needs to be removed');
		if (!both        &&  template.match(/{{{wasmPath}}}/)) ABORT('Template has unused {{{wasmPath}}} tag to insert path to wasm file, it needs to be removed');
		if (p.jsPath) template = template.replace(/{{{jsPath}}}/, p.jsPath);
		if (p.wasmPath) template = template.replace(/{{{wasmPath}}}/, p.wasmPath);
		return template.replace(/{{{js}}}/, js.join("\n"+indent));
	}
	return '<!doctype html>' + "\n"
		+ '<html lang="en-us">' + "\n"
		+ (p.loadbar ? ''
			+ '<head>' + "\n"
			+	'	<meta charset="utf-8">' + "\n"
			+	'	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">' + "\n"
			+	'	<title>WAjicUp WebAssembly JavaScript Interface Creator Utility Program</title>' + "\n"
			+	'	<style type="text/css">' + "\n"
			+	'	body { background:#CCC }' + "\n"
			+	'	#wa_progress { position:absolute;top:'+(p.use_canvas?'250':'50')+'px;left:calc(50% - 200px);width:400px;height:24px;background-color:#FFF;border:2px solid #19D;filter:drop-shadow(0 0 1px #5AD) }' + "\n"
			+	'	#wa_progress div { width:0;height:100%;background:linear-gradient(to right,#7DF,#ADE) }' + "\n"
			+	'	#wa_progress p { color:#589;font-size:130%;text-align:center;margin:8px }' + "\n"
			+	'	</style>' + "\n"
			+ '</head>' + "\n"
			+ '<body>' + "\n"
			+ (p.use_canvas ? '<canvas id="wa_canvas" style="display:block;margin:0 auto;background:#000" oncontextmenu="event.preventDefault()" width="960" height="540"></canvas>' + "\n" : '')
			+ '<div id="wa_progress"><div></div><p>Loading ...</p></div>' + "\n"
			+ (p.log ? '<div id="wa_log"></div>' + "\n" : '')
			+ '<script>"use strict";' + "\n"
			+ "var WA = {" + "\n"
				+ (p.use_canvas ? "	canvas: document.getElementById('wa_canvas')," + "\n" : '')
				+ "	error: function(code, msg)" + "\n"
				+ "	{" + "\n"
				+ "		document.getElementById('wa_progress').outerHTML = '<div id=\"wa_progress\" style=\"border:0;text-align:center;height:auto;padding:1.5em;color:#000\">' + {" + "\n"
				+ "				BOOT: 'Error during startup. Your browser might not support WebAssembly. Please update it to the latest version.'," + "\n"
				+ "				WEBGL: 'Your browser or graphics card does not seem to support <a href=\"http://khronos.org/webgl/wiki/Getting_a_WebGL_Implementation\" style=\"color:#000\">WebGL</a>.<br>Find out how to get it <a href=\"http://get.webgl.org/\" style=\"color:#000\">here</a>.'," + "\n"
				+ "				CRASH: 'The program crashed.'," + "\n"
				+ "				MEM: 'The program ran out of memory.'," + "\n"
				+ "				DL: 'Error during program loading.'," + "\n"
				+ "			}[code] + '<br><br>(' + msg + ')</div>';" + "\n"
				+ "	}," + "\n"
				+ (p.log ? "	print: text => document.getElementById('wa_log').innerHTML += text.replace(/\\n/g, '<br>')," + "\n"
				         + "	started: () => WA.print('started\\n')," + "\n" : '')
				+ (p.args ? "	args: " + p.args + "," + "\n" : '')
			+ '};' + "\n"
			+ "(()=>{" + "\n"
			+ "var progress = document.getElementById('wa_progress'), progressbar = progress.firstElementChild;" + "\n"
			+ "var " + (p.jsPath?"xhrj":'')+(both?",":"")+(p.wasmPath?"xhrw":"") + ";" + "\n"
			+ "function UpdateProgress()" + "\n"
			+ "{" + "\n"
				+ "	if (!progress) return;" + "\n"
				+ "	progressbar.style.width = Math.min(("+(p.jsPath?"xhrj.loaded":'')+(both?"+":"")+(p.wasmPath?"xhrw.loaded":"")+")/("+(p.jsPath?"xhrj.total":'')+(both?"+":"")+(p.wasmPath?"xhrw.total":"")+"),1)*100+'%';" + "\n"
			+ "};" + "\n"
			+ "function XhrDone()" + "\n"
			+ "{" + "\n"
				+ "	if (" + (p.jsPath?"xhrj.readyState != 4":'')+(both?" || ":"")+(p.wasmPath?"xhrw.readyState != 4":"") + ") return;" + "\n"
				+ "	progress.style.display = 'none';" + "\n"
				+ (p.jsPath ? ''
					+ "	var s = document.createElement('script'), d = document.documentElement;" + "\n"
					+ "	s.textContent = xhrj.response;" + "\n"
					+ "	d.appendChild(s);" + "\n"
					+ "	d.removeChild(s);" + "\n" : '')
				+ (p.wasmPath ? "	WA.loaded(xhrw.response);" + "\n" : '')
				+ "	" + (p.jsPath?"xhrj = s = s.textContent = ":'')+(p.wasmPath?"xhrw = ":"") + "null;" + "\n"
			+ "};" + "\n"
			+ "function Load(url, rt)" + "\n"
			+ "{" + "\n"
				+ "	var xhr = new XMLHttpRequest();" + "\n"
				+ "	xhr.loaded = xhr.total = 0;" + "\n"
				+ "	xhr.open('GET', url);" + "\n"
				+ "	xhr.responseType = rt;" + "\n"
				+ "	xhr.onprogress = function(e) { if (e.lengthComputable) { xhr.loaded = e.loaded; xhr.total = e.total; UpdateProgress(); } };" + "\n"
				+ "	xhr.onerror = xhr.onabort = function()"
				+ (p.wasmPath ? " { WA.error('DL', 'Aborted - URL: ' + url); };" + "\n"
					: "\n\t{\n" // maybe CORS error, can fall back to script tag with just js file
					+ "		var s = document.createElement('script');" + "\n"
					+ "		s.onload = ()=>progress.style.display = 'none';" + "\n"
					+ "		s.onerror = ()=>WA.error('DL', 'Aborted - URL: ' + url);" + "\n"
					+ "		s.defer = true;" + "\n"
					+ "		s.src = url;" + "\n"
					+ "		document.documentElement.appendChild(s);" + "\n"
					+ "	};" + "\n")
				+ "	xhr.onload = function() { if (xhr.status != 200) WA.error('DL', 'Error - URL: ' + url + ' - Status: ' + xhr.statusText); else XhrDone(); };" + "\n"
				+ "	return xhr;" + "\n"
				+ "};" + "\n"
			+ (p.jsPath ? "(xhrj = Load('" + p.jsPath + "', 'text')).send();" + "\n" : '')
			+ (p.wasmPath ? "(xhrw = Load('" + p.wasmPath + "', 'arraybuffer')).send();" + "\n" : '')
			+ "})();" + "\n"
			+ (p.jsPath ? '' : p.js)
			+ '</'+'script>' + "\n"
		: '' // default without loadbar
			+ '<head><meta charset="utf-8"></head>' + "\n"
			+ '<body style="background:#CCC">' + "\n"
			+ (p.use_canvas ? '<canvas id="wa_canvas" style="display:block;margin:0 auto" oncontextmenu="event.preventDefault()" height="0"></canvas>' + "\n" : '')
			+ (p.log ? '<div id="wa_log">Loading...<br><br></div>' + "\n" : '')
			+ '<script>"use strict";' + "\n"
			+ (p.meta ? p.meta : '')
			+ (p.jsPath ? '' : p.js)
			+ '</'+'script>' + "\n"
			+ (p.jsPath ? '<script defer src="' + p.jsPath + '"' + (p.wasmPath ? ' data-wasm="' + p.wasmPath + '"' : '') + '></'+'script>' + "\n" : '')
		)
		+ '</body>' + "\n"
		+ '</html>' + "\n";
}

function CharEscape(m)
{
	return "\\"+(m=='\0'?'0':m=='\t'?'t':m=='\n'?'n':m=='\v'?'v':m=='\f'?'f':m=='\r'?'r':m=="'"?"'":"x"+escape(m).slice(1));
}

function FinalizeJs(p)
{
	VERBOSE('    [JS] Finalize - EmbedJS: ' + !p.jsPath + ' - Minify: ' + p.minify + ' - EmbedWASM: ' + !p.wasmPath);
	var res = (p.jsPath ? '"use strict";' : '');
	if (p.loadbar && p.wasmPath) res += 'WA.loaded = function(wasm){' + "\n\n";
	else res += (p.jsPath ? 'var WA = WA||{' + (p.wasmPath ? 'module:\'' + p.wasmPath + '\'' : '') + '};' : '') + '(function(){' + "\n\n";
	if (p.args) { var args = ''; p.args.forEach((a) => args += (args ? ', ' : '[') + "'" + a.replace(/[\0-\37\']/g, CharEscape) + "'"); p.args = args + ']'; }
	if (p.minify && !p.jsPath && !p.loadbar && !p.template)
	{
		// pre-declare all variables for minification
		res += 'var WA_'+[ 'maxmem', 'asm', 'wm', 'abort' ].join(',WA_')+';' + "\n"
				+ 'var WA_module' + (p.wasmPath ? ' = \'' + p.wasmPath + '\'' : '') + ';' + "\n"
				+ 'var WA_canvas' + (p.use_canvas ? ' = document.getElementById(\'wa_canvas\')' : '') + ';' + "\n"
				+ 'var WA_print'   + (p.log ? ' = text => document.getElementById(\'wa_log\').innerHTML += text.replace(/\\n/g, \'<br>\')' : ' = t=>{}') + ';' + "\n"
				+ 'var WA_error'   + (p.log ? ' = (code, msg) => WA_print(\'ERROR: \' + code + \' - \' + msg + \'\\n\')'                   : ' = m=>{}') + ';' + "\n"
				+ 'var WA_started' + (p.log ? ' = () => WA_print(\'started\\n\')' : '') + ';' + "\n"
				+ 'var WA_args' + (p.args ? ' = ' + p.args : '') + ';' + "\n"
				+ 'var print = WA_print, error = WA_error;' + "\n\n";
	}
	else
	{
		p.meta = 'var WA = {' + "\n"
				+ (p.wasmPath && !p.jsPath && !p.loadbar ? '	module: \'' + p.wasmPath + '\',' + "\n" : '')
				+ (p.use_canvas ? '	canvas: document.getElementById(\'wa_canvas\'),' + "\n" : '')
				+ (p.log ? '	print: text => document.getElementById(\'wa_log\').innerHTML += text.replace(/\\n/g, \'<br>\'),' + "\n"
				         + '	error: (code, msg) => WA.print(\'ERROR: \' + code + \' - \' + msg + \'\\n\'),' + "\n"
				         + '	started: () => WA.print(\'started\\n\'),' + "\n"
				    : '')
				+ (p.args && !p.template ? '	args: ' + p.args + ',' + "\n" : '')
				+ '};' + "\n";

		res += '// Define print and error functions if not yet defined by the outer html file' + "\n";
		res += 'var print = WA.print || (WA.print = msg => console.log(msg.replace(/\\n$/, \'\')));' + "\n";
		res += 'var error = WA.error || (WA.error = (code, msg) => print(\'[ERROR] \' + code + \': \' + msg + \'\\n\'));' + "\n";
	}
	res += p.js;
	res += (p.loadbar && p.wasmPath ? '};' : '})();') + "\n";

	if (!p.minify)
	{
		p.js = res;
	}
	else
	{
		var src = res;
		if (!p.jsPath && !p.loadbar && !p.template)
		{
			// Convert all WA.xyz object property access to local variable WA_xyz access
			var varlist = "", treetransform = new p.terser.TreeTransformer(null, function(node)
			{
				if (node instanceof p.terser.AST_Dot || node instanceof p.terser.AST_Sub)
				{
					while (node.expression.expression) node = node.expression;
					if (!(node.expression instanceof p.terser.AST_SymbolRef) || node.expression.name != 'WA') return;
					var prop = (node.property instanceof p.terser.AST_String ? node.property.value : node.property);
					if (typeof prop != 'string') ABORT('Unable to modify global WA object with non-simple string property access (WA[complex expresion])', node.start, src);
					varlist +=  "WA_" + prop + ",";
					return new p.terser.AST_SymbolRef({ start : node.start, end: node.end, name: "WA_" + prop });
				}
			});
			try { res = p.terser.parse(src).transform(treetransform); } catch(e) { ABORT('Parse error in generated JS code', e, src); }
			if (varlist) res = p.terser.parse(src=src.replace(/var WA_/, "var " + varlist + "WA_")).transform(treetransform);
		}
		res = p.terser.minify(res, p.terser_options_merge);
		if (res.error) ABORT('Error during minification of generated JS code', res.error, src);
		p.js = (!p.jsPath ? res.code + "\n" : res.code);
	}
}

function GenerateJsAndWasm(p)
{
	VERBOSE('    [JS] Generate - Minify: ' + p.minify + ' - EmbedWASM: ' + !p.wasmPath);
	VERBOSE('    [WASM] Read #WAJIC functions and imports');

	var mods = {env:{}}, libs = {}, libNewNames = {}, funcCount = 0, import_memory_pages = 0;
	WasmProcessImports(p.wasm, true,
		function(mod, fld, isMemory, memInitialPages)
		{
			mod = (mods[mod] || (mods[mod] = {}));
			mod[fld] = (isMemory ? 'MEMORY' : 'FUNCTION');
			if (isMemory)
			{
				if (memInitialPages < 1) memInitialPages = 1;
				mod[fld + '__INITIAL_PAGES'] = memInitialPages;
				import_memory_pages = memInitialPages;
			}
		},
		function(JSLib, JSName, JSArgs, JSCode, JSInit)
		{
			if (!libs[JSLib]) { libs[JSLib] = {["INIT\x11"]:[]}; libNewNames[JSLib] = {}; }
			if (JSInit) libs[JSLib]["INIT\x11"].push(JSInit);

			var newName = (p.minify ? NumberToAlphabet(funcCount++) : JSName);
			libs[JSLib][newName] = '(' + JSArgs + ') => ' + JSCode;
			libNewNames[JSLib][JSName] = newName;
		});

	VERBOSE('    [WASM] WAJIC functions embedded in JS, remove code from WASM');
	p.wasm = WasmEmbedFiles(WasmReplaceLibImportNames(p.wasm, libNewNames), p.embeds);
	p.js = GenerateJsBody(mods, libs, import_memory_pages, p);
	p.use_canvas = p.js.includes('canvas');
}

function GenerateJsBody(mods, libs, import_memory_pages, p)
{
	VERBOSE('    [JS] Generate - Querying WASM exports and memory layout');
	const [exports, export_memory_name, export_memory_pages] = WasmGetExports(p.wasm);
	const use_memory = (import_memory_pages || export_memory_name);
	const memory_pages = Math.max(import_memory_pages, export_memory_pages);

	var imports = GenerateJsImports(mods, libs);
	const [use_sbrk, use_fpts, use_MStrPut, use_MStrGet, use_MArrPut, use_WM, use_ASM, use_MU8, use_MU16, use_MU32, use_MI32, use_MF32, use_MSetViews, use_MEM, use_TEMP, use_stks]
		= VerifyWasmLayout(exports, mods, imports, use_memory, p);

	// Fix up some special cases in the generated imports code
	if (import_memory_pages && !use_MEM)
	{
		// remove the 'MEM = ' from the import where the memory object is created
		imports = imports.replace(/MEM = new WebAssembly\.Memory/, 'new WebAssembly.Memory');
	}
	if (use_sbrk && !use_MSetViews)
	{
		// remove the call to MSetViews in sbrk if it's not needed
		imports = imports.replace(/ MSetViews\(\);/, '');
	}
	if (use_sbrk && use_MU8)
	{
		// simplify memory length lookup in sbrk
		imports = imports.replace(/MEM\.buffer\.byteLength/, 'MU8.length');
	}

	var body = '';

	if (use_MEM || use_ASM || use_TEMP || use_WM || use_stks)
	{
		var vars = '';
		if (use_TEMP) vars +=                      'TEMP';
		if (use_WM)   vars += (vars ? ', ' : '') + 'WM';
		if (use_ASM)  vars += (vars ? ', ' : '') + 'ASM';
		if (use_MEM)  vars += (vars ? ', ' : '') + 'MEM';
		if (use_MU8)  vars += (vars ? ', ' : '') + 'MU8';
		if (use_MU16) vars += (vars ? ', ' : '') + 'MU16';
		if (use_MU32) vars += (vars ? ', ' : '') + 'MU32';
		if (use_MI32) vars += (vars ? ', ' : '') + 'MI32';
		if (use_MF32) vars += (vars ? ', ' : '') + 'MF32';
		if (use_fpts) vars += (vars ? ', ' : '') + 'FPTS = [0,0,0]';
		if (use_stks) vars += (vars ? ', ' : '') + 'WASM_STACK_SIZE = ' + (p.stacksize||65536);
		if (use_sbrk) vars += (vars ? ', ' : '') + 'WASM_HEAP = ' + WasmFindHeapBase(p.wasm, memory_pages);
		if (use_sbrk) vars += (vars ? ', ' : '') + 'WASM_HEAP_MAX = (WA.maxmem||256*1024*1024)';
		body += '// Some global memory variables/definition' + "\n";
		body += 'var ' + vars + ';' + (use_sbrk ? ' //default max 256MB' : '') + "\n\n";
	}

	body += '// A generic abort function that if called stops the execution of the program and shows an error' + "\n";
	body += 'var STOP, abort = WA.abort = function(code, msg)' + "\n";
	body += '{' + "\n";
	body += '	STOP = true;' + "\n";
	body += '	error(code, msg);' + "\n";
	body += '	throw \'abort\';' + "\n";
	body += '};' + "\n\n";

	if (use_MStrPut)
	{
		body += '// Puts a string from JavaScript onto the wasm memory heap (encoded as UTF8)' + "\n";
		body += 'var MStrPut = function(str, ptr, buf_size)' + "\n";
		body += '{' + "\n";
		body += '	if (buf_size === 0) return 0;' + "\n";
		body += '	var buf = new TextEncoder().encode(str), bufLen = buf.length, out = (ptr||ASM.malloc(bufLen+1));' + "\n";
		body += '	if (buf_size && bufLen >= buf_size)' + "\n";
		body += '		for (bufLen = buf_size - 1; (buf[bufLen] & 0xC0) == 0x80; bufLen--);' + "\n";
		body += '	MU8.set(buf.subarray(0, bufLen), out);' + "\n";
		body += '	MU8[out + bufLen] = 0;' + "\n";
		body += '	return (ptr ? bufLen : out);' + "\n";
		body += '};' + "\n\n";
	}

	if (use_MStrGet)
	{
		body += '// Reads a string from the wasm memory heap to JavaScript (decoded as UTF8)' + "\n";
		body += 'var MStrGet = function(ptr, length)' + "\n";
		body += '{' + "\n";
		body += '	if (length === 0 || !ptr) return \'\';' + "\n";
		body += '	if (!length) { for (length = 0; length != ptr+MU8.length && MU8[ptr+length]; length++); }' + "\n";
		body += '	return new TextDecoder().decode(MU8.subarray(ptr, ptr+length));' + "\n";
		body += '};' + "\n\n";
	}

	if (use_MArrPut)
	{
		body += '// Copy a JavaScript array to the wasm memory heap' + "\n";
		body += 'var MArrPut = function(a)' + "\n";
		body += '{' + "\n";
		body += '	var len = a.byteLength || a.length, ptr = len && ASM.malloc(len);' + "\n";
		body += '	MU8.set(a, ptr);' + "\n";
		body += '	return ptr;' + "\n";
		body += '}' + "\n\n";
	}

	if (use_MSetViews)
	{
		body += '// Set the array views of various data types used to read/write to the wasm memory from JavaScript' + "\n";
		body += 'var MSetViews = function()' + "\n";
		body += '{' + "\n";
		body += '	var buf = MEM.buffer;' + "\n";
		if (use_MU8)  body += '	MU8 = new Uint8Array(buf);' + "\n";
		if (use_MU16) body += '	MU16 = new Uint16Array(buf);' + "\n";
		if (use_MU32) body += '	MU32 = new Uint32Array(buf);' + "\n";
		if (use_MI32) body += '	MI32 = new Int32Array(buf);' + "\n";
		if (use_MF32) body += '	MF32 = new Float32Array(buf);' + "\n";
		body += '};' + "\n\n";
	}

	if (!p.wasmPath)
	{
		if (p.rle)
		{
			body += '// Function to decode an RLE compressed string' + "\n";
			body += 'var DecodeRLE85 = function(str)' + "\n";
			body += '{' + "\n";
			body += '	for(var r,e,n,o=0,i=0,t=r=>h.copyWithin(o,o-y,(o+=r)-y),a=r=>(r=str.charCodeAt(i++))<92?r-41:r-41-1,c=e=>(d||(r=a()+85*(a()+85*(a()+85*(a()+85*a()))),d=4),r>>24-8*--d&255),f=c()|r,d=0,h=new Uint8Array(f);o<f;n<<=1,e--)' + "\n";
			body += '		if(e||(n=c(),e=8),128&n)h[o++]=c();else{for(var u=c()<<8|c(),v=u>>12?2+(u>>12):c()+18,y=1+(4095&u);v>y;)t(y),v-=y,y<<=1;t(v)}' + "\n";
			body += '	return h;' + "\n";
			body += '};' + "\n\n";

			body += '// Decompress and decode the embedded .wasm file' + "\n";
			body += 'var wasm = DecodeRLE85("' + EncodeRLE85(p.wasm) + '");' + "\n\n";
		}
		else
		{
			body += '// Function to decode a W64 encoded string to a byte array' + "\n";
			body += 'var DecodeW64 = function(str)' + "\n";
			body += '{' + "\n";
			body += '	var e,n=str.length,r=str[n-1],t=0,o=0,c=Uint8Array,d=new c(128).map((e,n)=>n<92?n-58:n-59);' + "\n";
			body += '	var a=new c(n/4*3-(r<3&&r)),f=e=>d[str.charCodeAt(t++)]<<e,h=n=>a[o++]=e>>n;' + "\n";
			body += '	while (t<n) e=f(0)|f(6)|f(12)|f(18),h(0),h(8),h(16);' + "\n";
			body += '	return a;' + "\n";
			body += '};' + "\n\n";

			body += '// Decode the embedded .wasm file' + "\n";
			body += 'var wasm = DecodeW64("' + EncodeW64(p.wasm) + '");' + "\n\n";
		}
	}

	body += imports;

	if (!p.wasmPath || p.loadbar)
	{
		body += '// Instantiate the wasm module by passing the prepared import functions for the wasm module' + "\n";
		body += 'WebAssembly.instantiate(wasm, imports).then(output =>' + "\n";
	}
	else if (p.node)
	{
		body += '// Instantiate the wasm module by passing the prepared import functions for the wasm module' + "\n";
		body += 'WebAssembly.instantiate(require(\'fs\').readFileSync(WA.module), imports).then(output =>' + "\n";
	}
	else
	{
		var src = (p.jsPath ? "document.currentScript.getAttribute('data-wasm')" : 'WA.module');
		if (p.streaming)
		{
			body += '// Stream and instantiate the wasm module by passing the prepared import functions for the wasm module' + "\n";
			body += 'WebAssembly.instantiateStreaming(fetch(' + src + '), imports).then(output =>' + "\n";
		}
		else
		{
			body += '// Fetch and instantiate the wasm module by passing the prepared import functions for the wasm module' + "\n";
			body += 'fetch(' + src + ').then(r => r.arrayBuffer()).then(r => WebAssembly.instantiate(r, imports)).then(output =>' + "\n";
		}
	}

	body += '{' + "\n";
	body += '	// Store the module reference in WA.wm' + "\n";
	body += '	WA.wm' + (use_WM ? ' = WM' : '') + ' = output.module;' + "\n\n";

	body += '	// Store the list of the functions exported by the wasm module in WA.asm' + "\n";
	body += '	' + (use_ASM ? 'WA.asm = ASM' : 'var ASM = WA.asm') + ' = output.instance.exports;' + "\n\n";

	body += '	var started = WA.started;' + "\n\n";

	if (use_MEM && export_memory_name)
	{
		body += '	// Get the wasm memory object from the module' + (use_sbrk ? ' (can be grown with sbrk)' : '') + "\n";
		body += '	MEM = ASM.' + export_memory_name + ';' + "\n\n";
	}
	if (use_MSetViews)
	{
		body += '	// Set the array views of various data types used to read/write to the wasm memory from JavaScript' + "\n";
		body += '	MSetViews();' + "\n\n";
	}
	if (exports.__wasm_call_ctors)
	{
		body += '	// Call global constructors' + "\n";
		body += '	ASM.__wasm_call_ctors();' + "\n\n";
	}
	if ((exports.main || exports.__main_argc_argv) && exports.malloc)
	{
		if (p.args)
		{
			body += '	// Store program arguments and the argv list in memory' + "\n";
			body += '	var args = WA.args||[\'W\'], argc = args.length, argv = ASM.malloc((argc+1)<<2), i;' + "\n";
			body += '	for (i = 0; i != argc; i++) MU32[(argv>>2)+i] = MStrPut(args[i]);' + "\n";
			body += '	MU32[(argv>>2)+argc] = 0; // list terminating null pointer' + "\n\n";
		}
		else
		{
			body += '	// Allocate 10 bytes of memory to store the argument list with 1 entry to pass to main' + "\n";
			body += '	var argc = 1, argv = ASM.malloc(10);' + "\n\n";

			body += '	// Place executable name string "W" after the argv list' + "\n";
			body += '	MU8[argv+8] = 87;' + "\n";
			body += '	MU8[argv+9] = 0;' + "\n\n";

			body += '	// argv[0] contains the pointer to the executable name string, argv[1] has a list terminating null pointer' + "\n";
			body += '	MU32[(argv    )>>2] = (argv + 8)' + "\n";
			body += '	MU32[(argv + 4)>>2] = 0;' + "\n\n";
		}

		const run_prefix = (p.node ? '	process.exitCode = ASM.' : '	ASM.');
		if (exports.main) body += run_prefix + 'main(argc, argv);' + "\n";
		if (exports.__main_argc_argv) body += run_prefix + '__main_argc_argv(argc, argv);' + "\n";
		body += "\n";
	}
	if (((exports.main || exports.__main_argc_argv) && !exports.malloc) || exports.__original_main || exports.__main_void)
	{
		body += '	// Call the main function with zero arguments' + "\n";
		if (exports.main && !exports.malloc) body += '	ASM.main(0, 0);' + "\n\n";
		if (exports.__main_argc_argv && !exports.malloc) body += '	ASM.__main_argc_argv(0, 0);' + "\n\n";
		if (exports.__original_main) body += '	ASM.__original_main();' + "\n";
		if (exports.__main_void) body += '	ASM.__main_void();' + "\n";
	}
	if (exports.WajicMain)
	{
		body += '	// Call the WajicMain function' + "\n";
		body += '	ASM.WajicMain();' + "\n\n";
	}
	body += '	// If the outer HTML file supplied a \'started\' callback, call it' + "\n";
	body += '	if (started) started();' + "\n";
	body += '})' + "\n";
	body += '.catch(function (err)' + "\n";
	body += '{' + "\n";
	body += '	// On an exception, if the err is \'abort\' the error was already processed in the abort function above' + "\n";
	body += '	if (err !== \'abort\') abort(\'BOOT\', \'WASM instiantate error: \' + err + (err.stack ? "\\n" + err.stack : \'\'));' + "\n";
	body += '});' + "\n\n";

	return body;
}

function GenerateJsImports(mods, libs)
{
	const has_libs = (Object.keys(libs).length != 0);
	const has_sysopen = (mods.env.__sys_open || mods.env.__syscall_open || mods.env.__syscall_openat);
	var imports = '';

	if (has_libs)
	{
		imports += '// J is for JavaScript functions requested by the WASM module' + "\n";
		imports += 'var J =';
		let added_one = false;
		for (let JSLib in libs)
		{
			// List functions that don't have an INIT block directly
			if (libs[JSLib]["INIT\x11"].length) continue;
			imports += (added_one ? '' : "\n{") + "\n\t" + '// JavaScript functions' + (JSLib ? ' for ' + JSLib : '') + ' requested by the WASM module' + "\n";
			for (let JSName in libs[JSLib])
				if (JSName != "INIT\x11")
					imports += "\t" + JSName + ': ' + libs[JSLib][JSName] + ',' + "\n";
			added_one = true;
		}
		imports += (added_one ? '' : '{') + '};' + "\n\n";
	}

	imports += 'var imports =' + "\n";
	imports += '{' + "\n";
	if (has_libs) imports += '	J: J,' + "\n";

	Object.keys(mods).sort().forEach(mod =>
	{
		imports += '	' + mod + ':' + "\n";
		imports += '	{' + "\n";
		Object.keys(mods[mod]).sort().forEach(fld =>
		{
			var kind = mods[mod][fld];
			if (kind == 'MEMORY')
			{
				imports += '\n		// Set the initial wasm memory' + (mods.env.sbrk ? ' (can be grown with sbrk)' : '') + "\n";
				imports += '		' + fld + ': MEM = new WebAssembly.Memory({initial: ' + mods[mod][fld + '__INITIAL_PAGES'] + '}),' + "\n";
			}
			else if (mod == 'env')
			{
				var mathfunc;
				if (fld == 'sbrk' || fld == '_sbrk64')
				{
					imports += '		// sbrk gets called to increase the size of the memory heap by an increment' + "\n";
					if (fld == '_sbrk64')
						imports += '		_sbrk64: function(increment, increment_high)' + "\n";
					else
						imports += '		sbrk: function(increment)' + "\n";
					imports += '		{' + "\n";
					imports += '			var heapOld = WASM_HEAP, heapNew = heapOld + increment, heapGrow = heapNew - MEM.buffer.byteLength;' + "\n";
					imports += '			//console.log(\'[SBRK] Increment: \' + increment + \' - HEAP: \' + heapOld + \' -> \' + heapNew + (heapGrow > 0 ? \' - GROW BY \' + heapGrow + \' (\' + ((heapGrow+65535)>>16) + \' pages)\' : \'\'));' + "\n";
					imports += '			if (heapNew > WASM_HEAP_MAX) abort(\'MEM\', \'Out of memory\');' + "\n";
					imports += '			if (heapGrow > 0) { MEM.grow((heapGrow+65535)>>16); MSetViews(); }' + "\n";
					imports += '			WASM_HEAP = heapNew;' + "\n";
					imports += '			return heapOld;' + "\n";
					imports += '		},' + "\n";
				}
				else if (fld == 'time')
				{
					imports += '\n		// Function querying the system time' + "\n";
					imports += '		time: function(ptr) { var ret = (Date.now()/1000)|0; if (ptr) MU32[ptr>>2] = ret; return ret; },' + "\n";
				}
				else if (fld == 'gettimeofday')
				{
					imports += '\n		// Function querying the system time' + "\n";
					imports += '		gettimeofday: function(ptr) { var now = Date.now(); MU32[ptr>>2]=(now/1000)|0; MU32[(ptr+4)>>2]=((now % 1000)*1000)|0; },' + "\n";
				}
				else if (fld == 'clock_gettime')
				{
					imports += '\n		// Function querying an exact clock' + "\n";
					imports += '		clock_gettime: function(clock, tp)' + "\n";
					imports += '		{' + "\n";
					imports += '			clock = (clock ? window.performance.now() : Date.now()), tp >>= 2;' + "\n";
					imports += '			if (tp) MU32[tp] = (clock/1000)|0, MU32[tp+1] = ((clock%1000)*1000000+.1)|0;' + "\n";
					imports += '		},' + "\n";
				}
				else if (fld == 'clock_getres')
				{
					imports += '\n		// Function querying the resolution of an exact clock' + "\n";
					imports += '		clock_getres: function(clock, tp)' + "\n";
					imports += '		{' + "\n";
					imports += '			clock = (clock ? .1 : 1), tp >>= 2;' + "\n";
					imports += '			if (tp) MU32[tp] = (clock/1000)|0, MU32[tp+1] = ((clock%1000)*1000000)|0;' + "\n";
					imports += '		},' + "\n";
				}
				else if (fld == 'exit')
				{
					imports += '		exit: function(status) { abort(\'EXIT\', \'Exit called: \' + status); },' + "\n";
				}
				else if (fld == '__assert_fail')
				{
					imports += '\n		// Failed assert will abort the program' + "\n";
					imports += '		__assert_fail: (condition, filename, line, func) => crashFunction(\'assert \' + MStrGet(condition) + \' at: \' + (filename ? MStrGet(filename) : \'?\'), line, (func ? MStrGet(func) : \'?\')),' + "\n";
				}
				else if (fld == '__cxa_uncaught_exception')
				{
					imports += '		__cxa_uncaught_exception: function() { abort(\'CRASH\', \'Uncaught exception\'); },' + "\n";
				}
				else if (fld == '__cxa_pure_virtual')
				{
					imports += '		__cxa_pure_virtual: function() { abort(\'CRASH\', \'pure virtual\'); },' + "\n";
				}
				else if (fld == 'abort')
				{
					imports += '		abort: function() { abort(\'CRASH\', \'Abort called\'); },' + "\n";
				}
				else if (fld == 'longjmp')
				{
					imports += '		longjmp: function() { abort(\'CRASH\', \'Unsupported longjmp called\'); },' + "\n";
				}
				else if (Math[mathfunc = fld.replace(/^f?([^l].*?)f?$/, '$1').replace(/^rint$/,'round')])
				{
					// Function matched an existing math function (like sin or sqrt)
					imports += '		' + fld + ': Math.' + mathfunc + ',' + "\n";
				}
				else if (fld == 'setjmp' || fld == '__cxa_atexit' || fld == '__lock' || fld == '__unlock')
				{
					// Field name matched an aborting call, pass a crash function
					imports += '		' + fld + ': () => 0, // does nothing in this wasm context' + "\n";
				}
				else if (fld == 'getTempRet0' || fld == 'setTempRet0')
				{
					//The function is related to 64bit passing as generated by the legalize-js-interface pass of Binaryen
					if (fld[0] == 'g') imports += '		getTempRet0: () => TEMP,' + "\n";
					if (fld[0] == 's') imports += '		setTempRet0: i => TEMP = i,' + "\n";
				}
				else if (fld == 'emscripten_get_heap_size')
				{
					imports += '		emscripten_get_heap_size: function() { return MEM.buffer.byteLength; },' + "\n";
				}
				else if (fld == 'emscripten_get_heap_max')
				{
					imports += '		emscripten_get_heap_max: function() { return WASM_HEAP_MAX; },' + "\n";
				}
				else if (fld == 'sysconf')
				{
					imports += '		sysconf: function(name) { if (name == 30) return 65536; return -1; },' + "\n";
				}
				else if (fld == '__syscall_openat')
				{
					imports += '\n		// openat for embedded files (dirfd is ignored)' + "\n";
					imports += '		__syscall_openat: function(dirfd, path, flags, varargs)' + "\n";
					imports += '		{' + "\n";
					imports += '			var section = WebAssembly.Module.customSections(WA.wm, \'|\'+MStrGet(path))[0];' + "\n";
					imports += '			if (!section) return -1;' + "\n";
					imports += '			return FPTS.push(new Uint8Array(section), 0) - 2;' + "\n";
					imports += '		},' + "\n";
				}
				else if (fld == '__sys_open' || fld == '__syscall_open')
				{
					imports += '\n		// file open (can only be used to open embedded files)' + "\n";
					imports += '		' + fld + ': function(path, flags, varargs)' + "\n";
					imports += '		{' + "\n";
					imports += '			//console.log(\'__sys_open: path: \' + MStrGet(path) + \' - flags: \' + flags + \' - mode: \' + MU32[varargs>>2]);' + "\n";
					imports += '			var section = WebAssembly.Module.customSections(WA.wm, \'|\'+MStrGet(path))[0];' + "\n";
					imports += '			if (!section) return -1;' + "\n";
					imports += '			return FPTS.push(new Uint8Array(section), 0) - 2;' + "\n";
					imports += '		},' + "\n";
				}
				else if ((fld == '__sys_fcntl64' || fld == '__sys_ioctl' || fld == '__syscall_fcntl64' || fld == '__syscall_ioctl') && has_sysopen)
				{
					imports += '		' + fld + ': () => 0, // does nothing in this wasm context' + "\n";
				}
				else
				{
					WARN('Unknown import function ' + mod + '.' + fld + ' - supplying dummy function with perhaps unexpected result');
					imports += '		' + fld + ': () => 0, // does nothing in this wasm context' + "\n";
				}
			}
			else if (mod.includes('wasi'))
			{
				// WASI (WebAssembly System Interface) can have varying module names (wasi_unstable/wasi_snapshot_preview1/wasi)
				if (fld == 'fd_write')
				{
					imports += '\n		// The fd_write function can only be used to write strings to stdout in this wasm context' + "\n";
					imports += '		fd_write: function(fd, iov, iovcnt, pOutResult)' + "\n";
					imports += '		{' + "\n";
					imports += '			iov >>= 2;' + "\n";
					imports += '			for (var ret = 0, str = \'\', i = 0; i < iovcnt; i++)' + "\n";
					imports += '			{' + "\n";
					imports += '				// Process list of IO commands, read passed strings from heap' + "\n";
					imports += '				var ptr = MU32[iov++], len = MU32[iov++];' + "\n";
					imports += '				if (len < 0) return -1;' + "\n";
					imports += '				ret += len;' + "\n";
					imports += '				str += MStrGet(ptr, len);' + "\n";
					imports += '				//console.log(\'fd_write - fd: \' + fd + \' - [\'+i+\'][len:\'+len+\']: \' + MStrGet(ptr, len).replace(/\\n/g, \'\\\\n\'));' + "\n";
					imports += '			}' + "\n";
					imports += '' + "\n";
					imports += '			// Print the passed string and write the number of bytes read to the result pointer' + "\n";
					imports += '			print(str);' + "\n";
					imports += '			MU32[pOutResult>>2] = ret;' + "\n";
					imports += '			return 0; // no error' + "\n";
					imports += '		},' + "\n";
				}
				else if (fld == 'fd_read' && has_sysopen)
				{
					imports += '\n		// The fd_read function can only be used to read data from embedded files in this wasm context' + "\n";
					imports += '		fd_read: function(fd, iov, iovcnt, pOutResult)' + "\n";
					imports += '		{' + "\n";
					imports += '			var buf = FPTS[fd++], cursor = FPTS[fd]|0, ret = 0;' + "\n";
					imports += '			if (!buf) return 1;' + "\n";
					imports += '			iov >>= 2;' + "\n";
					imports += '			for (var i = 0; i < iovcnt && cursor != buf.length; i++)' + "\n";
					imports += '			{' + "\n";
					imports += '				var ptr = MU32[iov++], len = MU32[iov++];' + "\n";
					imports += '				var curr = Math.min(len, buf.length - cursor);' + "\n";
					imports += '				//console.log(\'fd_read - fd: \' + fd + \' - iovcnt: \' + iovcnt + \' - ptr: \' + ptr + \' - len: \' + len + \' - reading: \' + curr + \' (from \' + cursor + \' to \' + (cursor + curr) + \')\');' + "\n";
					imports += '				MU8.set(buf.subarray(cursor, cursor + curr), ptr);' + "\n";
					imports += '				cursor += curr;' + "\n";
					imports += '				ret += curr;' + "\n";
					imports += '			}' + "\n";
					imports += '			FPTS[fd] = cursor;' + "\n";
					imports += '			//console.log(\'fd_read -     ret: \' + ret);' + "\n";
					imports += '			MU32[pOutResult>>2] = ret;' + "\n";
					imports += '			return 0;' + "\n";
					imports += '		},' + "\n";
				}
				else if (fld == 'fd_seek' && has_sysopen)
				{
					imports += '\n		// The fd_seek function can only be used to seek in embedded files in this wasm context' + "\n";
					imports += '		fd_seek: function(fd, offset_low, offset_high, whence, pOutResult) //seek in payload' + "\n";
					imports += '		{' + "\n";
					imports += '			var buf = FPTS[fd++], cursor = FPTS[fd]|0;' + "\n";
					imports += '			if (!buf) return 1;' + "\n";
					imports += '			if (whence == 0) cursor = offset_low; //set' + "\n";
					imports += '			if (whence == 1) cursor += offset_low; //cur' + "\n";
					imports += '			if (whence == 2) cursor = buf.length - offset_low; //end' + "\n";
					imports += '			if (cursor < 0) cursor = 0;' + "\n";
					imports += '			if (cursor > buf.length) cursor = buf.length;' + "\n";
					imports += '			//console.log(\'fd_seek - fd: \' + fd + \' - offset_high: \' + offset_high + \' - offset_low: \' + offset_low + \' - whence: \' +whence + \' - seek to: \' + cursor);' + "\n";
					imports += '			FPTS[fd] = MU32[pOutResult>>2] = cursor;' + "\n";
					imports += '			MU32[(pOutResult>>2)+1] = 0; // high' + "\n";
					imports += '			return 0;' + "\n";
					imports += '		},' + "\n";
				}
				else if (fld == 'fd_close' && has_sysopen)
				{
					imports += '\n		// The fd_close clears an opened file buffer' + "\n";
					imports += '		fd_close: function(fd)' + "\n";
					imports += '		{' + "\n";
					imports += '			if (!FPTS[fd]) return 1;' + "\n";
					imports += '			//console.log(\'fd_close - fd: \' + fd);' + "\n";
					imports += '			FPTS[fd] = 0;' + "\n";
					imports += '			return 0;' + "\n";
					imports += '		},' + "\n";
				}
				else
				{
					imports += '		' + fld + ': () => 0, // IO function not emulated' + "\n";
				}
			}
			else
			{
				WARN('Unknown import function ' + mod + '.' + fld + ' - supplying dummy function with probably unexpected result');
				imports += '		' + fld + ': () => 0, // does nothing in this wasm context' + "\n";
			}
		});
		imports += '	},' + "\n";
	});

	imports += '};' + "\n\n";

	for (var JSLib in libs)
	{
		// Functions that have an INIT block get their own function scope (local vars)
		if (!libs[JSLib]["INIT\x11"].length) continue;
		imports += '// JavaScript functions' + (JSLib ? ' for ' + JSLib : '') + ' requested by the WASM module' + "\n";
		imports += '(function()\n{\n';
		for (let JSInit in libs[JSLib]["INIT\x11"])
			imports += "\t" + libs[JSLib]["INIT\x11"][JSInit] + "\n";
		for (let JSName in libs[JSLib])
			if (JSName != "INIT\x11")
				imports += "\t" + 'J.' + JSName + ' = ' + libs[JSLib][JSName] + ";\n";
		imports += '})();' + "\n\n";
	}

	return imports;
}

function GenerateWasm(p)
{
	VERBOSE('    [WASM] Process - Read #WAJIC functions - File Size: ' + p.wasm.length);

	var mods = {env:{}}, import_memory, libEvals = {};
	var splitTag = '"!{>}<~"', libREx = new RegExp('(?:;|,|)JSFUNC\\("~(\\w+)~",([^=]+)=>({?.*?}?),'+splitTag+'\\)', 'g'), imports = '';
	WasmProcessImports(p.wasm, true, 
		function(mod, fld, isMemory, memInitialPages)
		{
			mod = (mods[mod] || (mods[mod] = {}));
			mod[fld] = 1;
			if (isMemory) import_memory = 1;
		},
		function(JSLib, JSName, JSArgs, JSCode, JSInit)
		{
			if (!libEvals[JSLib]) libEvals[JSLib] = '';
			if (JSInit) libEvals[JSLib] = JSInit + libEvals[JSLib];
			libEvals[JSLib] += 'JSFUNC("~' + JSName + '~",((' + JSArgs + ')=>' + JSCode + '),'+splitTag+');';
			imports += (JSInit ? JSInit + ';' : '') + JSCode + ';';
		});

	if (p.minify)
	{
		VERBOSE('    [WASM] Minifying function code');
		var libs = {}, funcCount = 0;
		for (let JSLib in libEvals)
		{
			let libLog = (JSLib ? "Lib " + JSLib + " " : "");
			let libId = NumberToAlphabet(Object.keys(libs).length);

			// use terser minification to make the JavaScript code small
			let res = p.terser.minify(libEvals[JSLib], p.terser_options_toplevel);
			if (res.error) ABORT('Error during minification of WAJIC ' + libLog + 'JS code', res.error, libEvals[JSLib]);

			// terser can leave our splitter character raw in strings we need to escape it
			if (res.code.includes("\x11")) res.code = res.code.replace("\x11", "\\x11");

			let libFuncs = {}, libFirstFunc;
			let libInitCode = res.code.replace(libREx, function(all, JSName, JSArgs, JSCode)
			{
				if (JSCode.includes(splitTag)) ABORT('Parse error field code (contains other JSFUNC):' + JSCode);
				if (libFirstFunc === undefined) libFirstFunc = JSName;
				var funcId = NumberToAlphabet(funcCount++);
				var fld = funcId + "\x11" + JSArgs + "\x11" + JSCode + (JSLib ? "\x11" + libId : "");
				libFuncs[JSName] = fld;
				VERBOSE("      [WASM WAJIC] Out: " + libLog + JSName + (JSArgs[0] == '(' ? JSArgs : '(' + JSArgs + ')') + " => " + funcId + " - Code size: " + JSCode.length);
				return '';
			});

			if (libInitCode && libInitCode != ';')
			{
				if (libInitCode.includes(splitTag)) ABORT('Parse error init code (JSFUNC remains): ' + libInitCode);
				libFuncs[libFirstFunc] += (JSLib ? "" : "\x11") + "\x11(" + libInitCode + ")";
				VERBOSE("      [WASM WAJIC] Out: " + libLog + "Init - Code size: " + libInitCode.length);
			}

			libs[JSLib] = libFuncs;
		}

		VERBOSE('    [WASM] Update WAJIC import code with minified version');
		p.wasm = WasmReplaceLibImportNames(p.wasm, libs);
	}

	const [exports, export_memory] = WasmGetExports(p.wasm);
	VerifyWasmLayout(exports, mods, imports, (import_memory || export_memory), p);

	return p.wasm;
}

function VerifyWasmLayout(exports, mods, imports, use_memory, p)
{
	var has_main_with_args = !!exports.main || !!exports.__main_argc_argv;
	var has_main_no_args = !!exports.__original_main || !!exports.__main_void;
	var has_WajicMain = !!exports.WajicMain;
	var has_malloc = !!exports.malloc;
	var has_free = !!exports.free;
	var use_sbrk = !!(mods.env.sbrk || mods.env._sbrk64);
	var use_wasi = (Object.keys(mods).join('|')).includes('wasi');
	var use_fpts = (use_wasi && (mods.env.__sys_open || mods.env.__syscall_open || mods.env.__syscall_openat));
	var use_MStrPut = imports.match(/\bMStrPut\b/) || (has_main_with_args && has_malloc && p.args);
	var use_MStrAlloc = (use_MStrPut && imports.match(/\bMStrPut\([^,\)]+\)/));
	var use_MStrGet = imports.match(/\bMStrGet\b/) || use_wasi || mods.env.__assert_fail;
	var use_MArrPut = imports.match(/\bMArrPut\b/);
	var use_WM = imports.match(/\bWM\b/);
	var use_ASM = imports.match(/\bASM\b/) || use_MStrPut || use_MArrPut;
	var use_MU8 = imports.match(/\bMU8\b/) || use_MStrPut || use_MStrGet || use_MArrPut || (has_main_with_args && has_malloc);
	var use_MU16 = imports.match(/\bMU16\b/);
	var use_MU32 = imports.match(/\bMU32\b/) || (has_main_with_args && has_malloc) || use_wasi;
	var use_MI32 = imports.match(/\bMI32\b/);
	var use_MF32 = imports.match(/\bMF32\b/);
	var use_MSetViews = use_MU8 || use_MU16 || use_MU32 || use_MI32 || use_MF32;
	var use_MEM = use_sbrk || use_MSetViews;
	var use_TEMP = mods.env.getTempRet0 || mods.env.setTempRet0;
	var use_stks = imports.match(/\bWASM_STACK_SIZE\b/);
	var use_malloc = imports.match(/\bASM.malloc\b/i) || use_MArrPut || use_MStrAlloc || has_main_with_args;
	var use_free = imports.match(/\bASM.free\b/i);
	var use_asyncify = imports.match(/\basyncify_start_unwind\b/);

	VERBOSE('    [JS] Uses: ' + ([ use_memory?'Memory':0, use_sbrk?'sbrk':0, (has_main_with_args||has_main_no_args)?'main':0, has_WajicMain?'WajicMain':0, use_wasi?'wasi':0, use_asyncify?'asyncify':0 ].filter(m=>m).join('|')));
	if (!use_memory && use_MEM)       ABORT('WASM module does not import or export memory object but requires memory manipulation');
	if (!has_malloc && use_MArrPut)   ABORT('WASM module does not export malloc but its usage of MArrPut requires it');
	if (!has_malloc && use_MStrAlloc) ABORT('WASM module does not export malloc but its usage of MStrPut requires it');
	if (!has_malloc && use_malloc)    ABORT('WASM module does not export malloc but it requires it');
	if (!has_free   && use_free)      ABORT('WASM module does not export free but it requires it');

	var unused_malloc = (has_malloc && !use_malloc), unused_free = (has_free && !use_free);
	if (p.RunWasmOpt)
	{
		p.RunWasmOpt(unused_malloc, unused_free, use_asyncify);
		if (unused_malloc) has_malloc = false;
		if (unused_free)   has_free   = false;
	}
	else
	{
		if (unused_malloc) WARN('WASM module exports malloc but does not use it, it should be compiled without the export');
		if (unused_free)   WARN('WASM module exports free but does not use it, it should be compiled without the export');
	}

	return [use_sbrk, use_fpts, use_MStrPut, use_MStrGet, use_MArrPut, use_WM, use_ASM, use_MU8, use_MU16, use_MU32, use_MI32, use_MF32, use_MSetViews, use_MEM, use_TEMP, use_stks];
}

function MinifyJs(jsBytes, p)
{
	var src = ReadUTF8String(jsBytes).replace(/\r/, '');
	var res = p.terser.minify(src, p.terser_options_reserve);
	if (res.error) ABORT('Error during minification of JS code', res.error, src);
	return WriteUTF8String(res.code);
}

function ReadUTF8String(buf, idx, length)
{
	if (!buf || length === 0) return '';
	if (!length) length = buf.length;
	if (!idx) idx = 0;
	for (var hasUtf = 0, t, i = 0; i != length; i++)
	{
		t = buf[idx+i];
		if (t == 0 && !length) break;
		hasUtf |= t;
	}
	if (i < length) length = i;
	if (hasUtf & 128)
	{
		for(var r=buf,o=idx,p=idx+length,F=String.fromCharCode,e,f,i,n,C,t,a,g='';;)
		{
			if(o==p||(e=r[o++],!e)) return g;
			128&e?(f=63&r[o++],192!=(224&e)?(i=63&r[o++],224==(240&e)?e=(15&e)<<12|f<<6|i:(n=63&r[o++],240==(248&e)?
			e=(7&e)<<18|f<<12|i<<6|n:(C=63&r[o++],248==(252&e)?e=(3&e)<<24|f<<18|i<<12|n<<6|C:(t=63&r[o++],
			e=(1&e)<<30|f<<24|i<<18|n<<12|C<<6|t))),65536>e?g+=F(e):(a=e-65536,g+=F(55296|a>>10,56320|1023&a))):g+=F((31&e)<<6|f)):g+=F(e);
		}
	}
	// split up into chunks, because .apply on a huge string can overflow the stack
	for (var ret = '', curr; length > 0; idx += 1024, length -= 1024)
		ret += String.fromCharCode.apply(String, buf.subarray(idx, idx + Math.min(length, 1024)));
	return ret;
}

function WriteUTF8String(str)
{
	var utf8len = 0;
	for (var e = str, i = 0; i < str.length;)
	{
		var k = str.charCodeAt(i++);
		utf8len += ((55296<=k&&k<=57343&&(k=65536+((1023&k)<<10)|1023&str.charCodeAt(i++)),k<=127)?1:(k<=2047?2:(k<=65535?3:(k<=2097151?4:(k<=67108863?5:6)))));
	}
	var r = new Uint8Array(utf8len);
	for (var f = 0, b = 0; b < str.length;)
	{
		var k=str.charCodeAt(b++);
		if (55296<=k&&k<=57343&&(k=65536+((1023&k)<<10)|1023&str.charCodeAt(b++)),k<=127){r[f++]=k;}
		else if (k<=2047){r[f++]=192|k>>6,r[f++]=128|63&k;}
		else if (k<=65535){r[f++]=224|k>>12,r[f++]=128|k>>6&63,r[f++]=128|63&k;}
		else if (k<=2097151){r[f++]=240|k>>18,r[f++]=128|k>>12&63,r[f++]=128|k>>6&63,r[f++]=128|63&k;}
		else if (k<=67108863){r[f++]=248|k>>24,r[f++]=128|k>>18&63,r[f++]=128|k>>12&63,r[f++]=128|k>>6&63,r[f++]=128|63&k;}
		else {r[f++]=252|k>>30,r[f++]=128|k>>24&63,r[f++]=128|k>>18&63,r[f++]=128|k>>12&63,r[f++]=128|k>>6&63,r[f++]=128|63&k;}
	}
	return r;
}

// Fit len more bytes into out buffer (which gets increased in 64kb steps)
function FitBuf(out, len)
{
	if (out.len + len <= out.arr.length) return;
	var newOut = new Uint8Array((out.len + len + (64 * 1024))>>16<<16);
	newOut.set(out.arr);
	out.arr = newOut;
}

function AppendBuf(out, buf)
{
	if (out.len + buf.length > out.arr.length) FitBuf(out, buf.length);
	out.arr.set(buf, out.len);
	out.len += buf.length;
}

// Calculate byte length of/write/append a LEB128 variable-length number
function LengthLEB(n) { return (n < (1<<7) ? 1 : (n < (1<<14) ? 2 : (n < (1<<21) ? 3 : (n < (1<<28) ? 4 : 5)))); }
function WriteLEB(arr, i, n) { do { arr[i++] = (n>127 ? n&127|128 : n); } while (n>>=7); }
function AppendLEB(out, n) { FitBuf(out, 5); do { out.arr[out.len++] = (n>127 ? n&127|128 : n); } while (n>>=7); }

function WasmReplaceLibImportNames(wasm, libs)
{
	var wasmOut   = { arr: new Uint8Array(64 * 1024), len: 0 };
	var importOut = { arr: new Uint8Array(64 * 1024), len: 0 };
	var wasmDone = 0;
	WasmProcessImports(wasm, false, null,
		function(JSLib, JSName, JSArgs, JSCode, JSInit, iModEnd, iFldEnd)
		{
			var fldOut = WriteUTF8String(libs[JSLib][JSName]);
			AppendBuf(importOut, wasm.subarray(wasmDone, iModEnd));
			AppendLEB(importOut, fldOut.length);
			AppendBuf(importOut, fldOut);
			wasmDone = iFldEnd;
		},
		function(iSectionBeforeLength, iSectionAfterLength)
		{
			AppendBuf(wasmOut, wasm.subarray(0, iSectionBeforeLength));
			wasmDone = iSectionAfterLength;
		},
		function(iSectionEnd)
		{
			AppendBuf(importOut, wasm.subarray(wasmDone, iSectionEnd));
			AppendLEB(wasmOut, importOut.len);
			AppendBuf(wasmOut, importOut.arr.subarray(0, importOut.len));
			wasmDone = iSectionEnd;
		}
	);
	AppendBuf(wasmOut, wasm.subarray(wasmDone, wasm.length));
	return wasmOut.arr.subarray(0, wasmOut.len);
}

function WasmGetExports(wasm)
{
	var exports = {}, export_memory_name, export_memory_pages = 0;
	WasmProcessSections(wasm, {
		's7': function(Get) //Section 7 'Exports' contains the list of functions provided by the wasm module
		{
			var fld = Get('string'), knd = Get(), index = Get();
			if (knd == 0) //Function export
			{
				exports[fld] = 1;
				VERBOSE("      [WASM] Export function " + fld);
			}
			if (knd == 2 && !export_memory_name) //Memory export
			{
				export_memory_name = fld;
				VERBOSE("      [WASM] Export memory: " + fld);
			}
		},
		's5': function(Get) //Section 5 'Memory' contains initial size of the exported memory
		{
			var memFlags = Get();
			export_memory_pages = Get();
			return false; //don't continue processing section items
		}});
	return [exports, export_memory_name, export_memory_pages];
}

function WasmFindHeapBase(wasm, memory_pages)
{
	var findMax = (memory_pages||1)<<16, findMin = findMax - 65535, found = 0;
	WasmProcessSections(wasm, {
		's6': function(Get) //Section 6 'Globals', llvm places the stack pointer here (which is at heap base initially)
		{
			var type = Get(), mutable = Get(), initial = Get('initexpr');
			//Make sure the initial value designates the heap end by verifying if it is in range
			if (initial >= findMin && initial <= findMax && initial > found) found = initial;
		}});
	return (found ? found : findMax);
}

function WasmEmbedFiles(wasm, embeds)
{
	if (!embeds || !Object.keys(embeds).length) return wasm;

	wasm = WasmFilterCustomSections(wasm, (name, size) =>
	{
		if (name[0] != '|' || !embeds[name.substr(1)]) return;
		WARN('Replacing already existing file "' + name.substr(1) + '" (' + size + ')');
		return true;
	});

	var wasmNew = { arr: new Uint8Array(wasm.buffer, wasm.byteOffset), len: wasm.length };
	for (var name in embeds)
	{
		VERBOSE('    [FILE] Embedding file "' + name + '" (' + embeds[name].length + ' bytes)');
		var nameBuf = WriteUTF8String('|' + name);
		var payloadLen = (LengthLEB(nameBuf.length) + nameBuf.length + embeds[name].length);
		AppendLEB(wasmNew, 0);
		AppendLEB(wasmNew, payloadLen);
		AppendLEB(wasmNew, nameBuf.length);
		AppendBuf(wasmNew, nameBuf);
		AppendBuf(wasmNew, embeds[name]);
	}
	return wasmNew.arr.subarray(0, wasmNew.len);
}

// The functions below go through the wasm file sections according the binary encoding description
//     https://webassembly.org/docs/binary-encoding/

function WasmProcessImports(wasm, logImports, callbackImportMod, callbackImportJ, callbackImportsStart, callbackImportsEnd)
{
	// Get() gets a LEB128 variable-length number
	function Get() { for (var b, r, x = 0; r |= ((b = wasm[i++])&127)<<x, b>>7; x += 7); return r; }
	for (var i = 8, iSectionEnd, type, iSectionBeforeLength, len; i < wasm.length; i = iSectionEnd)
	{
		type = Get(), iSectionBeforeLength = i, len = Get(), iSectionEnd = i + len;
		if (type < 0 || type > 11 || len <= 0 || iSectionEnd > wasm.length) break;
		if (type != 2) continue;

		//Section 2 'Imports' contains the list of JavaScript functions imported by the wasm module
		if (callbackImportsStart) callbackImportsStart(iSectionBeforeLength, i);
		for (let count = Get(), j = 0, mod, fld, iModEnd, iFldEnd, knd; j != count && i < iSectionEnd; j++)
		{
			len = Get(), mod = ReadUTF8String(wasm, i, len), iModEnd = (i += len);
			len = Get(), fld = ReadUTF8String(wasm, i, len), iFldEnd = (i += len);
			knd = Get(); Get(); // Skip over extra data
			if (knd == 0) //Function import
			{
				if (mod == 'J')
				{
					// JavaScript functions can be generated by the compiled code (with #WAJIC), their code is embedded in the field name
					let [JSName, JSArgs, JSCode, JSLib, JSInit] = fld.split('\x11');
					if (JSCode === undefined) ABORT('This WASM module contains no body for the WAJIC function "' + fld + '". It was probably already processed with this tool.');
					if (!JSLib) JSLib = '';

					// strip C types out of params list (change '(float* p1, unsigned int p2[4], WAu64 i)' to 'p1,p2,i1,i2' (function pointers not supported)
					JSArgs = JSArgs
						.replace(/^\(\s*void\s*\)$|^\(|\[.*?\]|(=|WA_ARG\()[^,]+|\)$/g, '') // remove a single void, opening/closing brackets, array and default argument suffixes
						.replace(/(.*?)(\w+)\s*(,|$)/g, // get the arguments in triplets (type, name, following comma if available)
							(a,b,c,d)=>(b.match(/WA.64[^\*\&]*$/)?c+1+','+c+2:c)+d); // replace with two variables if 64-bit type, otherwise just the name

					// Character sequences in regular expression can contain some that need to be escaped (regex with \ is better coded in string form)
					JSCode = JSCode.replace(/[\0-\37]/g, CharEscape);
					if (JSInit) JSInit = JSInit.replace(/[\0-\37]/g, CharEscape);

					// Remove ( ) brackets around init code which are left in there by #WAJIC
					if (JSInit) JSInit = JSInit.replace(/^\(?\s*|\s*\)$/g, '');

					callbackImportJ(JSLib, JSName, JSArgs, JSCode, JSInit, iModEnd, iFldEnd);

					if (logImports)
					{
						let libLog = (JSLib ? "Lib " + JSLib + " " : "");
						if (JSInit) VERBOSE("      [WASM WAJIC] In: " + libLog + "Init (" + (JSInit.length + 5) + " chars)");
						VERBOSE("      [WASM WAJIC] In: " + libLog + JSName + '(' + JSArgs + ") - Code size: " + JSCode.length);
						if (JSInit && JSInit.includes('WA.asm')) WARN(libLog + "Init uses WA.asm which could be optimized to ASM");
						if (JSInit && JSInit.includes('WA.wm')) WARN(libLog + "Init uses WA.wm which could be optimized to WM");
						if (JSCode.includes('WA.asm')) WARN(libLog + JSName + " uses WA.asm which could be optimized to ASM");
						if (JSCode.includes('WA.wm')) WARN(libLog + JSName + " uses WA.wm which could be optimized to WM");
					}
				}
				else if (callbackImportMod)
				{
					callbackImportMod(mod, fld);
					if (logImports) VERBOSE("      [WASM] Import function: " + mod + '.' + fld);
				}
			}
			if (knd == 2) //Memory import
			{
				let memFlags = Get(), memInitial = Get(), memMaximum = (memFlags ? Get() : 0);
				if (callbackImportMod)
				{
					callbackImportMod(mod, fld, true, memInitial);
					if (logImports) VERBOSE("      [WASM] Import memory: " + mod + '.' + fld);
				}
			}
			if (knd == 1) //Table import
			{
				Get();Get()&&Get();Get(); // Skip over extra data
			}
			if (knd == 3) //Global
			{
				Get();Get(); // Skip over extra data
			}
		}
		if (callbackImportsEnd) callbackImportsEnd(iSectionEnd);
	}
}

function WasmProcessSections(wasm, callbacks)
{
	function Get() { for (var b, r, x = 0; r |= ((b = wasm[i++])&127)<<x, b>>7; x += 7); return r; }
	function MultiGet(what)
	{
		if (!what) return Get();
		if (what == 'string'  ) { var n = Get(), r = ReadUTF8String(wasm, i, n); i += n; return r; }
		if (what == 'initexpr') { var opcode = Get(), val = Get(), endcode = Get(); if (opcode != 65 || endcode != 11) ABORT('Unsupported initializer expression (only i32.const supported)'); return val; }
	}
	for (var i = 8, iSectionEnd, type, len; i < wasm.length; i = iSectionEnd)
	{
		type = Get(), len = Get(), iSectionEnd = i + len;
		if (type < 0 || type > 11 || len <= 0 || iSectionEnd > wasm.length) break;
		var callback = callbacks['s'+type];
		if (!callback || type == 0) continue;
		for (let count = Get(), j = 0; j != count && i < iSectionEnd; j++)
			if (callback(MultiGet) === false) break; //false ends element loop
	}
}

function WasmFilterCustomSections(wasm, removeCheck)
{
	function Get() { for (var b, r, x = 0; r |= ((b = wasm[i++])&127)<<x, b>>7; x += 7); return r; };
	for (var i = 8, iSectionStart, iSectionEnd, type, len; i < wasm.length; i = iSectionEnd)
	{
		iSectionStart = i, type = Get(), len = Get(), iSectionEnd = i + len;
		if (type < 0 || type > 11 || len <= 0 || iSectionEnd > wasm.length) break;
		if (type != 0) continue;
		var len = Get(), name = ReadUTF8String(wasm, i, len);
		if (!removeCheck(name, iSectionEnd - i -len)) continue;
		wasm = wasm.copyWithin(iSectionStart, iSectionEnd).subarray(0, iSectionStart - iSectionEnd);
		iSectionEnd = iSectionStart;
	}
	return wasm;
}

function WasmFilterExports(wasm, removeExports)
{
	if (!removeExports || !Object.keys(removeExports).length) return wasm;
	function Get() { for (var b, r, x = 0; r |= ((b = wasm[i++])&127)<<x, b>>7; x += 7); return r; };
	function ReduceLEB(i, oldval, amount)
	{
		var oldlen = LengthLEB(oldval), newval = oldval - amount, newlen = LengthLEB(newval);
		if (oldlen != newlen) wasm = wasm.copyWithin(i, i+1).subarray(0, -1);
		WriteLEB(wasm, i, newval);
		return oldlen - newlen;
	}
	for (var i = 8, iLen, iSectionEnd, type, len, iCount, count, j, removed = 0; i < wasm.length; i = iSectionEnd)
	{
		type = Get(), iLen = i, len = Get(), iSectionEnd = i + len;
		if (type < 0 || type > 11 || len <= 0 || iSectionEnd > wasm.length) break;
		if (type != 7) continue; //Section 7 'Exports'
		for (iCount = i, count = Get(), j = 0; j != count; j++)
		{
			var iEntry = i, fldlen = Get(), fld = ReadUTF8String(wasm, i, fldlen), fldend = (i += fldlen), knd = Get(), index = Get();
			if (!removeExports[fld]) continue;
			wasm = wasm.copyWithin(iEntry, i).subarray(0, iEntry - i);
			i = iEntry;
			removed++;
		}
		i -= ReduceLEB(iCount, count, removed);
		i -= ReduceLEB(iLen, len, iSectionEnd - i);
		break;
	}
	return wasm;
}

function NumberToAlphabet(num)
{
	// Convert num starting at 0 to 'a','b','c'...'z','A','B'...'Z','aa','ab','ac'...
	for (var res = '', i = (num < 0 ? 0 : num); i >= 0; i = ((i / 52)|0)-1)
	{
		var n = ((i) % 52);
		res = String.fromCharCode((n < 26 ? 97 : 39) + n) + res;
	}
	return res;
}

function EncodeW64(buf)
{
	var bufLen = buf.length, res = '', i = 0, n;
	var Get = (x => buf[i++]<<x);
	var Add = (x => res += String.fromCharCode(x < (92 - 58) ? x + 58 : x + 58 + 1));
	while (i < bufLen)
	{
		n = Get(0)|Get(8)|Get(16);
		Add(n&63), Add((n>>6)&63), Add((n>>12)&63), Add((n>>18)&63)
	}
	return ((bufLen%3) ? res.slice(0,-1)+(3-(bufLen%3)) : res);
}

function DecodeW64(str)
{
	//Unused by this program, but left here unminified as reference
	var strLen = str.length, pad = str[strLen-1], i = 0, o = 0, n, U8 = Uint8Array;
	var T = new U8(128).map((x,y) => (y < 92 ? y - 58 : y - 59));
	var a = new U8(strLen/4*3-(pad<3&&pad));
	var Get = (x => T[str.charCodeAt(i++)]<<x);
	var Add = (x => a[o++] = n>>x);
	while (i < strLen)
	{
		n = Get(0)|Get(6)|Get(12)|Get(18);
		Add(0),Add(8),Add(16)
	}
	return a;
}

function EncodeRLE85(src, compressionLevel = 10)
{
	var res = '';
	function WriteOut(buf, bufLen, isLast)
	{
		// Encode groups of 4 bytes into 5 ascii bytes
		var tmp = '', i = 0, iMax = ((bufLen + (isLast ? 3 : 0)) & ~3);
		var Get = (x => buf[i++]|0);
		var Add = (x => tmp += String.fromCharCode(x < (92 - 41) ? x + 41 : x + 41 + 1));
		while (i < iMax)
		{
			var n = Get()+(Get()*256)+(Get()*65536)+(Get()*16777216);
			//var n = Get()+(Get()+(Get()+Get()*256)*256)*256; probablb
			Add(n%85), Add((n/85|0)%85), Add((n/7225|0)%85), Add((n/614125|0)%85), Add(n/52200625|0)
		}
		res += tmp;
		if (iMax >= bufLen) return 0;
		buf.copyWithin(0, iMax, bufLen);
		return bufLen - iMax;
	}

	// encode the total file length into the first group
	WriteOut(new Uint8Array([src.length,src.length>>8,src.length>>16,src.length>>24]), 4);

	var RLEFindMatch = function(matchRange, buf, bufsize, pos)
	{
		var numBytes = 1, matchPos = 0;
		for (var j, i = (pos > matchRange ? pos - matchRange : 0); i < pos; i++)
		{
			for (j = 0; j < bufsize - pos; j++) if (buf[i+j] != buf[j+pos]) break;
			if (j <= numBytes) continue;
			matchPos = i;
			if (j > 0xFF+0x12) { numBytes = 0xFF+0x12; break; }
			numBytes = j; 
		}
		if (numBytes == 2) numBytes = 1;
		return [matchPos, numBytes];
	};

	var rleMatchRange = (4 << (compressionLevel < 1 ? 1 : (compressionLevel > 10 ? 10 : compressionLevel)));
	var rleMatchPos = 0, rleNumBytes = 0, rleNextNumBytes = 0, rleNextMatchPos = 0
	var srcPos = 0, srcSize = src.length, bitCount = 0, dst = new Uint8Array(8192), dstLen = 1, bitPos = 0;
	while (srcPos < srcSize)
	{
		//RLE look ahead for matches
		if (rleNextNumBytes)
		{
			rleNumBytes = rleNextNumBytes;
			rleNextNumBytes = 0;
		}
		else if (compressionLevel)
		{
			[rleMatchPos, rleNumBytes] = RLEFindMatch(rleMatchRange, src, srcSize, srcPos);
			if (rleNumBytes >= 3 && rleNumBytes != 0xFF+0x12)
			{
				//Look one byte ahead if there's a better match coming up
				[rleNextMatchPos, rleNextNumBytes] = RLEFindMatch(rleMatchRange, src, srcSize, srcPos+1);
				if (rleNextNumBytes >= rleNumBytes+2) { rleNumBytes = 1; rleMatchPos = rleNextMatchPos; }
				else rleNextNumBytes = 0;
			}
		}

		if (rleNumBytes < 3)
		{
			//COPY byte
			dst[dstLen++] = src[srcPos++];
			dst[bitPos] |= (0x80 >> bitCount);
		}
		else
		{
			//RLE part
			var dist = srcPos - rleMatchPos - 1; 
			if (rleNumBytes >= 0x12)
			{
				//Encode in 3 bytes (0x12 ~ 0xFF+0x12 bytes repeat)
				dst[dstLen++] = (dist >> 8);
				dst[dstLen++] = (dist & 0xFF);
				dst[dstLen++] = (rleNumBytes - 0x12);
			}
			else
			{
				//Encode in 2 bytes (0x3 ~ 0x12 bytes repeat)
				dst[dstLen++] = (((rleNumBytes - 2) << 4) | (dist >> 8));
				dst[dstLen++] = (dist & 0xFF);
			}
			srcPos += rleNumBytes;
		}

		if (++bitCount == 8)
		{
			if (dstLen > dst.length-32) dstLen = WriteOut(dst, dstLen);
			dst[bitPos = dstLen++] = 0;
			bitCount = 0;
		}
	}	
	WriteOut(dst, dstLen, true);
	return res;
}

function DecodeRLE85(str)
{
	//Unused by this program, but left here unminified as reference
	var o = 0, i = 0, n, RLEOffset, bits, code;
	var Cpy = (x => trg.copyWithin(o, o - RLEOffset, (o += x) - RLEOffset));
	var Get = (x => ((x = str.charCodeAt(i++)) < 92 ? x - 41 : x - 41 - 1));
	var Src = (x =>
	{
		if (!nrem) { n = Get()+85*(Get()+85*(Get()+85*(Get()+85*Get()))); nrem = 4; }
		return (n>>(24-8*--nrem))&255;
	});
	for (var size = Src()|n, nrem = 0, trg = new Uint8Array(size); o < size; code <<= 1, bits--)
	{
		if (!bits) { code = Src(); bits = 8; }
		if (code & 0x80) { trg[o++] = Src(); continue; }
		var RLE = (Src()<<8|Src()), RLESize = ((RLE >> 12) ? (RLE >> 12) + 2 : (Src() + 0x12)), RLEOffset = ((RLE & 0xFFF) + 1);
		while (RLESize > RLEOffset) { Cpy(RLEOffset); RLESize -= RLEOffset; RLEOffset <<= 1; }
		Cpy(RLESize);
	}
	return trg;
}

function ExperimentalCompile(p, cfiles, ccAdd, ldAdd, pathToWajic, pathToSystem, outWasmPath, outObjPath, outBcPath)
{
	const fs = require('fs'), child_process = require('child_process');

	function Run(cmd, args, step)
	{
		VERBOSE('  [' + step +'] Running: ' + cmd + ' ' + args.join(' '));
		const proc = child_process.spawnSync(cmd, args, {stdio:[0,1,2]});
		if (proc.status === null) ABORT('Error while starting ' + cmd + '. Executable not found at path or no access.\nCompile command:\n\n' + cmd + ' ' + args.join(' '));
		if (proc.status !== 0)    ABORT('Error while running ' + cmd + '. An error should have been printed above.\nCompile command:\n\n' + cmd + ' ' + args.join(' '));
	}
	function RunAsync(cmd, args, step, outPath, procs, maxProcs)
	{
		WaitProcs(procs, maxProcs);
		VERBOSE('  [' + step +'] Running: ' + cmd + ' ' + args.join(' '));
		var pid = child_process.spawn(cmd, args, {stdio:[0,1,2]}).pid;
		if (pid === undefined) ABORT('Error while starting ' + cmd + '. Executable not found at path or no access.\nCompile command:\n\n' + cmd + ' ' + args.join(' '));
		procs.push(() => // false if still running, true if done without error
		{
			try { process.kill(pid, 0); return false; } catch (e) { }
			if (!fs.existsSync(outPath)) ABORT('Error while running ' + cmd + '. An error should have been printed above.\nCompile command:\n\n' + cmd + ' ' + args.join(' '));
			return true;
		});
	}
	function WaitProcs(procs, maxProcs)
	{
		for (;;) // clear finished processes and wait while there are more than maxProcs running at the same 
		{
			for (var i = procs.length; i--;) { if (procs[i]()) procs.splice(i, 1); }
			if (procs.length <= (maxProcs|0)) return;
			child_process.spawnSync(process.execPath,['-e','setTimeout(function(){},100)']); //sleep 100 ms
		}
	}
	function GetTempPath(base, ext)
	{
		do { var path = 'tmp-wajic-' + base + '-' + ((Math.random()*1000000)|0) + '.' + ext; } while (fs.existsSync(path));
		process.on('exit', function() { try { fs.unlinkSync(path); } catch (e) {} });
		return path;
	}

	var clangCmd   = pathToWajic + 'clang';
	var ldCmd      = pathToWajic + 'wasm-ld';
	var wasmOptCmd = pathToWajic + 'wasm-opt';

	var wantDebug = ccAdd.match(/(^| )-g($| )/), wantRtti = ccAdd.match(/(^| )-frtti($| )/), hasO = ccAdd.match(/(^| )-O.($| )/), hasX = ccAdd.match(/(^| )-x($| )/), hasStd = ccAdd.match(/(^| )-std=/);
	if (wantDebug) ccAdd = ccAdd.replace(/-g($| )/, ''); //actually not a real clang option
	if (wantRtti) ccAdd = ccAdd.replace(/-frtti($| )/, ''); //actually not a real clang option

	var ccArgs = [ '-cc1', '-triple', 'wasm32', '-emit-obj', '-fcolor-diagnostics', '-I'+pathToWajic, '-D__WAJIC__',
		'-isystem'+pathToSystem+'include/libcxx', '-isystem'+pathToSystem+'lib/libcxx/include', '-isystem'+pathToSystem+'include/compat', '-isystem'+pathToSystem+'include', '-isystem'+pathToSystem+'include/libc', '-isystem'+pathToSystem+'lib/libc/musl/include', '-isystem'+pathToSystem+'lib/libc/musl/arch/emscripten', '-isystem'+pathToSystem+'lib/libc/musl/arch/generic',
		'-mconstructor-aliases', '-fvisibility', 'hidden', '-fno-threadsafe-statics', //reduce output size
		'-fno-common', '-fgnuc-version=4.2.1', '-D__EMSCRIPTEN__', '-D_LIBCPP_ABI_VERSION=2', '-D_POSIX_C_SOURCE' ]; //required for musl-libc
	if (wantDebug) ccArgs.push('-DDEBUG', '-debug-info-kind=limited');
	else if (hasO) ccArgs.push('-DNDEBUG');
	else ccArgs.push('-DNDEBUG', '-Os'); //default optimizations
	ccArgs = ccArgs.concat(ccAdd.trim().split(/\s+/));

	var ldArgs = (wantDebug ? [] : ['-strip-all']);
	ldArgs.push('-gc-sections', '-no-entry', '-allow-undefined', '-export=__wasm_call_ctors', '-export=main', '-export=__original_main', '-export=__main_argc_argv', '-export=__main_void', '-export=malloc', '-export=free', pathToSystem+'system.bc');
	if (p.stacksize) { ldArgs.push('-z', 'stack-size=' + p.stacksize); }
	if (outBcPath) { outWasmPath = outBcPath; ldArgs = ['-r']; }
	ldArgs = ldArgs.concat(ldAdd.trim().split(/\s+/));

	var procs = [];
	cfiles.forEach((f,i) =>
	{
		var isC = (f.match(/\.c$/i)), outPath = outObjPath || GetTempPath(f.match(/([^\/\\]*?)\.[^\.\/\\]+$/)[1], 'o');
		var args = ccArgs.concat(hasX ? [] : ['-x', (isC ? 'c' : 'c++')]).concat(hasStd ? [] : ['-std=' + (isC ? 'c99' : 'c++11')]);
		if (!wantRtti && !isC) args.push('-fno-rtti');
		args.push('-o', outPath, f);
		console.log('  [COMPILE] Compiling file: ' + f + ' ...');
		(i == cfiles.length - 1 ? Run : RunAsync)(clangCmd, args, "COMPILE", outPath, procs, 4);
		ldArgs.push(outPath);
	});
	WaitProcs(procs);
	if (outObjPath) return;

	console.log('  [LINKING] Linking files: ' + cfiles.join(', ') + ' ...');
	if (!outWasmPath) outWasmPath = GetTempPath('out', 'wasm');
	ldArgs.push('-o', outWasmPath);
	Run(ldCmd, ldArgs, "LINKING");
	if (outBcPath) return;

	p.RunWasmOpt = function(unused_malloc, unused_free, use_asyncify)
	{
		if (unused_malloc || unused_free) p.wasm = WasmFilterExports(p.wasm, {malloc:unused_malloc,free:unused_free});
		if (wantDebug && !use_asyncify) return;
		fs.writeFileSync(outWasmPath, p.wasm);
		if (use_asyncify)
		{
			var wasmOptArgs = ['--asyncify', outWasmPath, '-o', outWasmPath ];
			if (wantDebug) wasmOptArgs.push('--legalize-js-interface', '-g');
			Run(wasmOptCmd, wasmOptArgs, "WASMOPT");
		}
		if (!wantDebug)
		{
			// adding '--ignore-implicit-traps' would be nice but it can break programs with '-Os'(see issue binaryen-2824)
			var wasmOptArgs = ['--legalize-js-interface', '--low-memory-unused', '--converge', '-Os', outWasmPath, '-o', outWasmPath ];
			Run(wasmOptCmd, wasmOptArgs, "WASMOPT");
		}
		p.wasm = new Uint8Array(fs.readFileSync(outWasmPath));
	};

	try { var buf = fs.readFileSync(outWasmPath); } catch (e) { return ABORT('Failed to load file: ' + outWasmPath, e); }
	console.log('  [LOADED] ' + outWasmPath + ' (' + buf.length + ' bytes)');
	return new Uint8Array(buf);
}

function require_terser()
{
	/***********************************************************************

	  A JavaScript tokenizer / parser / beautifier / compressor.
	  https://github.com/mishoo/UglifyJS2

	  -------------------------------- (C) ---------------------------------

	                           Author: Mihai Bazon
	                         <mihai.bazon@gmail.com>
	                       http://mihai.bazon.net/blog

	  Distributed under the BSD license:

	    Copyright 2012 (c) Mihai Bazon <mihai.bazon@gmail.com>
	    Parser based on parse-js (http://marijn.haverbeke.nl/parse-js/).

	    Redistribution and use in source and binary forms, with or without
	    modification, are permitted provided that the following conditions
	    are met:

	        * Redistributions of source code must retain the above
	          copyright notice, this list of conditions and the following
	          disclaimer.

	        * Redistributions in binary form must reproduce the above
	          copyright notice, this list of conditions and the following
	          disclaimer in the documentation and/or other materials
	          provided with the distribution.

	    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER “AS IS” AND ANY
	    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
	    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE
	    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
	    OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
	    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
	    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
	    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
	    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
	    THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
	    SUCH DAMAGE.

	 ***********************************************************************/

	// Generated from Terser ver. 5.9.0
	// Source: https://github.com/terser/terser/tree/320241024c236d05fbacab85b55dc9c50413697c
	// Date: 2021-11-03

	return Function(new TextDecoder().decode(require("zlib").inflateRawSync((r=>{for(var e,n=0,i=0,t=Uint8Array,o=new t(128).map((r,e)=>e<40?0:e<92?e-40:e-41),c=new t(70118),u=r=>0|o[(",OQjGT5>/5KSz(=K5-K2C@4Y8uO[^j}7d?.TCc8:rIFo<X<Ov@B8iaWO_0>2X_41dy7yAS/p77N*gX{,4f/saol_YJpz:j_r8Z8UGg[9@g@qaxkTEZ+32MBbF}aR5RB*933Yveb}OXg0XebuW}Dwki<c,`Hjea0uKh(ft*u3miF]Qq:b?J^5Y:0:f:j[N*En2-0oL0IWKcBqn5w[kiok@-kde5eH]_S4YN:P.5?T3C_L|f/8kX4cA7WOr-qe9.2lg@6lQ}WW])ui51o[KaT}9Lh_H_axw^xF,@RS{P+K8_H68gI-.eP7-7jN18gl0]T-,(`3u]+L4IS_O9Cb3kz664?Y7)Mx>pzR_YKSEJzK-CU5kY(8*>dd-S_y71R7RAiHE.34;Cux-A@I<9_+RV6;ZmF4f5)nvypFiAD@OYnP4aOkeE{fd@NKsPo6RL7zj;(:ih8yd+3QoB:1M0I{.h}(`@(MV`Wh.0M5I{/58+S@Y51/KveH)M{/3h)90CEQ+joHW?1Xq*1LYT;81IRKljG9DWL8]^=W,k;L?fs-xnKYNn6G,mPFl`Fao=C}7Xr{2THG{b3it{/Ou2=-0LT;Vl[F2]k8RjGCasZN3()wB5kP)zpqo`Ob4s;ci[S_-Cx)+T|GS}yk;/RTAL+2(|=xxW}Y`=:YBCVy8>`y9wDG{8Zj11kPT*rf9t:A_M?e8iIt/H@Qkus11c.gJ3YnkM.]Vj;yVnhDQB|ECQ(8K3j)RE^w)q65IQ;ik}<pMU:D<wWqZSb}L+i69ml3dOkFilJGWtTMQC)a>+cMOI3b1),Mm1Ud:62EYusY6NZb)e(kI8zR:GA]Z.AoOGcx:+P:14WKcBg72mA][b]D:^NV6aQx75r[TQnz{EgWhw-ZB].QiJzQbyvF4IFH5qXdb:[Pkr2uE|0*xB4v45zR,cI;`j,tc_;z0B7Z)Y):(z@SL@Q2Y1{|/f2yk@h@fSb0w;<o;X(fca7?PNDhKTsJsyH^DBgmza]H1G|A(abJhNnfUxL+lbQpM)2JwdkC}YkUO[JN.>uI(ZwY4tvpf^bjDoPLv:*1AI}ix4{`uFOU,mY>,feZ0u^c/}Nt).Sun3]]w?,tL]y6]H0?4+0nfN[Q|?;;`vXQ8miW9(i/2ym^`;9[pn(J+0<7[wVsl@A.O^73Q3Cgm_r?g^4[VVn18-iE<7]PAjvoUw}fN|in7aZ7<w973(u*bI4ZKAR/n`OOHNk+UB1h_`e*o1*`VS)M+.N5=c=]ECLb^sDw2L,CYbN3V>O3RkF<?ifh2,-7wwD8UZ>s5Hbpq:w[oVB_DN7nhU,.829i=v68;TBQ)|eN,W9K*PH.KxT|5Q4.qfMKfQlKssS6Aq,4taGi}ls42rrLgw3X/P??`Wi;WSLK:3jGtFnI}+q`md|>KXWzamKfZ(-6{aUrt,,xaJ8CkIWx9G,tvNevDyD+JKdpr<ZdB^8}nb,O4nV8y;mZ7<Qly<UvGh]</yi?sDU3xW_CX*kW(AvM:cZxpgFr+Ei(/a.>cPDT/?s,4F-Zyr_RbCZ>Fi=9f/3k)OkKE0TOt+{R@<:^IOFrgxnW:2rCwF7rsxc^iPf51r,XOQELAL|3D*Af7Q4.|(.@;5]waLMg;_pV`jQ*7|{0Y7`NW>x`QE>9O8t>JDk8;lViODnrCnN=/HyAumxod{mBqD[MfR*VDl?:M6wjrVJBew]]?vy5Ox;*6BM^XVLF2LdH`s+-LJPe:k8]9lqp4+D{i3dFD90jaQ(TDtQ`|?WfF7(;-wi-f0[9L*0<Y`Gv}7h]uirf_5g4e|]Kef/xxxyw*vy0OXTlg4zjVC(l3LSh(OKKhYT}Pm_<8Ck7:g3U..qdxPvq.KcYjTQuPTHvuC7AA4kOKAjy5=-z{n/WPzgq.84VgsDbip.1p+d:+hx)OhB9VhPV/ij4peY2=ynp7TM<avmx)X,SU,HxGt6douFulC;;6+_`C{>ubpURJdS^Oc+Of9v],}yugsPu:=arx-m^zC9w/cFm=L9j8{|qym*mM3cVU`F}|2,6H|PMx99sSVsjd/6/+Kz;yjR]Qp;HH6.d:msXs=*d,4vD^D-C}dEB=SFx.g.8O7[:b[|quw[gXcI+Nn2/@dS9ukJgD4lF8Gjq7OXIi>SjQ?2AjQcF2|H.k8,B71M,q_P}>dU9}L9H;f.tF*13wgj>_lV(OUy`)E[y2P,:[U5M,r:_xRV@.*yQ=fB|DTlMWq(GO6We>[W(mTiyyv[fGR*mDjX;taEvPYV^Y_v?TbhGZEvIZY{zL<PX@|qrG^|kq*BffEpCDl67bPz<,K;apMb5?nLCkOEGzB@>T(<UJB[ku+oc.nU38:^}yLRqGdMe4(>xDO{hD`T-r,Ki^v?8T30n.`v-bT]0>w.v]O/|3|zWV<:)=M*`9`Ny)f*L4.hpctwmy<0rHR]{waUqZM*8Pa<bBq[+_+6;HzQQOQ<`eCPtp3S>ew.lO4i4E6VXidv{)B-IS@JD}Ymceissj}0_*FHJ+A:5irIB<A`.e<Vq6Sj[5J;*bmOA}U-S,JEd]X9jgikhG[uoYdsF?3YTo|AIV74;`pa]sqz:c1kY/@}e)mp1NxHy=<]50Vl2=X;P[U)esm+*j=ALn*0N4GviO=v[cTI?^4h=xAc9tehZs>_86Yj*3g@kCtF>^deV;H2bm)[St6:F=sk*{;U,Fk5ErdC+GnKF}s=a04,{*Rvf9@eA7(d/w27D<TclGcl2Y>7p23tVyk_Ybh8Nyg6RG<YuZQ.mcLgVtV.h58axi[Q*dso;ng9I:};:z7wz6}80e+b,50|yaFFa.{dUezI^iA=homR{LVV(<7J1WLAn[o6*7u|CF0F2{])j>h>cIh0M<K?6wLrJs@9_rSxUw,6R6gGq4;J3OL=mNE9EgG-0hx,NySDsj*<EKzf^E83(BjXPTPs4=5xOaujd62?3FuAvxUCiBuHdA,]HE5@:<OyC2BeNwkEMc{YD1A|b(w6QPX^Pvy=u+^9[zQ0VNJKV*y>N1M@g]8{:})Jn?Z6*xz?(OM]QqY7tXkQ[cO^GQScQIfG/@8p6G)t(fE8[tSIz3+9yC3YVQc?z4iAK67B?enDO;93}-(Ew62YR3_)uOG|C,+TFg[;3Q4>)3{.1h2+SNGQwx=T,lfA3*.4XzW.Cr/a/VMK8<cB^L[)=(?yvu8DKOX/u5ebc^>UyMQQvsw4AUvBg+JnY)1o<fk*LD0Wm=Sq[aWOdWd6-SW9]IaG`1WWhcMj]opuvogdm}DZkQ<Da.03^kwl0E;H05UmFaCu6Wmw,u(E,]mwIO+`.N^-KWtX]JN6h@Eib@bwY>/tAC-{q0;agnOw}=[(cmPs4DO?(c+e44yq>khKC1L,)Wf/Jd}*9SL_?Yk@M)+g(,Rswq6Kg@6LRB@,+:jy5T?8k7u2,PZ98^T=v)oUov5/dVJtMQ^e+?[P[+]OnvInar{;325noPKyzy8VPd-8)Y*4lif7N6B){WMGt|/bQO{Wbm>L{0yLhZP{0U_m2cTHQVTItrT?J6:A@PZ9T]twYBGsPWVKy7hEyqtgE]SCg7t)ZSVv];Z}L=@w36vWNw)frXh*t5+tSFDDgs;8/(P1,}BUm>d*o8MuNS7(=VS;R7Ib,4;9]wYf1?w;9lq_S-hmAbp0507w1FiRGzP9+PSi|zC}hR9yRjjfV`uVQX.Y[7sCcakh4pFnTe`4TzQGJvMYanSQMiAg@ipP_[2db4sv3Es9|hGSivR+1tmpbc8?0[Us4{+<jqp)-:f?/,qu+Y/G5BqdGQmiFPsV5HaF,4nWFC2X|37kkD6-W(Qr+k2C``IY[.p,bt`efcsm/+ga.P^fI5f0i-+Ki3BsZ2Wo0UUEI3*9G=TRHmSN@|)gN>P,xm3x4u>lYUgU?bn7)co4uOof*X5@I2`FZtPE2By7uPt)Oqx/p:4It,Iyv=p-6^hVF4<U]|K7h7wuQMoQR3)8gsUe7+5We9M.g^_s8hcLI@)P30CH5W(*{^-XTG]h+P7Wg)]C,+v95mBju{^@c?)V*z>U2L:G:eeyktU|;bxTmU>L9]FeFhg?oto6`u6+aKd2e?f.fKDhVJ2>.Bc923lD6YjhE-1.KUxZo1dy_EVn3ve3t3vDl>M8<tXz:?<HL+|NH^/x(8]|n}U(CE:`w0h?0U*dXFUHpP,K,-W9:r|gFxit{2h*:>wBi@Hq[>W85^:2;9v{v,+i85rM40G=u]psFCLUSN+DBEZz^Wj@l3u-=[0f+,;c1]ZJx<ks@K_gR7.GuJZN,a<PmI4?IIO50t^F{DPM^)BfOF`qJtYk^1>Q?;Z4-^lJ`iW-/E_hXJU}G4}J<1:ZQGJq?.VWSf5S}Zlky-Q7dWVAO]w)]<II0v+{gM<F2U0UGrs+LnBNc]zm9t(e]GV3aa{0ZseY:GxmvY,s?83AdvSKyw3Dhj}Rr2vDdyFX2u1aeHon*NeFr4DE9U7Xg/q0h4xQ*cSoC(c[f=cYdJgtp;aWf-[zb,-jORE8r@(.r8v<N|`j6dq=Lt,EtzVb2P:L.S-1ev99S<[(X)T-E{k9/eQ8}7(_KK@;b@Fa(rWQu<Ps`W1W1e[`J8dvW,uDlxX<CpW2fD2mhFZ}2EcvxJweag?g;P]Inm9okbAZFfioc)bY7G<+s4K)GSr>aJy)e:<R/^?jAJf`bN>sZp.MnZEWeVf1(dpxGCpqCFwc<BG?:F^vdeGf[v+gDNZJFmY40*bvIAvOw?XJW)8:p>ykohz=sr-4fAsU<ZbT7@8N)XZp@3Mw}g4i|Q.gTos]4Vl;V_w+IVSR6GY3C@EuPb^ztybL6au0C4<LO;AF+F[Q*[s|UH*97}kDADO?C?c2T0@8n*^(pVMVD0XHA6=u>XI*NR)NVze3NP]--dKA.[^y*_3I294PNMwijq`m/2bk9`a/E-koRjW(c+?sAb;ucIFS0M2c([U]1]v3YW{54;0FzrbH6hkcsIr()MK:2>?3Z=lLqlk6n8L7QQ=|Rr.o?r8Ule2}_{UY-,agD)jx`mz1B2TGXMSsA0CZZH:[macE(2EGtW}LnC2k,8>`gOI5uyipK78d9T.I8vtJ05T*H-+JTwc97^OWg7j9N=-;+nMv1nTwhmxKUg`Kw|HjU>sBc<r6gB9SgygVjkKywxBpf-*jD8F>[{B*O?9lWZgxV4Ea+>2_,P9VI08tCPMu:-7koaMuwAol[:1Pw1u1Z4<?c`:bWy`*`(7>S^AO:yq;EerQ7Mg0r0I:m?Pe;m+Ii9@9@.uf?WI(tq7k?jC*,LU;CXjmvl5`a`:4OB=14CRn|-1Uf<?((r1}yrPwT0egHI*bqkG@N=0Ru?SC6Zq>rg_w6UTt6i(jd7E`3Thb|V_NfXrKqf^No(cjR[)D</iL+7eyGJES;O8qVfdiG?UdvIOMWd5+;zW[Li:@@|FhYqwl;`pppg=Q<wCWJn2<jTXNSs2pbJSXG.]b-w:D;>:b{E,`lK/|],HmPn(c{wnvCHV,lHsMl7D)vqtiIydJkE;{X>l(L<wJ_+SaOg=e7vwJ[o<^;kv)V,q|[J*yp<=kT_,0c?LG0oxQ-a0x[QtV6Qi,p6b-]QVr)D9_mAP71Xyq[ZsRRN6*{ak,+H-+TB]iduRtj5NZ`WQlrOx90u-xAUzaGeFb;*B^-[iMYGKR(C6NO/a?f)BFWqJwePq?_caWR>GMi3D[}Cx4vtk_9ad`YsnV[i,;@XLyNt5;^>@Gm}eY]-aNOjY?9z<q+QHpXsq:675dUj-Sx`l4b2IZW`WNr;xR8sWxXgwGPFw]DgP+U7r9:KIg/)euVL@5^Iz>L{w0YJlzzP<i/3*qJ4,+5}N@yb8|C/W3i5Unmy6b_BKVRNRlG9,>LW6Y{a*u,Eq,Q@y|F<_V}S-vC-RL8YP80tHex}Qw=;)Pn@5PZjISZ/scQfxxD1p)MAdW=7ZR*e}*6jf/azman/t;8uQyI|Tr=h=54g+TVINnJ<5SLs9p5n}B-e>T[cO?0LaAV2uKYQYmmQgVug39<MT1BKZpR6w7-.UXV1v6LNrxc;^g6i<94EbiCJPo{khs)dm[^[c/30os[9*;^PLeBz(.Xf-0Q>jRlXI52>lijI(H`+nNG:=`Lij-qvA,h<qXQZVVWIEDvmNu:d*E1jWd^cww}lX+Fe?n-?}]O>f4sk:P{xC8e1WRmC/aD/d(kZ4a(ZI*yn5Upr/WB_c+@k-|xTXH;A9+QExd|h/TfuCaus,^3@FWX>4cw]D{_KZ53vuCHvl>ww={t[^B>cY:U@[yNH0oH?aD9t2x[w.O^5:c[27IZOi=RICeD]=P[lQQpS[6Pzm`aQkn(,3:O=1On4_@f5:jcf>?kg/{@2rn/ije_0>SPB];Dl_Y`r?>.[tA(1jELci5eVW@His;@=+J}|{z2kC=CZ_5>W{(bPhM<^qXW}Z*8D^bkl(dUB^?1Y7HfV<UCl/)XV+>yD/UPIg@qH.W)(huLiBaGr]UZ*z4aSL3Lo-x]krN:*^2cTua-Vo>4ua<<=CN;b?kO2]GD/0YvS}w99GQj*SzzdL-5xkLzEn?8>2^uf*3,slVbH-)BCj2A?TU^.-lwl/_kmkCB44Dw9iqxc)k{zWZnC0Y?Cq++qtT|?OhB2?g6e(=>zHWX3P4g>)[wW,TG`t/=<Y@7Q{O+LVp[MyFQ>iNjsq*AJjSyISx{N+;MkPx4E7hSuqt*/nGxT/blg1azwYV|K9)R9QXUvm52[KK4h:FU=Ixd2iJ5J(:X9^loVO*in^i52fmcb>C.)WIC8rNs_ohz5VfHfe0cu8P:S>>[C>].Iv1G,ZWNKaY:8eBWae]:i>s*3()2`3SL(nd;gW-:og/zPO)?a};^DnW)-CA+4@P<Vy6Yhk2E+t0>N)PzEGWx|*,=SWN/<XMNR-1Y?8v?Gf+1zHj;Xx}@m<U6}SJYuDPQ80C]y5GiOP{M2;D9P1PW>*l=oam-al44V`oc:kIkCZz;4EsRT6Npq0P.MZ<=:)yVucc3(959;*|pH/_TvM<KB>a9]40]^DKIoR]5S:E9B{WE^5HpuefodyW|YD54*yT?7|`>6L3Al/4ivPV/seoI0bzYB3]MmMKz49R5]yzGr:_cT2C8dImO[hlNvzDGl@3?Ri/uq/AnMr@w[Pm].o-Z+(MZ>G)uo-e^<37*.ujymRdePY7WASt6gK3FFr4}eABTb0VSl:XP+-q=QHY`>w--=|l3>lN_-aD-@[swG2OI_1u-Y*T)2Et.cE]?]5Fw>l[[_Eem1Z+n<*;E2(egq_k|<ZsjpYTvp7k>j4v9oT=|^+r+84/XL/90lslnob3}|IB3=T_6)v[22PE{d0Wt0(KIrN9j1(qCU1yKH]_bR;0LnyvcXf>72hXA14>Alg}an}2=UMmStw5E4_rZCWj3MxN/i)r/ET@IOXu6B@CECE-=9GW`aC}]tPlgeJxSZ+Z[Ry/}fe=O8fWGs^L(8v}{RlPl+bqVc2N:`Yz)Nan}dfS1Fn7ME60bNuOk`P}h0Ig:/vskSfow`s3CCS)nobdki)CtbJNmtY6du<YPjxKlYXjzAoqngdq*H6>j0p>>OS/p|(5GdCl..BFfX>Pu}WCU>DEU]n.C?79L=;{:feCkDNJM4?URf*PV,>G[PfBGpq)uiQ<=iM030}3eB*7{Yqp1V(Pip<vFGUX{?RG2fDso=/fz2.GNl5MlaGyI1g`(?t.l>)i=9HPEXXO5TRdGn3F+{mxqxLC-j8ds8qsyu*mol9TAsd.O+BNu<1h2`ARev6{FAD2;(^3vs?)9TZq.8yjRSLrImpOMs@/j_N|hno7:g1L):A.:r_)(={bOy(cbOel5)`pu(v5{7-wVT<(M^Sx@/-toi+ZA>T]Op+`8lhM;1g>Ys=dN<|5hs+VI4efkx=Tsns4]f*`gohBV3aE|*6k@c9=/G}24zRiaj]XY4DQ,c7-6Owtl.27;)rrdR?2(jV_r{?s3pT9ol<yl?/PQInf(;T5QDIXOx95]O[cf}wk)b`BqzO]Xavu|Gq)Tj/bdln4wWmNp2Z?(Aeb@fd7)^/GzN*`8,itk[ONF;ZIX*m/*/`a|>nm<zs;XpjU{yNJTxY-^Wi1.bnq.JR+RZcK[vESK3+vyhj>^RoFw,+_+hR:LuPh`k/<(_b^FGJS)+(AA,wHXYZ)GrV]O;wUSkwGkF;UaR:c7G,Qd`>jAI-CHi]Pr[DmH3491NjlmevvE`ji^:Lw)JebyTRBWaW=_3[;dM[o8^r*vh=:QR)3w:{}y-pv[kM1.gGZDk4X]5D?ceYuaqfF5P`@[|xe6b:iqjL@60*+8r^(TE5G=g)=}qbnv]9L<ooeED8*hy5HwJNyT*bpx|vLZQTN=kR{;j/;=lqPDJRx;bm1[TWr}zbi/E6c@*fP1E<`yFg(mf4j18Tl1ZonNFSBDET.RQQGhvJD6rOxKYco_Yc?n4e9uCaHoMxTZp?u1?Xx;3hyDZP[Xa/,od(3E.k*{|eoHuAZ9EF6N-7)Pj5wS]wxUog4J}CzC6,5esHr}ya@EfK.S>dM(v3vnk)Xq{Y|[H_>mY*]Wo8l0]?v-yCG4G2edV4HwsdMT42:`i6fZS2)RqPtwMz;ai9[C1>nsr.Ecr+uU4OM_qGNJg@Q<w{x|XKvnx5S68[syPoy=S-M)Y0H6=Lq)W7hB2Ie(:AB(1c@063(P<lW1b*zt0316rnMP7)JS]QWsPeHnlxdxvU,Rrde2NswtZn;Mulq=x5>2az(BCTp{ZV/x^+Y]r95;a+S{ZNPz_c0I5qcFG@J[*u0`FnlmRSP>pad+Sp>u`D/Q3i/O_AYZP+DRZm:wjvR=EXgwx5FN@?]k?JDTa({sNM4:m[OMF-cZyf8)2axQxvxHIxTQBt4RS>>mH+|Rvvft[sGs3)3zQD0u89A,Gi5p.cjVj-Q4)Oi-,6GjQ4x}5*F:/R]ou6JRyo8]=+W/+=*Obhfg/wGmP4c`K4tNS2,>E/5kTE4QuEmUBof(u<SbQmh/]M3?Qdd4y9Thi[@K6:rgU[1jaR_^CGTPFXw-tzv;lLAsl<o.@;@CcG|G5[VDhX][^0eNln51WKXH+Ic1FY9z.1.?Pzt9ZgnIxn<dC+]A0N]7Yq/Wa,*O^}PkGbp,H3,zp2It)@}xM_LQ<G|`T2@T_7Bn(q0WjG-=c`6<5px/p`lFIrwIZ5chQU|gW`<gc=:b|6aX1QTqhm)ZYhKVy@ou(J5aE2MV20Y3THmh@+=qjN-Wc3)fW,GZ`d[DWW5eLb|my*K({>U5dDagA@Y;s`MVnpZyfWw*EzX>y@icOiMxcIy0B(r=bIcLDW@+<H]e(<]Hq/w@@`p(e7B:/645MnalL^;d?m_]YJ3RG=mHzgRLKTCaX86qkZ(@;nZZZqF>ovutR8Pmu1WsbAqWT|Mu:P_Hv8J?a/a_8{ucXPw>ODNU{`s|Qjw*1)c{TgjZjvdvBHD/Yl_I.=iF{o^j|ynkS{dpqbCIuO.OR:{c>M6=0c<557sRwN<Iebc[o<iN*P,vjT;sv/7p}46xGXq@J,<)l-JFcP8+-toTj_uBa6R/3s2?48YR^Go>KTLjUl*aMWr@xj[w{+SD|W_df3zx7bhZCni;2+PP@:p)|>bRV/wZPJ>`I*uO3Ns.hQ8}RTy300p8J>GC(sLWW9bge^mA=rUFrWiPBD,N1(kp1Z2Yli:hC=AC`|c^;RN+`DDeI:8:NkEO4D0=MEuhTHzG_-XP>Cqv;MWoZnJE<<>5YRSClLHKt;yLwfU1a{?t8E)OTXcFJa8fY+N3>t|^4wc^u/jpJ]?epf0_@FlgAeI5A_W_SAPx]1b_-nl`0*lEZ[MTGT^DHYzI<c|m6[SMFMB(e^scsHJjiF7<WP{)]OkuIUrHQaq=4L,ySUzBV1oWs/VM]/8t3kNx-RQv]Cx2^Sei,r>9XhGthb[EIR?(HU.GMy,3t`,+^H*V<hpbA9H<[}3(<thnsF+SA[6]wfG?0WJS@V./=j)16Vps)ZJ]vI-(:V`;53oc{pVuR`qkzPHIPr|(M-phnYLwo-lcHg?poFJGKcdFw<Yk@4tgm{SlP{rUyG/jelyh{wL*4]koM{BW6EF9lA[**QqzgRgnV_`6S=8mdOLA4izg0z2OyTIl(9XOB?Waj.?)-F0T2mL/ymhQF3k5UIG?3)eX`[9RNT5Y{rg0.Z@ZFl[rr7}7*UApwhTV`^Hr_I;?oXkB8:_GQ7FnVIXOD@HwvboVwDebbqiP[v^_9+IqlTOG9QqjPwZ13OBKJfOrNt0j{4MMhoL/USJ4Z@I{7Lj>U]kMtux`i5JE_QP(+k]=@CSxuF[}PhFz4z.W+-@2i,JFHC=zXu+B9U3|q]yAXQK;QXB.3hyhn++XA*d]:FJFZA8|Yi72Y?a,ry<(ut|YEmsy/5IxJp+ou`0y}rjMa:vZD=2oOsh`H_+CnTlaNwD)(W0{nE+TsVnHlLE5g0wP/Kj)l6Nxq^(TXflx^_[VcRuuJvAp1Zz6u0HL}5GfS+bC6^/{wrfY(Ql<_.>=]7HT=S@?+WPGj1ERV=JM^9pevLZ9KQ=*O7jS(4K>Aif45,^wD85NAdB;`63{_vOA=F?.Z`XIVXu)-9G7Vj>/n@3Vi<8L0c>^*4K>kQ*8/4k8/Y9*pL`Sa8<{2T9+m/g@2V@.i1s-J[cDuAni01sMB(:Xa@@}v=dLE:NJ`VE|V<tMFa/R4r=zPf}1k|6RvxQNvm[nb+iX8PsA3i[=_uVm_IS0:uyYHi{-hNx7N]N=6@Z[|L+<*qrMfzD1n{e9s8XBlNJ8:u)/IN:gPS/=NrYBWjEq;C5?r){whhf.MmJG}({/u9gDAnJ8Gkll1l6h,sE+tk5E|hraz}m4=,=]3]AeC;X3hf2bB2T1MFF/q0y2+P8]/UUt=l7g4gR3f,{NMt.,TQeYY.zYcVvgJycE;;wcqb6`c;3RzizbMaW8^=}*EsHPNy/WJ_w[]^gpDd6uF`x.GE-27Sq-;xP3:]YCg=Q^<T{y|cyVkF*Xa;,V00zdy?8COWafN-).MexvNK/F0@B84ne5NT1XZhU>7AmA/YFv*oMNw-75M?Yi?]NI_pW)vGevpEYH-6IPoy(}tZ{mz:bHbqow15V=-eZ_k0aj6GIYuB?d62}gF8dWAs)]lLF;2;p2G.dP*UtK^LK2nPvl^i]ieY/UU^E*YqJq+Yu},iV,[,W.*z/5D`A*d+h},*z47Ot[b@GV[HuW`KGN)24Q@S6zbKCY<Kc@,({YPU]px0J9:rs`h3aGpg2R)bH5VvH8,3Ft?_q1A@Oev<@|2V`G9hk3B/.Jo+ogEio{=M_fv(`r`;j|Xj5Z={3H5n,zXni{.@5zLNl<}in_6Y+30aXXsLle+/O^r/NqUkl(OG*:x]RnNXnu^b62.iSXem)T*lj*m-MSki1Y8cuv{aJQVHiM6Uxy(/*3KwqYANb*gM]1N8W|B9dV(_*yT7Drk6WsWgs}AKA>IAfV/Cq=Hl>0>m]sAOa9_c|s1]a;bhN4P:Uu2Gtb4IDEwm;;,2s4da/Pi<AZvYVzKh{oK(XkmQpAM_QkZY@f]p,d]pY@m=wKT=?`?Z9@ukK@@KASg@C>jv7ym`|h0*6-YwARFhlc|D^aZD}o7<_rZ:yO0p{;I4rhx54Tv2|MX<2ud*;*9SdUXkPW<vX5.lo5yUAQhcoFcc/LTpGkfw51S;0i,P@-KQKPDgb4EOM.y]<f{m7^/}.FpiM8eDm^])Cu|wSu,Zy600E?)}j|?u2V{Ca_20to,irtnn0dN3k{YUl+i,UEP-x<7Eq7K+FOm=v4f<xfRB>0NqWj,Fz>P51+?*7((li>Z,18^hu[BIpUc?kY_t*zTH]M-=S-4sg>v*l0b,QY`;STi^jOcJyJn>YrJ@8JtamhO77PU7{?bm@W>VFTdC*]E8w`;)n+P=Ap60nXTCTFJ]B_*MZx>3Za]BjAFT7zAscuG}mp6P`33/IJEEU*.?Vipob26R^UPy13[>V*lYqbt*7JO*9w]oBzcx]Zy<Ww^]X-OarK3;wN0lleJ9)vkRTP2)DEwxl}5^Ah}X=M>yXp=LWSl,B0P=LxI.0J0N60wJyPKfkq`Hr?|hL4gj<{K@OVeOHM.*y29r6St{1a.O-JYQb,gjMr(Pk=}aKMf6E-fPRv(7|=g<3VhiAGa}y5/V+(u9D[-sm|gP^nF4eR9jdr.-@QH7RAz}FPX*P(LTH[_BYdQ6xYZzFCD9lW{?jT-7V`<y_l3Wnz_8Aoba3vvFkvBPrOr,[r@IZS=Q[};WkK4x`CSz5dds05UN<75.IbT3[Sr;bdvM,^N1xIs_/qa8y0Gfls7y:/(n^`O7*a}ui1}I6hh9=NDD{RiVC@nb:Q>uQ8BbRBz?,@/iL,ck>ZG6X@NXo,W6+bu78_iS}cb4(XzLjpZkTa-J*0kw{|uDJ-`uAhyi@=V9/5xWS@?CpTSYh,w:;UJ)givmfm2)pr[|?l64G.v[BM9o[jW8{Eq9*?z<_3vim7=>fH,LObVl4a>Q`|NLU_l:e5EZFxhQd_eVd?0K_}_nBV+s*dL-OwE|)Yhcqr).qj?L3/RNLosxzXNRg1Rg-Ghz@ezfNeGS).v8:LC@A.0b8^ZkZe>H7mv;2hgDsAQz-BD6Voqn[iuAJKBKZfAfX`kHpGAOx<efd*Pl(,T<x4m5Ek)YbZ5HUa}At/cVrk?5;0qc9kHE*`_BYW(ZE>]Ubg0G4(+Y0|=j`Fn2(E=`lJ=jj_t|n4<xvpyBulfZ3/.;_abL)Kluq)t:V{fr(t{HYgf]p@upe_I+GbUv}p?XvGl-WU;0pB0N8PDBJ-}-gp<i|OdNwgkk:>fZ`][@/<:}DV?Z)H1Q]^l25jsXH9/;oUa@ARfVo-/u-]U{l(O:+be@wA93[|6db?**Ur;H2wNcw|{1*Y(z9tYHLsF-P]hh+/Sk@5-u0(XR-N1R>z40z85btar6gzZu7f4j:3GmibleZN@2e1u4J`Z]WMmsnz,]nl`xl<4q,wjL`qx(W*v4eLhqhC|tD+)pCKBo,>Do/sLp3O<MgKcco25{+.eyf8ITH0GfmY.pi(i9Jhh`L{>+Hw8?K^TV6h538Ltu.kF3ntW,_PFJzFs)gjSKBebibcif.<B<UjPP-mus9CqK+{fQ]W_(C)n+?CBFf/9oIqOLfGU0w8K.=aXrSf4.jT|`K*QN76vzpU8+p5GW82P^>M^Lp(gmEmYzG|DLy_iQje.f1JNhRC|ZC/B3)kMF,;;y0VdhmQ|2wEiDGO>G@3v[UVjKYny2jZ-,km/`2ZTP:?<lSjJxz|Veg5Sh[]WiOcV[;jRS]CjD,r<M7YJK(w{tz_[{OpB(.+-KAwLjDG|q9rqAech=70/iPe@5pHRX,nsLE,^4czK[LHwk(e,QaqJ6Ka}3YQvzPKw_G[nx5BlN0WqsdpSW}j]r-4Qwc@M4r)f59(i.,9Hg@N1/|C5-`U3,pA5LJHEm9=Bz1S)AZOlg+{n3^+n-Avz@>[.snPAmjmuExP:0-|2AAf+:_I`c]JJ(JilWIw:xFj[q526ncTB}ALdKc7tfk-PCzqWVfeR(ut/u/cYC8K-Zf)U}l,.q38LFM5lUJfq}ySpU?UFjor.=54Q3_VN7dc}MER?cO>iza=^Is{IH2>-enC60X1GQiC0K*i;uuvc.`FUnWVT|o(sy?*.oX0iaUhRkN@GYAa^)Y:6d:jzU_g)L*Fk(.LWY7nGA0FG;I;*4-[jPw0HU}}W[0p3bMC(y(C)Reh4)_gQ4rp4+@.m-C<C|i@qRFvE01[D`T7<hrJ>VOVVY4cNLiIoKvkn+|[p6XmDa2xvc=>66+_1/O).Pal,EZhnKSYHD^Qi*{|MfX>/P.[,h>n{R/>U1*-0ZT-/E_9Gx=F59{[Vb]e=@`?MjTs-Tg-.iU.XpEQj_o]<y.gUA/az7Gvai-;m*7?8Rn{[a8xK-}yK(p`bV*dtd15dr`cz^[w|cc7)WnE-]5PZavY.zTdR(aWutjFO+;*H_)+xjk0XoBKwl<axJ4dq.NosFhMc7ih19{Y2l>W^@_3q|zfKP]XmV[@shq+BB1PU/?h*OATUrtwUUrJeAt/7+/oZg4Ej3@]:K*F2,U{Vp?poU,tguJj>`{mhy@oOiXfoAs5Z+I|mwX5kFkfo?,,kh[lc{mz>2U2A:`W0zwZtLUSC?}(HWt8@OCSX+*nD|ow^?-[8[ovCJzvz^_=r-Pd/N=1CY/npmdorn]D:qPDkr:vBPi=`yxZ:Fr3nU`|xG@:|w}mH<dirJ}SIUy|z<mqcHUI+*ho[fFPi}>ahJ2dafOCzThQC}Pl(=._0]Z.*h7(i=Em8V<e:Z1cxt;ab;CeM@x:--]kK;LytbfoaSl,BY)h7<(3u7)6a3fL^o8e.6B{F^Vjcbjl7xS,@d_]4ga9La9/^>Pzv3ASoMc1kDo9SP>:P`rZRC}K*:Y|VNsXoOtC*z7Z]T}Dqp_GDqpYMUivJask2ml/=oNg;,lId>:6SDu=iCi[UUH+t*)UCTI/x{Kg>i3*,01v[yO;rh`d<:y<*JKOCdKg,x,AVw+dP[P`xP]fq3/?K)6fAyVw_uw*EdQy^eW:Dr`7Ghr1CNb?mMdlcT2VJP4<B(RA{j,PM_os<?dQdHVp9m;DDVQcse^/My63}KWlWRX3ZD216mynJ={J1DAMigO43.ErI|8PxvAvz.-Mt4Xixm60`DBNX,2[Hu8?6Jy>p^E/*hNk9u=v+tq|SM^d(cez;2CDpU0;y985?`D+<K[jeDg*{3`T)Kcz0m4DLAnm7Xp45u`4fvE:bewvjb=azk9x9C9o|pN<zF_bH.?b6LM?H-9b^h])s6BM6UUIv,DZj1uK;iaJ7kQpLYE*=Xc,LE+^Y(<(H4.EWv=bGbPE6gU>;.NBxiUH_o*JazLW7N7k2y@rA=^xnpXpY/Mr7auQCF._n7uZ-q9*y+{R,/WL:Tix_xr-t_/zID9+8G]rsB<KmbonTllfOY)VlBi.0VcgjYtZNU@8LsPyE:}+ma_wp|qj7qctl()F74pLGk>HgOqhuFj0;{(Ukthd{vxV7zGKs38s)jXn*:n+c4Z36H{x9tQ1,EIZx;3NaW[1sOGl643)cy(a(5U-dz(191=nb)ym4I-G3DTXU2WQY/^T4Y0fvn<IPjDLI8RcFpuAjC^1RM_yt.Vo(HdDs@9Y|]fK?_;)z/NZH9-`3JmD}k1rg.^}(Pg1VnkVY4u<v`o/)V?pv*^}(5AdYH;/^|Y<Ka8GA;HMtgA-*y7w67*RLTfOP1Q7AkqI]TWVNGANMOe1v?gqfM=GzAgF.fsmUs;-e,ddGhj;yAf`@<laNnb*FEv9yqI)m@26H+PKDN0OX<<JoK@*<|JY6V8HvB2@nYk=,]h*V4X3GOOiPfe_U=q]po2Gl`lWDgs?OKY:{R5sVViN16X0c[TFtawJO^3nV|tuI_?gOiw21q2hopuy/J/V*//M]nK2jvYxbdYsh_Yf+A9i>82,ZtIL;^QbB^FEHCVj4?bbgYhTbit8uW9//}S`+P4rB+Em2ocl1:HslCjO;fFHXd)vAUWDi_JiPzCexGu[27-2<5KB2;(mtvskT2GBxHQS`]GI)nnU*-*T-vF2@vuvAO-{xH:B/9}b,Oe^]<l|KGN}Dsb{zlY-XMRGHi-1HE)Y,>H7O(YBBeBu?P_u*ybjD6IcQ;y16Y};LR@19Kz1`JXhq>_E>yB8=ZWzhhlwqGiIj|XHz/r(-nSay(S7dhcWjTjuY9Qfi?4x2E-Zw[^fNn],d;BnrQYpoC*WDdNM(+9t9U,?]xmabfM;V;PfgNqO}TB){w(}oC=8?ZpEhQ-{Xp,WDGOE^S3avgseCg3bCP[^/]KK633(3lOrC^:sC1Ia>A5T|*e6.JulgkOlX]:)ZLB93Z6E2NHCY/jLq_N7dzCdD_3zR8{<xmrg^0zarw.@4tf)J.b+mQ3T4*CV8(:tIi+YlwD_Qy1MJr?Obj/}M1YAlbY>y6GQ@|RXF],TH3I1q?"+
		"GtGobG}*9W6]9|o3-q=5Kfjh7_;KLgX2KAK/PW?n3>8gU9*xA^z4XJi>Y-8,QJa)hU`;EYH{hxsPUgi6GtbM9z5;gMY_dczGrn+}g7^Jqxm^*,vv/nPHdOd9>}2iL<Ry}MK=ExC_Gh=nz,_uzK?Zrd-Auu6UXv5Z-z,-@st.X>];?Wyr3P+aK1.Yd^[wpMGzf64+UZK^j<D`<OT/HDb3f<f54oxXTvGi@JC_(rqDs)@DIhNw(cq@nE;0n|K8MZX<DA.rmkkTwhC@CNEby<KdUnr.I/r*wPbL4Q;vf7|^w?]x;:LB]viLuMBuV{EdE{`;gX+SvB|lO_,3{oejuM;fz`YmAiuAbBaoU`HQEY^L/|zb5qQ0m*ApbGpu<(Zq`Vqoo*i3FDa}2ZWvUondVl_Kg.=.05fvlP))W[G-DWJjZadb)HrAsB1k8?BjFzZ1QG0>U(qC)=Hz?++*79,xq/<7.W7/iOwLDRz(>Uxy?GrEc<ihDlzz<yUi_Px7a^U=y:rKMh}h[duZ,ujWx25vp1;|<B+37Ps>xU{rpxsI4*>daBG7|@q=aQat/1dJ]{RM6B}*Jkw=OL,y?=GrP3VVz>1c[cd6QekUhz;o?ov=}VN;)<VW3pDYMj5Nyxf+(+KSUfHr>zit/57_>r>9zI}T4fPbHMHxJd1?m;*)1>v+lB0d7>Rumaz<g]`]_Xh]g5Qwj]w*/q=8z/K+VRpmsOQ}tWwV*qVm4qK[}b8{[TqDMOjobyb[vvDQVNdic?MhW9?/4mN[lyu8BYS;/uN@?;5G9xO{b2bY[43YECvp0ePpxGM=NSdd7WRbEPde:PWg(+TgQQ(7cf4?l3Gd;t@>-Rg:j5k/wIwtpfoDf0G(vE>5:DL3BN}[GT1)hpRS]f:B2J[OyDyxACWdKp^.sEcC;O-KD7R`y>(ImL}tq=hmjpjekbat]MH(0KpfJSpI?vw=:|uM=eSG4J0;U8iaQWLooMUWM,pb4{9(ITMdsVdwAsy-Lx93+rHa-5LErQW}-.pwcND(4qEDB2JuTl=},F`.=^{wvkrP7Zv,Gvv(o3YOsS?paJEwV9+/V3r4VNge_b^,gj1<h4}V7lzLUR*/,/5,]E(1ewT8+*Bw016Bde^NRr1vN^VZ<<LT*?2SiEb`1+TA_vpVnVzhgXeDX[E:?^Q;<5i=U{SErCrtr}[}+9QlJw.*I.NteK.GUJ_uECOjISLw;N-aGFP55giW/HUjXw[-Z[Cx[q(T,(L_KY?yFDI14+g}]d/+pE28FL+o:`EdQaM0msh9Sp6.d6LL>9b9iEV71q<5*`t,ci62-U7*Zp>i;Q,-gc1wqsLHpza-|U7GLy?b^:PN@KydWtG|0kU<C^ig^<+N2@bKLw,4p^z{^hoP+_|}gjj;y6>oiRFfs{qs;5WuxGivnpfq8WNb1^muFE4?z.R4xcO2cU2Q=wM`z<UW:s4*,gXO9qwA0y*pf<I1T}jPs)]:R9]9a:)-UeM9lb0dye2o7qFNBQIM<40M68sm1IhF1[H=uWv.n<U[99gu0BDDoDG<OD;gIX52xZ]_SJO.OG^5ZKag@E+LEjo5r(UCBS4Hsm:s,`kp7CzonfVAh]slufqkoJJ8l}mNoESCzO5u*rczIhDT<yLKu^Q1Zq;)_U<5ZZckfWlg;+{{_+bgm?vhZo*np,=Q6:9<DsR*G7Ws9-ueFL3cQSM5}]t@78wV/Y/GDS,+jIPD@Ckm1-kQh_mzd^F=4>>@m_`j1FJfECifa=>9s3sx0c,JMowJH0|/<+aj0v2|iAw.FX9E_pw-_hWTDv>VD3s2gMU46z+im}/BDVUXql.7T>*lMrxd?:<IqAH|O=[>>_]BIlFVu.SL0:LIW7rc9;oynF->`7Io^K0[qj;39Ak=gfX=g45kh>vqu-B;`n-L,Yq,]r^`(GNm5>?cWCen^_r(9`:8]4z=J9sHk6ouvc2.tz5F,t.Q3=?ItXkoYPpd(5tQK9KZkxZuJ+eMA<i?7o1uH0F:c_eJDBviX@b<;A-OkPR;tq;+72F+eN@wdSf{t*kygjuhU0R4R-@JiR86x)l?*K,6cHJN^;eKmz2HYgoyW--of6S]J`FqV^WB(s<OQBFVe*1qu3NDx6[>7]hf5(<(/>:]AjFoBH+YZ8UMX-Y|?Ssk/RFffx9fN5FN1urLg26Z7adT;sY{pg)Bywm1GV1A_mN[O{slS=wn.0E]5`_ey?L[r=48zn8IJl=Z.yt*^L.h<kHLY0[EJ4(D^^3bV6(ghU(^Ea}?|FzccFE(RvZG:TH_h`X9YNstBm}`ZVkJ^BfCy1>p*jwX7`Xj/}a{L4>:*?L9rE>JK=33hXoP<Z(N@Jy@3u*vqT]@x[=iE>.@TB{z3JXM}SbB*M16Je0{zd@+aHIDtAV7;g9`g75sbBbdHeTw5}3,IRaK:p=PkQ78Xj_fx6vk52ji;9NRyV{np4nL]O)z+)Nl1h9}vVpw,E:V9F>76?}k0e7^V<gJx:o(3^oh2Q(f>snh]M+Oj36,B1C,[uA__p*V)[<@4j-SSa4q1Qi.bd3V^8s0)v)]geYs)8E9+ef*D6vX;FW<40)5yphL52vWWytnzdh,h*5S9,yo2pcXr5[nNlNI)m6uM+x6]A4?/rX-3I6@03GYcu/4vwM-o]o)*E-n*5kwVCe6/Ft;x.f:9/a8r^lt1e=5z[,?vM>HORjDQj8a_@=i:b81o)?)TlnM`zI@]`NOk<@h*w3@otC)Q=QI`5vhpfO101{.Jt._++P(D))e>EL5xbSF/>Wyw.v44wKfy{nJ`D-Y9N@Wj0,yV*F(Te(M<<pWfGP=ba?w[*z?R52cu3<eL`s_`u:GaLwTtAqJoJ8W*5D@4a;@/Zp`=t_H]Nh`2=XYJy@.}:C^FK-@[?T:lv*0<HC(*,>,<-{},r3(=NI;WV+dd-W.lYMa`TDe(*_|Mi_U/^0qGmX0@smL.,Xq[c?726I6ZSni<dXmj=e/15yP,dJ_,cha*ncC+UKR)smeJ:?qAbI_G-,[3f5KUnk^Gf=bTOWAgjif)+^+yN4ag`JNBvCr;(pnj*sz@jyH=Esbvt4`TKhtw*2Zw}aKf=WR>8Y<ChJ-O05ZniQvdJP34IWE2_T[6H6bTuK<StwfqRn/igP@Bp*7z[JNFeq.lh;z/*7g=^@0q;V/rt8}2Mr+lmf1>KuphGyPb]]gRo`s?vHLjtPDAiD-y)eXDrrpe8hz4@HYp|`[2ZA.fF{?7qbht(RKC.lrQJ[<Lsc1;+fV]E@x{H,>=jyKhi65,=(6V1fe?s`x@j0l1[08Nhe65PcrIH3A9DS[xC)joP{:r=1Yp{?<Q5T5?Zj<8reXDa^jA3VfKkydqj:jfsjF2/AzQ(ZheyB6BQGGCqMzp0Vnxo5wO7nZBH8aB?vmug/v{Y6tO0=r;(+gwd_W|FpQrDqQoJF[`l|Z]@nma9t_r>@st0RYRAivS=3A/ZH9I(34TZTC=7KRYZ39]N>w7C`t=2YkwQHOfW>oapBup4u8)Laqqo`e_7wV7hT32?LlQSnERXl=KAZO7O+9<5,cE?Ak@-9y+rV+ifduLggMB6TRek?NIczQuzN|RP|[mE*/M_Wg^4Meu{5Z-C9bP?w^-N@OxkE7:+m)R6BUir?H@cE1,kU/ca]nA|?lN]QN7U{_1r4N068`Cno/A16(ms@OyEL+V(+][:Cz8DMp>^:*Ve-7]]re60Ig7?h[JfLlq*<f4qLYINX^:nFi`LyRvh.W+Fv_HW+=aKZ2c*JsFO/wh``o{Co6+>JF0bb6vLTQeuA<c*ViQ*7s1]45PE(qC@[]b^aPPpus(llOxg^<r?>D.n^_B=0nIEjCjzFO=<>8lHm2=.6)jo}`dl}9r,O7abV5:T+haV2l(ooj>J3|ORdzu?*+udQ4jUm_sP;b7rrh_wL+8l/D0G]-xm8FUq4:AO7zm:[JsU6a+gwKp5Pf;C0inkVTGGCE.B`wlhFx|1g5V_2({6|=BS^GJBtWp^pBzU33G4n[W<h/,+feXtGrMG:9oqG45bpi)i+VFG8{vE*AI;jX7;7d@tS.9kSo<.0gIEOlt>{27@uwns1|bU3-)eHM]ee^rHkOdnj47c@<ww{,KF2juH*`5qDC<:oFbscYgxs_j5AtCm[cDWA}gv9`?`>`l*GG+5IOVXrRh+H{==cN?MQQ)pKQQdj|]/V:.T4G(^[SsG--LkhccSH5eLo6T`o7z3J3(}=G=X/w,0gF_N?g{NLnk5dc9E-Zef8nnh_yI+@c;3u@tXq:q8I.,32^xYZhyTVV_QwDuwtB9|pw)2TFL;:iQYd`SW-j*<5xj`602|B]5;lEvf.RE4q;v29Q^OR[z/1Y7gdEP0>wH>.K,:{Wb;GEgjjPQsi4EKkMHI0I{1qGn`F(L+f2@R<+,iPD0Y`}GPVN))L1,w)eC_XKK7@g`,)/c>j,@EaqJ^c5^*cGfD=S|K0]M;ghU|:M{*I=);r*)toODdCnd`n(ouC|^UVKyo_mZNsSq|,u6VRS3qbn(_cpn`L)y7YoNMbn.o,-nSWvmX`^nZY=tB3M4TBdfS8J,LSvFzmE+tjhYgNP^xmp/{Hk6|PpnL4H8BR]q]f[Zd4_uWG>u`dOC({s|)K3VCLi:bQ<Kw+V9.`mYv9]k_EgBB-2L]tjlxhViLL(rHRL2MWD:RY:bY/CZ3.{MUF>Gy9?;tL22egZXC:1pcId<B:vc0K,PL8U.E>df<EQ`6s.{d4x3iqC6h<bCek9f/Y9UqOw>+BZgyLVcDl(nsNSO2vpfgf<rI|>a87Hl9n/:e@I6gUb^}Xt`-^id}6-k-:QB8qW_-vNh0*[tu<w0DGB8=;pHafrOZB;lbUf+5.6bH]vMX_*,9;aw_HV0WsYegCmf=+488j{+>A0lihY`x[oqG;vhoIG?x?YI=K_H,GlZxQNmGcXTAt355VOW0>;|]LnF:u<MIvdD6-zE*O9LW1uoU3Egnw*IM1BiQ^L:|pbV:W-A>cV[(O=Z-EHWZ1=j9{vC)Ah1yF?4;,Fpe_]TklTi>v|>4UiF-T4He/gV7mPS^w5DNxyiA.W>6VzhBLxtE*XB-aVrkxGDlX,KQ0p;91v]U5|T-bs)uYoSiTKFC7I-eGmWB5.:miuEiW7ej)@m*`[g>*w4?kuJ2>`>nd`p,S:h]2,=.?SLx?h)f/il`4{.bHnQmV.l^m/Z4`)?.-[?`b/(OE*sR(@rX0O@g?ihQZXM3VW4T*nsf0)_Z,78To0m=zV57j=fMr)mFWCH{.:LSLb)zQ(<B>=(`[O;G<MI{M4u.}FC?r{sslSzRrW2<PI/]Hut9TxrZ?-Fs-MLy.(>AbE`Lxm,i^BF/ayI=z_8iabn4UNC=`4|sDaj7=FBFzy2ATQS@FHT4ZK8eooCiiR;SeEEttMy.5^vLR8>(FnM;>Lk^<+XUe8/o.Pk(((5H9w^Wym`eOQVP[qd]|bB{,V)?(NGYHcgGf]gD_3rBjQBS+{O,zF0]c//EjieZv0gP7iE-:,5p>2a+YMuEi66+RwurwO{@:>XAC/{95KmWet0^Vq>)V[|6Chz`_Rs,L{^ipbFhGAN*xMQ3kN}]9^Nf@D9>}rFo:h_sc)g43p3/J1m6.iQ}OF7vGBEQhhTjbGrMaXUW6rcmrRLQ]AePiH|]<^?`{GvU11q4jD+yEt],Qb:TS^dJR6BIH9OUpO5Rf`4e8zCcx=DX[}wd9h*).8Ix:0p{N:BWK`/A>{l4PZ`=.7Ga^Y{r^+ts=cq>UnuaAq:M*8N@+T4+tRUa-7pq;k=DP@Mt*K_65L3L<f){(aU6)?+MtlOKiOI:rw=a/ow_`]ZZ6mzRgG/eGsY:niv<Mg9+/|Gjerz_,pX|A|:eI./1`Wl]o8kx{VR9dg6kZ|h1QzjafTU^O*f7vezV5|LS]fUu0w77_(+,3aJ8Bz;qAK<{g_w*a9lkuB,DqVh7O/U,nYVq8jN{0NT,BOQRy_`RS@KlKJA(2Gbwf;TNu3T_M0)ssl]/7MByE`dWw9^XB)C50*KoA|<R6}`+fW<mLCN7BIiwczCWNg],V3HiEu7zZT_`0l9bvmDMwq2XExw>``xzyNa><FCDm^BRlu,u2L.3U)3@T]|t=Tk6?2L3*p3YuSyCX)W>v@<w.|5;w]W/fQU^|(QSq64DSac[+[J3.DDl-]^nR*JJ^SZ<VQ)6zSMniSyyc):^d]a,E9)96l_4`w5Mtt?oV4BrmvH1VQm[8uol9AKvR+@;pjO-y<VU^c_c?BU00.i*HlNJaF3Tdz:M[HhV4mZtrMDqeKDai|3ug1/62dn3e94(zAo^zJz1_v]N/:uVlO]P,b30_n@P3xiBx<S{zascT0PV:tbsYaT|haKXe(6)bP4@fvs7J3t@SYnPF5`L/cqDIOaMxzo_@,MRXf|[wjHhs)I2;T?Sx3nLn2s@.ucEbL5>-Q5Qu}chws_x>3(^R`xt|EDU]X1lffN,^dB8=)ZM79RVEm7;ZOzF?^PwY;wu}o`6TquUpRE9W20d|Sho:Of}sSP{TT{yxTy_t*e/T>^nIgF5BhlkQF_nd/OS)lA-^kR7WwVe,?+GvERxN@j7@lWtzDW,k)/9wia+ycGV)I<T+SbbybhkO}CH_>I>3yb-zz-UH{<JQzcxbw/_A,9ZBxrGOsc:W`p+d__8M5O]a(Wf{)nt+Lz*uv|m*ehLohHUxnN`@.Tx=x:MB]kdagAdbaNT+Dxzv9HKcM]fhOqPOtt@52m)bQ]t.jviC4l|d?;KeR`>d[eC;8mVs:KI.0av4[+aQYF5*9LW{NDhW5Uo10L/lH<=nb><xKEbfo`7TDl+)b*3AaizgFRCVXg`sBaxYN*>(4vF^p`L(=?,Px[9g0N45]D_f.r(<Jq9}S9-vT*`oB_D_d/7(ph6c5)dl+t`OFEx>vOmwx1Yb5j5LTlYeOp_?nZTU]K<RetN4:DdD2cI)(0gD^3hd.*[L=>4AHT*REijGB(:@vJo4tHw`-k3xJXRVsohCa1OujCCHAT}wR}@L?CJ{s|baU9oak6,;ee1So:jB=Qx9Q7hLKMfihHDaiuKu;s/x=ewLK.K7e/{*7<bKA]2e)Xuy?5?MqK.Bkc9|[(X23@Y<7z{Eb>K4lw;81:>u?5:H38BIMjKR2)51A^4}5yNaH8W<J;|7kKIqGvV:af=2rfStj=D=lX7_}kBG;Z|=-lfFT(3M=SHlvPzT|cfhL^A,k^[VbA^g;(yU;@FOsT?bp}bS.5q}iZtc{i[ETu-?2ro=hTl}K]9X_*hFOyhE[LXt>^}67x0=6Wj:bOn*[N8TboZ9B(>H00kpbZSmCsle6g[Gc_cq2>`SX^u0l-d)0Q>g,t`AzUj*a-MxK[e31J<L_CXve:0Pfkk(gp;hP,^/mll<o1hrEx/*WJSyC0[_y7b[a)Hoo3OEC|B0<M^A:G6JlD0GRS;dnDQDqi/TQwD39I(4QlO`@)Wa7QS5r27[fbaHy}WA^l91YOyTGT1`5WYccNo=Xv154N`8lp02XlKz/|Ou1]-asYp;2mCTZLBV..,/Xm|PLZ{tMR4PBmV?9t;|X@pVNqDF-w-02,Zv^5XuWqI;1N<BM^qN,*L26Oo)OIyNzjhn4Zd[5zL7IwECflzB<hGYO<-r0puh+lFw>PS:<v_Kjm(^<b)qs-)|H^ciZ^URHlCi}@|uz_qT+Vs|fxqWjY)N+/8WVLk}BWU0lQ6ptomuU0ggZZT_cMYN;S3>lAYq3Hdio(h+_(}krZH35BpY@ZkOW?}>)Lg.EqMp5q:Zy9;I,>CWhha{}SvRckvUE=`az6,8>R6RBZeQ0m<@o-t:{]`m2fAIk6-}Kk@Lq}N]5+oE95FiR0<P2o]-Oc]zkbRud0J{mUx7HT9jCi7_BVbYlss{W/Gt2xRhP=RG|;NOn_(IWqoZ11w10V^yvQw[h|9JF/NH(aNY;;QhkfptpmCS<?tiEzrEFII6K`qOi`]nSN7iGp0K4J?WlK0ClI@nqEln;j(AIk`7sv)RP2ur=JRc^)lG29RxKFV+qr.x{oM8t9rw8S[Rd64A4{(.XYLm@.vC<FfL=c=1OW41iO3i}MKZO1ZUvCb^BG5hj(rYrSfHccRR|;c;x+ac6e7hX>e4sJZ[Py1iyYgfS;:TUoy5>(;bGty=jUI_7ShWMy78sN::=jD3mQ)X5cLwmK[^DD?VRZpJRp,yGx)b`_+k0)*8O:VHFS^_M+fnsw<VOHhN8<g>.|=}KJGO]4+wbulQNs;fDw9XZ:vdJDiK)vuI;D)e]FRtNu-ddSr?Q,(uL@3CJKg1*-a84ddXRPc8}Z)uoZ09K.xw*`FS7RHIuZ`@S==BkoRmc>7BRlUusdjJ6*i7+EIzf]+_<)YE6|>^9NyR;k7_/gD3Vb:>f}gTvDJ>4{:]Ci,V<=YbjN01(pUkYC6HFRMDS[O]totl`kp@?ss>,`eI.2q9(+BM?0+Tf7<*|K*N3i3f[B{9qmK4BnI<?OytW<oAZ/`AtUc)t0NX>hweABM?9(HbAlwH?Ih,rfh2]rha^JKDsQErflnx2RHlf>sW{zwF]x==RBJTz?wULdJUTp^5YcEDMuBj|aS8w2=Jvb[3+OnFo/3;D,zNb@+,x2U^c7v7WR?XC9YvQ8j3k>aX-[wHV):RKS2Td[vvhQbAEEz_^[YNUd:g=[Z/OE=e20rDo`g<[^[?+UA`:2r?YXpsU>NkL*sxLKLr_V|z@,MTHSy_htFtEUHJma0kVIY(qKY|(moB7}Zo{CLJFAp1jA;t|Ir^Ptggw48BA>CA6jFNg@>01,1e}ASUl9KSXs_j93Q;eAyNhm)3*UzRTL?jx>W.[7:uekn*DB)*OdXE?Tu1KD^@m6HBWvy;mjaW*lVt0@0w(odbNvwWl7rbk3q.CUyzn`=/SBr:ZARRPO)z]=p.0WZ]DzOAc7|4Cs;Ok^4+ck:P;FbCJQmw,J(+]0:qaC5FN^qE8o2HTO??JBDytk{sgM.Y35?YG{qSH}.;-L+<.B3BRUNxOT/dneHT5`pLywOa,F-=m_FYrnTnlrsC3Q>C{cH,Plc3?h<abVxM2tg=>N1gc[U^x{YEA;_-xGpZX1iGi_-/<y,8X1hYlQh`T9X/;CjWpRUURcGk:DTjj[Qmc+Ge+sYL4iSJRaIhQQk,MXO9pfyW85n8+982D>W4I.{u7w}/.x5msqQ8]j8/VBU>]oa,r|0R:[<rC1R@5m:vuHZ9(X5XsHZaaN?m7HJr]9sNOGUdF+d}ZYP5HFFiaDFt)iIq4-YOhIt>Sb[ZPY}o77JA(L9-[2iS3UtSgv4Dr[d4h}d5JpRV6dZ=_B.vdwetpII-NymImjSDw]bogxu>QHZ*zF``MkJ|1H_GoW.i:=U9pz.kM]H=r;C-erA|H>OBR<1mvfL9wH),{4`rh+Hl;pdX65wWBm?T;H>sxf}l=j]_hk=/JG3*4IAXQl9W*n/BOUh3@Tie017EKF_lcZ-}<fc>+a8sLE0J)mrw3,|l}dAgBLpS]}9Em`0l|hgz6flfkrbj-pZc43dnHy5v)kD*Kl.us?(x^>6{mAVfz{h^ZuekX(OUi2Gym30LeI6YKTAC}t]c]5(JVxn0an1D;@VN1|GS:O`.(3wswlQ9_bg8RF:b*Z.vn;IX5gTpKm<PQ}X=b8hwPM]qQEe1d=]M0pC<9w(]ui*iWImYX^+(=J-67>(9.w}g(hnALhXn/?.[r-3ssWCEzTC{(lr/z.X;02X;)v-a(3H,qyms.Y1Oe0ZXyk}z^v:u7EZ*YqAqR1:B,;sXJEMhNHM6?rP@z.>RB4)AxN7]n*}2:2}D4m_[TBSQKDO^Iq`31B0RJ3^;/zi+s[:,oiX<{jt+oZB{P00S}97c(;+urVQno1swO1df_`KE}F??;x0f_o`1WHqNV`cQ.((y-Lj`[S}PhdLro3{ZaG.YHi|TfQv*|O.+<B<*PySwvaJ;+`.eL2/;7mXz;]z7jN[?MbGC37m)8-Y6?>c`[*c00|}YQOoRPfV=bwvq88fR[A](wE|)cT1h]8r@?}}e/Qt}6F7gS*9Sp,?y71v@qyksCzl.ozH6;hByYdv}:g]gRAI1:{T?*`GaZYgIpkDC2Rxz6gfqL,M3I|FUzK{QG<9*g3FWKwu3uWh]A/qqBj{JupuBnURm0aG<Bfi.PS5RAGd9lZU+[rWlB>X9r,Qna0w@.=On?mKT_Hguwk)p^iH{nuL=MI)lMK@gobXuG3-YsbK<jj907nc-[V3h50v^v<XU[1Q`c_ntUA;eH;5g5:Xv9mK72Kq[>rcodcqy<JhT7>[CP8sx7Aul)26*WusTA8l=KMnl)*=LRE089}NOXOE9[,OSy.arVmq<R5ek91Yjv(R.7,q[BYy?s@|EcLbu3Y-H2GP>|8V-{^-9IqrT21,_MMRhi/ihpi@-6AP4effpBLI0+g<62x3?^_qY;-DDu`L;sbN?3w3:Ww9[;5owSzhRg;TA83en|D:YxM0_<i.REw9p:tCOY@W;p)d:CuUKtxt6M{|n,Wsr^EG?Bg/Sx>@d456lSCwndpqh6=Yd9::NJ;sOt`6Z0@p/XwP)pOR{g>Sr/A,rR7,/v[>ykuZhXGe(P1;8czKd;urPwv1EIpZbmOIk[[+}Mai3,/uiZwN[TLik8R6EC@=5qjCIv[/eI]12-[[wf>rwLZjR/xuJr)W6JNM`mxt..r2}caTxoz><NoJ_*78HFJrK7dF+=0;7U_z5{)D/JxzRC13zO+*3r,<LLGc.5|)C_5VXpCGpGs1JTl9i^}qy4OT=JKSzyQ(k4VB?PDeqG}SIoky;3Z1h+rw*>YcRnTouC)2`UyK3?Yy?,Nu+Yv-GBQqu0U(>s*gIo/ki<KPq.qRBP(p1/,?0Uc9]5D3c,kb.0^?ffM6V7zul*RbIQD7JTtfv4.ZEc0HoZ)Eetyv6nVGlNm,,V9d1).4v:YFt2?o)qgv/I;Dt-LNffq.1`(_khbHT`b[T1wQq)LO+(m<5:;P*rOEw}p]jLJeNlk*PCdO4dUqT,aR<h]_Z19;2u+qDxdDwtynJz_Gg-j-T5>+^+Wki69j/|1cgLVC8-t?XE.NSAF_:ZkQy`aF.B-YI|15p[R<45RN6tSSdR]/dislgmDFvNERS:hvkI)idb;tA46VgvcX1,iQ-3J[TK73T/56638]W[Jtkwj@.x22iS5>=?gpS(qP/T<oJL.Ez@o8Z5:[Z>y=(Dgrv8k4KAWPsD@VNq>nzcF,{Vik^.RpEU|:}tZ[dnxr,[PzpW;gnm7x-:/Y=zyu,s>K|{}OE7^PQXRA7CjT.fSk2Q0SNjM.sU]0eKy-4(G*BK:;;rIrT*R00FTJNDrPm:7N.L-d7;@5qu;+x@-0:qb`i90_g`pt-aZ:.}6b>{`3HtV/Y,W9oNhSKoCvwhA*Jy+a{q/AA^k;PyiA(:TNiW(*>o/`@QS=Nc(Edx^t?8hCz@w-OX;:^<6WbhCO5*AAaeL0LnS)c.X,LJW[G0:d[n:Ii>RlU,,(+yAS>)a8p*kdV6)[Rlo]qj/qwiCu_Z7Y0KfI)]3:2>S`g:<U^:X?3;9xEQ6o>`tk`:jy0T|hEzrg35lf};9Vl;uBS.D>|06>GLM7d+rwy8h[Yl^*I:u6?0{c|_;@R}GqwyAUy4WnmZBWEnDgk7Z`.lt,Q0Hcn`w7srhQXDD>><GE=okrLi.d(F]_aloL8_J5zIZ+c+B)hIE(rt5PQ[TYIZZQAx>:wjN(3a}1ig,^`:>?`jR.xaQ4FdZO7u];yy;-gP?E}Oe9PHxmW`0OJ6_MLcoX;0c..V5ygVM3^DeVvAww`tHS.zfuXVgPVt,p1NP(5qOG5`D*[wC<TTL{7NFD-lp.Xl/=;XxiS;OCt)k5ZS>};O*td`n@WcfQd)eCcvryyD2R8=8E]QZjijy>]j_>)nz7ficKh:T5yfK+Y/`psv4j71wvX6lE-I[->(pzUfP|XokD/4EC*6S-8UA-Ea@zy5kpvXAn{jc`f;_]CWc232JaD-H3FIC)1A8WAZ=rWhas3>sRia)eW)Xwp1`@*Mf+0cVokx+<J(DNr5g)n9`[trN@yEPI)pL-{/?s|ZbP.i+3GG82A`H0C??;b<iEQ|yVE)cnDH}j[24P1x:W2K9s2/141@MEGM@hY<+q={MUin.K4t`bTC^t>S)Cv1:JkXswKk(G`VV5aI5*l=0y_Muxevvds{rfXokEQUxA@b5[^Ao=P/BwPaSD)RB^gu5^L:+1hxnJBbI3`0f]:gv]TDz/THJwdkM:p+2Vs=hA}1aD/Z*O2bkxl/>*el6TVfsnMf378;y2v1:cBQCjx=5FoC,8MsRV{7M9Td]SCi:sM{2[,pg)LzJ}9xmmxI)W1ey?wrg3]TiiJIJYG<en+mbTnTcw(*^h}XyZVU;+<5+4Gq`=is^._x2TdwjntzCf/G0^IDDj0)qV)=tk+oPM[na/yx*+h=sI4:WluY^;vP3jH]x>DeHp>*Jsz1Z>y^:9w*PWc0]>ybAEB)wuw(@=GXgDwrQvc}XN*hCB4oL5j>X8*_)=K}svueW*ep1AmGftXDLxEjQfS,Wg2=f;+2N>S2NI|vg35]X4PoYH@rK4iBfOi`>*`Z:0Fe)8=blxrR)Ro??Q/rn?/IrE5SVUHaK(PBdpxdOPCub4pixJ>g/ruEI}UjrP{9ox[]=26>>Ru?Uk/dTOQHiJ`u9MdTv8Q?V{.TnbIcf{D7ZFcx=S:4i[swTDhT2`AU?`tC/X/)+|i+|H`.|BM:Z_Wj)qJ4nIv63bWjJ19=vb{:N]R6>{S=ADKoiGw,pjn^YkbyoprI@AQy]kY7;`U`pM;BR246`RZ^:L9{e01}utCtvru@hN]Cu6-]j|u/A+ivTgyf>M5McZbbIn+d}XK0,b)?5B2caRKi]iZU;w4Ot6HPyv1FXC,.-;_UppJ2x>shpXy`I7p*X:qnUTm;y[lO)W4t+66:gkN^n`/[F(./cATxDZUXMUX0p2}JZGTuz@/+s2ZB3x3+[b+-{^;4lvA:1V<WFccRgIu=Eqs^asuL_IpFSF/EyoiR}+nZNz<m=rvAURzY3I/(PqpF,}-5E)Vf8CX(7[nQN8fm<lFI*/d[X0-1)m==:CVN;NF<uPA_.oH.|T8WoVvE>`z]kOsh(t364mzUlz-J)pd<?Lw]WYe,YBQN`^dVoq1A*jpgq@CL}?`E`3_RBjp,3q{8h}2=5C^,iJx;Ko9h6KM]5UZ=3V3+(i(i@w*@Uk}HI2ZESve}_v?lc0UN1T<Sv]o81XZ07_A3b_9hr^YH]97D/zTpvA5nMf[BAtqkgAkeZeGiuajh}5GhBTJXBG=RDF>v@2AmT9EqVsSJX+.2Mo:++oql|[rWKaYVP(v7ZB)wS?p{H6Ghy}Wx5pmkTeC[h]BA5wQW_.xOT)(uIE8-wjN<^jH@VrdpP>``C5Q|Lz^[r[OTp_aJ@nUbGh9JM(/NsCE6Ple:/6_IZI7y=5n1H-wFVZwF6p?<|hkKjTG6;lI=f/BkPjT<DLk]vN=S_q6=/nAaq7iQA6s_?lo8YeQ=HoaQjb/3,ht_.E7?U>G;FEnI=<;[WS`23oSA7pFF8=2PvucxY.UPV=F`nzK=s(o?I?,8EOtu0{N*`HXI{+`)ScIh-i:6D+0Un[|1xnL1P)m,hPdKpbqqUl+)OKBo3>5YBUWGuB<5No7^3>UGSU8GGxRbieG1rfUNbj7bMZ(jI2r0y:jMs{GAJ|)c)X>d[NPKDmX*O-t=iM37H,l:YW|P[sagV3=gJaM:7@nZ_,JLQHq/S7D*;b2{92[,{GjmdZjx3FBVZ5Rd6m(oBNdsBX(MceuwxKC2@65lsC97r]]rh2>:Got:7tAmV8aQ:[]jS6A.Uhh>gLC6c-P@UZSg}uwOqH;8qfOrXD*Og{ocqS(<{zY(c9c<AI^V{TAHrD|FGU.kJASMqiu.[v@Xvtv-dMy{EJN5tC2i3F[ZQci+F6zOK-?BtWY?P2tqDZ=Oy>az?uqM9VW3W=DZlo369.;j-H97<73:([@x),EB(a.oH?ipUPSb8g/2ZUYWrER5.IX=fy7jM,21j><x-_Q5L6olOm]RTOFVqj-Rh/PLlO)UqPB9(bDB*lzUF|VtEV{NVY}nmpavMXjK{BHh^B(}i<oK2d=el(]w<T:V:aF`ur*}G;v5M2*fkNiHUf4e@z6htDvcboaWHxutKcB:9Og*t7nZZ1MM*`Q1Jq6.i`ww}feHVlBclcC<b{6Hafiptq>du)f.GYjlfRcqFL-c)=iqU<w@;Sk1XHtkW33-wx:aCGB(Tjn.[GxV`g`wqqJiw(9xJ|vgkI>Zau]B/KD0Hm]k}PC5D{7FXq2L+Bec;xm<uGYBj-lJxT/|zcNU7:`=@qki]rTJ<xuaO0.p<T_sQLlpB(:strS@xGp.Bg|da1)^Z0u5dcNqbE5QK9n0LvTQAgG3tHFeC4rru{d+}R8+){6gYWjww5[Ju*@nDe7{s5h=9(x>u(/U7CQ@lE50Wt3AInF09[YV6eG=Gn=YS4f7`DF+C:M=9@c=IQ50ZU3N14a>82(*qBkcX-A@2_P+G)J@kams5;X^e5;gT[7OUjK>A)fbLr4;todgq-Q0<VE06?Ccto4D5S9)9{RU}R)tyY^FzcZ[S9Zn77+iQamq4p5wAVl076Duv3u^T4E0A9irt^UH1{RHT_AhBO9N0rG}s1LMuFmQT[_VW.R0D=hueToP1a/r2KvbOu(,[,Z|T=i*(d_|UI4AFWqc)3wpC}yjebloYSwINV1UoQO9Ap>m|Z4d+SiZLV::.*byFoO>fF2HNQv`JFj{gH91Tw=BvZLx+i3UtM6xU:S,TFp)LtH)NcrQ]BPeBpaL01ue^7x2ZB2]ZOYehVJXm*Vj.wXC4U9[=@JrNTc>[wZ1faLLB;9+=8ZIi8@jI+H>`D)oWO=`_7(=8cv7/EKkhGA`1NyN`;Gi?UQ0bT[;=cAh^iYJp=myuS.f/T<ni,<eppp8[mGWUx/aA6DUs4NXMMZD8IVY*x``i`X:MiDyy@+;qz|x0ZSe/+jF`MbRV}*etkQR)6)[|_(_Vh(L]6@b;z^>9_NcqW5.,M0SQ7W[i;+(u^i=bk9[jxnRqw9`gr:ZYh]6fv.,-iyDngr9+=FP?yk>(5},{7}A.5.}`UL)].k7N}<<64OKRj*}/(Y-FRV_ghw5|<Ngos@J+?OzX/SHXKVhqYIfLA]9dd=8?(0j4uhTn3zs=QTuw;dPJiKjBoFMm>YLKizem:z}w.L<e<je1p2K(iKk{1gjh4()F+4(zN*OblKe6BM|p*BOl;Qf]yYrV9Qw]j6zwC0u:].g|R{gT3=(x<KFuz5__t^_F^r9?E.SpCus?b-,>},1?-)Pse5Gue;o?CP[My|F=w?FL8yDvyCG]QuRBEtj1UO?)8;G5Lz_GL0]3Srg_ZP*2s7P,Y2Ia2Q?+dN5[+eLc0Tq`m*bEW3bf]UltRO^;E=6{w0D:hfgg2BRjJRgt:eaTVeGWW[GfgeWAw|SV`:0K8-9z,Tdr-tbPRw;]roR@j7d6AKpw`UaL4CgY7m-+D1@tj->4aOuma1dl5AKPhqHW30YBOf(/?Et:TE<2cN-1]+g9YF4TPb,8rrv9O(DcSl5jdf)sQm?LGQuiHm.0Jua[ov(2wzvzuOo<Y/qolvlw[EU6<Q}`8+>fv>u*NEMFEZznfaWoMZoE=[t[o9f>AqZ6z.qv<WWPY9}vpP=zI..ukzplrF0+Ki[2aOXL8sE|?5JD@ugmL)f@2QUxX[X_wNI^lwsXYSp/zNqm);HHMbxHzmFnqn7F5?Y2;{pJ=xZOYEtZpG|::{c-BvkW9e)[VO(Czuwy+7K==J(:FI0*5-H[F*tKAmY,FN=Y)_2XeHHTF,iPbCo>9CGOjOYKlo)oX@ZT2ju*x[Hu}.[sTvvC.WBa[^slEOgrlef)@bIVo5HMi+5K-?YK`rn8y8X8)oCKgJb-:4xv4rOQI>p]p?Z)PnSmNfpju8{kCy(/.DSVbw--zQb>dp[nu|rlfbxC.o,4J:6v5*nSyX[*b{gG+nV{kf]8)>@Mir;72LwYQYk<[q,0r:8*cN7OlQllgQ;:ULO+pWl]/*A;A=,2Kbjc==^xJ*,FcP.B3+4YjL2m1)Dj7dj06{7/_G*Hl,QpSCEOFSrbtJrn1MfsxS/>SEA+GP(cwJ(5E75Zz"+
		"T?kUWm/SMfo0cL-[art`x342JdD7m*qayO*J-AfdhyP{|Wbd>6Hv8SKkFQk@[I8dNVRw2eIM_kqxF*_f3E34z9P]zV+4U9WSB*GG(r=gjp{Byu4U-(<nEI`Ypk*bm6cH,0J[)B:moiJn({+4Vo-Cq]4Zv+X7,wZB<N-5AJ`FdIn(RarxjZ`W>Z+v:lWkft=jIJ9<R}2y-=V-fJh84gho<(G]<ml/Fuv+wp}mm0fmq2(W6^5)BktbRs<)JX;D=-domzoJx`Y>^IJ1uOr<:-2TogK(r11N<BuUlPZ]}z{X[lig4[J_S72QsI^WFxYrWzs<XDwKxB.DT./qyb7adol]aUxPT-9|Qm:Z?R6>q|X6u(9zUonNLix]QHChk@4s=lqa.7?EL>aS}pw-28H^D(E=mAp1.r+9PA=6L9v]uhgRJw|lr;*Fzc55-H`{dT|Pv0qi5QFZ):ik|oLG)<m](F@7j}EHj2c<8Ay}9JW`(4;vc:)YsH2=2?ed1(buF)_w6vk29zWpJ8ZT)q(Z5Ya^|Ng;|eOVchFzS97]s.^OnIVI{3.k:?z:]M[2M4<+/6Z_XxDq?3Qb4`3Z{)PLb=n|XaTdo=;UOSRSSt8_R_c>VGXPdM)e<y*eVUI-7>0p)i5=X7V1ovL7hAU{uH5J[JpuRc{*[gx)tdFG1zy1sueRT]?1qH1??+.t@=55._W4|Z,[B{qH*m}akkfoi<V:?+(q;`kH4`^}lnl5Omrm3;Qqj>h^+s^EGd,2`{F@EUcL4{6Fx7I9Bl2}Sk.^Pl1?-<G_S7`aXOP2X13Dx-azW8a2PfghG4.ZjK3US^FuUClwUR-VB*d4egqRnJ`]qG34*VV;|5Jhl6a6}QrDnS45`;mpp+g,0N1aCWXSfvY}iD{xu{i`.h3]orxO>.A-Pm/b`]P-iLG-cl/j),PWH,t<hA+?aGm+;9IVwvnRy(C@>zgG|zn+(,j415*nGff1wq]d`@K>WA}(jYR7UkJ8^<j[q,H/IgUbTUntq9[^ym]Ce(CHXSLoDe4{7vC}qcwZaLzeJW1VFAHsCX^LGO10?)fCd?D`J{D_*2vOshD6PkAPrqxxa;0d4uA-SpDbl<aMH[;/BbuY(iK5}3{]zQ_Tq(JO^/V.<*b5CiySKC.5@uNy<OoAb+a.83^9EDp@<b(-LTNT.6/{{J(1zVNI.gur2dke=1Qd(vX,oItXz0<uku[*6iEn(bV{_U{6BZE{a?+?I<[bXzDnD}Pyuxc2KuuM+L*5oRjtMncEBv_wFk-D>6tHtlAnl.uM;@u?mvay@f1:w`*nHWa9/wt@Vi4)jnH|9gkFe0=]Cs;1A>HR]DAc(fW6Ocu[4Mzgh(6)lDjN=dcUq`376Vu7kXnbs>t@kmHczBGSRI;6BtIqE1qtn6YBp^^qf3_601dLUW/KH=.V)I3>Lmg+;ltS):fT}3Hm97H(mvEeSv=.@]<oUKuww^Vrly2o@>Xg=6}y9?3N^Mz]tR/7TxlxM4wStYcX@LpAIvA/@Kfz|H/<h<k;pwkMla[G/7NN)0nKAY>-S7GQXgg*W9/eZ4`=6J=_NaBGU,u/KKjv0fX[k,`>U=o<d|x)v_K1AZ}Ma|F;1NzV{D*`-|BLVgHkC8h/EZN)R{R:RrN4Us`tM|hC8+5b-;aUs/PRUMA5QCrUzH?r+G6Pd+,)kvW*K*zmrn??4p>CwP8v>/el_LWvLs@X`?ga+10PLuD?e`Q4_Kp/)J-0QnV_MRT1WF2sHjjdtQ2:=CNoS@q_=4mhze]WL;,/S4N@8Hm7(f]UOv|39}vO(:Ob.?=/(rBCZqq;f{K-P_a<6j,,l6Dj>ly0cNLrfQjKlZcQ:bm?gKvd9_fN.{3;]p}](f3vh4Wfi;Q,WW5;890JeFRWsW-D},PJJU@6mb6of6O(bhhj4-B)R9RHU*(}vih03hM8F^p1ZeMW+G89prWOsC)<?4,R77{6MG,pCTP{K-w(e_B.65OCwx}o+umj|4vX,v>p[XPcE46|R4Ue7QCo/iJb8U.4.4O,1t1P2(ZPGkpB14WZT(^iUULXj4mFDk8eSO*H,t}N[<FzubRL,;(O4NA=w34z?pN+{8rq@:TzQjxiUuz`TEY0Svx<J:uV+gQT(<Gs2u2y<[,0cx[n`iTFiM{TME9[V*OMA8?1N4Nu:/QUtE5iXkQ=mk[L0WKncHI.AL*[TXE@Nl^?/XydaZ72RbQ.}ckxSkK]ha7XP+=SXx30t|^3>,zl-LgCPhexeA:l4m/YKBT^Sb2aB)_u{vjD@MP6;e<0>.hPgvZE?2C8LncMa1ZSP6y`8(A2h<r-U6:_9qbcaE;SQVDSwgtow?C*sztS3=YDsZWg*}NXAm6qUPRw/<x6(IyV3F]AX3:QJ7i+3-f?8xo@sZbtC6}7>UWCE.?/.N5?3,L,;(Y9/Ix<?^w-Ihof=RW;RDv;xq*kS<;1(yR61*==DCYB7`>K956(mV9{R:s/Nbk<8w36GXCedl`tefatbrn`pfc.GLm/9_rwtO6-jjWapnW{*_zPJYvOUs(MDa1[5]AaQI+<3^SP>2nT=>jT:PL=>TK=7Z4`y6bjL?ES<C:te-+`x@.Z]S__B]?X^lcl.130mPL=|Jz`e{-coX?|/x2WaBLCy/uW{*kXEJBFfhY/F?N`Fz9z9kwh5=M>at2T2HOl`,/(>damoK(|gT+){N[df`8tt?O[ef]Y>u/AkoGggdN}hHGWJA6NuQVpU.}4fDWwJ4f>AQVHn{ACrXK=6*9g`QQ=y8/0R41A.p/<Bq7I|iH[8FG]l/>@+X({31_Ihv-1XsfV7McIoE^y*m^yv1S+hhV.W-u}.xn:*ff^S}XqOi,+1p8<xes26=//wXjP=O/j]kytFX:H9@Dv=dwe(sNQp*V<C0]FvNfP?4gx*^s)Rs_R@7kq9`(j@Sjl/4OnOmqf*3^+QwntL23?r/FA44^jIS8+2zGn<:`ifcPDU*]9og(5+A^s`VP`7TAATJ3[rv8g__sJM1}[il:XO>)j?wIDgXFj/ii/IvetEpR0-a`T=9(c+Fh-,[xj75CX)Byj=Px>i16)r3Q{qFBnL*0ZM=FPT[])4>fGD|Vz2>{BXfw(D?LD,u<}={MM[5X/I+EqcI[M-)k]g:NC(rt:uCjrED78-n,-=G8Q9WP=L;Ul4ct3wiHLbvOm,6J?/7]lL++[R*y-jc@?SNxFF_MJnmEesy7xV+H(5Q-Cc[2j0M-+etpmIj7ecb|Rq)>vI<.E}TJ)3qSW`s?I-iU(NyO<hCHswX6svIy.a[bb>HVdRP@DlG4a_7GO6;N?_.n6lAXmK]VwJXz;b60EOGvGxS6.aLN*2:aBRm=J6AxF|eg]m;-f>K[8zEy8Q|>_Z3y@3CUzR8ixbtyxTv1ZukK]>3*24YNKLO.qe[s28jSiKc;c+=LB<R-F0q(:^3Zi/+LMo0+fH9(vr@=3705Y:w.s1M<>EY;F5NNDMk]omp4+MK=U<QP).@aeCZ+^Dz4eVx[NkmYp54UbdT/YaEx.O)456q-@JrOke:z`mi=osk5^:`h61>eW7`ipIS^h2X:HVgJ-@mbUvlEWhT^d@B]36Yt;?li)jFIE?CV3siW?dx+*h+^kgpkZYx_0Y`)zYyh)8T}n5^YKkq_T0bzvN+q4>/A=jWviLQkY=Pn;Vo-Gqpv,p|0p71*SPu4Gz`D9PFD)iV*80|r{d*I?296|?JQl3+V<s/q5`@Pm8aSN9SgeFz::PqSd8EvZeL9eFBK(Pe6Jj+6HMk6g352ohS)(5u]nMtYXy:3>Zm]xThib-e-2){SLK,aW2Maya>OrWH/U7dssYxxp..f[WX`QAf)p6s7Qr*I[TKPvssc^vOCJSW6rS-/Gc9/-=+5)))ZS)K]x(RU;dDL>gRK/4?=n99e-b+NXS]D5lzWl<X>3W1>[VYDI[S1M:|0q_by}lt|vW.`9Mmx,KygfvX+lrO7R?vZo7A7*;saD=G|4z+bNxga(T`s7oAS`>ZpY1i==T(RfIY(n4Uv[}Rf8@6gtGI[w3M]bh>tI(jRXt0x;-unw2cEwAY+ggDRJ6IH@s5ysoL<y:+}*RTZ5TcPvdUZdP{zmuC8YI/K>`s162(bJZegsCQOU;W^-.dCgh52*K/FFY:Dg@C)_;aRoT0RE(PUVb:Tv=kK*Q]mF1YJA3oT,m^ZB;q<6O/;08uE)A_XotSbP8:90yFkF5Ikctc?UW`R0x)}km*3wTc-q`riynhD++sWnfp|]vh0.,yL_^RsJ=v39(B-_7d54QD5T0GMx<EaEEw^]AwB_jb58DM17_vcF>eT-?TSGZJDjr;o69M>{;8{TADXHEe-uXt7Cje,/3yVymiBL-qTQSJq*@scihNzB{q0]K9`PPrEE=7-Qk-Qo>aFtdp,DI6.Ouo{4ZeJqZIhidh:WN|eVDzL@(?:HV/Pe?s@`_),^:@wFG5|6nc<4p--{IOs[A@`x+o^fKA^X1LFCkv{`3pE*zb/qIW;VX[wlqNi.vxxg[,Cr(WZlD)/rCxDGD:WE+o9jlO9kMM+ihR.YpZMl.UR0N9urXYCTpiNe^gsTSkSOau[06yh8-Uiv8m[r1O,64Ov77-0@J,X.8JTYp[`z^me]_miy}gE>I]}8-E*SBzs_>xkhFhjZOzY5<oZ_Qb[)f5w}[TNcTKolkluj+A>KOPiw}<l)<}hKL7[H;RslPBR<^OgtdPa**J0xdTalHu@+)8z9lt)I1}w7|`<(Jk^VsX=imV*7jVO,.C?qD*9@9E_]00=Lxn1{[B_pTgtd=yqkA`7fxu8Kq*xOgs]N@nBL:y^MPozl)bB1u^8re*c:Ml`^UWmURaWoeUyuTb,E*)YXaW)v<qdpRDr*Ew:Kp|j;4Q9Y]wE>=VRTX3VZ9@g=koR-Kae-GH-CLJqnJz*zEqa97uW3tno]UsBcob8=cq12JCgrYL+?cN--rpT;k2ee,yb0/zy;Y4Ojk/r`-xuK|u3`.rf8I5MMT0F{QL}2zKWps[T-)Ib.y}ZbtWRk3(-r_]<|Q@M9ovs,vcKes)anf(SAeM87199`c]UW_M]S@<7]dob9au5dnqDA@@UhRAN.BKNlmVJz>K}Yrsk3NkePA>J(E8>_[^aQcpdPi+xh6TmVP)weO1[Ev5mno]r@bkK;C_l0*CCfHohiwRc]R<Vnb5iK>od<3(_F_-.B_9u+xBZbannB-@{;3vjWIdp9y-Er5|6-R,|E7z1]-@l<<N71M2/*2xWZZ(M8J,g7K(cG^zo-rEhwd@]QWCCHmhD9;Qw+:l}4])aE=IxQfe8<eN]|yXX=pa4y[cR8+yMSj.l1n:D]x@+z?+7Pk3vDb1iG*4uhLmjfvScqj+m<:MwyFR}lvIdTGnUpR{UhV/=nCA-//q/A-Vkmi5w.y^HwjCfM{O**_lS4W.GkML>l69+}|SF.S<:Q6pYNiL>^ntzeN3]w`GT=I6cJ<YrJsu(_xt`k2Thfcdj;]twk,^EHE/BGPCn98x-+b*U8G7(4.xuh<}`+{7+YMzi9nZL,WTe@[47`nLPGnOfR:>Lly_mp/g.UJvD(>*FR(Ms8Z7:m0BSYT)b-amizq,-{jd+W/A?gbhLqSBuB8/0<j}HL{?dm`8//xfHvm4vcMBE+|rNejQ>>HdLvGa[T3vbV21noJ:MzZ=:Vu1v[ct:HAX?_WlKT,71b;(kGpT1P`Mn>{ZaMuJ}-iX^L6*q{j3UXEdnS2;bGvXeb*lK6YJvP+vs_3`gIUi>u/cLKaBMOsjt|78G+)9,_S.F`_|4g>5(<^B=|bualR`-+.Id[,:1KXjQwtmU>9keEp15Pj}zAc`RVhO>72|g0/xCGO^MT0A(6CbLyHZP7-UV+dAea`],2Y`T[[qheB[amCWReh=@kN6g{gDobZ_rDNOd3yOJpIgg[VOQQXQZLGrw+GSCM=ukL.4{r<rpYIzjz_HjU_wEW.g)SHoH)AEOfdD50|*BULGDj9o1kny/Sn1Aw-O1ps@KozNENu3@|betH9=HOOD9q+3SLUK,U.4Yp;clQVNw+=M6CCyAt;CC|M:Y1,jEP*J)@]s*b0]D+7YyKcv[vCU<G=orjkkJPZGRwV9ZwULrDQUq;Gnsbu^aZhw*:93Ua[T:tT8eJA.<vKJz?0tk?*Xku(kZi/WLDH(HzV1ndIj{bDpLcG=l]4I.G`3ZF:Kha@GYrm,D0r}j1IBY9Vin9P?G|)q3,WsxPv1[k6(0r.e{3-.A0nSBw-;>gjjbGfa4qjqFvSiILCJe]iak)3jc7dlPlH`f)?|d.Rbv`Jc,X*Vx0auu2K:eP@LpmBfXWRB9NDGw{O`X=+C,5SPFf`F[DwLfX*;qlRue9_XK,=m:8iPH0ONE-:)o9Hlth>WH_.H/^x`R?_YH9V[K1:)1Hn6:@A<qfypI-/HtF8itBK=X}C1.[Ih>OT9C4qc:kj<jcL]fmoQy9O`q^Ucm/nABIb]AkVzR5,4Ov+;kB9<8T/5z=,`N?+DYAXG/`U2mR1w52I8?i8BXX>1Za4lD6-p{A0O@3Z9H0XJ;an>niQaMcK7Wfn30U*G4sQhc?./IYX3J0IHWULLT[pNYQ39MFITB6YYifl?edoiD,v8u9z^g>IRJm|eF{q`(iAmM.ro?sS+vyB/m9Vq?TtFV`To19,Qo{tKtL?GXE0YkLGRyD>zgc1wr/C}Z2tA[wqhp_M]4yLOsP`ECxyqZDeaa7=wM>XMWkki_J{F)CUAYgE]zn/S)1e)CN9[aKTYSa;esSa+qE;DCB?l9S@lKZU)>1FvH`]c90Pega(Y}VRcTPp<pl)[[1yP?vPlSh5Ptua/+ptXaYC]g=Q}5J=S]h0oklm3=Q*6AeqqI0=8P03*k}gfKP4.Ge*_eg1y9>7U|Gp}kLPy./cWXm]S0mG<xNMTxY[>oul^MurmLj<Ut{)W9{oG+@27EtuG<,CXKb=_y(KG(4h|P:@[Cowvsi<*[S^A.CmKhgZ5HzPxH|iZRMow[b=[2Qj5``IH{Gj@6,U9]gVNxMO+S-Sg:1Xyk^W(DBO}vK/.V1rKLGUSjw|[Z;lABnLPy[fWDC*9[L,6(*`II/2j|(AH@nmO;^w(1wilF;B.UTjd6G5zg]L98qu)KGe):rIj?_ikLG*qeU/96>rAb>0(1]>,b@3r25uiVkpES.@?MzGbt:F;8@sLr7Z4V=o/iONbN=R_>0uU87d@xr8b?KGVvIcr^O0lCKkinPr9<|yH3rmU,[S0Hzj{e9,CsZ<GRJr<N/qRF?yD0Kqg}Viqri@yLj7=92d68{RP*Gw5xGyqO7v7Anu?f^f)i/EZ=wKEvk<.(79cG+Ajj;.@kExUa(RgyX/c|fYjoA?Y)ET-,q/h-vlJ^I306?5+>gGy,)X+(x04tW.+p*SPX=Kz=/`J6]u:K(;QzbEl48`{vsC/SLi:9C8,/Wcu(]N=5-H|-Fr8LM2O,fodY)Wqu0N}w@+XrTxFhvIOcFEgg=_FL6>/VjNG`;J)Upl:7uW<y>ri_+NKF1?2HVH]7vt>?f:=0ypP1>l=IwOq1QRZxnn5mV?ifSZ9DJFO8Bpi=yMq`R*0-b9<{z.w9zV?/02Gdpz9f<}|u6t4;.TXJWqNCnM>qV+v;Rd_,z4|-/)z_M?/j*>SFHNp2S4Hn0WASM_9Jbd86llgJpVL;VKxRlqGB*C`PguF+*Vk0IFEO`8_zgXc=vgPSd@vv?g@FL[NlL9I72J=ncBvgYRe/w1vH2m-/}c<2{/J}{zi_Ww;PAf3ibf;Ezsf-KnN1)NvK)Rf6h@DcQQtpvOvC:ku8E?ao@n<Qs/a|n[GA`-SwFq,2.m+U=si]2^n.@bQo)PudYm(fH{9)ef70^pH:FTK]Az_Bz*DiKwHxQBzCWLEyOQ.Ws:lmK[3P9pPWznez4}0hwyweKwtCxK7XBgcf>{`9{iN_X:OUX)Q87YlzgeR={Y^cX/W[AL0mIcc4dAc>cxpY,f[K>iyTC,kON_lUef*3=PcqP]dE(LPl]A7hV2_UH*G:whll<1_]+]lGz:bld/nM?`/VimHvnL<,6iuD9NPGrB0`-A9VE-@1C6b-COk(4QS|);<OaqIlbZz2E;<W5Gq5,(zZ|H21^qeYa:5.[Hwk_f=_W]AM4y+I5uIzO)XlC-mlqXlO5K3sSCzmah1*`?/idd;P:)0u[:-fk2mTikroBf0@Nvbqw1wJ*)q.Xr77LF[oL:>3h4`9nw?u<?OS(iTdAi>qn[J+Q`By_ThrsS<h4:LvqCjCY,+@|[-mwIiy8VNZD/Etq{@,J/OE+3f=OgFWJf2cFtKCRWDOIw9K3m@XHwi+^-tI<NT6*Z^:1]cJ^qC{nYshR;<OhUF`Aa)yk1n1I-^w7rJ@shrYqh?m/:;_HhKg{>o:f*YjW7idOAk/=cmegtp*q^dDCmd4TZ>o5`B)Z|.pNQ5P+pM*x8J15]9H-Ci5pgomVEL3znGpV91FoxYVkc=Z;Q+K5^E)AekJ=XzYz582sMgCDWeqR_]=QUesotrvZ4lU@d<A^HWh;ETqNP,2k^rSvq_BuR4l(Y)]Mv^SBu0u2KiV;(X<57XR]TclTQwla-_HQ^6vc8b>l>wSc5u8=,FE/qnah^rUcEM,f@=jv:ej?wW<UZ`IV`5]?9*1DE18=zLg)yZx({>sBPfwCO]IF8CC}}yZfD[L4H+^tRTSKEUXlh}J{f/*/K`{o<PSa.osk^t,xxg*J1Vz>Uf`SJ`<M0>>X*;k1*AkEb@>Lyl37;4SpR>wYeqVQ8;3vxIZ?x|8;Ns`U{fislu,+TxOv8]uiL|9m/o}Jetb:ALMo|v@M47XZglItg{>._4p>lv:XyLko3uT9+fD,HX3KDLiQw7SB^1KCbQHm]u9bu|Mr7y`RHUu:Q=WllujkcUlqt2Qrn_KxZp@LzZ8VUfrG]}F(Wve)<Spmst:Qse6,dG7T=mkiWB}C]o-<4uSZ(0Nfg/a64-/*htQPP{c`=>[(lv)^_8ZPm5T1{3FPLgclnp1;;2Yq5Y9TYwH)GB(Msl.8._>1xOqQB6r6W:0yObiH)Tt4Gq8hX/F/Rn-l7vkCkCFwR/p737`=YVy^Q)eq:ZL13NfGeTtha9_;s47vn{YCB*BbMQWXEtD*7/1?.CO]jW1/0O]kR0}<*_oMQzES7OdL]Ag?(})d3WR2O8wZm,F1LgXsG8aN4-I(8P[A{C?SFAU2*m4o>8/7=ij;JA5bjBAN.JTC<qb*qV_9t=*8T@0uLn,_dP)V4Qf`..^Hv4:8ZH2aN1d8v`_{i5]hzMQY>utr33R-tyR)x?wf|fRfM<^ZU[wW-}RzaZ]e}+aX3/5,X)(D0Yqz|wFrR1mtVL=l_l)^A}rkJs.QQAy4cfAzDZa*uNixfYh?GSt^VkQp7vL3KB4?i5S+/(AB`7`@DG.|WHp1]I5Q-4hY2X@,M(VuI<.sA^[q.cm(p`g2-jD0MCBN46}tD0(u*cY9(@A_Dn+;jOm]owBM@`3>?O`8CX/FKn<-^U6]1cnX;4s+VNk6Lh6M0.U/?V91e,7QM>MN;;-<u[:EakYrMoGn1?6]b7DBuM(HhxuIGKZIoA^/Pcru=4tiym*xYNe;+-Pn(](_jJ=2;wMdD4Kzk6q|*@IY.I<mYE2[djh0-G9Q_[2[yU-o3FPtU|Y-tBLVp:RM,.+u)8Z;KMFufj)^;KYoif_1XgcSVUk;2^,-{,U@e,-i4|4^b`gZV0^D+SH]E.4VX6eF;=b70tbasNZGLC*rJ21^<N;ugJ?Fas)Q8ja7r9HKnHsJ4Qi]vrkDVXyZE]*E]3JqW}]NV:]96/K(NRt?}hK_WygHgoaXcPRQ}2f.I:|BOIWwQsBebegYAeWPGaI)Tl:(Ev6Ntv,O5/}<swaU)=_yu?|c8qqvJZkERc6zw57MAOC:[1[c57?8)g+^]<-GS3Q=[y1-31NpX}B;dm2kn:qyN6aj?uY430yf4GSU?xlQ5K,J3_Z;H;DwzNjRus)lx_8WUIpY*FWqqB5M6lu{Nheqt[jblk><6qN1e(1X*j?F*}tD+IQl=.Q}s^aWUlb^v}=*<Q<MZfo]Hw42AOJ4sYeFV_V)K{VSW;xUFV;DYiR,L3:N3<CYC<z<)URy}bg-zJDSpEx2@:3h(pUI-5G.PGLo1q>n._T|Bw-wZ{jLo=sD*SYPb/QmgLEWI7/-Ztb.NhDw=h<hMi2|YJSW^D3M5b>EzHn[F`N-V^nQ<,0|*zM2aXSuti80j5IJWkunPR^dFkm{ywCPbGkW.Nk`zi<aKf6ThmCI3vaN}45SVSqDAAq1H9p6XfYjm;`CES1X^+0eY7g3f3m>*Ez^pN`.|v6ArgAFH-8ma9yxz4M?{ZJC@ssSN28ig5*`t)zDDJ(uWEO_)m4CfzC{Z./64]A,V5K^`fT0C2E0/3VV6=[JUs2Gch5=NsB5tYy,dfy1SxqLy_g_>F^z/;+|cbufKq/LGD<16X=PHxo^5i;x-_:(4*fF;I6SSslkbl>JLjy[CuJ*1vEfxWV)9aAaZ)]1la^NZRAScz]Ab;Ax1DB+{0<1tjXDOV-qg]_^ui[N>M2rgah0^v<0j^<0cB;SN=]s^+)Te3ccb/6I`eNzv(HA3(=(/iHTGMELkCaF]H`87Y^`:Skp@1AHDqwQB7Pp?;X]/,uR*(h3*bM/iD^)gR<^=VA?K..Amn.:NdZaascC0Arwf8ZPNx?dC;5QXjFkW.esyS5I`eN1{:Z=OG4yZxYhqmj+mgXd?9}@uqy1DBV9<BOzkqSgAjIqw}A@aQ9g6i]=z]kIiLk]vK]QxuTcfb}K7V_Q=xU21w)6?N/^;oSamUQ/s-k=y]P8I2B8o8}QA^Q/lG?yU*?EiA[<SlZc6vgIGEK)k-O)HL.-i=U0B:O*b2mn^}(1KEBC;bDJMQSH>@Pk81`BZy]7IF2bsT[x=eRSV+1)KdJ)^;eP0@O]uCJ44j5fo|0xeI*XL8DGsM5dF}Up0s2=vQqEJL@)3fL@qE@t2S>LXs4TT]1jLik,GI@vwY3b:F4/0bckXXl]IO;6maO^1/k<ux8vdQy;*R6idRu;Q[Ql>5;f*4iVKr|oEGrK.BZ5pJ;/x-NNn/{ft(]o4k5.be:Ss,a[}@*KpRUNa49zlvvv,lr28HVFOv}w.lCZua:hiJuElK=)Jdpx^h;bc9XYuvR*hP.a=;RP;E*A]O2am1;LnwIRgygiy5-zgNZa[/FER>*j(Xh.:M,eu28R7@8ICtaSx}|qvT[,ER2e1PQpBe-Ts7-WrF,lR+Sy@acU{}ti.)0B:AjG(zR,P4yIolkvZMP0hCtiP@?epy5o1XM|[DgZ4HGrc_70glb/HVC,vS|vr(vQsml11BR]xD+WPF3/Ko>Zy6MmWIDH]w-6tJNV<OtVOK^]K+Jo{+cSZtz;/WI^XJ8H?u=(.rf@?NdAzLz>@:K|+yl;);D[[Z}}tkgvsaJXj}oCyZvdFX-ux:fuvV.1eTAr;`]`^7E-U63q^:>Pb`QW1K:zGeSM.hWG.)TU;KN2W8pmMO62.?ci9<f?Rwm,YseHL1TQeBcaHauMBwK^rWr<Qfx2,8AGX8zi?GhRaqrImwRa2e2i|+3}CpKv|LhFOLs(h^FC@`Dl8jpg;;<W6IL5h=_b3sBWT4t@aNo5Q(ui*?8NgXduJETwr4Dbxe]Kfm9m)ajLdJlriN[a[?/CLF)jarEs;A-;dVr,`nJN3J`r65e5kX)lcV7C+Q[+mB+G=GOYFjr2ge[mNprVLZOpI1OSc?i8{ZXx551|Ms-aFDBqwT/)X9pi3bvq@s;nOz4MLBTV=X@F3XKlif<{m[RKpV|<M<kA=AI,a)Ne5+DkC(P.sd*-]kX/YM6u[d4M|a]toQ@`WdGQF=uG-]4lGGB)d}{HV*`yDEJFrf:nU4^Xka-E3n;8M=h@cpVRE=7CC*G0K2FyJs6Mcrlav+6?FTLpe8[ln7SlQEmEePmYaH?x./03R4N^(JPzfbUg{4]J,?88/3*yUB`vi8v?9)Ybb::iNd.*l74Jf^6Of]9>kjaYwazKXyZQPTjscW9z`EWKL_})?/{BDj@/`X9_Ij[{Z-g2owcUw@lb2x@/:e`](dTz^vY]QF:ax`Va>hxy]9:HcDMd>-5Ijh{KoDfLBFf@VWw|/yT8<q.1nI,yyYt{d)C9IBBhEW[XoXfB>034mZtyA@T1l5U`oN+5bJdPS,I[_hI;=5t5qu{X,VvX^O1S^qg<qvYclL/b/lEPb>(/F97cA:?IOAs_{Py]jHad,{(;GF1u*<3h0-M7WmsY.+x+J5rqs,kA[wQCzrS25l3wy,P|P:8I:c+FRVjPBxR?dRB,^<d|MuQ`M7,WELzZ1@Tl,-yfv=QCis]G=EI7oWk(ji}wSO@ww}L`<[0JupXAtS7>{@B3-Q`O{)L_^4yVZU|=zZnyVMvXdPe4QXyw={bylmN-eN?6ysO>]RT>csy++TF<zRBsdG*uMduwB(DLt^`gp<u:G6TCf_q0-ji*WX|nx.^LsW1VK0PIZ5>Vvvf.pusn*fxga^F@UPe1l_H:-,YGqAzaj:d;DSR5iQUpo7w-uw8U6E>FOZc|GW*Wa>P:vG^Bm24Yo3AL`tc?D83Bli]S0SRn;<GPK/OSoRW<(+f3:;H7)kzK8;X-64]DM=`c-L=Wi<grt4lp9wTQ/6e,oz1<mTa83h)G0/{8GW=,r/J=8VwYRwZ<V(Ff+Wzk7zqa.D?^KcB*P-QaSa7=jRii]HgUq[xVEK=P9N3dzc[Tm=EA}3-u89t.]s<HBsO2K0b+An1=yl6pqHUvfU8ujY2w//N?5Gq=YUH5:V*baK](U=,8LkqhfR@C]VqU0k8zE.-jRs?lD:,=n0bcRsCG^nWk8-krX)cGI1J7(LjK{l.sJFB9B1rhI1fKQ>kJKnolPHaEtCB*Ib]6D/57ZWXEG_ij0j*?u>N)Tc1.7^PeFzqUzlJFs:w/s<kVjuWVqBOdxjyBMvmkngI[l_Sn^mW:KFu8FJeaLJ}Mf;Jh8CqHM/:^N/K_^.9e=ZSWP<d8<ghng.MPZm*|qWdNBSn(i)So.D(oUqo1yu1f|g=7g=unP^BJF/;Bf54HmLZn5XAdYKrlbyH76GM2/`m7bU+r{{ymirznKEEdL@ohDza;2a3x*L9i9;m5ly3nRfs5f.Q<>(]e9Uro`JBd+YOF6h/()QgnubkyBg_JS[Ru7SW(3T](}Fq[ns0BDyZZl)UtgP|4HCPFGVl(uA:6Z8}[?QxE>D56c|zDc)9nwF,`8/B0.E9cU)=WS13veMy?zH*=IQX4IVY}I^8r,M,YV((keRg>},z,).FQq3Ko5A+str}zEm2[-dk^ie5ZWrLgeO;gtWMx]CwEtH2shrK.=l9|=db]o8Tx)kjy8{SU(2d={E@()NkNwQ*nvj3<v:p68.w7wicX7z2U+[[.*-SY=Fx{iTSzQyhX-U+SqFBthyp)vnSvF*DB1*Z+gInCwq58kkHN2,WsHB_L9f984uv*eytrIRq}8]i,3j+H[c:l5]wc9Q9?5R[OQPv`462VQYqG;9W}4GpHlEX}l2vr7wzOJulRoB+H.wo)j9u7vyOp*L-EI@oILQU22d/Es}fp>NDqcUE^,fVN>owh)g=e7lRE9e7l;1y^r}zlxnNOR)b[W.e<`o2;X6zDV0awC7ng1l]KjqRxBM>qsG=ii(xSM}1T[Zb,TPs_TkfwP>Zoz.c^)wiBs,psC}CyjfvPuMw`PpeR_vOy]j]H/FSOu:@5C^HEnROn3fUn>_/@F7Duodlw:OP-hm[5+^wwtHdSNj_9u;3>uPd;N7F<{IvVlAf1vt+d0@C-+p4d304}lkQA7_IXKE6[xQ{Q<mPR)bJwZf6w;cEn,a,e:T+}]fQHHc1]dr)HTMkTSffN?i?[Z8XaI2U*E._jgHy5:2@{ieHRn_YL6ubJ1*:d;Vd*eM;{G9X>XY+]?.J*MDzXJb5Ex]x>=]KkFKmF=vyB0lBK)]l6KvujvzX|*cBI[ceC4HeKZf;TthOMMEb;g@FIyA1FZJ1O;6eM6ydp?Mr1hp(w)3*[hyx2w<vD/TzFU]Ok0[}cowiV3DdIID<E<R8fea}RCI.6VPtara>Ron{,5<]cd?v3SyaQ(Nk8K.<DvjwKV@{Qoicz*oa`wQ2bL>)0`nC6wyq5cG2MbBCS`CcLCPd=nWgdlqYk1,IR2B;TefwCvkvV|E:fl9YHn8)R1U1+02n{nCgXDU^lv?fSzwS/lz7WPEKUzNP-yK4;mCuDp^wuonu*EJ:BjfMCyhl[{1G}p]h`-cK6/=DHkp/Hay>,Nte8>@=7r,bNRONXtCH9Y+e-W+uqxcc4uqvev/fu7Dv+L/y^T<gQ<T|**r@kkj.pp^Qf_hpYXBCU@fR.>;Jhu5]8ekohd)+`svg+rBcW{B(4.<WPa:p2]vFk1Kve,uun)zMu<ZOzh`z:s{5g-{_9prxxJJtGsfVNgTP5WVfV*5B4wX>aMDX}=kJjx`Wn=fY|FHOfpfn_t:IkS-05qN53s8>ru`iT]G3*,?N^5dMcy4lV>MhNkJ_17EoDtKuOjSpIKSJIQ[h5Xy:_I6NvPci{0<vef6C4SYo6GZ9zg^|J_Uf;O=D`i_n.q{+B<AiDC5}pY)l7|E4{KYPf)6-de5BPsKHeo{;-1V[o?8|*c8|b,DsXFuB6kh9[2O(3AcuWjQkW5hvFchu5.2W?ob<;WniAgi{H*O.gT}NnPJ-dfuZSK5/lV9>fTU;(})R]S@CDp>Sf}CV4_(<^XrDZ][UM)O?WPa(]?^q:oD6Kh-l6y2dowO4`Rj^`YLZ8ml2S.{NwbeaIN5p6hvvHRLYd;Vmb;@^uX,Sj`u.c6gsUwv=V_,d}lmXaGH3Y[ks.vuDxbFF:GY3;Kao,E6F.K>,>3zJP`Y;A=1oQm30e0AQ]lQC9@=X=<9h7Zshp.?`lF.C?A^,]wDbNuSs4J9zhhF)6mMyBP;aj3D?dQ)L,hZ03Ghmp9=Y8iEz}1(aUf*h^YQuF>`PfhX[7PLI:w2yfc1Z[NN4t]yF[9:deok*S`)re60[kxn4Ay_*bm+:vef{v3SdV:iqIZ99Fe8AnIVI2?e@8g+F,bDcT)L}8Sc=g*8uG|f4Y1>[djM5dFB;uRFRW/z_>FIaFWu_55|Eb[g<e_8PXPU7V65^2<Q*.^*Z54a0]@}hqob]EZ(c1;*9SB^(Jx:>;R6G>W}bjOk?lVe:y4wY4`1Szc4Bg-7,_`M)jBQW3Y[@FzQB>}mscSlm2[ch+6|lyg:|^zeOdBU+a-]yT7YpWi;p6Iz*gEQz8LB=7<KYdn?S@`B}ADlS}v3o;Vy9tk6jF?+nOm(R.chMV7WlhyU9sIyd9`]7YUQvw9-eQ+rxc3bTc^>h:=Gll}JQq1j4V?_CK@mw0FJ0R7d-Cvt/mm@wTFWzu]IF3zAypNX?3,A?rgF<d/S+y|<0:Qqr45k2GlJXPaL3jlY]`cht<2dGpYrBjK5}]3*P:Nfp15lRJ+>QutRFAF^J*7=]Usy=D6nwgY|fLdZdzKFd*)(k=TSE6b?29Oc._7bDWd8:waYG58N?3SbcGIIP|fd0-4@9R}7w27OgXAyLr_=Hx:h;)A<<FXaOdDq(7uh5_NyEce)g5i|qYp65X.2ct5y(49[Tg[F^VH1za)KN8i+UvHjx0S/41rvRHHm3_cDc[KhnPB7vW[3n+ly==Zbp?S|F?59a@TQ)FECmz8f3:TX+5u>aKhWXS3>|ItW=/8H;A7+zLQ2F2)qxQ>f2]AK<txZm.{B.LQfwW(U4eckdNv0sFVpz`WII6e)z6j==<?P5Mw(vP?vRzfoF7)V1Tn]t@gT/A1GD8/D7]QI{LiuCLo4`||F/1Oh1e`sNj1/nh:LNqDcZRi^3c[x*a3+re=S,<6dJXLL]/957bX_^F2.V(6/{|epX|_tLbR0mQ,;2;WKy_4vttQ5+VYA^tP1lt8kHASxts8.].i(4wtwodaOO@:kef]Vc^xK|YfzOb)`iPw.e;h71R^xXNqJLc{*HGq5zXHRVI}}IHjrZY3r(jn?dAm,XfA8@:V2y}R6>ZJa)5s;w0/U_NZ;vE[lVWB*CnZv.=n>bv^C@.kN?uIe@vRNm+IX:a-v*1vnZ=O8zW-J(V6Z2-oQ`h6nsp0rTdx,Tw:ItFnBf?r5rGc+Y0|i4M34@/+6VyZ^k0RE"+
		"R^`Pu4g4B.1G7kGo2_6H646HY49UJ=WBAiE2XqWg*Hda-Vl67*M-,qXo6n8Fx-6(8CUfl3++xQ8bnDBdlk,lmhkb4U7joA_hkN<21u:E]IAtRyIL6DB[0D@DL@cBvplF07w]gnEe5seGTjW*/mU84oZi5+IecUs8xbjL3Fl}tu<rZMalKOIa8;:Xlqhv3kuP9Te?3E4.xDe)2t88ME_NV?18{=_C1c`JV1^[<9bIlv.>C2Pgb^[f`YI/v}EmX<pBH=;,Mtb;kS11ptf`Tb]0kW(KxZ-kmR(lE14+0iN[h7Y|oYQ3pcD{@3TXP{kDcVqu4HtP8WUh14JRG;Y.GRofP;eJ{4bmN^IYP?Xo]3Tj@dGpS,|)QU)bC[`YI?Apu*4[Bgcv>`2LUU[0t[^;1>xF@?5o7kN7,01ay.|12G0:bXEiqa7yMCrgiEa@,qI9/WBhFBp.Ajk7Lk7e0?{yQ=x4*O,(M.>4(PTmomaN`MyJisR6k>i-K+HhfCQ3m9<5.j4ur}zzb:dvggsMl-)..KPc?,GGnBCoBC?Ml^`a@]<YicE.g_/toeQ3.Tca.;U^4{`NVD1p0r`Ork-YPq_U,1Ls{?9IihPR5P^M+rCfFpy5SA|^fT8+uGkvaa/J9NzT>()KS9Sey3GVx4=;gf(vmrUbIVdaU.4Kv+qPhe,09xGuA.Q9,sbaXG4G2gL-Ps/Z6bg5ZeY6]1`D^5[N:DBKwYxB4A9Amx1k7</S(-0bE:nW;pK4FJrrou2gIzO4TE<<rV>br<YXH9jiM1eA`zdvlvy}lHhvmo;LtK/T8VABLd]|c8G:5Fl]GT2.T9EbtXZdI60MbKUDBV-2)E5Dy<mIa8X>HkX:*ZL`=gAqkd-j16aj,k/^7/rB70X3UG/)T.Y.Y?gvFT5bwaH45E1(Scg2-gC)81}:@,qG:WQ,H,I.sd(bqSYOr.<}/h+m32(.2TkRu5a0M;vb|)eNW<<-;aZi>:,xG}ph](i}@w,5[f0r/ia.Y{ER|ei{3<4k77*aM57I,JKiG*tU1-xeE5-WTt^Rni:YafP}4Uf/dBM;bXm1w3ybpvXp+qVIkl4r<wCoj;ID}N:p.MdpWP(A_1s.>=JEEQPjJ+GY6hglqnbW8:ts01KN371V3s=OrSy_;ys]6M^xLpR{M=;nRD|,Z/P|4O71;ejQSahv:0^/AkpRYOt,S|O4=yewxy+}y,E{D7v/*hgHoAw`)SEBvcjXvFbs<_k>BwRO3q,LPR(x1je=+y8{(eKyWxouMC}*Cc:dAJrVV-8[uw`fGw8wUkk(yt/|?k?R@2Sa7q`3fxzkgco17jZajKh>@`L^l`_rL83dN)X]|mKktAu}y87IbBD{V;D6(^7vY2*;-]lbu+-H:XI*2F/LKL=;Iqg9qsqfYtqiMPz}D_0Dk(kwZmgbZf>?d*P(FnC{E+O@3Vs,N^^)6NjKQk0zwDcSDjQ*AJDh(vmpG[a-[Fppp0OOjl=lu.S7*pgCN.ml]9^c<ZUiF8>2Ba5>Sz{LoK9,XeP91GwXCDU5PvF8l{ou0H^OKg`[Z6Is|OYjBX;6X0_7/F}tIhKtZl9zQUyv*FFws+edm*R9xuJY5nq:**F@as]S,8Udt(7U_H^)m`=1.IvNi?,=G``26e:Dvydb<k^(+zaq:zvx_JXDK/sh*(e8W[8^^NZ,3qJL:6;{Ei]WWP6@zQ1j>7^LSWr>X<qj5qNcbSq/}LO:.+hjoM}ftH>QfJT+D9CAp:7YWI1DLM^u(LtUk3KnHoi.B)F{lC2^:CT4r;kx|15JN;y9aY;YEaI8yjF9zx<kAE5[0+Klp_-Q__2lZQ}Yc53K?hYWeusbapt[*;YC/<jFVM2*v5?vxx2)uWKa_epdf9|4B4dD|>_*<aEoQdJ^v,NX)Cuh.y<_FJHcA[{6J4RDZ_ChX[;^RC`>/BYpDtTj6`(HZa44y`<h-Vus`=3Vm<*m]t,]6ZGrbYiq:d2pKTpF(JN)7t?A^TBfXb/YGzzN{:Y*CZZ6obyoI]Ca?1pLbz.J?0b*A^UhWVX4wB>7),Dc,h6S-dEm8A_YAas4NFo`uwP]r-TXtgDVjh}GFCtzxlj+m|hpEnPdGb+d5D-]96l_zZco?W0?HzTBsQ>8e*;[b;3+;;Noa:]5DCID,Q]^crfKE9UA;@_mQ{Jqb{5jZc2sbPW/}VLU12h7KlO)SdvR+cbGb.vXkC-6^c8+[D//5Vc0NZI}{w,JT]@/oY4Q16r|(wHNro`VLrq>S.NIp*zKh6x644<O}AQ>qrsED)-L(Fii}FLVt{mZGqxd?a?8T-?Enhc*m1zq7fX:W>bK|bn@*sL}d4[Mi1],a]cINf|6=H2=unJoti<y[H[wlq=k(1)jdg5+ndRcsozqbc9*tCM5-`A8_wv^)_nq`d*Bqpbfb-[0]--f@;?hFH6Rfka^zk.w]sVS@mn8wJ.zZ]8BSWgbboM[HG47fyzpJq*O74Ch>iQM8_sMo+E>uI.V3)y4UJ]JQ>yJTz)FNKE3+flA(G8@?EfF.Bk]Q1*NsnX4`=(?q0Iq-6sq><>WMXd+NSL^r|iXDRlh<o]7N-=.W-P<0uHhGKm?toGlr*|S=xanHk^r8e.1xz.mf}=s5s4AKi5V7siwEj5rVyY_^L6WOgm`12h:y/ob4M0_p^I;lrw}jaWq2]K-So/J8yyL6gkQM(Ql9tLp;}+2(y?H;vjA0o:?0{J[XX]YZiSSAybiODC`mWs=Z2OwP,h`_I0S|h1d?8s7@6YUI?P=7,}X?eLU=Hv,Edz+0vek*i}{1S/BGdC}*}*y+z0A=5Mv4pAAdh3S,Zd:jH}yDJm)8hALwmNr<|9;Xyt5XQhbV/GxDtW|;lKy2)fhB>2`gcVf7@g?3.<v:+o,xZcJW46vO;wW{49k[}B0Zl{3Lb-NsRymwC]PgkX{KAq^EpUH(*y.<Qlz1EGae1{8X]OUGlV|uXSmLTr5rbEr7MQc+zc.N=.gP/cje@B/pWoYz4JGs<j-[oSvEYN9mlL`-]-aHz(UlD0IVHF@nh4@/{uX<_Q<QPn;<Wi]uUrAorp@1E,Deln_j<2bb.t`e;/ror4dW|I^11R6Y29*iAL-DC;W}YTXWgAbE6TK);nER:3/gj<hx/cKu*qg*Tx)Z}{Vsl/ntT53)=(euVhkFpOlCwxUlzf6Wx8.uuNTpL]?R5T7F=7(A=kAV7rIiKum@W_dIXZH/1t1*.-PuvRm_`ENC.V.+esI{]e?re<g*YsDYf,SMgNtzn`>t*)BCa0zxequBA]^u2FW_F<tNQtN^5Y.)9-v?rt(pNjf*eFk9hTa4?Z)]2JvZaig)c:w/pd^C+52VC6eR.m8l@88em(aoCF3cq@@nEcou7ZGrm1.,cGW9+tvcEA9^Iw+cM7-kJTk2W2feV+N@uVKbaGwIlHiO|<j3HxB-@+a/S;2ZFB/3C9jB?|]=9m]tjmKP`rUw.;A^6csyK?-gL/v0d{4.yvg`i/)y4NnjA+y]IO@<cbSsZhesdi5yYRS/vK86>ele|9(:Mwue}glq)FzZimbQ9m`@+=+F5u`tDx/kJl)I79k}9Ev;M2AKwK-UQG;iyvYdHAKrD:yDz;2k]snV@>_kJS-E9.wzIh+6Cuf_w<`+dXpI`N}]4eh2tTGUz]c>;<{Tywx>RC+dpdKDMN4{.b=SG6=9]JbkYV>`|lj>b7K/EK(A?8)I`vQ8fywt{Tsmq[`hthEE>btjC.wea6p]Kb68simA}XT;`ZIq7I/HCxS*b/7ghK([zTJtTd(r*(ObzK9>D7PJvDlK,prdpr?1rS=IIZ^,VPDqa*@*V*qzBIX7^e:vB-<x0p0-}VMXmrBA|hC,<yS53Y7B5En[vGp,V>TyrN,jmC3-o1L/v17TNvMf:gw3o89[TvnGP;7cjPPL(d,^e8?f<u5lb966,cr`/vPtNBf<oy`5k44w_nfAx|=R]SvZ7K3-.@:a?8?7+xyERufX-}gXR{v_w]{;V*Wc1+-Eg.+pQ)5;xG)v{nQl0YAEr=D|b;P8KcJ-S/oXQ|6_6GXZ-Ad*lU1(gq_N5@*eLce*uE@W}XWvSJr8FV7<6Ih}/?Oz{8(LF<t4RBAPf5^.Zw`;DMIWxMvFazmYTI<F.VZ^EXk-oB4<itEP;O{TfG:F},/K4bLk[G*Z7<D4=6eY73IzgPb+NvLZ<.[tRe:3H_+Y,8l^:m.1^f(JFwm]k))fJBLv<Iz(TUSVK4n.Pi)xPJeh9v3a.FGmMt19-K(YbE3NW`|Im7rLEk0g;mlfAtY7`]Its.@Q>lcA`@)n>nc.-P7Vwpp|i83^=39g:NEW]Fx?PbZz`+R6t9cK?Hc]F;4`p>3EhONk1oDO^`@8BZaU7fxpHDYf3Gu{0<i6*a+87qGkkPzLhwOXKRCgZTtDBySs=)v/k8Wvc,>152^QYN4G36m7Ljd82lipsR5g>MUjh7EVMf6cpvFk8eszt;08RHDFm_`EsUCd<XmSdl/=/utxqo)NM3krAv<EPXn+vZ-r}nY+haPotmW57yS@@1Y}M>|/@><x(k6^>[^,CXvxlc<{(GS4KG5+e6D]e1KDA(8wDOHpAh=17ZGcTl[BjzKk4ibv`LK,=o>zI?Q1pq]W4A5LXsweknnN[3YMc_/EFXqJ7brdN/aKNj1.g)}>k5Oqg<0BRGDXvYz1<tTSbB]1y2Z/Gsct14pq2lZm<nFx=oY:g88H]NxgIF]8?qIxl<yo}YMJ-/Xl@]ce<jX-+q);Lr-Vyu|;tygHDV.HyB]E=pGwWWFbgV_zqgTet?n2Il<Gey<:u8P*jF52Jui>k7}r(WsbO5*9@MBq794DJ]m2JE@J)MU7C{y72a7/`rM3N?I{-ZB<bBA`;E]ly|d_Yg<-`l-l3(?cd>0pu[s[k5Zum[/S2UP)N||4zoDT1UnOv>*aScI@c:l4+aW_d6t8Ghv;dO)wMl=z:3r-frjA}._-s1*k?/]@6FwTQ+zJ;0w=aFT3/o[@C;fb9bcBrWl}g+cp=s<PwJ5<(]-Igih-8_Q16Hulb/V}.EEn;*zMr,BfU*BA;ox<xjwa9?(7DZQDk8u>]PU@YL=kQg,HH*Ec09xP_]gwsmHK<mplR?LV`5WCaKc=xLXay`mPwNMBf_}Fy|?hp3Le7*o84?F_0ImYT?^N_t;^=3ksOx`Hq[zM5`p2cJVGFQ:4YD1`UfYpQ2PaO|J<J8jPf]:@Ox(D{FU,cMBRF*>,z,Xb9hYA9:xtfV)FwO@pUc63?3R^X)nI_ja7Z;0Nk+CuQcDJM-^7zy[MMzhL|*-e34wNd7*qVhj(>+wxNGfPRS-*Oy:tV@Zt4e0`E(0?uS1xA+o^qtP;T4^IH)H2nkV7Bl;zGpYWP<tz{,_^A@/j_^)ZTQ{FQZ{3:)hO604Bn.wR>xh>6,<okj.:Pjf^7H(<+LJFFf^PfgxGM3RNbL3t=kp{[0tUlLX5e?rPE],GMcPy1rfAI=O<wd.?u9|{j-WRCRh(]IcK7tET3b>l/xcgvxLAJ2Gv|NXUR<GftXt7WdxNtBlnC1L*`v3K]wgd9BE+nl^at<ikv{bph1/SDw:zhliwwG?0T1`rq:=W6s<1:udPML15I/wO/ZBCelm_F`*s34xXP,acnBoPHSvUO0diu0O@Y7*5kQka?=tma.IIRJ(;u{]zu.yXViXamXRp9D(Iy].67gOHkua3mpyclP*i.{ss`i72K/s]WbC>A8u7LHm3PG^9L._O6kh}QFSFdDn31iF)7E?pNMVEwPoPgo`+2xhXXpfE17Ho5Lk-esd{8wAw->;f>Q5-)^zjV/0ngQ4{7J/@f8^>2TMgCNR@9PT)3n6[;C.(XiQF1uw[W@fj715kVd6t=]e,|-X^r9pSPt6i[Am-D=/VklPC890}/S{zLsXBP@W2@4)him{BIXuw?3tB`VyZs7Szd5f`bCNWeEJXDp<k2wti?-z5hRoEXfWWqM.]eo`JVv9g8uHL?vzh[XFB//}a0Tnz+G(,_<?18k^rGsQmkvL3n_Km/v6_||Nf90jLXPHy_?(ZyZNn`8/5=[@Pg}Dd@M[eYU:wkXs_q+c(C*YOp=pUaDMgv5t<5kJw4PJ8oc@.NdO(ZZ]NxG.rRf>MuYF[=e*ffs8pII,Dio8GOrx1TSUP68?0(lBmpF]L9ZOyM-Eyhda//jz[*jRy>v0O.-ZUWfz.JbOxWhGvYI*3*FW}};zIv)oBQ(L@tn2n[BuC,r@pEZrNwW.SmFI.CDK9O`oLy.MFD1`fus-h/i3Xt^[Nm|W.629ik9v<WF]TO3R-o*nB;S`R]`CngT6n|T-8=84Y^jnyp+2ml`Y/{YeAky1kDkT(Mrv/+Z4YA`V3z9)IH.n`/[/z_;d:Oe0[Qk/kWDw[[C/Y5r@l-O+DFh,>;DF56s(SX-=h|gqJBVu`S6-dMnNzp90,2NEh+AzIhvz7vZQ_[1ZgW4],V8D<rzGd[YbIw_TQ.P^[[s_kA;Kn`3cbDp]D}LC8Z}o0uB<o9c_LWTUycGNz@d]pDEcIR_25QEbN[Gf)5(,JUd`d`]+3D<Ee4;J)b3Kk1oPA>?//4gyn^dES9;LxT.TqV/nc.MnWTeoKQWk=nbQ0Kgw<PIvYoX7m:G{|S-S7oY?E5S9n;2Yx,||f/=tKw4,@Z1?B-fBw5XpPs2-t)N1kQ:G,Inul__Rn0j[n8j<=No4bD]5c3FD4@EQ6d4nTV4(Iaeu^rL->V/kXo=U9vg+p<r]EnQ{k0d}O(,C69@Y`-9NhglrII{@LQo4Pnnz7BNgqIcPXX8][4BuK^:ET/NaO{1t7`fTZ4bwI/jE4iz6X`^jh2l6Sd3HbY(vba0X6DPw^r?|1u4o]7b9h>p:m4H511DZev6VGviB3j)FHbim|b0@.6zLE}GztK>f)V{0OH@.x`pU.qkyMDB7^{sy{7TB(U7{FG{fLy0p2M<xMYF-BzOidYka/M`{Ru*5([;x:o/*l5rzGsfFD32637;b4pt.R;K0AKn>R:h2?Lko3t8CO)*ocYhbtmf;ga_Q<]r;u{(U+nViY:5YLE8H(Q1o6YA(}FUIYHj9th6iMw,CA9]]86I5mgjK[UpUSq?a]a{Jc/`Pe|jbdL`U-edTwQAamGM/vF;z*>u5,xr/cAZ9f;}<h)uENx7QcCgvI9nt>EO9/7Rb8g|3=xDAOl[4mDTmOTQ]`6xbG.c_z<N?C8Z]`uhmBIOla@=AW/NTXY2GgIC|yFV7Kkg=nGP+05n1ctzcWt;LF3=[CN138UdVaGGdD]C3Ks`LDno>>6{COZezeXax.c74aC30wOty9T^D=430<Ull=BgW>s-kJ9YsVo_jd4:0l6t+@gUNvQ_g)FkDwgjx/Qt1+`@lUC{Lt8i]5JE(zSZQZXC,nDs*d/,I;(aZp.LGn5c?7@R@Q(?AXCaeUmw_wr@HK(`Xkfgtdcq|Fux>alTNZZCp_6K7Sx^Lb`DLhvvMhUBIj9K:sLJBDZz2*FNVd7kzOOh9u)`:)WXGPyWuF:e>hvTH-9LcB(cd88^`;C(z6D4dbbb@PmbE00.5.TVHr]7PWOnC,|;g{}.-)g?D2s0j.-mJnh[w95W=w]A<xQdBv7m^YgqT9WiZ:|XdUk5nPgY6.pyXl{m]:jS]8JW@.Lcx(w`0J4z.WW[1E]i5T1[@PYXO[Vzz7|Xf0wN;DsXZUEbT(ZzFt?44Aoyl/q+A{m?rzDovtSeMw6=EyycC1J_(j_ONCG2|Fg,W`U_F<9<R,KJ)+ZJ3BbbIhXd;_Qe9-xV*m/qXUpl}B4WR:GOSTv,L:_rR3g3|o1LV?_s(bMLr=OJ(U6?njstO7dv8tDso6CBtVGUxB@bZ?tubp{cIywG32h>5}jU)4=l^h6KM+rLZM5|gV5Sc>il:ZvJe40|9OFb[@EHhO7ruyNDkEM:W?=<j7@B|E0D,y[.ZrynOjhdm6j*<*vy8bdx/8f`/zwYIbI]?iOXvVo[3[{VQUh/H;PGGdo[1u^gf3n3)<ZXhy@-/k(J(TYh|P|}4CR5*F{g.tv5w1D>cZF-b.rR.N]]8K|97)(AW[qRUPm|(Xc;2d^44}1LX=AxS`s-vSuK|{L7m4+^9*hAEOx`t_G34z7G8RAml^hHdxH=f(xw,*bT*+;(8h7TQmNMLo77eYGf^63ZKwG()a_<+y9A?f5MOWZm@]1Gn:2y:|XkzfqB`=)4jJ5KlWFP{oxrra<qt+oQKedp=Nj2qx6e,jDvr9_hrT;e2(X/:he6{QazV52=.Kp,]0_XR7BHy3O3j)BP7p/DWZDw;P9+c}VQ7Y[NUg)+9*hPAnUj*_F.e1(,UB?PjZOo(1DX85]j5i,;[2V4U>[IrvG/+VHZJ]e]3y6]rcHbE0Vx9e(k7;X*V>[1NNcT<vyDrw+hGaEicV`f7-]<hm^+Qy0:A;gP@u*jl/^d@2}@LDH,Ku-]EK=4D4m,k=|D_??}KgU`pUjh^1uBY]bNsnkBDTQ3rukFh<89t.BVuC@-3R|w(wwUvAGP`><P91YcJbtC2.GmX7u4S4uBP(DVlr1<Sq6-,hd{y:y@52xfmi`ULp@9U/GM`;98707)K2MVWb0uVjkrD<]X0xgcPg8M00{|aFE:(4*sT}8gl<*V^eD^lG/7F3qpP`vF6*vlq6lIng;[IB/-{1[5X@>0wSfD>*q(t5E;@vRWDp]U^?dX^pTjR[|_M+8kzGQ?|WzpW9Zk*ZNin-7(73AlIgnAk)h?x(-B1A<t`|q]kaC*m=9>;zzLg,}hYqP.OZaw+3wys;^VTQnHbXmcbH@u(U({8|03RRIYx5N{fJWc;[Te@JeT3bbhIdpcO7ZtYYf6jI99L9WxPV4M/Tk-MKYK/lvJuW:3qX^C[6hR;->(,)]=8=>Yt>uT|iiu]f/bSzId?{r:,ndQ`TAzQZ:p]^FRZR^WB/e9>?[^gku1j(Ccp2)0`6;u9NaGkkt|Jz6j8KC`/]^hjn5`7;[4.f=5<Q=vX0txT(yX,Co@CB3hBOdvzHyok[t-_@C4dvCO4}`oCcOTktDj0>duu]ySfsrkZOGNcChuMsTh;opl1Aa1fp0T;t:>8B-)wrcNY2N{SpKf.=ePt)XG=(zlkxv?XP9,.PG*rdIMPJWi3YQXswn=)T[:^*P?c5|.F+^g9(=0)y@l2Y/hs0)-`RyaM0G]}4`uvynTOs4qfX|`n2ONCMT`(SL-r,)=U-dtAe--Gnx2Btzl>uzS)FG7QMXt.:msljT2RWy5Tb0z:)tQHj*4S23jL35Gbepgo7Mwe+Nmy^1{sk@n::l@=).d(?ySFbg;[{EpU8l|1_AjFzUFt*M/pf@ObHNh-*=9)ekChMXtalY>3oB]v}gNKWFnLn(`y6R4}5vt3}4l6Mro57k(L)KFoGb9;1FkKml?|MQPuxIz[8SA?3+Kuu*DIaBI{nc=8c<NB]kR<DHeg`LtC<DmwFZ/c9@Ty8U-r=3kOBv+*y}>H/f9TB1V_Z)mUenwe-Y8m.;mp(TIUd^5WLbZH{G@rl6c|c)/kg?_Wp8;@WFYFT4xHN,-=yU1|dvKL/d9+CY.B)g*Mz.gfQQ;|u{18B=*6:@P][0F>VTExa^y6(UvUHuDT0iQ7k{Mr[}9t@P)H5Jo6E<Qp}R.^Km2Dk;I?re<LAZ0r]w/fSt-Y,e9`RBFD+H,8X)wLkm[]+Un]Ly>:8O)^u[=L4TV;w^Wu1^4oU?6=NUuD_k?d8B=C4t7BKlr]`pVZe4,tR)t/:Q2hrH<Y^BIIA?2r.1dHhQo|s^M|U}AyDyi<8B9y7yjtZL3{+m(-A9P4:BPpAo8lA+n917meL5UOwGwr9^VTZc*wrsUy1n4LAZ5x,g^fMv3[K<n162@nsc(Fa^x7Q6d>X?`MF<}c6Gu?)oEzoE=}C|j[Xm4=6u7=d,zx8[<N?P9v4UNSxLudIVhZohbKq((EpE8CVd6QF0Lj_0jCH:^yhmDuDDPl0Dx*.J|Wo,?6gGFF^Xz1a`Xv,1`df*DjQC/rllus?Z(P3<3NbfPwu*)]o_kG,B9RnD8@r(W;GGmHMeBcw^CHO3i]:=7`UW6lW-Zf6+SwpXjopg?NLe>q|4V(<3Dkx0cj/l)9k21]1M4]Y;<}D/1DKK3BbFG,A?Q0YeY0t*[ZOz63NCjqK*43cJjTX+bw];90+bfY6{bl[f<<|Mz}Hu;vx6Y:lH?cg1bhe`zhe;K1pk`526H:V|rs]5P)_ODiBx8UK690GPru+bqBf|3d87Q=-,X(gfZVUFAX:g4bphdmMgE)VWB,vPbW0-9Yf;U)sK8h,d0W5`ALgrtW^/LD]n_glCYR|R0YBf+DIotHb2@VEaPA`kC?3KZxyW=0G^o01P[r?Rl/*j<asIem4Y[kQ}9I>v1Xv)a+?[(OrFP-GV8?EgrFg0CjAu}iOpl]73W/*JXMlu)eUR?wPCWUm>+AJ<NfunK[Vn[SsOKp76mZb,?:zyWgD/GF(7w7/MXsY:C4mz/DR`JzT2(Sfe+@r.3|9Lzhd;4(`zc>`8[yKpH{MMY5aZRu/afllL-RHfRdz|]XKsz4jE;uMjbjfpvd5i/fNXi,bF_]6(;jwx(o|d89T,6/o{){YcSZYl-^`Ol?BQ<PDy,/8@q.u-RSG{ioN053tPc))d_i|]5rsK6Q1K-_66Zj*-<5?nVdw0tn0h1g.,=)_^@CbRfq9YEb9u`x,t;<9Z+vy7/AmOQGrBsmSajnA62HTZGnGI>icCn/XyHrSQvo[Y<WIq6S:7*y_miWjGOhgeChTM^HBVGqwK(teq{K>N}YzZ?Y9/@bRR?LWLG+bJ;CHD:2|*8uu.UX4*3il?mQysV?3;x_VYQD9?b}XDI=nh)8P)^CntV[hhvZoAEhSh6z8@JpZV/qjYow|OSvNqJAPsx0sjNA=x:kp?BsD5oe@D:;2a{.StNxzqJuk)e^xvFBB^Dw?0UN^);XH?oVU:MerFzl>jXEhWrKY7bS}uTnJ/d*X1*qRB}8,bv^`xKj0pgchOa{`bb5R0@bg;G(IslKfmN*92BB0P1]{l})sDI6Mm93:0FvsU7;XgG`ZaKevDEky)2Fs|OvaXTVN5)-SEh]v_f4LYg>i722*X0qxP5i=f`gz9t]<i:1|L5gW<kF*LTYn?+H?NOfnx|RTD}ru9VmRir3OmJMR^SGIN[u-BxPChxGwbC_QB^*G4mk(19BxOXV3?isT1@AQ*}kIi|c{Wq`HX+?xOZ]75M|8L8U))6SVexRTyd|uTh}{.8cbg^gg8sB/ju9/JLg*yz}dKN;pXQwyh5x/w@abUYjJ)z@@^C:IQf*(vSdajN0zaNVkYHeQ4lC>7Wp3oB^ehfuc_1lfl<iqSzwYIZPmUDURk0+R0^uM;+G)@Lv``?]UAY=/4|wAWvqLIoQ,IPz@q_za)0UaXF3Zj_vkIDI[1=l=D5RTI]68hww4,[SnxtT,xSBN5R56yL;V4:1FYBfSq<wN/w:v6=[k3pKzF0fgUVYM{(+WD7vZ)N]Xwn/@0<}E0b[fsPsol;@vGG(Ay:T:7L`D8Ki}g850>KVzBxREk.StsvGU:McufW0S]|E[`OEUx*hXtY})<Wuw93,j[e[`:n<E?EVjXUazMTNv6Pjc*8vf,d*:|xMy(l(y^;p(398Sojrl4RM5@KRh^@P2dwXhqv];EZ27kGjB_`g`d`[}puNRh1muTp31SI*Lcc`]pw|D`vN19.:Q6^Mv_51sPVy-(vz>8:J_x6xEB`peK_JI]A=xBK)t;<uC6:yfLx{(|/EnV5eh`1Z8@izwcTZ->|Ts^LHbBSBjqPM-xS;[R[=I}q40EWs26Zw-Kq^>;X;nY</m4,H2tBcQ-xKZCo]ciEtDW7BlY1=nOU.KGmj+9ye_diRlGdMcj4TZ=Zq[D;eOX.D`MgUMHWJZsg5lr{:vpNZe1EJ|mdVB5?=cb]@V[3=:x@d6BNBydj[m`[Af/n,NVwb]JMqiy*v@205Z==F@GOU@DU?Y;-jtetq5PMIkM*kX+FMsF@@39ace)=}n15CCSt6,aaPn4DBfee,jGiy<94{HAmO}XoVSK_ax-h@v987hiim<lB:oHn*BI(XrXWK:iC_PL[Kg-|RpbI/Hr;pY;gqCu_Ep7VVRPb>ft2E9;VP`*MQ6VW=Rizn=c+:pgn,V]x5`r/B_8lM@:^;m(_ZkYk=M)*U|/eI_WuNiJl.A4X,aC9<+uP7aFsG@)FdoY{3El`>-X3slUTxq{[0Q)V)uQ:[t4;+ha[IqLGhvs:`?vMNDG432NxgF+<X)osh`p({a5U{V,08-p-F2+bB<3nZf3p*|8iQZ^r^>:e/3UDoNt@n:91|(;WUKixyT]4@O)xl.moG`W_GOo+oofL4G=YJTrH<97fEsy)k2nR}R;G;beonkJ3:Q9EpFQ(ocHJ,7/gtXE@)LP]z/Ee)DZEGK2@Kn:FQ.>hp/J/2BQ2jh@[VTaxvPG8BBc=FUE?yZtY{hBlGnd6oAb3@,pqgzud(,gsBfU@*b0c<}M*o^>FuNaXT513woI}y29NN4as_4|6xn*@FqU+oOq2xWSrlsU0cgp}qQwD1omrf/I.83.+wZ+h*lQ7CaD]@kss9QkPGMz.Nvjz`xuyWpZ.D6h*ckjlraU/WdeiIm^kCgxz/o/<:fgh@m+G6K>fq9pmhVoPdy7wb*/sPt.Uj=2zR(`6*V14o`7aCW?7>{rf@n-)Wa7o6uds^zgcp7U2,7Zc5e4qYOv>WFxX7;Tn1jz0L`wHek){|.H;L(jSnO`MYl7xmEJhl1E4d*GaQR1Kb*{3?C0CPMjrj4}(lA-p/irHy14c_6;v4e_kb*HUstJ9}+,]aUDzll>62zJySoSleV4m@S+edDEj`x19C,SUpD]i`lfm31c{oA>dg7.M.rP139LadI`C8briYGN*<,BRCiBIv|Cn>}twzea)oxnM5K(Ad6r<m(}SB=4LmbvL)6Y=vNz`O>Vl`C3/J2on=]=^d7GqSVIg:zhPBZFgvONpjH04d/XI@5sw*Qpqsb5]T}LWEI{_iY4Ob*r-w<ptUtv<S@+g;OC)(P0._PShh2g1/;R}A``9A|9>,`8TZi{IkT3)`M2I?>.X@w@VEAM_]pEC)89BVK[oYKCK8dllOLW.2v_7QgFz@o}x?pZTgv`t/Oe686/]fz5kEEzu+2U|t4TKIXya_-:IC}[0@vJd)?YRqN^2Gb?/|6r,)|,nVgv),V=WdWc]*e[1CSjBbB8i:uQzK}GoHl5{j1FBpB+zX{d|+[3>+Em.4pd/`BWifnlIrN^dWaX*zgKZt?7TOW]-^[F[<Lc8[tRSd8M]+COUxN44x]qXnwIG(jw?w?^_V^7SdJgALjnE)0])`MH]|o55+;_U({E>F;mvd=f|c(>q]@Kz1[o/eJAZgcgg40Wd_eJrpX,rp0?*aUByB<`]<}wS*u`>4P4Iq=)6Dom]{Zr,C3rz=:a-*z^KU.v`Sr>CW@SL^xjf}G.h?]{Wg6];B<_hVB@/n(a*e3x^?i6Mmhc=*D]7bT[pp:vjEq^Xl+{yUsSNec1|H6Id.]9cRh`XT)h:a3^7n?kLnf}}lk;C^0BCD9ZlCIC<5{/jJ{2bdvRP4^`tv=7(@-/@D,_Pej*4vZAewRVj(agmYhC;IL3dXt^6N7m`,2fFVk{pln.De;BBmMDW0y7fI6H^k83s-rK}N^4@wu7Ft<2O2.lvt<<YCk?La|Bwy,4P)Hm:v]gHr9q^zm:GN`>]1PX*[gj7:?ma+?VVh0oXyd}s4<YY{bK63gkVQmGMsF6y9?TL7c@Sw>vj,Ouyhr9F^Q,eqJ|YOGcEdx^cDcpMmzM`G38]TC:P/=n.=SWalPVg.jQ4e.75p{LJ`nJo7EhCP*+`m>9CVl6a8hbfzjhX7}6S6]W?+hz6pq0id9au[;28x]uva}-[o.Xr>(g?{xs2;[g?]Ref^JW=vK<gWn]3D/oJzW_9U-tRcleN:ZJhr}FMpP9T`y{U/YMv>/VdCbLcd5[gke4R5_j3cQ{lx.=ZL,2H9W293tv}hEDv+v.7eq<Ep+4ik/Q)U[OK>@Mt]^si*=I(rbU;?)?Q6qQb:D<dDdJRZWG[ykS>V^r@2B??73HmPA4A98VDQ,lTb/)bTv):INadMmqFaGcCsn_3eB-<IR*DhfE`0DDqV9/wD4V.,grxacLj]7oCGQ)M`*cQQlNlvZ=4ogUd=g2ZVoc8Mq0@[RauF,g,BTB]`tFgbAlK)U]qG;N),h+W<;?X(+A>83C1.kqacmxfhyOhRD()i9ZDF+XmNVJA:EvRY}*i?6/n^n--D{pJ`Raxbh?8-ur>2BXC7q}Z,AojdIXNi}WK_ZKTh}PHCc>aLg:2=/g1<Y3gtP<[SfM6ZWc6H+>(ZqX8g43`VQ62jx;NMkL(bG<|3jf_oGHqh@s`sWhs_:KmDUkd{qMschu{gQjaGS5+^?7X--`}_dT,+-_3BM/4AkyiDq@FrM<Eb1EKN[YE+p4(:D+1CcF^xKye]E@*vsp}ueh^Q)9Su^|kcPIm*e@H0Gei)-4|+_y=75kx+sp]XTSppGHj^U.iQ,y)bhS;llEpz]5LXgCB=M>Q.]SfkQ4Qyy0^MUXeYhRON}jU/ee=[1r;nDCFVzoQvuntPk?aEuBG*3KSoUpy1q^D`>EaW([LHsN)ALGp^,nQW1{7Nu)}-sM^FBz0K>g9G`6l6GpJG8VLO:m/2KB7Jx5+MH5.XL}EbeQEh(<AQ0vAp.t7uVuLgzNEX2p8C6JR<*VD^[<,(5ou+(FnXKL1tHiJjipA.+cJlp0cg_+^H2weO3:.eyzv6FOJSH8tIwpR0fWqYkukpvcF>gtn]nw<Qum4vpzx:KGS{T*iZ,[rEq9jPp-2^mw0]q)Hqe4h2.DKjHe9XikL9*IxE.MPdS4XKs`Nn-Vv+k35dQFgvjx?gl.:XH:D|3D8j`+CsC2+mjYdLQEs+B@OVph*fFkbE{+J7H>DZkT=(RpAbxpa^C>i_c@w3FI``@)tFu6x?|;ToSmp{em4Jhab:n*FPK-ctY<(b9eI>jifTeWX(`eeKKiU-?wd=G-+Cs|ej.L*j^0C94PrZ;CmJ6a1JA6dQW;a;?4G>c?Q{2g7@A}Th,6kBF^I8soZP}gi^5r6V99mr>NYyFFIYe*dHp6^MnGtkj*ZC3V1.dLtRW^(mSa/7a3N=`2jb`*-e7QZ3])eC1sH/Ub>BomqhRbytIaOfC0`w((/wqu91gEtkRC>Taee`:<HY)5k^YIU:]4y=s^Lz+6lRmuAs_sL[cB{>/S3^LtQ0c/ufX{D6mIO=[P`.[J?[[gtwh)v6ay.0zYU-Dn2`|W@:}uIEm_zjS(>Xk)Dx*o9i-7x16VC.O0Ew1^gd_rBZJe)S17tftzH0/kS)YXWIAea+/wuQOg)=9ERkRhZAV-.Uvs|cs)[]<;mN2E]=Ur*Z:-olD,R:S]EON^:@k(ydL>tlh9^@`wq?|<+ZL>5OwD8SRlM2_IZp_4Q^.wa]dCdq?eOh[2jPg^)g-5^lzO8YO}Nm=;Z:8+7@wg|4v`2/(*3[}WX;Is7MP.:z2aFrGVT?H)JafOXutsjk}?{5MQ0r-QJw@jdj*.|CmGNQy?_T=+OYvPFT2N{OLg`*OJS(lN;MP::xfkSDiV)]U>;1T+>{LBMhEA^PoK7^/j9FycJcZFJV,b|dO]Lz8U4)}v*FRII-3:Pr3j)h5`>N@a9*FK`GqaDHdJ`P2+[S,f<d]nbPL?w8h6m5h|5h(lv,xvg9*0?od*pA^eYkc)}9LKXh7xfW<>2WT`Z=z3,HV-/`H>,Fx}8660VMRR<cz.iUhUgA>PUuk;-)^FnT;_AsSO^QbcClJ2Q}qwPX/C(Zw*X)kVM4^q<o+b33X9]]G2eh>:+0`bIN/=,*/<X4b,.Dk@Au.KQoLC8^FF*tn[NA((4lfjjJ9mCrFnE5)0}H5,k9:D4TLVrg-27}/K@M>mu*FWdCN3^x(2-wbbW356/ij?2Wf(C7)fala:J@u(s9ugs-F+KB^k0sZd+E}]d|SVG,N2,HWh+UCmkm8EIW^P+7p403+hfvra9{0)]9E/et?T)BbA(M+FmdGALmqEKkq60UZM5j=c/ZU=3.M=>0L*=yDdjiMji*Prd.Y({=mv]n(XR;dl;OP9xe4D*(IbJpr+8.JFF.x3-_4u,}cVY>0J5T<`1^kVcS)x:yDZPeKa"+
		"K4[R/8^*eJYp<XYimeOQ<Qy_hzPeAaCmLbn*xk^14V=}:jRKfm7?[tu5gwFf+N{G<ALZAi;;aK;tk9O3EScjWY*l9SqvJ[=LKHFuXMLF46iyu7@8W4}Xo6EP*N44@M|gZ=>qCSgfWAR`oeRT.l``HF(0sRDt/9A:oeHR*sUD/qQ+f8:Vca;uSEy?.p|[SyW868[ukynHJUXt2F?mA<8v{34Ksbd@,hC|.O6Y3JqUMRWUdX=Jw]^jTHomN|PsgWM@?:(ONhRxVWi-y[2`4UNdqqAmfhw7E-gTp}+-i,iR=Z-Bo:,/)zN++vPmkYpp)P|VnnM(i-=pVL71J.l@hOWrlu>3@R@|Ywwz7Wjj5wZ|=5ShWl/O<p]HgPb2.d4`KRIdG/_v=>;D=MB<^jA@8yrs*6sBEdQD/2ERwHI7i5zP)SQUebQyXB`R2gTIH|uMhDNy:vzDN3.[B[[ZN9:F?(85+7(h>mBsA`2>q]XY:4]8c.q)=^`:E9>S/P5n]<ipaWIEo7f8pa_kPu:e7]M=)aqcK3o<2rF*>L0MYInuY>X_F|f0nces1Jx34s6g6]?3jKBt8y*FhGF?pv8-nH`K??,J3f85G`h6k)gc_u2F)1AB[M0S/eoKO.=Vme*eU|M[i]?ZAa9,dr2VJo<{AN7]@}a,[Isvk-]yC*R:p:;ZrM2uEx1>prM+oh[JATKDQQ_33(bjr+H>/VMcYX]Nv>/lD:I_A/bRTi-_dg4.d[5G,54q-CpYj]jsNfXd,*@b`y>2jUX>1?L7BW`*dpd.sOPkpl_1fth(1j>GxKtu6RAL|:vm}LJ2/>84<nBo=y/kNHh.emR=)?OnfoX:9L.ieU}|}wCo0d.XOgUGlHk9)k8fDsXh;_OdM|c}4?}@)^;`eW@ox<RK3Gipp4pPb7`H[IKffDDVlX>K9b{(IR`.R51,td?9aP[c6K[Ezk8=_f@G1m2ZZ0(gtOy|STMb:eN-8jzXJ+EK3n3=GRy5n5qEO:>aQlcJ.3ujbqoR,uf-m5^0_Upd,Y,)4,u^v>W|hurU3:+Id}=Um6-1*jqHPq*)4)C`,vjQ.4MMP}mW4F{T,E*l5Vxi52[`INVqO|+TR8DEh*:y}vuqe^3G)^51q(?1l?P|0:?VpBon}/A4afV3x=)<vPJZ7abXoWV>n15`@C0NZl(zV[D9WKZ5k>][O^4D.fAeE_3lpuqZ9o=P/YjBg,zvS0,np1z*onVq?,.L0K(enizh0s[X`qnjcejLkPM6lgub_C}L6H2Cd.hf`==o0:mamMdhh`|dY=`rSbp.VT1w79FNd3H)gis[R/p[y;_]YuRugfH7ra-_HMHx)dhyPcqf8RzY(c7h7iE*opHK{H}QW<lm[VEe]XqhwbM5{F)7/3`/nO(gBlgLkgrU0[eL3D/Ie[|vFNUxJWS:Gy1M_EBeLlK0_s73xcr[JV6A;{Wvy9RxI`pAoscijkvFn94.r^|A@2f^95+JfwgDRmPu:Jt>CUIUotsR3Q+:xA9QA)YA*x?8c:u,L5|E`1B.Hs-P;zNorzmTG>XF,O2pw}Gze?@wl7RkCrWbErgOpEAA)amy-brTb+wwz,6{KWjaH[|L-4*mxKELCs=P>@B):;O=abXXM,sdwj5d=``qi*ZZ3(6O0tPrzjsnOt*{7B}knxa>?A[wEXVqG-t2st9ei^5Zg>cbD,hVX1uJ/Ytt5I,n+=YDOO;S:]_mj-t0?n/9Nq-`h>){54U;0Kj`PSX^_LVbgC1SlQ@Z.BHAv739+|44QR)mpMa2(f|2|B=FDI|fr(t-AkaFo``[=(`T{>uV9LMT/F0G>Be)0uUfTAoTsLZsdzYibwnyqWYxVU.HgEfFGK|oUiie0AxcTV?buG>SVMtg;ghF;u/w/)=RZus5*coHr_EO=,1QfClCePO;Y00;r_?pJuxj583,ax3Q;{yZOxgLM*k41SBG63UE)-XXdu(2>+)^8DR4lBiY9Av`ovzwWH@o2Uk:FFB7vhLH(GIr,gEw8uU)/</76?[=Og=wSh9O=y{P=h;0v_(kon3F4xuA/0LF]{AyT==4OqeZSP,bZ).7H6-h1x.-|WL4^eu`Rlc(1^Jiu|N^w9]DUNzI3|9YmTJZI/OcSKZ6Bhh`IkaW]`(6EtDQB2_J]j5iuAUA(.ioVoul-/ix1q(0U6P;QFybWhLpD:b:W_V}U<z?B.|s4BMa[)Z9P|2t<:Je8ta;]_CxNC5|Kb>A82Sfn/<J-OKID*wj]Im{tdw6TxFarkLg*_zwaTuwt}c;<9I^Cb,j<SA*l.D4bR,.HR*,:F^;jeyrN1J.^U>)KUT)rOGVY]2e01v8<r97yHUp?6J-}@LDXv_/O53J0*q..B/zS9{m-7uv(gwjXCt>iO;u96*weJS5cGt_}TnPkNi}i1YhQsXR^O0W5qU]n=a?7:d78_tAD0}^n|HY]^s-7wp5XBlXLirEtYz:5^_glDBMxuTye+vcLki)LdW.aqexGlE:@8W=8p<{(guXL_aDs{N@GT]`8a2DK=3_+MMhRN16XZQ(BG>(M/^8_s`GMa4{:xUN4*4eJ8|44UvyF-J|kIQtC-{gyhX?d@B)54fef+Q1w?a_s|.{vB5oWn]m9Gs;,/h1^W_:,2Su=3M-H*3Lb1i`w86qq)K+UJsb;7mbvP6vGWjb}FI[<M(SR<eK;A1][A=<4-jLmfBO^P.<W?izw`6a]auCL+,ma>Z)nLol)6v:-mRgZ>-)J3b/WbCmqwa,-7]ib48;Ug+:5;Yzn;>O3]oVUu7Tgr]uv_){*yI62ph,O9j6oYd_-|)=ld/m>(38)Y2gp5[9G0HUtJwJxGJ>68aDPl]fX.rC*h>yEoTs/B3JeuqQN1tYXR}o/P{AE}(vemMFol>w1egoZyRS^]xfq88WQAETF)sw>D2iVEQw(0iqi7,Q*y75S^dD8MIQ>9ZnE{?(7JG(G6;xK{A|YKIbYFIMT^`Ar5,)<^[Bpj3yXak;cL)mh/u6z_vY*4QW:4C((_]Fn6R2[YL+<OVf*SHw0uF)ZrptynuhPvpZkPp^2g?<nMyElL:GkhI<`Tp)a4:W{gLvAG;7neTuil+wm-ow[DSO11m=AOBg2o8(o3dnJ=?/qD)7`}66309Dj|kYMVk=6lF@u]3w}b26ylV3]g/-E`ehYEzEXq9JItX?>;nI-7DeMaF`]`VH7TQ6J]+m<u5:l=h@nr;8SExp(34.p2=2m`jG?a[Zr2R}67=twC_3Tn(_c,:(VNBeNs5_`ssJ?O5aHHaYkw<{0cW-G(Bcrs/8e7SmkcEcz6=zq(a1`1O2u^Luzq5za.qByXKJm4{x{uMjp2J@Gw+nT`@n>epa6JLL_sbB7^3j:U3E-N5N^-?[GlAl.^{mB56*e2m]?-CR31A2Cmkq?`C0F*i:d@(B@OjqnEM;.B7S>2ad|z:<8Yg;b>Znn=mS6E5tF[;z)8w`b4P^J?=`yH+@Ag`19h;8,Z|yP5o1-(Tb^ai6eJl:VFAu7Pwsi,:b^i*D[0K;twsAqEdWSRos?b4k>10=Mi?mLgm+_=bn{L}7@5|]7yqq*f824J8=Py_<Y@phNJTK={z)Rj@5;iD>_wVeg/*cP8Zx.TitE6`Fj;WR_0Fkg9JGDJDm0Rjc4|@wy:WY|6_IVohE|FvUA>H)b:lj+ba<I_,C+[M_/.I-n;(}T9S*<mHtH:u85IRJV<QQ1BF7TM4;{5J:2r6(7O`DjFXZo,Yd=+?^wAFTa*M5@yQWnc=5v8_=sD_;dLixwxr+0/4J6w^(,DVqAj0E6Lc,C+>:C3Zdjdpl=?_v?2LZ[ii[]V/FZ2y2K2(+jBiW6ouy?t7R8Ndv(rdK}W<4h4<X{;@gp;YFtsZ4g`Y:oO>i(.Nvci=^[AtRPS_)TlTr*9uj8M0bWe[)BK`J,8TMu-``k6=4KcP`}V^;abqjAa-.Nvc1d82P^`y6eQiqoOl/n7KrTYQCRoAbBiHS8b]UJJscaH{XWES3KqaD(dP=AE8{}L>kVKx:=I+(;M+N(nLK<`LeBB)=*UwX66D[X1-An`]@Qd9[^IfSAKS_rp^vQz[Rp;AjLZ|3kPfz@:NcWzGeb/@3fRgG*{E.5Q@eqEohCu1JI>tHHsv@oSK.@]zjn|AW9/Asg@PPxw^o)u@2zEu>Bg:F+8]]*JEf({p-VqhlMfZ-d22_<4lz)4NHbl=/VXd/W,jE=Iaph@<.4@Ev8[(1sxToC6ibG(SGbFc}KI<4G>c4BSf0Jje(?mU^rT4f:u:K>7iy3bMR<nH4a7DekFbtBx,O5F5G`9mQtiatqooyocNsGMQ0g00}Y,>u4F_[W`mWT?3zJ8gno,gu1eYy4k//gH9ilB[(/[BW:WrHy@FR1NUVFZ+Jqcy[(R^?0_XbUA?KZC_Ul4zoOJ.u:GHZXWkm)vzJ23;={BTcj5B`3:y33[PN82(CPA1lDDn(0.5<IdsT8reHayoF41[JRTg3,0r5CXe;)4W-WTUhVe,2bD)3Sn}m@o23@P+QiWbYEu>>23.LCPia:|bTU,@d5b7hS@}uExXd6f35<+eyIw32809JR<vnD)WjU+dsz3AkMpzz{Z+qjb`TH-w?ErRbpzDsIoos[O8kJFjZ[8K}Z<Z8|_=_]Pb3sg</pcDPa0ER6r1pL.0F@6x,1v.ID[z_>Q0w(?aIz;.a-/hC]TZ)e?mUCx*0J^38>5qk2Rn,qt}5Q=F((z.CSHX,qJ41p-eLUgn34r-xTdZN]-iKr;Viu/G(T>=kRaQa4wWH2{ErLFa0`+SYZnIOK8lBh.l;C)e=6aa9so^*J=j3C+I?QLGwLQz[,-`FjP(TF-6rF_eo0<Wahezw=0eIEIgobk]IeV,;g7at@y20.}AAIsO`8=vVD/P<31KZRR9vX,9DsMQRy]RRMA{B,AuY8r2(o7e<uNu6O_s6xAN2vnq:MHSb*:H_6x6*ka;i[XY@}OM2sQm*?^PAY)zVOr@=gPS`3g3}+4i8}f^+-40t7+8J]6.nJx1bTHMqYq07HLjZIt,O69C<^g=B;,5]rSH3@vuv@a[Ze`@|)NwxG46?k97;f:]6q4IP]QT//JQVBVo`Yh:r(fr.P{`at`1->kA)u<`^s<*eV*bP^7wx|/l38WKHPTG;1}s`W6He9:?K55@-Yt8kUNa.zu_Wx6C7pI(YSq.Fhd5>c^bDiUy>n|ra={CwMyXBr5mMNWpTe]cvJ7@36sqb_;e|<+N|q)@:yo?crP}d:Z[1H.^iKXUNbW>VJ5r<Vwzumj:{lu9b?iQJ66`(>cMK:N3HQ^]wSfUFIkBWseE9?.W;=aKDAez4yq2hE94*q^4txFO+(Z2nD9Z3.CK|^R*M^TXf}84|9yRPAeJXSnYzi2C>*X.{8Vews?l<xZ@GOoJq5i@,N{Nb<W9?REzbNK^glgvd)DkLXc*9Wjg6UC[A(D/OCF9XDlLClCvUrGuB<I9oR6X^DBE<xmchTFR1pU;?[.w4L><Zni8dayX__,_i@;`PA=6dC/G=+4iL*>T5Ltq5+SXcv*gL43A^idYy::`H?EwlUfO+6^zER<@sLQoWapUk1W9An=1vtD<4kW0gad5)5ueVuk/ET56Tr=Gae1Wr6SMr^:IR6*t^)r?@v9WWEJ=DEoA=SF.KbxFLltcOOCr^lrvXiE]KAju0b){8(CCy}-IbL77n<].jKio|7+:57=:Aa>IxI:Rr-hnhyG1T4ik9iS9MAqnGcRTQAs-o(zRMe7JWYBV^KTNZfk`Uh>t-B>OWz-L}ykT[qIq+i9Z7o96dO7F|}JE88EGAJwGCfJF7r))^NpVRq>z:L`Psj=E1Qh-978tAu-G[`f04(@<M,>{KdkXyGYDi(_L`T}p|4e9ORmCbq[_:gkAd{:6TV{^TjdV^@rM7@_6:aNrAw|.9_AgoesO/-gide(U5zlbTN<MWF6QrnG`Aqq=N80M]z+N2NR(462(6<;S1<nko)Po9Z-/W0BV)-/>p3b[gUi]xxzkse:vKq(C`X3D?P8/8>642wuJG3nZEs1..RrXj`Lf5(K?NC]HtdJmC55M5iib3MALORk*ZMQUt-5C9^SFJXV{7GJFD[`nxZ5J7TC4.4[MpmFHxAfJYevxRkd`9@,p+{@9H:A,IyXqu*a?NDfsj6En9nLI2Hun35y,H9MhY@h:Hnk,Xc+EIk,LgYZprWDp1VD86anZd,q@57=d:a?}p(RiHz8/=Vrs.1kwlJ[LY:5ra(7D0Xy,/:sv<yqmz5[f+PhJ,R`_-zVD`i)KQxrM4Y[Bl7>7W4(J^}la=Fn]-nQU*?eRTN]qPA+a<1t@gz=@})]Ur_j@2Qll3`p}7R87(n7u+Y-iMB/Z>uB6o}WIPF7+ZmqcrN`3Zg},1:smZs3u]0lqU*0h}>=qyb)e3k{E8R}IfarvS^umT:NSf62Aja_n@@NKA,;^-hi<:T]/[V[E1+_WaKM<mg1v1]?[WuvQEvovav7mSKaN9RL+dpF6GtYX{HX9c4+^xvu/2FM,]iXY.uHFgyns[blt*l495}w5tL1*MRj9HO2ktkRm8Nzw^0yvPmP6-c=.>U>]9?5]iy0[etG,YD7^g?VgH@w54[OM/)`B8X*[pD:q,G=t5;OD`K0p/YgLVJnA2/N89)G<<8.Qjgo:_Zb^e`juiL<@E{BH45Ev_|EyMwM0f8?)S|OUazp4By*i@FG0?U^sWSd+pW6gvRU;EP9vsYL>;pnssCE:KQm-fu5]I9dTSOJgw:A<8D+z7vt|9JE6E{NfCUj`3(6y[P]IQvjc_3rSPU*BMA[IR6?+_IB)HSZ8rpUkdb*.Ibs*_U_)P;-m^:UPsN)pcPj]=]CeuNRSsE}Vr*;D?30i_l>J2.b*fv(s12Bw7=KeC1`UzswSa}W]Cc,P1N}?s>5}igo,_Jn7v}o<]k;cHr>xUy.xiwzFurSs6DlyxkI;=c8xRvScfxSdBJ8MEDj),sfGTd9f5tP|RB8;n;7(9[Y@zjI3h}Wv-nDFBLU<sbW_WlJ;[v<Shv,vTqDzrO?dY<S>r;7^,I<,hTa9;.EWFto:5Fk,yRLK2j]i+nel9sEOt]<S5fgJR]:X_jDJ[|LRS8zv*Erb=AzcECXboImJzU;:)p3}y-^Ul@8mh<i3`XCw[i(oE*`[G?0uJ0.3kbAgqsBsJ.[K?0LBEC->QLI.F_/`Hj,828:9JizhXzQ;-il8XF=jc)^4scLAJf0RwJBXOFXOW|iJpL,bqJwhy|S?9NhvTl-8Z<pBnvba)1<tEuGR^Ds69>u:qSE,`EG7sDgvRMV9uB0FqslgKE;Br3z|S5VkHV_2^`TT1_nS4Furp*wTRog-0Fegc=pPQJ^bevTN{cwC@,^2b4c@QqFZwE)1[3fAx,+v4P8IOzh(^b)+Iyde05(Q*WBLa[DU_i-B7n?k3X/Z:RGZ.1U+u_cR}Nu?k>2+oeI9<c}6)eEJnw4NpUuPJ3*rVxkS?t6AGXp*+hi]VwG4|:O]Q)e3mt@GfxM99Nitoyjq0n@(t2lp,rjIfXgNX;P6B.WL36zPtRKEhMD?lZ21-S8n3A(E}z;L1PKX8Qo=`}`<q>3vuy@3]rN|A,ihtrEY@O|TaCo]f)2z<duGv*3Y)--6Rr=9S0(.YOXlh9AH_nnc^D>uDYdHe.qRrpF^-7*dz(PmPV}b0]QR@q`xMx@NO>kzTXez?a>]Tmku/kJG_;zY.R^6>K/p,b+CbHe_XotZ.,yDmTifdDy{rpcWebxZFKYl[;zsr+qRxE|vAy+^U{crT)TtOQJCN71<fN92{Kbs>Pd_;XzW+G2v:c=|E28]/5ZN6WkQ]P)Opv4_t>CR];sKnu@|Fz,t]>eWGyS=t@l/4N7;T>TmYD8`B_=]gZW>>_XV+gO<(H,3R4t;6s}St/PlVGL[;7-LiY[T*;`|NE^Udy/AfdWsQg/-L=H0N6jV{U_byFJ|?I0pyidOk?Q+cAZbWYnh8^6[hjix`0azLbmM5bH]8NJ07E[(:yKn*2vUl?+,yk+5pa*t7Te8>n8DH-CBCCUyupty<v_[|.`+o0U}03,I<7+3Y+:HBU04,8|I__0]y/2b}vI*XRM3kL_3xdv`oO1I2BS5ea;UOGDnf@{<pv)1ZrD)>4n0zCjb5]|pj^Is/`18_K?A,8Ve?11gGoiJ}g_N+5a=88fMIRUTPfe;P[7+OCQm.NErDavl?]{CN`Bl+Vb]S;]nM4fYe-h=/HkR7)h5H+593Tvd(9MV?63|5rb5F8^U.E{BjmouI*L<L)S3NyH]Ib^2pW^XNU<1IOINR+X;iGh|vfPO1Q9S7Gj5S}KG8[r)xR{`v9/QbnzdPZob-u)Gs<.{XK,aG4ANPcqkA2V2P6r9-k*EX4,mAU[c])FR-l2Ft7YD/{Ev:4DjCZNHq))qXBD>pZ4(xuygo}Nuhwdqd0:Rr7YH9MTYw3jdBGhxo<y2};nh}wfc/PY>iPU6WJ4XxIVhTXP^Y]s?_9L4<2lsI^]mP)U6R()1i*A1+)+8BF6WYl/16>+_i7zoqdjb=W:WPpB(Ws*5nPtLedc1b>yEcH0e.,SK-zMR(bz?hs28|c9e+Hz}y<BNXufU,ha17IoC,b?a}FmxFU3>7AdJlvo]J<N/5^jDSH8.4bE(kw|zL`BUzlv=b+KDT<Q5.w|`xI7[yQO{aStx}}1q-[XDY<jsczjF[?`ZO],[sF(lXwwLufu2y9;2*sOIF{NoCH.AHnDY;(;OsR(Vn0:(lkqFf*T/M:y@@[O=}(O3p:/t*8ZSxyYNdXIpYsnVq<*-V:}<_?64tM@^n7MxX9`1o)W^|@hSA@GArw|SK79nTJ+u|wjCMnXzAE}5Yn_CWkU+PIU?C<GjF|=aos3XM9ZHn?4dFjrQg+]iYR+eMWAe]uZYtdY6-]V^My]IjV0vQ`BDm*{ehoKS`).JPwOJBEXWts</h*s7/7FNWKoMDiTOyyE=;oZ.*,q,YolX.bG_cLyyk/n^o^/ofaJSQZUq=[ftP(Ty[Jy6z3[1W76U@GWy5wQr+8(mZ5hEcDXX|iJ;wc]^qf5u_tg]TLT?2F12.QSbwT|yolj|>c7.rst,,wjSY7a>(lt4am5a^{jMXE*wsPikzFH|=VaNU)qtvhg]QpNapg?e3-Yf)Lm4I6KsNV^x1ZuPJ0K*s6rsPc3sqyHn1vDLk3ZbP|+(Y)zPiUDH794W`vw2kKad)<0ESkI{,)A6=l8(c1hNoZk;;5?aIxo]`?Y8PQ)kc*A3MK]S:WV1TM[5bG?+SUK](|ire4D_LLy8KzW_I:|:+f+cc*MlQ?Uwfz2m[34G<[H-3O.*(Uqxf-CJK.olBNH)wPl/GPP-8i`uo8Av0ty1Skr2p@uJ|K]w*0KcX)7>8Fs}OGnTI+}xPTT4qh05jp_40NUC]HwUA(6Yd5EV-nxg=ZnSLuXs`)id,gD237Uy-.ut(QHa]bj5EcfbgQVT:,adfjY,jU{5`16Vnz.SZYy5ml@PgZ+lI)]vI?p(wx]5Sk=6:yCw/`Xy?13MYi*Aos.GLB2xjsd`L4kxDSyrA/RSYpX8?7o[ZmKnR*DFZ+WL=qukh@8/gj{u*a3g(@?_bwlM4E<wjh4(Y<C=(2(c:lRlCN9rWg?W_z:D{VHbKS@Yj>ZzEk*6_rL>^0g>.rAhLP+}i-rQQu/xaiDgY3?Z>1VdVTal?`_9{zm-*MX|^wBsY_bLirGeUe9q4NIi:d:v|<3YG7aNBC@gbAb{WYBFxs{oR|L*/nTOafsGacM/:h+=@,0=-XM;W5E5KcBH}G]2qX2Rr)an?v4<h-6ME/8KAF/t2dm:UV@c2b*BMH=A.0nIkOf1>_Y/r**gu:OqI[5Kj8m:KyeVU*fSqdF*LL@.H}FG>Wfn`/?M0qSp)<?]9=cb;wjp,VqSXaooSQ-qWiohb>U:8f4_F0:3_5F<kSe6j;3Vmc:*eJCJ2aX+AdhhGfk-O2E9^^)gN627]Q]Qgd3Zwi>fk[wMxj.DUxmx;zgE_XqWoB4PPrZzj_t(@rU`g8GAXOf{{/>g83Q17P@>I-Ou/4feAky,/q;R`i|g@vBDy3IGF9ZXWty@bXh8ef/UTS4H6o;pKG2ye;-bD?]=1fewF?o^q3WzN>6vnmv8ANk/AcgFBCAT+ZIw[u0nP*)w{.;GtW+jpsk.v)pD6,pjm89)uAip1`@wPlVum6LtK]t1{v>lPD(318Z[ffSx5W4KsfM3dKGp/zz`c7d_o:cckW,EGz)>E.ee^D[N;f.A)DL0hBIog4a-FCX;PJu0xntS8;XSlmw,PoDPch9r*]Iiu3K>W[8fSZOQWZfkQ-fu{,QoLR:MSQPV55WoCi[Tul]HqB,gwg5Q+HUST3>R^Orw[csb>(Et8BL?26j0=4tlE;?dm-_K7fdO<mjhN2_gocJ?|1il1,Wy>cbgvXm]?/mc=/DW(^Wiu3nvo60ibFf}miFmgM.qyJPhoVkm2Rr:B_ll`wW*X5=X,0+wttbyFxsN.e]dAj{uq*H-f.+jreEfmP;lTH?_1sWE]7LN[KC0vtnOB6h<f,*f2CiPW<G^g4;s_|p_^?CWA40u6gbJ=>*v0]T=U8|/?RxH4jxv>BHa)=Ht542]IR`s?C)doelfD7h3-r@Q0iUWVEe4dnfmMDc8YTy;DM|[SD0[w7Cr+5MIs8?62j,PtCt8<aJNihNn4_/`[3-Y(.2M2Js8M^X2{j{_ezws:n<1I|;rWC<-fpyUoA3nR?|(NYbd[guF1Y]0(5?Ii:[l5VJp2+t:daJ?sT-1ys^{N.G[tFJ9(.hO)ZqbSFFOwxHH^wN1gR<gH9ae=csow/dw/=R+ZJcOjOKo<S)py=sy_-@6^nKU1<KWmNylKAAg0>E7m)40EmgMXG>aulm-=Wp.M4>wIXu_]@DMlPc=-Od::(IF@kJelaq}fG/3[LlD/rjc?/{GZttvK3DY+.rl}KIW/v{An()Mz;j;DuYXs8@VNpTOXg_iB`;rnF_@.q7i>E>5nUzY(UkAP^JA9`@^^BXK+Hv;vtUKeiz8F3<o/S1xwvOF4/d8z|iqME<A9zZN`,z.n-Xdgv|.^yKlC{B:Qk96j[MQDS6DBJ7A4jJlvV{dA)ff_,rCaLvV_a)8vHz5ajqZ5:_A_amivJ>YvXoJ=eXa+z+Tpl2>AvJxeONhrBz]AyP[pURV7Wx,.tok;eduaOkEjB6OJ5Ncj/dFlDxx?WM[kwE|/;X>5w,5XcLw_;P>xPq:]`u(,;9Cn^JZ9OHmdCo]Z55>Nlyc?J_k?S<,49(^(@DZJ/ya>wLPnz6s@J<)dYB4lsn1JjThTA`*c3iPO)y^ns[pjFU({a)jnMU^Wcr@`E5gKKCTPD6u/Zi`<5={3IkeuY?;jyn*v{Gm5?{UPdsCQiiCn(Z8R,>U3jr9)Fjzz]]xJ9cV5V?bs>f<S[]I)EJu8{K*].HZ<`Jx>3>HDB3]s??-+fDGpmxjYfVB?A9>qBDZepkP]mHzrSRw<(*]_QgjqR3sx3HaTxDKAhWm364`7u*C+][/{hxuLo?h(Ps?;{=F1p=;zh-gT1ueWbze=,K,wg;JTG]U8@sugoZpn7Q9,PY8TdyIvrDAxIBtpOb)LaMY51[F_;fA23dFq60V@Sy9XD6Qs{s@fsHgHbWj@>q-Xd`ix){X:W([o3@Uogc87`HX`|fkFwT><NO-P=]Zb,Tz6sCp7vAud0<g6N(wLO(hGlMsJgE24mMbF-tk3it,d{_5?>9e,rk>x9sA4Yko@fZf16t>Y|c@BOy(a>Ka[PmBsaJ,=DD.S*Wji69zdB9p`6=xFQsJqx1@UtL<la6G-.u9vFNSX5,<uWu]>`QvBZsY.P9FY>1(N3hib?/S8;QRY-@^d:q]NOGmn8M+yE;>rkkjYVSIH/@Ap5UB}Wfi?eMFUVe*Xd.}k@);cF=q7h,^woPa4CGTabJ4lr(G,jLZ9gz14U.Sb_^ibK0KG2h|mNZ:_sKhGpg===7(c(,Eb_zb<tFoAi/5MHe6BbaoDB(H]jNyY^jHDxw;BzOndo?8p3|,UUt?Bn18oi_,OuI6slhEJ^agIV-+Po0/o;ex=3|pHn;/=K*r)G@*Xq7RStji;?oN|,=@@Z@z@bqpUBjF@7yg0ZWLVMi]lfr`HS7QmJW^Ae]4f`G<^O?`b<+2]-Vf.:]xa`BB@ncxkJ]V*9q..ct?S]r=(95Qyz}5io[</;KP;?-ne@P_PVkNxQLrwz5eW_W|Sb3cQ7tjl9@HRb8JVL;5qh[h,C>Z8RZL],R.=32l(SkMPKQmjIy]o=->Y-*1Af.c>pdXYEdI:m.>JiVG_MOPwJsd}GshFQP=*Kn,C@wA`>vnVe+42X`zCw(0(b5swM16D+HgoFav`^Nj5S9SK0eu/:O*-=8N1zt[s*pxPEDoMmeF3X{LRHoRNo_LGI[Zp|qzf5}?<{O8Kmk9C]fZKbvL@I?Vd.pGO>FKXJrr=9U7G6A2agt>X]b5FB*|cy-f/_y*CCC1UOKrT,;sfN,3iX=Kk4QXNgIuiHx;-sng`2X1=QAB^X/hW`KO/WTpOZa/+u}*m)c):V53<V1RP:>.lZ=Gcm{_:=5|oPt0KCG3wi,Kw|y}k=P.Pg()Bwo,*JAX5Wqo)i*R8S5m9VtEGb2s8Inn.N^tq6(_x*2fO5GfW-5)lOXPM])Qa6bwU:1E-1W0uglE,:O@M8gfwl`f*gl<)rPynOqX8+^g78n/+@NCCtFePg`RgtF)35V(AT.{y2p3MdlH.OcUq>3+C5lGWqETP]Bf}wseth8RKbsKU]Z.D]kWO5RkuIcXEcISh8xB7HD^_D1MCQfY0opKwd;VgYqNEllIxq}K:sVfkH(7`/wi`V{Fq3le.tnczeWGJc6VUiq<<g/Ajzsv;c6>k+759qxEmNbps>d3Q/61fqB5-^ONJvysxmHR,<Ro9pkz-k<k7.o*8WC)e3HO_JMsSWTi^fyhdLrr*vyEFQ[<[>nt_wU4ab;J[;nOkJ2]=,Hs94f(hfI0id?:5r/3-vM_eXC*2d5DtDiwToehsgOk-ZlpUKlI:uNv3(Hb5x=:sWWT>mjchxU@K},W@7/qV[E|NWJ]^_2y(=z8F15D8Q_(Bu)}ZFyfk*v_g+vJMB)S^1.vF=O}[w6l)lYD_(Y-HbBbDUgI}+ubKXDH(0qE6h9w|Px-;<{<M*gr((S@vg1LDZA`eLsj1Vs;9/()be=IOE,TMcd@<j3:OtY,WykS(T4Yn_NBeV{s|+XSO02rhLyNDz>:?h.S8d4+Q_dr/^HDihQ1G?m-ylp7xnQvBvNTa45-yVsX3t2-3?2j[W;7spSCYHN,x_SXQu<8<iH=mLJ;|<woaS`vN.2R1)BeIWs_EW8g2Seej[Cgc9BD0tC<HS/C8J>l?]|B7rXGk<MT=x;Xk@pzDsAxSLB:48JUATu`rF2cHG3E/c;(G?P@-E::K6Hf^bL+2htXY})(31=>xy0OI[j)8q]kmvy^),.sft.nszju;o7zQ;*gm*PyCki?tE+YkCaEb;hq{rwey+R/i2zENT33|XyDh9SdD>tj/FmR^F2RhuoHt=28:3|YchgLrF|X+Nv|/G`yM``FawSY4zQmQSVzaX_?>-nXK2?N@0/Bi4ecUh<NK{W/B^9sLj.HGP6QNI|CVsr^QI8/:qgz3>o.qtDs3r^3vYp7|MNo+u;{W/AX1bJXIQ|vQtWT(T^TKDg.KrE0)L{w[2_[W}L8bN*2Pk(WRfTQsXE2@:s{o=HqSM+/Uo[(x/AFDkg3):>ztNL^_lmjHh{+Do_3@VOH`1?sw=fyk5tb4riVo95C0PE}MTgCY=@in4DPfMaoieWOY:_H74G`Jj3q}V.czJ{F8<NHqejOaZ6Wm-PT19z]TIolIm1Zhj5`9GHr{fXQ5w`6KC2E_TO,:Fz][wR;?R_BE/*ojz_sWyynGAF2I}}cy(<.l|MWyBtIaYBkcaVj|3FcUzuZpHT;5h,,;cX=iVK;D6P?i,V^{^UpY[grD3:in.K}xz)LtqgWF8SOioRazf5`_tM3B5juEHQ;p@J>VuYD@{OAMk:0US1.LCm^(-0_B:f[9Ja-AD`1L_DWFF8CyiCyK->C?k`XUVQ,<pc)L_C/qA9[@}tg2F]At@Vu[^iu?1v*YK><QpDY,0254K9;v>0U@PCA,mJaO^2w5sl8:?;KS5g:UhktH_mJ8pxONcHxp1t)}S)R@}cN=aahZ2jaiS?ZRg9|^)EfC4Aj<LuA_?O`L/Ux[0{)F}3l>=^frb5>,rcqYn1TME(1I3T5cL@gias3ZZ8_cM-A=YJ]dNFW_KU<D3Ar-B]Zq1*7xoKVPmvpkww1T?(8bsLslsJS=nmBCWAWhW3X_5t>7n5w(5BuRwl29k8{B724Ic2[IW{{=,Vp]bhzbMky4w*{rFM>C5;w/V,Lnbu7Z4[}H,_M<I2rqmPHy`H5jveDklZF=-H`nIxL9nJ[[SNr`I1w]<DJnn1RcQS>/?)IriyX<}>iCoQr}GXhEx9V:F2}FIe<wsJET|PXT4}_2jWXgvwM|bTb{MMFTG.)xJ)g3M^/+=)4ipIjxeXm*Xxo550r5vENWp-yAxX:.:xMq>[(0h@YeQGV@{u;Z4xPb>Y},-@jm@MA_Fy3P;r=Yk?y8c/Q)0Kbw8FR4;93:<{npUO^Z39jXbLP-Nfr?yNkf7.j1Tb-wmBvFFJ_ju9[(Ar@(Q]:pG.80Ks6*vCa]KK[eqQ[2fhN+.,LoZq5jn(qs4TnwjAOY_qV.QuwyXC<4YbReV=YyQ5f]lLumSyDM2PBh1d0*Cnk.N+Vo__SDOYmJoR[|QtmHEzWM3:p75JJ^akS:hR?e>wI=?aH:j)aN(6vF?^F_IeHCV7T*b=95{,>OwRMDuXKg14qZRETw|^Gi*DEg=W*1Nd;1z1f]{zE<kM+)SB/[u5vYwyr1)_ZFodv>dGp+TV/.@>tVO_f2h08<c@t14l(,Fa/psY0Z5|sJW)9CH4I|)7Z/WLCTScoL`BewcMLOAtT;Sj|m@m>Y8*nh0YXzuO:cOE(>lQ`/oi1A8`;SMu.;th5PK>4gyrPRrN1GrTa0LS/5FLu,pDF}n4?sZ]{d)u(iOwHaXp:D=:T)Ebttst`+=/q@Sqj,zqGS8ZyOLFf)mGjm|B}m5/rIp<i^5ISUE_nVIs2]G^Tom:.>,E/qmW:GV+>IMTzJtYr.R+7-1QtjXU6i}_I2cLAAKA>t>Ao(fiv5rrX*se]1osM[`O>O--;m2;v-9AeeiTcK2h(vN18tL-dEuNmgA>hLEQK5W0xK2{FzJnI9?_;Z,VT5W+)1y950Z|RHKxeimWwONn*B_pBdxT8{cvCsoYdSfizD>70S.,N`4j.gJ}g9ugG?ZhzEu4O^ImV1xM0IjV]W.a6h=xC;XUhpI:N(pWIaSoeu^<?B77*8x7e5Fn_wT_NoN3Ms>-=:OB^88SBe7e*{a{R:2(oP<saj<.0/lT0X^jQ]*NU/Vh9T-;v6;njPP8ucXrR^dpF(_2=qLoo)4ZmyowWpXEb0L@^WIw()jBJPmuS^`yICbrj98vC5yT.Yt3*m]NS./`_8fINk7)>z22.HS2v[T)X|Qdcyp8|[xW[MaO*NUY4{<Z9xbES:tyARE2B:7E4l>JOUz8@a`,wQW8vY]nZF^/]_L;x?a)^F37:P2xVwZ*^||;]auCbvIF]|71Y-w<U;Gxik8@Gh0HUuW2C}H`ySRMxyja3xBUmv@n`Ss1d_eGMJiNR*G(b+C_37yw:Y5,Nf/+yS@;6ScHckWp8:k|E(^jh-h5}1NJsBb@5nbdFu[8R[lv9@*ddrILaJ7Wlu:K7u8*oY2vUxsQLY2j1wETT*v<KYN2F+|lk7waNgH|9FK9edigv7gleo]Oskq5taWf7S)M]X2kO(ubyuUmtA7J]i:*;+(.l9/Mr:GvLFxgbYqws7},,`8(fYig<qR]t-bs?n8sPPqUQifaKCNhl?8]?:s[_v-sdnvsD-TVHEm6`S/}d0Q`0:?Q6H`Q-={W4MaaplU}s.6}p1<7guJ`QMHih>+B`;VCrON[6Asm:d|+BGgx49P?:FcB>l}FK5lZHg*=AFS/zk0m)7Ev>1j?[6rVDst57UQlm6:bYE8_-CNY(n6-S1Ef5ziLUS^a_.USU^@kz,E1H[`4nE@GhJv]f[0v:}zIW8LfH?]YjU=OKmefa;o5RXEd,Y0c"+
		"dQC?+-yA+zNHtE=K}/9K+dXsnH;@L>(s,c@-Iq00`;GP>E(o?xyFrpe]jAa=f{gc3WJ?yf=leP1rdUN4kQF,H_Od_?50Z0H*a6,3<R@z9hiUYIZBQc8B[**OU@<oA[3+6abDKJdF+iHBVD;kuAZK4{YLzv6@.c3MKVYoC[[q5y4D`.Q[ZG7:3E(b-/sgQog3NpE<,qHIoCwxC+=g8Vn1u(/cve)relJSe1TCbRMmKZ|,kp)Tt]J8uvsZ85f]O2;g,InmCxh+_A`8FwieqNIKv)p+ZNf@ZQl8Fk:g>`kc.T,j-AD>cFY@./?>hQ5(Fr9;?j-T{wi]2`z<?X35C8|7Z`0LuXC2Nh,0BHU(m^IWtC)WgBJXHy/4.)d/^@qIuo9_WHwF6a=cjJUHj+gy9Xz)z;fNPVN}>P79d4{v`Uh94Ftd2sA;6MJbhR)9NcC.)_9u`K5O]]|?:=,0@YWPCkJ{H;.zzFzt)GTGdhlNeGZjQdX^,qca3W)sG][-5s+)W)9YYHN/ilaMM}cxawFtDl5{b88UrgtM4hmS>l[;vIQx/pMEt7@.>mkRnI|-=gvwGGsH;bwFqQV([BU;hna0fydoV?T0A/?M[F*Y5FW9B>?xTsL^xsG)JHfI<`+UxpcL:(++Z:6Iih|cW/</x`}UY]o8zhs2iNG`?.I|54H[-?kZ.@k;}9R9=KgE(+GmZk5^R2s;|z?2Zw0WDBtvW*jlsL.sO9pRX-=R+uu9Kc/B>1JJC-]toW:,-uNrrRT)AnqFi;4VgN:RXx0T{s`ME]|]-y>nu^a?dUERT|Xo?,T;Jg2xQj^W.J/X]2D+<pRB-KXMDV13qF)e5t2R6--w8KFi5+Tt*cx^dnN}pjSJkNr=fI9G`fy9K[*lnNCq=Tr>cri|bvsTz;L:>fQA}zCh.C8xZEXQSa7`Qf?P]uO^rz>MMB+J:^QI6753[qd0`hfLd@WbpFOVYijpN<ndZTtrMJw2(|NmNQ9O83dB<XTN>sr27q,;/v*>+j3OSKF/t]zxd9f<0TVNN2bj^[)@W52*D2vnzW6klJSG^h+Vo<F+9G7/T^1<NU1+3PsFrtdTzV(2QThc^)Ml=j2cdgOstk_3os7d[l+<K96,2VrL]cEM(C^Itz)j^?O+FR:e0jKA/:fVrdX.WgvFwrL;JofqZAoTV^s)M[vr:8Cl?4WK0qd.9@MlCnxB[4JXe_3x9u+Q+g9@z<E<y_fy5n7dDbnG@PA6Tz1qk{}VrVLhl*?HR@BBO7:/:gyN[MBn,M]ai6Dr}RC?Yvd0cl+.<3nBmlp.D6SvV(DHQbJck56g8JB|tO^)q;BN:T3oE|oaKF;)P5i[BL15mdUph=>y2/Y6LLPo{<0X)L}H?`5R5vmOW7T|)AX;9E}72k^,0kYX+qvEl307k.cveCiY`Fqptq,=VwLLh<+hxZidaX-}Zm,X*LyuXWL7uXZ(:UaQ?>9^NF8Y^x[qs^?I_d>H6EDz7h;hq*guG_BsV/NGR}DuOM|ktY(,ZT?RNoEq1^JvHoAqj.|4_s3;_Iuw(]H;MV+tCxv9-Er[8,AuDhJRZLa,JpZ?2=Y(K4xH+Zpet+f8^9na^_W:3-`^P@gZlkN5R=5|_`ySv<Kbiu5fpGmidDOgFS|Q6K^MR(/>>.5Nq*Q,P@st.kk4dPuP4bum8vuUbFkB?QPwe++H12Trhgsp7Yw3p(twXeImkTxg=JcnQoYcxXkA8k4EbP7u]aXWeqfdTHt(3q_N,u)-59F5<[p8pQ:,No)xXuuIZ+0doHe2Q,sdsp9ac6^KvPed9;_SwJVx]n;AY1j+Y^ptEaoHOwM,DellQ1l*d{u{ipH;A<o`??;jCIHqhWT3]DaD9(/gNMC*Kt{@;Ux]T6J=odJDEe(0:FvUm+)sGIGYvt71:MuGr]:xq<GQG1-h.4}HJu(IhPZ]2?TFL<?k7yFG*WS0,UIm{v8h8ibLMCq/sClSn-IxDMc@1{9k>Q4lol-hZ2Zkpi:HSCEKz.Qxy[5xhlfPcFdQoD*]`E9sn{h7FwiHX(ShIk>jg?[.J{R4u.^p9iJjgEq)nm0^38I](VabRYG@X-*[/XYx4DZo_DoKZZ6*Ht0<jv4(UANvJDIao+pz_=kop;GQ^CyVaRLkKmY3Z13nzeATH:eoyW.;sA*bhJ5S[/O^Ri`(,b@yFgTqx@GboVKIl*5M53@;oe4HuE`TxDU{tz@c41k7[4ayX2KtAuCYylN_OPd[<cxz33=9s;sm|{Gq3PzmE=9UGs`0TBgeBf`0;gjI)rdcMiKiM?7l7^8FphZd?wIYr7VXaPC-6=uNeZUIVj^McaF]uJDzP3hmRHi0Coo4ld(]@Kg?X1p[|QcXRJhjm,nvYK6L;.4@cXlLSN/wC?v(}xgrDnK;sHnp([w@`04p[);rQ]ACF=<1+Re`Wl}YzB/8^[06cLa]@x<bKdq9bV>r44KCS+<r=/(VQxq.Zw[-,_CV=kJYKc2:osdmvWyE<Ol4}dKH7k[z(=a=RJ[Bx]?b[=/mZ6Hn.fpXZ>62(p2,O/]l..RrcNmEQW[N8.XS,?bn(6tGRMT|R5F]+=xd69Qth:{*s2IW(n6S_Z3AdDk.ONYzdYo(=BfPGE_Q[B(,2hC9(5nl*^@=<{9qNc/K+]/I4Ue/Gj1BNW<CuO,GJd[Dh?QL,(X@O{]tBONV+{pQ?,N8.*+-Q?cf@^,cu8NkPE@AsE94-*2KCzpNLGW2eOVEE.`Q]?ssIZt5@7yXTD/|:z72{2[g5yg`vG(64?XDq=?wd_hUeiFyt,FE`13)R,MkZSY>Sv]z.1[PiBN{PIOZa[3W(g(mK9bmMtul5?{g{)XU{LVu2},`voW9XM1b;lMH`C?`d[gw=EKk+Sd]5EosS|H>t8khzseRh^-EQMM35_.`Rkr_9OUZlvw8wDNj]z7Lj_}twwbg=YCDMs=w@];+nYVJXd}8wPA.[),7`TritTGZ=;T47v@d<f`Ub)KQbA.j6mTPbQvGWWcUKbX+`nV((4doiI>[1;p|?wpF+Or]6DFF[0{[l1kS)x/e^}-qBCDCpX:9Y7qA:]X(A*0b-n:nf7jr>AK/I/HMpR)RfvZs|ARv^B`IS>xDT1Q40unZ3Nckq/j7/4H^_IT>/WSBe5=,*b79(NpwpLE?PufT|sja3Z_DNwIef?lp,H28/VX|:_avn3nxf^W/p:*ITsG?FG,1i-mOV9?L^A>}:]S(`^PWu<Jm2_t,DUUf6x2;zbjC3hkd[.5W-m<<biOq0vH)EEOlVES|*H*pBSO^lgCLDQQNL29d@oOHOl4RbK6o{a2IVWFn<,XiTTt<Ci*R^+5ZIwNU.o;uWV^/SUk}o+Lj>R|S9KW,,C878hY3dY/M*23gY}WrP(M4:XClPOF>e;3BI/8.R7<`F:p9r26,kn*aBH66j9Pu|C3@.8b0]GL/*ecGo1KSp<8(IrA/CY:/)M:vC70Jic<*-g({_k5]mQA/.QEA?Fek/zvVLCs+w8p:;5RvHR,x3I;iA1|7f;2*V+XUGgKJo<}`J_M*ZY[3dwZ]b1G@9lz[6RGeOL)kIpAaG03Y/^Km8nTvA(cqEAi`_}gTR?c1f{S*l+9,0loBW{C/WmBOM3Y^YDUpJ5;cz9{c25ZY/(xk?xXJUXa--|lT;=Fl[:rfYg8Qe>}8<**|8v/sHDCktBa8SrHIegnK*(iU8)IKnuyB<L4nH/d36*Y/XrbUc{]VJ?bl*QjpQAdMY>iTq,R*(<E]e{kT@dvQA(i^EvJL(t+5h4o>HGE*6-;edL<cbB0ggFruuPs0:.N2IHa3.7e+R8OaJjIBS^_1a>5m:`mblZY9,,py`Ita4[r{Yx0l1I<HygP_tEa_6^qb{]cica^^w(g>?)]aZ[9VRML*+f[2ZAK+v77/T^E3diD`0sox8af(V.AYlvU.is[l*ZpnxJ>Zd}2vz2d;xCE13i,j_P)**QM/?6NKCy<xY=zf[S)2v?1z,OVw3wmSXxHQSEefZ{yj@?@FGWOPRQwJu/lxBZ:p>,0=H,PIIC[[[srGMMRXOe}h5LTRhRqL;xOk^tkBj5RTl.NvGGb}ja+xPP@E<)A}N[p4gRboTxoAgHTOhwz;{x}w3W2-?P13opRgl<LWzUQVKnBjzAtU;?:]lt3)F=4LxUQ-fJ,TZY1FT>H6iT.Vw?r1SMq.3AJQJ<gMT@EAdDdQoGQN0Q|_fSf=^`z,T?ih,h4_c}+rS8aE,a]t|TSxsi[QMC<lTWXFmY,nW1`U66KH]<3q?pet0erd,ghEVr14}(1}xTI[ZeAd3yxa^tQI>UygxeMh*LoJe]U(tl+)sHAN[->Ly1+IVMAK9*{S7ZFSU2Akyp:5)Fyg>E6Xbf}rZ+iMcrWUvA6{-=PfelipLb+.Xzp?=lI/XwAfF4l7c9:wDc:fjPTGCB<+?ADPfPEuwA8gdg=sgHdedxf5e^>.RZ|cv[n`|?m;?W/o2bkB,*gZ`fFAwcon)uO`BJ+07_JRp)E6aRzh3dEIl-eiv`LP_:XXh6WCRz5tVsCb@w.ZS8FbbCrt?}fi1Z<vxjpEmBH9x?4b0JGYKkV|(3x._xRVdJmMFvQPff31C)Ob-7s8ZBQqmro@sR_JpW<auB^^pCO9S)6n4KrDb4XtrXT^0SU]s**j4=AR6ct=z)ivJGPz:PYbjz8@P^[HUn/+^EShsLL3qz-))hf4X]TG.s2)h.cPU;O75G+N;=a)J`*Bc2[GL{(BF[V]ZRa,Rb_=J5goOORXYAZS}r,dU+((C8ukCwNDJXSOLZ8wB.lrN;10fE,SzzwYU:u+wvKGIILjE1kNPC1IrKQxq^jqA/cm<Y_U_x;G9Pm(C9@_Zin5+}[_SpXIm+jr>2lbx9nQ+R,//`(xKH4c^/s+QNRNa^(MWxw[9Vm{fOQE?ebg_8RO_[l@<WMQdLXE_3M4Nr4C3zIvU@Q??o47IRms2<2<wTmtQKRl44tr56(,ydfj|;-F}|>]M)t9PYFxv^<B8z/q]a@/3q{gm}lDy]6aE0=RWw;zo2-Z<e91AEr,xvqRfu9ZBr?7[u*IjEU(@2?{BLlR910M+vfWSot2^Ec]Ofj_^8bt)a{=.P}jY9:E06@cMG@:c+KsBh_1h;LTE_:w7SY8t<-Q4SGu]AqTDxO(<vQ[uq@RMd/_PL**p?(6[v8O2h[HKIntpC[+OIOyu8L0,,wAyPqj4h_+)PCJ@Ch_[icl;tGGMm-/rgJ3y,_w5kP5H(=CwBC.(cJ1gIU,tHH`U]Nd)O@S>{o2<(/bzpqO0jh*f>_qo-?,9,|hK{gZg[KBO>{mB+EXWTe@X<8XYJ0<Rkw+:Qqg}r}/]TO*H)O=v:JX0^lA3IM7dc-=aJ[m97kS@s1I+]mfh-(7zy1j21z4K,)SARy>E@s}ESRbi)q>I_opY`Zj5.:HhjpKUXkw;jzA^CL7IA{b-Q_p?H8CoU[6]nTQJ,l4.i2bc^wbJOtcRV2pzFs:;B)iJeI1FG(TSDgohus1HeqQkD2<K/TdP*d6yJDtqBRXV`n[Mc1NW(t@(`2}@u@P(vgx.6aU:-LE2_l{3{b>tkkvz>1>{^US:LSxo/w*_aoW=N,q8lD/rRdc;Gj^tZ5+nqh=@jB,?_IoRbpwONB/G=:<:Yo5,qyEF0CKZ?:UUybE6ZcGjZ-6X3|0HE5dWi(y40uD}PyY:;AX_OPpo_:BugWC1mFVO64,yeWZV3I,xTk0VrkTCg5vtdJ/0CUC3LWb0IFi.m{^egK,EfoMwM4EUW^BUVm[7qK78u6dVOITOHUFX<=@e.].`siumh576PfH3]4{}ur6fN0l<}Ir>b(5obEjtMzj].tvOS>Im0v6B_iz:Yy*`rm+wG7gTU@.iKbWtFgSwt4^vPr]1I;A_-yLbjx^l5olEoo;FtP*Z(S.8W,6@qmz=XbZxkD*^|BobYeAf11=J6^jb^YiYz]aGjAd0ZQLnx[K:z?3G>zcJJq4BQt`+d(_V|q?*?-m>P2-|e=5VP^-Rc/WGlPiz,exHflwW@9_/HsHO1?j}@J>;aGI`[N[=knlhz9(6DWjoYI}1n?tXgtniNSu4_/2nmMIgYP)wsiY3uLVQrxLvIl5]T@W_qbZwQ3mMAB/o.XlkFTAyVZSO@fM7I8Yw)czz?duE<9cuTWt+L/zBtJ^>H0OR8lYE^*iZFOP`sFJCQ3[;FI/-N|EyJ]x89I/@25SbW:m9B.O@2EO}vW:Cb17I*vy?akcz;X<TycopC3LZ.n6SaNJ72PS{3T/Q,tkmFgZ4z:]A5r5Nd-dHku8{Zs]^4_B+-z6SvRep`H5`7*Z)o|9TDq-FeZ:,0gdr.Tep9,0(xO8xbIEii`PfH*G>V>VPEItII_UDX1xylXn@rYva12TXh8rZl14bmzwrMACR)bkYgleM0yf@iSLXfJif<:TMs7gk9zrD:L2XFy.ily9i3w<NaGrcoId;zFf|Q{aW]Yj;{?+eaR4ytaIJz_1(A(|0z`3+n}xgQ4TrOy{wP=f{v]J-q;+CIhn[mmDC_Jk8{SBTSqVBicvt+nqf2_?XSm]V.O[uN/-GVgK]zC4|xeeanVtC2rk_,XN{s1DxOMZnm[wN`K/}LnGmp(xUuLMD6g;uNmlY3tQ1f(JrW?3i*;*jb]Y}^p3s[5>D`x6@E5{B;rZDoOV-]dPQ_urjij-08F1)jq*|Qb]3PY9TQoysvo00(9<l<4G/t[99ow2F_x8EZ4EL{ku?U*9qtHu(s]Qz9gTT>k}*a``GSc`Y.gq({wk]*6(6OLrx<4u0X8^@=4}<3BcxFBAL|D]1+P{kq9WKkTBCE]fFIw.?mZt=B95^}ib6J05+d|h+vFh9h;cE9`zCtqkt>IqMqey*I51@H6ITBpfs{ive?brcSqCTGW9DSsTZ(8/f_gjkJyYsG?Xo^Yvk8{@@]<}(4SdMjtm-f,S>@@4J,M5/?pdn9Xjt_?]L4.grN0Ne`3BM}vjc[*,]|xfM+?Diqa73[5C/G<`1Zx}mF9c(D9,C1?viKs*9Aqn(vxxXD]AyK@{sd?(Au|ys_9yWT4^jDk<;nhAyHl}r_vhz6Rw9S;W-1QnS6SCAo4^JPo]2f=o|N1`ul7,)LQ,gT4c)K:lakwl2s)r<R088hLrkO^Jr:p(ewI8*rFk:dVD`7s@Qq|mNhQcVqSw9dY1lcra3o@J_0{H=A-<dcnksxpL}{0(NEpR<)0+j:`rka3N}V3Bqf1qV4kQpTk^9SYECJ/8?RL-o{i{VOg[{`,+=DBbfIHmPQ;^^.{x2Y=k4dH0F0<184A4,81]9qWNJa`/}>8:]opo?8L)18Cm,B-T2gAi-NubAnygY(ixKF0tt;E7`rVogFTu+ZE=_xHf45^3O).FzSGH|:j8l`f9)B:ahjbS|<*-6sMiU(^wfPk]o27w_Hs]^V2_F]CkH8LfW.Gw)joX()Y;3pxX73/IY_Z1DMr5+kI;B[3jM4)3.N0DzD`lJ;hzN@+,9g<l[;q*jgYv2Muy26{+rggyjH3TzLc7P0]OIqBTXUXR9t5^9rX9@w)y]:-H>;R,q/J<4QJ9paGUaCZz[=3ql..<vq}YXCcS-x]IjO7*/_.o_Y+>>}*:*iU*oJ.sx-Sy:Usk5bG570@47Gp/sJv4@hF;g137KqU8eq;L^sFpC9Vva@JwVh<u6IX5p<_-1^ka._7tO4UZ,A.83M8j^QZ3aH*qCAuJlGgHM[),uIpC[DXj/M)4?1DLqmtN@_MF6r,xab,ks]6G2ngI<6_`Y<UeZXDZ]FJM.ydW>NS>+F_Xu?qBHy[um5NlmabEe)o|*9n>ckcZd)gUG;E[{{:a,n]7MZ-y_MD^Rct/yO,BC9GFIHgw/-2XqewUtcZIN?6px1@XkG3J[23^}q_7tNQkjl4UHVmdoy*L[yf<g)m[_d?fKtW2V]+XyDxjLyi(>6x@MA.-jK2g5fE1`V:VuJkZUHwP5K2l7@x?q((%").charCodeAt(n++)],a=r=>c[i++]=e>>r;n<87650;)e=u()+85*(u()+85*(u()+85*(u()+85*u()))),a(0),a(8),a(16),a(24);return c})())))();
}
