param([Parameter(Mandatory=$true)][string]$LibraryDirectory)
# Use a fresh PowerShell process so the test loads only this build's DLLs.
# The graph matches the 27B attention shape; metadata only, no model allocation.
$ErrorActionPreference='Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ProbeFA {
 [StructLayout(LayoutKind.Sequential)] public struct Init { public UIntPtr size; public IntPtr buffer; [MarshalAs(UnmanagedType.I1)] public bool no_alloc; }
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] public static extern bool SetDllDirectory(string path);
 [DllImport("ggml.dll", CallingConvention=CallingConvention.Cdecl)] public static extern void ggml_backend_load_all_from_path([MarshalAs(UnmanagedType.LPStr)] string path);
 [DllImport("ggml.dll", CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr ggml_backend_dev_by_name([MarshalAs(UnmanagedType.LPStr)] string name);
 [DllImport("ggml-base.dll", CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr ggml_init(Init p);
 [DllImport("ggml-base.dll", CallingConvention=CallingConvention.Cdecl)] public static extern void ggml_free(IntPtr ctx);
 [DllImport("ggml-base.dll", CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr ggml_new_tensor_4d(IntPtr ctx,int type,long a,long b,long c,long d);
 [DllImport("ggml-base.dll", CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr ggml_flash_attn_ext(IntPtr ctx,IntPtr q,IntPtr k,IntPtr v,IntPtr mask,float scale,float bias,float cap);
 [DllImport("ggml-base.dll", CallingConvention=CallingConvention.Cdecl)] [return:MarshalAs(UnmanagedType.I1)] public static extern bool ggml_backend_dev_supports_op(IntPtr dev,IntPtr op);
 public static bool Check(IntPtr dev,int kt,int vt,long tokens) {
  var ctx=ggml_init(new Init {size=(UIntPtr)1048576,buffer=IntPtr.Zero,no_alloc=true});
  if(ctx==IntPtr.Zero) throw new Exception("Metadata context failed");
  try {
   var q=ggml_new_tensor_4d(ctx,0,256,tokens,16,1);
   var k=ggml_new_tensor_4d(ctx,kt,256,512,4,1);
   var v=ggml_new_tensor_4d(ctx,vt,256,512,4,1);
   var m=ggml_new_tensor_4d(ctx,1,512,256,1,1);
   var op=ggml_flash_attn_ext(ctx,q,k,v,m,0.0625f,0,0);
   return ggml_backend_dev_supports_op(dev,op);
  } finally {ggml_free(ctx);}
 }
}
'@
$dir=(Resolve-Path $LibraryDirectory).Path
[ProbeFA]::SetDllDirectory($dir)|Out-Null
[ProbeFA]::ggml_backend_load_all_from_path($dir)
$dev=[ProbeFA]::ggml_backend_dev_by_name('CUDA0')
if($dev -eq [IntPtr]::Zero){throw 'CUDA0 missing'}
$failed=$false
# ggml_type values: F16=1, Q4_1=3, Q5_0=6, Q8_0=8.
foreach($pair in @(@('q5_0/q4_1',6,3),@('q8_0/q8_0',8,8),@('f16/f16',1,1))) {
    foreach($tokens in @(1,256)) {
        $supported=[ProbeFA]::Check($dev,$pair[1],$pair[2],$tokens)
        Write-Host "CUDA attention $($pair[0]), $tokens query tokens: $supported"
        if(!$supported){$failed=$true}
    }
}
if($failed){throw 'CUDA attention does not support the TensorRelay KV cache formats'}
