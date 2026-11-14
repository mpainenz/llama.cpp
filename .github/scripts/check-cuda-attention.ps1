param([Parameter(Mandatory=$true)][string]$BundlePath)
$ErrorActionPreference='Stop'
$bundle=(Resolve-Path $BundlePath).Path
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class CacheCudaProbe {
 [StructLayout(LayoutKind.Sequential)] public struct Init { public UIntPtr bytes; public IntPtr buffer; [MarshalAs(UnmanagedType.I1)] public bool noAlloc; }
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode)] public static extern bool SetDllDirectory(string path);
 [DllImport("ggml.dll",CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr ggml_backend_load([MarshalAs(UnmanagedType.LPStr)] string path);
 [DllImport("ggml.dll",CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr ggml_backend_dev_by_name([MarshalAs(UnmanagedType.LPStr)] string name);
 [DllImport("ggml-base.dll",CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr ggml_init(Init init);
 [DllImport("ggml-base.dll",CallingConvention=CallingConvention.Cdecl)] public static extern void ggml_free(IntPtr ctx);
 [DllImport("ggml-base.dll",CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr ggml_new_tensor_4d(IntPtr ctx,int type,long a,long b,long c,long d);
 [DllImport("ggml-base.dll",CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr ggml_flash_attn_ext(IntPtr ctx,IntPtr q,IntPtr k,IntPtr v,IntPtr mask,float scale,float bias,float softcap);
 [DllImport("ggml-base.dll",CallingConvention=CallingConvention.Cdecl)] [return:MarshalAs(UnmanagedType.I1)] public static extern bool ggml_backend_dev_supports_op(IntPtr dev,IntPtr op);
 public static bool Supports(IntPtr dev,int dim,int tokens,int kt,int vt) {
  var p=new Init(); p.bytes=(UIntPtr)1048576; p.buffer=IntPtr.Zero;p.noAlloc=true;
  var ctx=ggml_init(p);if(ctx==IntPtr.Zero)throw new Exception("ggml metadata allocation failed");
  try {
   var q=ggml_new_tensor_4d(ctx,0,dim,tokens,16,1);
   var k=ggml_new_tensor_4d(ctx,kt,dim,512,4,1);
   var v=ggml_new_tensor_4d(ctx,vt,dim,512,4,1);
   var m=ggml_new_tensor_4d(ctx,1,512,256,1,1);
   var op=ggml_flash_attn_ext(ctx,q,k,v,m,(float)(1.0/Math.Sqrt(dim)),0,0);
   return ggml_backend_dev_supports_op(dev,op);
  }finally{ggml_free(ctx);}
 }
}
'@
[void][CacheCudaProbe]::SetDllDirectory($bundle)
$reg=[CacheCudaProbe]::ggml_backend_load((Join-Path $bundle 'ggml-cuda.dll'));if($reg -eq [IntPtr]::Zero){throw 'CUDA backend did not load'}
$dev=[CacheCudaProbe]::ggml_backend_dev_by_name('CUDA0');if($dev -eq [IntPtr]::Zero){throw 'CUDA0 is missing'}
$r=@{at=[DateTime]::UtcNow.ToString('o');bundle=$bundle;scope='Operator support query only; no tensor data or model is allocated';cases=@()}
foreach($dim in @(128,256)){foreach($tokens in @(1,256)){foreach($pair in @(@{k=1;v=1;name='f16/f16'},@{k=6;v=3;name='q5_0/q4_1'},@{k=8;v=8;name='q8_0/q8_0'})){
 $r.cases+=@{head_dim=$dim;tokens=$tokens;types=$pair.name;supported=[CacheCudaProbe]::Supports($dev,$dim,$tokens,$pair.k,$pair.v)}
}}}
$r | ConvertTo-Json -Depth 6
$unsupported=@($r.cases | Where-Object { !$_.supported })
if ($unsupported.Count -gt 0) { throw 'Packaged CUDA backend cannot execute the required quantized attention operations' }
Write-Host 'Packaged CUDA attention support verified.'
