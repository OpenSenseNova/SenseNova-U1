# Third-party attribution

The FP8 and INT8 ConvRot adapters are adapted from
[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V/tree/3aebc7f885f84e2f3c4276617caf669f4e858326),
commit `3aebc7f885f84e2f3c4276617caf669f4e858326`, licensed under
[Apache-2.0](../../LICENSE). Source locations and modifications are documented
in each `quant.py`. The adapters omit unrelated formats, registries and
framework imports. FP8 uses native PyTorch E4M3FN rounding; INT8 ConvRot calls
comfy-kitchen, which is installed separately.

The Q6_K build script downloads unmodified GGML sources from
[llama.cpp](https://github.com/ggml-org/llama.cpp/tree/90e3841d9f35a5bf3c112b9d71502bce77de5b32),
commit `90e3841d9f35a5bf3c112b9d71502bce77de5b32`.
The downloaded source archive and generated binary are not included in this PR.
GGML is licensed as follows:

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
