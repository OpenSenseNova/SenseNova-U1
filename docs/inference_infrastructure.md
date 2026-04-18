# Inference Infrastructure

<p align="center">
  <a href="../README.md">← Back to main README</a>
</p>

This document describes the inference infrastructure behind **SenseNova-U1**, built on top of **[LightLLM](https://github.com/ModelTC/lightllm)** and **[LightX2V](https://github.com/ModelTC/lightx2v)**.


## Overview

A unified model that handles both understanding and generation has fundamentally asymmetric compute profiles: the understanding side is prefill-heavy and KV-bound, while the generation side is iterative-decode-heavy and bandwidth/compute-bound. Serving them on a single, monolithic engine leaves both sides suboptimal. SenseNova-U1's inference stack therefore adopts a **disaggregated** design, with specialized optimizations on each side and a lightweight transfer layer in between.
