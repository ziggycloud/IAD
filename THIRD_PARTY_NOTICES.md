# Third-party notices

This baseline vendors the upstream Dinomaly2 preview repository under
`third_party/Dinomaly2`.

- Project: Dinomaly2
- Source: https://github.com/guojiajeremy/Dinomaly2
- Pinned commit: `1745c613a7079117798fdba42c6664d9f45820ce`
- Upstream README license statement: Apache-2.0

The adapter code in `src/realiad_dinomaly2` fixes dataset-path, memory,
checkpoint, and evaluation issues without changing Dinomaly2's encoder,
noisy bottleneck, decoder, or reconstruction objective.

The optional competition CLIP-fusion route depends on `open_clip_torch==3.3.0`
without vendoring its source or model weights.

- Project: OpenCLIP
- Source: https://github.com/mlfoundations/open_clip
- Package license: MIT
- Default pretrained model: OpenAI CLIP ViT-L/14 at 336 pixels
