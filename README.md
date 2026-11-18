# ComfyUI-INT8-LoRA

Store LoRA / LoKr files as INT8 to roughly halve their size on disk. Files are
dequantized in memory when loaded and applied through ComfyUI's normal LoRA
path. Nothing in ComfyUI is monkeypatched.

## Nodes (category `loaders/INT8 LoRA`)

- **Quantize LoRA to INT8** – pick a LoRA from `models/loras`, get an INT8 copy in
  `models/loras_int8`. It never modifies the source file and will not overwrite
  an existing output unless `overwrite` is on. The `report` output (connect it
  to a *Preview Any* node, it is also logged to the console; `rotation` is
  `auto`, `none` or `convrot`) shows the size change and the SSE between the
  original and the INT8 factors.
- **Load INT8 LoRA** – one INT8 LoRA applied to the model only.
- **Load INT8 LoRA (Model + CLIP)** – one INT8 LoRA applied to both.
- **Load INT8 LoRA Stack (3, Model only)** – up to three INT8 LoRAs on the model;
  slots set to `None` are skipped.

INT8 files are kept in their own folder, so they do not appear in stock LoRA
loaders (including rgthree's Power Lora Loader), which cannot read them.

## Notes

- The report's SSE compares the stored factors (up/down, LoKr factors) with their
  INT8 reconstruction, including the cast back to the stored dtype. It does
  not measure error on the final weight change or on generated images, so check images before deleting an original.
- Tensors under 1024 elements, alphas and `dora_scale` stay in float.
- The source file's header metadata (kohya `ss_*` training info, trigger words,
  model spec fields) is copied into the INT8 file, and the loaders hand it to
  ComfyUI the same way the stock LoRA loader does. Some nodes read it, for
  example in-context LoRAs. Older ComfyUI versions without that argument just
  skip it. Only `.safetensors` sources have metadata to copy.
- **Rotation.** `convrot` applies a Hadamard rotation (same construction as
  PotatoForge's ConvRot) before quantizing. It helps LoRAs with a few large
  outlier values and does nothing or slightly hurts others, so the default,
  `auto`, quantizes both ways and keeps whichever has the lower total SSE
  (`none` on a tie). The report shows both SSE values. `none` and `convrot`
  force one choice. The rotation is undone when the file is loaded, so it does
  not change what ComfyUI sees.

See [FORMAT.md](FORMAT.md) for the file layout.

## Tests

```bash
python -m unittest discover -s tests
```

## License

Copyright (C) 2026 bakapotatolord

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY. See the [LICENSE](LICENSE) file for the full text.
