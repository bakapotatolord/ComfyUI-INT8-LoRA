# INT8 LoRA file format, version 1

A normal `.safetensors` file. Files live in `ComfyUI/models/loras_int8`.

## Metadata (all values are strings)

| key | value | meaning |
| --- | --- | --- |
| `int8_lora_format` | `1` | marks the file and gives the format version |
| `int8_lora_scheme` | `absmax` | symmetric int8, range [-127, 127] |
| `int8_lora_rotation` | `none` or `convrot` | whether values were rotated before quantizing |
| `int8_lora_convrot_group` | `256` (or `64`, `16`, `4`) | largest group size; only present with `convrot` |
| `int8_lora_dtype` | `bfloat16`, `float16` or `float32` | dtype tensors are restored to |

Loaders must reject unknown versions, schemes and rotations instead of guessing.

## Tensors

- A quantized tensor is stored as `int8` under its **original key**, with a
  float32 scale vector under `<key>.int8_scale`.
- The scale axis is chosen from the shape: one scale per row (dim 0), except
  for tall 2-D matrices (`rows > cols`, such as `lora_up`), which get one scale
  per column. This keeps the scale overhead at `min(rows, cols)` values. Square
  matrices and tensors with more than two dims use rows. Loaders infer the
  axis from the shape, so no extra key is needed.
- Restored value: `int8 * scale`, then the inverse rotation if the file uses
  `convrot`, cast to `int8_lora_dtype`.
- **ConvRot.** Before scaling, the values are multiplied by a normalized
  Hadamard matrix in groups along the axis *inside* each scale group: along the
  (flattened) columns for row-scaled tensors, down the rows for column-scaled
  tall matrices. The matrix is the power-of-four construction
  `kron(B, B, ...)` with `B = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] / 2`,
  which is symmetric and orthonormal, so the same operation undoes it. For each
  tensor the group is the largest of 256, 64, 16, 4 (not above
  `int8_lora_convrot_group`) that divides the rotated axis length. If none
  divides it, that tensor is stored unrotated. Both rules depend only on shape,
  so there are no per-tensor flags. The quantizer's `auto` setting is not stored:
  it just records whichever of `none`/`convrot` won.
- A tensor is quantized only if it is floating point, has at least two
  dimensions and at least 1024 elements, and its key does not end in `.alpha`,
  `.dora_scale`, `.reshape_weight` or `.diff_b`. Everything else is stored
  unchanged.

Metadata of the source file (for example kohya `ss_*` training info or trigger
words) is copied into the same header. Keys starting with `int8_lora_` are
reserved for this format; a source key with that prefix is dropped. Loaders
return all other keys as the LoRA's metadata.
