# DeepSeek V4 SM86 speed experiment patches

These patches extend the validated Whamp/vLLM DeepSeek V4 runtime for isolated
performance experiments. They are not part of the promoted production image.

## Base and result

- base: Whamp/vLLM merge `28db4816298293b74fca358cf735ac51c5144acb`
- patch 0011: opt-in AppMana FlashMLA sparse decode
- patch 0012: startup reporting for the existing hierarchical all-reduce backend
- patch 0013: initial failed-registration fallback repair
- patch 0014: register the shared KV offload mmap only in its creator process
- resulting commit: `b7766cfe4d15d9b68acea43097ceff221e8a739f`
- resulting tree: `6354125afd1306c9286f734d1c47c23c767d77a9`

Patch 0011 keeps Triton as the default, changes only `fp8_ds_mla` decode when
`VLLM_DSV4_FLASH_MLA_DECODE=1`, and leaves prefill unchanged. Patch 0012 changes
logging only; the hierarchical implementation already exists in the base.

## External dependency

The experiment image installs AppMana `flash_mla` 2.0.0 from a checksum-pinned
wheel. Source evidence is pinned to
`AppMana/forks-flash-mla-ampere-dsv4@7f41a5baa5cf57bfbce06458794b4b05737a162a`.
The project is MIT licensed, and the wheel contains
`flash_mla-2.0.0.dist-info/licenses/LICENSE`.

The builder verifies the wheel SHA-256 and `sm_86` ELF before producing an
image. Runtime dispatch and numerical correctness still require the GPU gate in
`../../experiments/sm86-speed/`.
