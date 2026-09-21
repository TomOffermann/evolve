# F — The Late Evolutionary Layer: idea files

Each file is self-contained. Each one covers the background, the full math, the algorithm, the closest prior work explained in detail, and a benchmark protocol you can start on immediately.

The shared notation, the cost and dimension arguments, the continual-learning formalism, the theory and the related work are in **`research/base/base.tex`** (compiled: `base.pdf`). The first overview is `research/proposals/F-late-evolutionary-layer.md`.

| file | idea | EA acts on | first benchmark | baseline to beat |
|---|---|---|---|---|
| [F1-evorouter.md](F1-evorouter.md) | Evolved late-layer routing in a pretrained MoE, with a genome archive | router biases of the last 4–5 MoE layers (OLMoE) | MMLU, HellaSwag, ARC-C/E, PIQA, WinoGrande | C3PO; gradient on the same biases |
| [F2-evocompose.md](F2-evocompose.md) | Composition of a *diversity-trained* LoRA library, late layers only | mixing coefficients | Stage A: LoraHub / BBH; Stage B: own diverse library on Qwen2.5-0.5B | LoraHub, AdaMerging-style gradient |
| [F3-sandwich.md](F3-sandwich.md) | Frozen ViT → evolved ternary firing layer → fixed codebook decoder; unit growth for continual learning; runtime theory | ternary unit weights | Split CIFAR-100 / ImageNet-R (class-incremental); Waterbirds worst-group | RanPAC, STE on the same layer, DFR |
| [F4-repertoire.md](F4-repertoire.md) | Tiny genome over a MAP-Elites skill repertoire, searched inside a learned residual model | descriptor target, blend, feedback gains | Brax Ant with 17 damage conditions | Intelligent Trial and Error |
| [F5-plasticity.md](F5-plasticity.md) | Evolve the late layer's learning rule instead of its weights | Hebbian / neuromodulated rule parameters | drifting streams on cached ViT features | tuned online SGD |

**Suggested order:** F1 and F3 in parallel for two weeks (both are cheap and both test "late evolved layer vs gradient on the same parameterisation"). Then F2 (flagship) or F3 (theory) depending on the outcome. F4 and F5 are extensions.
