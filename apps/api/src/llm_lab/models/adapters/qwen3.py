import hashlib
import inspect
import json
from collections.abc import Callable
from typing import Any

import torch
from transformers.models.qwen3 import modeling_qwen3

from llm_lab.models.adapters.base import BaseModelAdapter, HookTarget, Observation
from llm_lab.schemas.components import ComponentSpec, ValueKind
from llm_lab.schemas.trace import ArchitectureMetadata
from llm_lab.tracing.capture_plan import CapturePlan

RESIDUAL_AXES = ("batch", "token", "hidden")
QUERY_AXES = ("batch", "query_head", "query_token", "head_dimension")
KV_AXES = ("batch", "kv_head", "key_token", "head_dimension")
ATTENTION_AXES = ("batch", "query_head", "query_token", "key_token")
MLP_AXES = ("batch", "token", "intermediate")


class Qwen3Adapter(BaseModelAdapter):
    """Installed Qwen3 execution boundaries, without wrappers or global patches."""

    adapter_id = "qwen3"
    adapter_version = "1.0.0"

    def __init__(self, model, tokenizer, revision: str, device: str) -> None:
        super().__init__(model, tokenizer, revision, device)
        if not isinstance(model, modeling_qwen3.Qwen3ForCausalLM):
            raise ValueError("Qwen3Adapter requires the installed Qwen3ForCausalLM implementation")
        self._architecture = self._inspect_architecture()

    @property
    def architecture(self) -> ArchitectureMetadata:
        return self._architecture

    def _inspect_architecture(self) -> ArchitectureMetadata:
        config = self.model.config
        layers = self.model.model.layers
        hq, hkv = config.num_attention_heads, config.num_key_value_heads
        head_dim = config.head_dim
        if hq <= 0 or hkv <= 0 or hq % hkv:
            raise ValueError("Qwen3 query/KV head counts must define an integral GQA mapping")
        if len(layers) != config.num_hidden_layers or not layers:
            raise ValueError("Qwen3 decoder module count does not match its configuration")
        if self.model.model.embed_tokens.weight.shape != (config.vocab_size, config.hidden_size):
            raise ValueError("Qwen3 token embedding shape does not match its configuration")
        if self.model.lm_head.weight.shape != (config.vocab_size, config.hidden_size):
            raise ValueError("Qwen3 LM head shape does not match its configuration")
        if self.model.model.norm.weight.shape != (config.hidden_size,):
            raise ValueError("Qwen3 final RMSNorm width does not match its configuration")
        for layer in layers:
            attn, mlp = layer.self_attn, layer.mlp
            widths = (
                (attn.q_proj, config.hidden_size, hq * head_dim),
                (attn.k_proj, config.hidden_size, hkv * head_dim),
                (attn.v_proj, config.hidden_size, hkv * head_dim),
                (attn.o_proj, hq * head_dim, config.hidden_size),
                (mlp.gate_proj, config.hidden_size, config.intermediate_size),
                (mlp.up_proj, config.hidden_size, config.intermediate_size),
                (mlp.down_proj, config.intermediate_size, config.hidden_size),
            )
            for module, input_width, output_width in widths:
                if module.in_features != input_width or module.out_features != output_width:
                    raise ValueError("Qwen3 projection width does not match its configuration")
            if attn.head_dim != head_dim or attn.num_key_value_groups != hq // hkv:
                raise ValueError("Qwen3 attention module disagrees with configured head layout")
            for norm, width in (
                (attn.q_norm, head_dim),
                (attn.k_norm, head_dim),
                (layer.input_layernorm, config.hidden_size),
                (layer.post_attention_layernorm, config.hidden_size),
            ):
                if norm.weight.shape != (width,) or norm.variance_epsilon != config.rms_norm_eps:
                    raise ValueError("Qwen3 RMSNorm parameters do not match its configuration")
        if config.hidden_act != "silu":
            raise ValueError("This Qwen3 trace adapter supports the checkpoint's SiLU gated MLP")
        tied = self.model.lm_head.weight is self.model.model.embed_tokens.weight
        if tied != bool(config.tie_word_embeddings):
            raise ValueError("Qwen3 actual embedding/head tying disagrees with its configuration")
        source = inspect.getsourcefile(modeling_qwen3)
        if source is None:
            raise ValueError("Cannot identify the installed Qwen3 implementation for provenance")
        with open(source, "rb") as handle:
            implementation_digest = hashlib.sha256(handle.read()).hexdigest()
        config_digest = hashlib.sha256(
            json.dumps(
                config.to_dict(), sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        ).hexdigest()
        rope = config.rope_parameters
        return ArchitectureMetadata(
            layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            query_heads=hq,
            kv_heads=hkv,
            head_dim=head_dim,
            vocabulary_size=config.vocab_size,
            rms_norm_epsilon=config.rms_norm_eps,
            rope_theta=float(rope["rope_theta"]),
            tied_embeddings=tied,
            attention_dropout=float(config.attention_dropout),
            attention_backend=config._attn_implementation,
            config_digest=config_digest,
            implementation_digest=implementation_digest,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
        )

    def validate_for_trace(self) -> None:
        """Reject unsupported capture modes without changing model configuration."""
        if any(module.training for module in self.model.modules()):
            raise ValueError("Scientific capture requires an existing evaluation-mode model")
        if self.model.config._attn_implementation != "eager":
            raise ValueError(
                "This adapter validates the trusted eager backend; it never switches it"
            )
        if any(layer.self_attn.sliding_window is not None for layer in self.model.model.layers):
            raise ValueError("Sliding-window Qwen3 capture has not been validated")
        if getattr(self.model, "is_quantized", False):
            raise ValueError("Quantized Qwen3 trace reconstruction has not been validated")
        # HF's capture decorator permanently installs hooks if either flag is
        # enabled. Scoped observations use the original attention tuple instead.
        if getattr(self.model.model, "_output_capturing_hooks_installed", False):
            raise ValueError(
                "Resident model already has permanent Hugging Face output-capture hooks"
            )

    @staticmethod
    def _coordinates(shape: tuple[int, ...], selections: dict[int, tuple[int, ...]]):
        if all(indices == tuple(range(shape[axis])) for axis, indices in selections.items()):
            return None
        return tuple(selections.get(axis, tuple(range(size))) for axis, size in enumerate(shape))

    def _spec(
        self,
        component_id: str,
        axes: tuple[str, ...],
        shape: tuple[int, ...],
        selections: dict[int, tuple[int, ...]] | None = None,
        *,
        layer_index: int | None = None,
        kind: ValueKind = ValueKind.RAW,
        sources: tuple[str, ...] = (),
        method: str = "actual module boundary observed through a run-scoped PyTorch hook",
        optional: bool = False,
        expensive: bool = False,
    ) -> ComponentSpec:
        return ComponentSpec(
            component_id=component_id,
            axes=axes,
            logical_shape=shape,
            layer_index=layer_index,
            value_kind=kind,
            sources=sources,
            method=method,
            optional=optional,
            expensive=expensive,
            coordinates=self._coordinates(shape, selections or {}),
        )

    @staticmethod
    def _target(
        module: torch.nn.Module,
        kind: str,
        *entries: tuple[ComponentSpec, Callable[[tuple, dict, Any], torch.Tensor]],
    ) -> HookTarget:
        def observations(args, kwargs, output):
            return tuple(
                Observation(spec, extractor(args, kwargs, output)) for spec, extractor in entries
            )

        return HookTarget(module, kind, tuple(spec for spec, _ in entries), observations)

    @staticmethod
    def _input(args: tuple, kwargs: dict, _output: Any) -> torch.Tensor:
        return args[0] if args else kwargs["hidden_states"]

    @staticmethod
    def _output(_args: tuple, _kwargs: dict, output: Any) -> torch.Tensor:
        return output

    @staticmethod
    def _projected(heads: int, head_dim: int):
        def canonical(_args, _kwargs, output):
            return output.reshape(*output.shape[:2], heads, head_dim).transpose(1, 2)

        return canonical

    @staticmethod
    def _normalized(_args, _kwargs, output):
        return output.transpose(1, 2)

    def hook_targets(self, plan: CapturePlan, sequence_length: int) -> tuple[HookTarget, ...]:
        self.validate_for_trace()
        plan = plan.resolve(self.architecture, sequence_length)
        arch, n, tokens = self.architecture, sequence_length, plan.token_indices
        hq, hkv, d = arch.query_heads, arch.kv_heads, arch.head_dim
        query_heads = plan.query_heads or tuple(range(hq))
        kv_heads = tuple(sorted({head // (hq // hkv) for head in query_heads}))
        assert tokens is not None
        residual_shape = (1, n, arch.hidden_size)
        query_shape, kv_shape = (1, hq, n, d), (1, hkv, n, d)
        query_selection, kv_selection = {1: query_heads, 2: tokens}, {1: kv_heads}
        targets: list[HookTarget] = []
        residual = lambda name, layer=None, **kw: self._spec(  # noqa: E731
            name, RESIDUAL_AXES, residual_shape, {1: tokens}, layer_index=layer, **kw
        )
        targets.append(
            self._target(
                self.model.model.embed_tokens,
                "forward",
                (residual("embedding.output"), self._output),
            )
        )
        position_shape = (1, n, d)
        position = lambda suffix: self._spec(  # noqa: E731
            f"position.rotary_{suffix}",
            ("batch", "token", "head_dimension"),
            position_shape,
            method=(
                "actual Qwen3RotaryEmbedding output; all positions retained for mapped context keys"
            ),
        )
        targets.extend(
            (
                self._target(
                    self.model.model.rotary_emb,
                    "pre",
                    (
                        self._spec("position.ids", ("batch", "token"), (1, n)),
                        lambda args, kwargs, _out: (
                            args[1] if len(args) > 1 else kwargs["position_ids"]
                        ),
                    ),
                ),
                self._target(
                    self.model.model.rotary_emb,
                    "forward",
                    (position("cos"), lambda _args, _kwargs, out: out[0]),
                    (position("sin"), lambda _args, _kwargs, out: out[1]),
                ),
            )
        )
        for index, layer in enumerate(self.model.model.layers):
            prefix = f"layer[{index}]."

            def rspec(suffix, _prefix=prefix, _index=index, **kw):
                return residual(_prefix + suffix, _index, **kw)

            targets.extend(
                (
                    self._target(
                        layer,
                        "pre",
                        (
                            rspec(
                                "residual.before",
                                sources=(
                                    "embedding.output"
                                    if index == 0
                                    else f"layer[{index - 1}].residual.after_mlp",
                                ),
                                method=(
                                    "actual decoder input; verified exact alias of "
                                    "embedding/previous decoder output"
                                ),
                            ),
                            self._input,
                        ),
                    ),
                    self._target(
                        layer.self_attn.o_proj, "forward", (rspec("attention.output"), self._output)
                    ),
                    self._target(
                        layer.post_attention_layernorm,
                        "pre",
                        (rspec("residual.after_attention"), self._input),
                    ),
                    self._target(
                        layer.mlp.down_proj, "forward", (rspec("mlp.output"), self._output)
                    ),
                    self._target(
                        layer,
                        "forward",
                        (rspec("residual.after_mlp"), self._output),
                        (
                            rspec(
                                "output",
                                sources=(prefix + "residual.after_mlp",),
                                method="verified exact alias of decoder output/residual_after_mlp",
                            ),
                            self._output,
                        ),
                    ),
                )
            )
            if index not in plan.detailed_layers:
                continue
            # Both detailed pathways expose the actual first RMSNorm result.
            # The MLP profile does not need any QKV or attention-matrix retention.
            if plan.captures_attention or plan.captures_mlp:
                targets.append(
                    self._target(
                        layer.input_layernorm,
                        "forward",
                        (rspec("attention.input_normalized", optional=True), self._output),
                    )
                )
            if plan.captures_attention:

                def aspec(
                    suffix,
                    shape=query_shape,
                    axes=QUERY_AXES,
                    selections=query_selection,
                    _prefix=prefix,
                    _index=index,
                    **kw,
                ):
                    return self._spec(
                        _prefix + "attention." + suffix,
                        axes,
                        shape,
                        selections,
                        layer_index=_index,
                        optional=True,
                        **kw,
                    )

                targets.extend(
                    (
                        self._target(
                            layer.self_attn.q_proj,
                            "forward",
                            (aspec("query.projected"), self._projected(hq, d)),
                        ),
                        self._target(
                            layer.self_attn.k_proj,
                            "forward",
                            (
                                aspec("key.projected", kv_shape, KV_AXES, kv_selection),
                                self._projected(hkv, d),
                            ),
                        ),
                        self._target(
                            layer.self_attn.v_proj,
                            "forward",
                            (
                                aspec("value", kv_shape, KV_AXES, kv_selection),
                                self._projected(hkv, d),
                            ),
                        ),
                        self._target(
                            layer.self_attn.q_norm,
                            "forward",
                            (aspec("query.normalized"), self._normalized),
                        ),
                        self._target(
                            layer.self_attn.k_norm,
                            "forward",
                            (
                                aspec("key.normalized", kv_shape, KV_AXES, kv_selection),
                                self._normalized,
                            ),
                        ),
                        self._target(
                            layer.self_attn.o_proj,
                            "pre",
                            (
                                self._spec(
                                    prefix + "attention.concatenated",
                                    ("batch", "token", "head_feature"),
                                    (1, n, hq * d),
                                    {1: tokens},
                                    layer_index=index,
                                    optional=True,
                                    method=(
                                        "actual concatenated head output entering "
                                        "attention output projection"
                                    ),
                                ),
                                self._input,
                            ),
                            (
                                aspec(
                                    "head_output",
                                    sources=(prefix + "attention.concatenated",)
                                    if query_heads == tuple(range(hq))
                                    else (),
                                    method="verified head concatenation alias"
                                    if query_heads == tuple(range(hq))
                                    else (
                                        "actual selected head output; "
                                        "canonical view of output-projection input"
                                    ),
                                ),
                                lambda args, _kwargs, _out: (
                                    args[0].reshape(1, n, hq, d).transpose(1, 2)
                                ),
                            ),
                        ),
                        self._target(
                            layer.self_attn,
                            "forward",
                            (
                                aspec(
                                    "weights",
                                    (1, hq, n, n),
                                    ATTENTION_AXES,
                                    {1: query_heads, 2: tokens},
                                    expensive=True,
                                    method=(
                                        "actual eager attention weights from Qwen3Attention; "
                                        "HF output_attentions remains False"
                                    ),
                                ),
                                lambda _args, _kwargs, out: out[1],
                            ),
                        ),
                    )
                )
                if index == plan.detailed_layers[0]:
                    targets.append(
                        self._target(
                            layer.self_attn,
                            "pre",
                            (
                                self._spec(
                                    "position.attention_mask",
                                    ("batch", "mask_head", "query_token", "key_token"),
                                    (1, 1, n, n),
                                    {2: tokens},
                                    method=(
                                        "actual selected eager-attention additive mask; "
                                        "0 allowed, finfo(dtype).min masked"
                                    ),
                                ),
                                lambda args, kwargs, _out: (
                                    kwargs["attention_mask"]
                                    if "attention_mask" in kwargs
                                    else args[2]
                                ),
                            ),
                        )
                    )
            if plan.captures_mlp:

                def mspec(suffix, _prefix=prefix, _index=index):
                    return self._spec(
                        _prefix + "mlp." + suffix,
                        MLP_AXES,
                        (1, n, arch.intermediate_size),
                        {1: tokens},
                        layer_index=_index,
                        optional=True,
                    )

                targets.extend(
                    (
                        self._target(
                            layer.post_attention_layernorm,
                            "forward",
                            (rspec("mlp.input_normalized", optional=True), self._output),
                        ),
                        self._target(layer.mlp.gate_proj, "forward", (mspec("gate"), self._output)),
                        self._target(layer.mlp.up_proj, "forward", (mspec("up"), self._output)),
                        self._target(
                            layer.mlp.act_fn, "forward", (mspec("activated_gate"), self._output)
                        ),
                        self._target(layer.mlp.down_proj, "pre", (mspec("gated"), self._input)),
                    )
                )
        targets.extend(
            (
                self._target(
                    self.model.model.norm,
                    "pre",
                    (
                        residual(
                            "output.residual_final",
                            sources=(f"layer[{arch.layers - 1}].residual.after_mlp",),
                            method=(
                                "actual final RMSNorm input; exact alias of final decoder residual"
                            ),
                        ),
                        self._input,
                    ),
                ),
                self._target(
                    self.model.model.norm, "forward", (residual("output.normalized"), self._output)
                ),
                self._target(
                    self.model.lm_head,
                    "pre",
                    (
                        self._spec(
                            "output.lm_head_input",
                            ("batch", "prediction_position", "hidden"),
                            (1, 1, arch.hidden_size),
                            method=(
                                "actual LM head input; local prediction index0 "
                                f"maps to absolute input position {n - 1}"
                            ),
                        ),
                        self._input,
                    ),
                ),
                self._target(
                    self.model.lm_head,
                    "forward",
                    (
                        self._spec(
                            "output.logits",
                            ("batch", "prediction_position", "vocabulary"),
                            (1, 1, arch.vocabulary_size),
                            method=(
                                "actual last-position LM head output; local prediction index0 "
                                f"maps to absolute input position {n - 1}"
                            ),
                        ),
                        self._output,
                    ),
                ),
            )
        )
        return tuple(targets)

    def component_specs(self, plan: CapturePlan, sequence_length: int) -> tuple[ComponentSpec, ...]:
        plan = plan.resolve(self.architecture, sequence_length)
        specs = [
            spec for target in self.hook_targets(plan, sequence_length) for spec in target.specs
        ]
        arch, n = self.architecture, sequence_length
        specs.append(
            self._spec(
                "output.probabilities",
                ("batch", "prediction_position", "vocabulary"),
                (1, 1, arch.vocabulary_size),
                kind=ValueKind.DERIVED,
                sources=("output.logits",),
                method=(
                    "trusted Stage1 full-vocabulary CPU float64 softmax accumulation; "
                    "cast to float32"
                ),
            )
        )
        if plan.captures_attention:
            assert plan.token_indices is not None and plan.query_heads is not None
            kv_heads = tuple(
                sorted({head // (arch.query_heads // arch.kv_heads) for head in plan.query_heads})
            )
            for index in plan.detailed_layers:
                prefix = f"layer[{index}].attention."
                for suffix, shape, axes, selections, sources in (
                    (
                        "query.rotary",
                        (1, arch.query_heads, n, arch.head_dim),
                        QUERY_AXES,
                        {1: plan.query_heads, 2: plan.token_indices},
                        (prefix + "query.normalized", "position.rotary_cos", "position.rotary_sin"),
                    ),
                    (
                        "key.rotary",
                        (1, arch.kv_heads, n, arch.head_dim),
                        KV_AXES,
                        {1: kv_heads},
                        (prefix + "key.normalized", "position.rotary_cos", "position.rotary_sin"),
                    ),
                    (
                        "similarity",
                        (1, arch.query_heads, n, n),
                        ATTENTION_AXES,
                        {1: plan.query_heads, 2: plan.token_indices},
                        (prefix + "query.rotary", prefix + "key.rotary"),
                    ),
                    (
                        "scaled_scores",
                        (1, arch.query_heads, n, n),
                        ATTENTION_AXES,
                        {1: plan.query_heads, 2: plan.token_indices},
                        (prefix + "similarity",),
                    ),
                    (
                        "masked_scores",
                        (1, arch.query_heads, n, n),
                        ATTENTION_AXES,
                        {1: plan.query_heads, 2: plan.token_indices},
                        (prefix + "scaled_scores", "position.attention_mask"),
                    ),
                    (
                        "weights_reconstructed",
                        (1, arch.query_heads, n, n),
                        ATTENTION_AXES,
                        {1: plan.query_heads, 2: plan.token_indices},
                        (prefix + "masked_scores",),
                    ),
                    (
                        "head_output_reconstructed",
                        (1, arch.query_heads, n, arch.head_dim),
                        QUERY_AXES,
                        {1: plan.query_heads, 2: plan.token_indices},
                        (prefix + "weights_reconstructed", prefix + "value"),
                    ),
                ):
                    specs.append(
                        self._spec(
                            prefix + suffix,
                            axes,
                            shape,
                            selections,
                            layer_index=index,
                            kind=(
                                ValueKind.DERIVED
                                if suffix in {"similarity", "scaled_scores", "masked_scores"}
                                else ValueKind.RECONSTRUCTED
                            ),
                            sources=sources,
                            method=(
                                "installed Qwen3 RoPE/GQA/scale/mask/float32 softmax/value-mixing "
                                "recipe; reconstructed rather than observed"
                            ),
                            optional=True,
                            expensive=suffix
                            in {
                                "similarity",
                                "scaled_scores",
                                "masked_scores",
                                "weights_reconstructed",
                            },
                        )
                    )
        return tuple(specs)

    def native_mlp_projection(
        self, layer_index: int, gated: torch.Tensor, projection: str = "down"
    ) -> torch.Tensor:
        if projection not in {"gate", "up", "down"}:
            raise ValueError("Unknown MLP projection")
        projection = getattr(self.model.model.layers[layer_index].mlp, projection + "_proj")
        with torch.inference_mode():
            update = projection.forward(gated.to(device=self.device, dtype=projection.weight.dtype))
            return update.detach().cpu()

    def native_rope(
        self, normalized: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        with torch.inference_mode():
            values = normalized.to(device=self.device, dtype=self.dtype)
            cosine, sine = (
                cos.to(device=self.device, dtype=self.dtype),
                sin.to(device=self.device, dtype=self.dtype),
            )
            # The installed function transforms Q and K independently. Calling
            # with the same operand in both slots preserves its exact cast/order.
            rotated, _ = modeling_qwen3.apply_rotary_pos_emb(values, values, cosine, sine)
            return rotated.detach().cpu()

    def attention_scaling(self, layer_index: int) -> float:
        return float(self.model.model.layers[layer_index].self_attn.scaling)

    def native_norm(
        self, stage: str, values: torch.Tensor, layer_index: int | None = None
    ) -> torch.Tensor:
        if stage == "final":
            norm = self.model.model.norm
        else:
            if layer_index is None:
                raise ValueError("A per-layer norm requires a layer index")
            layer = self.model.model.layers[layer_index]
            norms = {
                "query": layer.self_attn.q_norm,
                "key": layer.self_attn.k_norm,
                "attention": layer.input_layernorm,
                "mlp": layer.post_attention_layernorm,
            }
            if stage not in norms:
                raise ValueError(f"Unknown normalization stage: {stage!r}")
            norm = norms[stage]
        with torch.inference_mode():
            normalized = norm.forward(values.to(device=self.device, dtype=norm.weight.dtype))
            return normalized.detach().cpu()

    def native_mlp_activation(self, layer_index: int, values: torch.Tensor) -> torch.Tensor:
        activation = self.model.model.layers[layer_index].mlp.act_fn
        with torch.inference_mode():
            return (
                activation.forward(values.to(device=self.device, dtype=self.dtype)).detach().cpu()
            )

    def native_lm_head(self, normalized: torch.Tensor) -> torch.Tensor:
        head = self.model.lm_head
        with torch.inference_mode():
            logits = head.forward(normalized.to(device=self.device, dtype=head.weight.dtype))
            return logits.detach().cpu()
