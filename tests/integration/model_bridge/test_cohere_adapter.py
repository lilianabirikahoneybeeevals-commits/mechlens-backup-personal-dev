"""Cohere bridge integration and release-gate tests.

Source: tests/integration/model_bridge/test_cohere_adapter.py, revised 2026-10-12.
Model: trl-internal-testing/tiny-CohereForCausalLM (2 layers, logit_scale=0.125).

IMPORTANT: TransformerBridge.forward currently delegates to self.original_model.
The explicit non-delegation gate below is intentionally expected to FAIL until
an independent processed-weight forward is wired up. Do not xfail or skip it.
The tiny fixture has use_qk_norm=False, so it does not validate QK-norm.
"""

from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import NormalizationBridge
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)

MODEL = "trl-internal-testing/tiny-CohereForCausalLM"
LOGIT_TOL = 1e-4
RESID_TOL = 1e-4


@pytest.fixture(scope="module")
def cohere_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def cohere_bridge_processed():
    """Call conversion/processing; this alone does NOT prevent HF delegation."""
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return bridge


@pytest.fixture(scope="module")
def cohere_hf() -> Any:
    return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()


def _input_ids(length: int, vocab_size: int) -> torch.Tensor:
    """Deterministic IDs; avoid padding and compare exactly identical inputs."""
    torch.manual_seed(2718 + length)
    return torch.randint(1, vocab_size, (1, length), dtype=torch.long)


def _assert_max_delta(a: torch.Tensor, b: torch.Tensor, tol: float, label: str) -> None:
    assert a.shape == b.shape, f"{label} shape {tuple(a.shape)} != {tuple(b.shape)}"
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), f"{label} nonfinite values"
    value = (a.float() - b.float()).abs().max().item()
    assert value < tol, f"{label}: max_abs_delta={value:.9g}, tolerance={tol:g}"


class TestCohereBridgeCreation:
    def test_boot_transformers_succeeds(self, cohere_bridge):
        assert cohere_bridge is not None

    def test_block_count(self, cohere_bridge):
        assert len(cohere_bridge.blocks) == 2

    def test_has_core_components(self, cohere_bridge):
        for key in ("embed", "unembed", "ln_final", "rotary_emb"):
            assert hasattr(cohere_bridge, key)

    def test_no_ln2_in_blocks(self, cohere_bridge):
        for block in cohere_bridge.blocks:
            assert not hasattr(block, "ln2"), "Parallel block must not have ln2"

    def test_ln1_subtracts_mean_like_hf_layernorm(self, cohere_bridge, cohere_hf):
        ln1 = cohere_bridge.blocks[0].ln1
        hf_ln1 = cohere_hf.model.layers[0].input_layernorm
        torch.manual_seed(0)
        x = torch.randn(1, 4, cohere_bridge.cfg.d_model) + 5.0
        with torch.no_grad():
            bridge_out = ln1(x)
            hf_out = hf_ln1(x)
        _assert_max_delta(bridge_out, hf_out, 1e-4, "true LayerNorm")
        variance = x.float().pow(2).mean(-1, keepdim=True)
        rms_only = (
            x.float()
            * torch.rsqrt(variance + ln1.original_component.variance_epsilon)
            * ln1.original_component.weight.float()
        ).to(bridge_out.dtype)
        assert (bridge_out - rms_only).abs().max().item() > 1e-2

    def test_cfg_logit_scale_matches_hf(self, cohere_bridge, cohere_hf):
        assert cohere_bridge.cfg.logit_scale == cohere_hf.config.logit_scale
        assert cohere_bridge.cfg.logit_scale == pytest.approx(0.125)

    def test_cfg_rope_parameters_matches_hf(self, cohere_bridge, cohere_hf):
        assert cohere_bridge.cfg.rope_parameters == cohere_hf.config.rope_parameters


class TestCohereForwardEquivalence:
    def test_forward_returns_logits(self, cohere_bridge):
        with torch.no_grad():
            output = cohere_bridge(torch.tensor([[1, 2, 3, 4]]))
        assert output.shape[:2] == (1, 4)
        assert torch.isfinite(output).all()

    def test_forward_matches_hf(self, cohere_bridge, cohere_hf):
        """Original smoke test, deliberately NOT the conversion acceptance gate."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            _assert_max_delta(cohere_bridge(tokens), cohere_hf(tokens).logits,
                              LOGIT_TOL, "unprocessed bridge short-input logits")


class TestCohereLogitScaleEndToEnd:
    def test_unembed_weight_is_scaled_relative_to_hf(self, cohere_bridge_processed, cohere_hf):
        scale = cohere_bridge_processed.cfg.logit_scale
        tl_weight = cohere_bridge_processed.unembed.original_component.weight
        hf_weight = cohere_hf.lm_head.weight.detach()
        _assert_max_delta(tl_weight, hf_weight * scale, 1e-5, "processed unembed")

    def test_logit_scale_folded_not_applied_twice(self, cohere_bridge, cohere_hf):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            _assert_max_delta(cohere_bridge(tokens), cohere_hf(tokens).logits,
                              LOGIT_TOL, "original no-double-scale smoke test")


class TestCohereTiedEmbedding:
    def test_embed_weight_equals_hf_embed_tokens(self, cohere_bridge_processed, cohere_hf):
        hf_embed = cohere_hf.model.embed_tokens.weight.detach()
        tl_embed = cohere_bridge_processed.embed.W_E
        _assert_max_delta(tl_embed, hf_embed, 1e-6, "embedding preserved after scale fold")

    @pytest.mark.parametrize("logit_scale", [0.0625, 1.0])
    def test_embed_and_unembed_weights_differ(self, logit_scale):
        bridge = TransformerBridge.boot_transformers(MODEL, device="cpu")
        bridge.cfg.logit_scale = logit_scale
        bridge.process_weights(
            fold_ln=False, center_writing_weights=False, center_unembed=False,
            fold_value_biases=False, refactor_factored_attn_matrices=False,
        )
        weights_identical = torch.allclose(
            bridge.embed.W_E, bridge.unembed.original_component.weight
        )
        assert weights_identical if logit_scale == 1.0 else not weights_identical


class TestCohereHFDelegation:
    def test_ln1_original_component_is_hf_norm(self, cohere_bridge):
        ln1 = cohere_bridge.blocks[0].ln1
        assert isinstance(ln1, NormalizationBridge)
        assert hasattr(ln1.original_component, "variance_epsilon")

    def test_attn_original_component_is_hf_attention(self, cohere_bridge):
        attn = cohere_bridge.blocks[0].attn
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert hasattr(attn.original_component, "q_proj")

    def test_ln_final_original_component_is_hf_norm(self, cohere_bridge):
        assert hasattr(cohere_bridge.ln_final.original_component, "variance_epsilon")


class TestCohereParallelHooks:
    def test_no_hook_resid_mid(self, cohere_bridge):
        _, cache = cohere_bridge.run_with_cache(torch.tensor([[1, 2, 3, 4]]))
        assert not any("hook_resid_mid" in key for key in cache.keys())

    def test_attn_and_mlp_hooks_fire(self, cohere_bridge):
        _, cache = cohere_bridge.run_with_cache(torch.tensor([[1, 2, 3, 4]]))
        for i in range(2):
            for key in ("attn.hook_in", "attn.hook_out", "mlp.hook_in", "mlp.hook_out"):
                assert f"blocks.{i}.{key}" in cache

    def test_residual_hooks_fire(self, cohere_bridge):
        _, cache = cohere_bridge.run_with_cache(torch.tensor([[1, 2, 3, 4]]))
        for i in range(2):
            assert f"blocks.{i}.hook_resid_pre" in cache
            assert f"blocks.{i}.hook_resid_post" in cache


# ---------------------------------------------------------------------------
# Additional release-gate checks: long prompts, intermediate activations,
# explicit independent-processed-path guard. These are NOT expected green on
# the current bridge merely because process_weights() was called.
# ---------------------------------------------------------------------------

class TestCohereProcessedAcceptance:
    @pytest.mark.parametrize("seq_len", [4, 32, 128])
    def test_processed_logits_match_hf_for_multiple_lengths(
        self, cohere_bridge_processed, cohere_hf, seq_len
    ):
        """Compare *processed* model outputs; delegation gate below is mandatory."""
        assert cohere_bridge_processed.original_model is not cohere_hf
        assert next(cohere_bridge_processed.original_model.parameters()).dtype == torch.float32
        tokens = _input_ids(seq_len, cohere_hf.config.vocab_size)
        with torch.inference_mode():
            actual = cohere_bridge_processed(tokens)
            expected = cohere_hf(tokens).logits
        _assert_max_delta(actual, expected, LOGIT_TOL,
                          f"processed logits sequence_length={seq_len}")

    @pytest.mark.parametrize("seq_len", [4, 128])
    def test_processed_intermediate_activations_match_hf(
        self, cohere_bridge_processed, cohere_hf, seq_len
    ):
        """Compare layer input, layer output and final norm input to HF hooks.

        Note: attention patterns are not compared here; this test intentionally
        targets residual/normalization activations and must be paired with the
        independent-path guard below to avoid tautological HF delegation.
        """
        tokens = _input_ids(seq_len, cohere_hf.config.vocab_size)
        hf_values = {}
        handles = []

        def pre_hook(name):
            def capture(_module, args):
                hf_values[name] = args[0].detach().clone()
            return capture

        def post_hook(name):
            def capture(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hf_values[name] = hidden.detach().clone()
            return capture

        for i, layer in enumerate(cohere_hf.model.layers):
            handles.append(layer.register_forward_pre_hook(pre_hook(f"blocks.{i}.hook_resid_pre")))
            handles.append(layer.register_forward_hook(post_hook(f"blocks.{i}.hook_resid_post")))
        handles.append(cohere_hf.model.norm.register_forward_pre_hook(pre_hook("final_norm_input")))
        try:
            with torch.inference_mode():
                cohere_hf(tokens)
        finally:
            for h in handles:
                h.remove()

        with torch.inference_mode():
            _, cache = cohere_bridge_processed.run_with_cache(tokens)

        for i in range(len(cohere_hf.model.layers)):
            for suffix in ("hook_resid_pre", "hook_resid_post"):
                key = f"blocks.{i}.{suffix}"
                assert key in cache, f"Missing bridge activation: {key}"
                assert key in hf_values, f"Missing reference activation: {key}"
                _assert_max_delta(cache[key], hf_values[key], RESID_TOL,
                                  f"{key} sequence_length={seq_len}")
        # HF final norm input equals final residual stream on this architecture.
        _assert_max_delta(cache[f"blocks.{len(cohere_hf.model.layers)-1}.hook_resid_post"],
                          hf_values["final_norm_input"], RESID_TOL,
                          f"final norm input sequence_length={seq_len}")

    def test_processed_forward_does_not_delegate_to_hf_model(self, cohere_bridge_processed):
        """Mandatory anti-tautology gate, EXPECTED TO FAIL with current bridge.py.

        A genuine independent converted forward must not call original_model.forward.
        Current TransformerBridge.forward explicitly calls original_model(...),
        regardless of process_weights(). This check will surface that limitation.
        """
        tokens = torch.tensor([[1, 2, 3, 4]])
        def prohibited_hf_forward(*args, **kwargs):
            raise AssertionError(
                "Processed bridge delegated to HF original_model.forward: "
                "weight-conversion numerical parity is NOT independently validated"
            )
        with patch.object(cohere_bridge_processed.original_model, "forward",
                          side_effect=prohibited_hf_forward) as hf_forward:
            with torch.inference_mode():
                cohere_bridge_processed(tokens)
            assert hf_forward.call_count == 0