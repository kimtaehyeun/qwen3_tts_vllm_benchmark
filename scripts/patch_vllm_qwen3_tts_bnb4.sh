#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path
import importlib.util


def package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"Could not locate installed package: {name}")
    return Path(spec.submodule_search_locations[0])


vllm_root = package_root("vllm")
omni_root = package_root("vllm_omni")

talker_path = (
    omni_root
    / "model_executor"
    / "models"
    / "qwen3_tts"
    / "qwen3_tts_talker.py"
)
loader_path = (
    vllm_root
    / "model_executor"
    / "model_loader"
    / "bitsandbytes_loader.py"
)
code_predictor_path = (
    omni_root
    / "model_executor"
    / "models"
    / "qwen3_tts"
    / "qwen3_tts_code_predictor_vllm.py"
)

talker_text = talker_path.read_text(encoding="utf-8")
mapping_marker = '    packed_modules_mapping = {'
if mapping_marker not in talker_text:
    anchor = (
        '    Predicts residual codebooks (1..Q-1) into `audio_codes` '
        'and streams text via `tailing_text_hidden`."""\n\n'
    )
    insert = """    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

"""
    if anchor not in talker_text:
        raise SystemExit(f"Could not find insertion point in {talker_path}")
    talker_path.write_text(
        talker_text.replace(anchor, anchor + insert, 1),
        encoding="utf-8",
    )
    print(f"patched {talker_path}: added packed_modules_mapping")
else:
    print(f"already patched {talker_path}: packed_modules_mapping")

loader_text = loader_path.read_text(encoding="utf-8")
old = """            if any(
                target_module in mapped_weight_name
                for target_module in self.target_modules
            ) and mapped_weight_name.endswith(".weight"):
"""
new = """            if any(
                check_match(mapped_weight_name, target_module)
                for target_module in self.target_modules
            ) and mapped_weight_name.endswith(".weight"):
"""
if old in loader_text:
    loader_path.write_text(loader_text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {loader_path}: exact BNB target-module matching")
elif new in loader_text:
    print(f"already patched {loader_path}: exact BNB target-module matching")
else:
    raise SystemExit(f"Could not find BitsAndBytes target-module block in {loader_path}")

code_text = code_predictor_path.read_text(encoding="utf-8")
if "_uses_bnb_4bit" in code_text:
    if "import os\n" not in code_text:
        code_text = code_text.replace(
            "from __future__ import annotations\n\n",
            "from __future__ import annotations\n\nimport os\n",
            1,
        )
    old_envless = '''def _uses_bnb_4bit(vllm_config: VllmConfig | None) -> bool:
    quant_config = getattr(vllm_config, "quant_config", None)
'''
    new_envguard = '''def _uses_bnb_4bit(vllm_config: VllmConfig | None) -> bool:
    if os.environ.get("QWEN3_TTS_BNB4_MTP", "").lower() not in {"1", "true", "yes"}:
        return False
    quant_config = getattr(vllm_config, "quant_config", None)
'''
    if old_envless in code_text:
        code_text = code_text.replace(old_envless, new_envguard, 1)
        code_predictor_path.write_text(code_text, encoding="utf-8")
    print(f"already patched {code_predictor_path}: MTP Linear4bit")
else:
    def replace_once(text: str, old: str, new: str, label: str) -> str:
        if old not in text:
            raise SystemExit(f"Could not patch {code_predictor_path}: {label}")
        return text.replace(old, new, 1)

    helper_anchor = "logger = init_logger(__name__)\n\n\n"
    if "import os\n" not in code_text:
        code_text = code_text.replace(
            "from __future__ import annotations\n\n",
            "from __future__ import annotations\n\nimport os\n",
            1,
        )
    helper_insert = '''logger = init_logger(__name__)


def _uses_bnb_4bit(vllm_config: VllmConfig | None) -> bool:
    if os.environ.get("QWEN3_TTS_BNB4_MTP", "").lower() not in {"1", "true", "yes"}:
        return False
    quant_config = getattr(vllm_config, "quant_config", None)
    return bool(
        quant_config is not None
        and getattr(quant_config, "get_name", lambda: None)() == "bitsandbytes"
        and getattr(quant_config, "load_in_4bit", False)
    )


def _bnb_compute_dtype(vllm_config: VllmConfig | None) -> torch.dtype:
    quant_config = getattr(vllm_config, "quant_config", None)
    value = getattr(quant_config, "bnb_4bit_compute_dtype", "bfloat16")
    if isinstance(value, torch.dtype):
        return value
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(str(value), torch.bfloat16)


def _make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    vllm_config: VllmConfig | None = None,
) -> nn.Module:
    if _uses_bnb_4bit(vllm_config):
        import bitsandbytes as bnb

        return bnb.nn.Linear4bit(
            in_features,
            out_features,
            bias=bias,
            compute_dtype=_bnb_compute_dtype(vllm_config),
            compress_statistics=True,
            quant_type="nf4",
        )
    return nn.Linear(in_features, out_features, bias=bias)


def _finalize_bnb_4bit_modules(module: nn.Module) -> None:
    try:
        import bitsandbytes as bnb
    except ImportError:
        return

    for submodule in module.modules():
        if isinstance(submodule, bnb.nn.Linear4bit):
            submodule.to(submodule.weight.device)


'''
    code_text = replace_once(code_text, helper_anchor, helper_insert, "insert BNB helpers")
    code_text = replace_once(
        code_text,
'''        prefix: str = "",
    ) -> None:
''',
'''        prefix: str = "",
        vllm_config: VllmConfig | None = None,
    ) -> None:
''',
        "attention signature",
    )
    code_text = replace_once(
        code_text,
'''        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )
''',
'''        self.q_proj = _make_linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
            vllm_config=vllm_config,
        )
        self.k_proj = _make_linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
            vllm_config=vllm_config,
        )
        self.v_proj = _make_linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
            vllm_config=vllm_config,
        )
        self.o_proj = _make_linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            vllm_config=vllm_config,
        )
''',
        "attention Linear4bit",
    )
    code_text = replace_once(
        code_text,
'''        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
''',
'''        prefix: str = "",
        vllm_config: VllmConfig | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = _make_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            vllm_config=vllm_config,
        )
        self.up_proj = _make_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            vllm_config=vllm_config,
        )
        self.down_proj = _make_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            vllm_config=vllm_config,
        )
''',
        "MLP Linear4bit",
    )
    code_text = replace_once(
        code_text,
'''        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = _CodePredictorAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = _CodePredictorMLP(config, prefix=f"{prefix}.mlp")
''',
'''        prefix: str = "",
        vllm_config: VllmConfig | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = _CodePredictorAttention(
            config,
            prefix=f"{prefix}.self_attn",
            vllm_config=vllm_config,
        )
        self.mlp = _CodePredictorMLP(
            config,
            prefix=f"{prefix}.mlp",
            vllm_config=vllm_config,
        )
''',
        "decoder layer Linear4bit",
    )
    code_text = replace_once(
        code_text,
'''        talker_hidden_size: int | None = None,
        prefix: str = "",
    ) -> None:
''',
'''        talker_hidden_size: int | None = None,
        prefix: str = "",
        vllm_config: VllmConfig | None = None,
    ) -> None:
''',
        "model signature",
    )
    code_text = replace_once(
        code_text,
'''        self.layers = nn.ModuleList(
            [_CodePredictorDecoderLayer(config, prefix=f"{prefix}.layers.{i}") for i in range(config.num_hidden_layers)]
        )
''',
'''        self.layers = nn.ModuleList(
            [
                _CodePredictorDecoderLayer(
                    config,
                    prefix=f"{prefix}.layers.{i}",
                    vllm_config=vllm_config,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
''',
        "model layers Linear4bit",
    )
    code_text = replace_once(
        code_text,
'''        self.model = Qwen3TTSTalkerCodePredictorModelVLLM(
            config,
            talker_hidden_size=int(talker_config.hidden_size),
            prefix=f"{prefix}.model",
        )

        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
''',
'''        self.model = Qwen3TTSTalkerCodePredictorModelVLLM(
            config,
            talker_hidden_size=int(talker_config.hidden_size),
            prefix=f"{prefix}.model",
            vllm_config=vllm_config,
        )

        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = _make_linear(
                talker_config.hidden_size,
                config.hidden_size,
                bias=True,
                vllm_config=vllm_config,
            )
''',
        "wrapper Linear4bit",
    )
    code_text = replace_once(
        code_text,
'''            return loaded

    # ------------------------------------------------------------------
''',
'''            if _uses_bnb_4bit(self._vllm_config):
                _finalize_bnb_4bit_modules(self)
                logger.info("code_predictor: BitsAndBytes 4bit enabled for MTP Linear modules")

            return loaded

    # ------------------------------------------------------------------
''',
        "finalize BNB MTP modules",
    )
    code_text = replace_once(
        code_text,
'''        self._lm_heads_list = list(self.lm_head)
        self._codec_embeds_list = list(self.model.codec_embedding)
        if not current_omni_platform.supports_torch_inductor():
''',
'''        self._lm_heads_list = list(self.lm_head)
        self._codec_embeds_list = list(self.model.codec_embedding)
        if _uses_bnb_4bit(self._vllm_config):
            logger.info("code_predictor: torch.compile/CUDA graphs disabled for BitsAndBytes 4bit MTP")
            self._compiled_model_fwd = self.model.forward
            return
        if not current_omni_platform.supports_torch_inductor():
''',
        "disable MTP compile for BNB",
    )
    code_predictor_path.write_text(code_text, encoding="utf-8")
    print(f"patched {code_predictor_path}: MTP Linear4bit")


def replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if old not in text:
        return text, False
    return text.replace(old, new, 1), True


code_text = code_predictor_path.read_text(encoding="utf-8")
changed = False

if "import json\n" not in code_text:
    code_text = code_text.replace("import os\n", "import json\nimport os\n", 1)
    changed = True
if "from pathlib import Path\n" not in code_text:
    code_text = code_text.replace("from collections.abc import Iterable\n", "from collections.abc import Iterable\nfrom pathlib import Path\n", 1)
    changed = True

old_lm_head = '''        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )
'''
new_lm_head = '''        self.lm_head = nn.ModuleList(
            [
                _make_linear(
                    config.hidden_size,
                    config.vocab_size,
                    bias=False,
                    vllm_config=vllm_config,
                )
                for _ in range(config.num_code_groups - 1)
            ]
        )
'''
code_text, did_replace = replace_once(code_text, old_lm_head, new_lm_head, "MTP lm_head Linear4bit")
if did_replace:
    changed = True
    print(f"patched {code_predictor_path}: MTP lm_head uses _make_linear")
elif "_make_linear(\n                    config.hidden_size,\n                    config.vocab_size," in code_text:
    print(f"already patched {code_predictor_path}: MTP lm_head uses _make_linear")
else:
    raise SystemExit(f"Could not patch {code_predictor_path}: MTP lm_head Linear4bit")

inventory_helper_anchor = '''def _uses_mtp_quant(vllm_config: VllmConfig | None) -> bool:
    if _uses_int4_weightonly(vllm_config):
        return True
    if _uses_bnb_4bit(vllm_config):
        return True
    return False


'''
inventory_helper = '''def _uses_mtp_quant(vllm_config: VllmConfig | None) -> bool:
    if _uses_int4_weightonly(vllm_config):
        return True
    if _uses_bnb_4bit(vllm_config):
        return True
    return False


def _linear_inventory_row(name: str, module: nn.Module) -> dict:
    try:
        import bitsandbytes as bnb
    except ImportError:
        bnb = None

    class_name = type(module).__name__
    quantized = False
    quant_method = "none"
    bit_width = "fp"
    if isinstance(module, _Int4WeightOnlyLinear):
        quantized = True
        quant_method = "int4_weightonly"
        bit_width = 4
    elif bnb is not None and isinstance(module, bnb.nn.Linear4bit):
        quantized = True
        quant_method = "bnb_nf4"
        bit_width = 4

    return {
        "module_name": name,
        "module_group": "mtp",
        "class": class_name,
        "in_features": int(getattr(module, "in_features", -1)),
        "out_features": int(getattr(module, "out_features", -1)),
        "assigned_bit_width": bit_width,
        "quantized": int(quantized),
        "assigned_quant_method": quant_method,
        "weight_dtype": str(getattr(getattr(module, "weight", None), "dtype", "unknown")),
    }


def _dump_mtp_quant_inventory(module: nn.Module) -> None:
    path = os.environ.get("QWEN3_TTS_MTP_INVENTORY_PATH", "").strip()
    if not path:
        return
    rows = []
    try:
        import bitsandbytes as bnb
    except ImportError:
        bnb = None
    for name, submodule in module.named_modules():
        is_linear_like = isinstance(submodule, (nn.Linear, _Int4WeightOnlyLinear))
        if bnb is not None:
            is_linear_like = is_linear_like or isinstance(submodule, bnb.nn.Linear4bit)
        if not is_linear_like:
            continue
        rows.append(_linear_inventory_row(name, submodule))
    payload = {
        "quant_impl": _mtp_quant_name(),
        "n_linear_total": len(rows),
        "n_quantized": sum(int(row["quantized"]) for row in rows),
        "n_bnb_nf4": sum(1 for row in rows if row["assigned_quant_method"] == "bnb_nf4"),
        "n_int4_weightonly": sum(1 for row in rows if row["assigned_quant_method"] == "int4_weightonly"),
        "n_fp": sum(1 for row in rows if not row["quantized"]),
        "inventory": rows,
    }
    try:
        inventory_path = Path(path)
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("code_predictor: wrote MTP quant inventory to %s", str(inventory_path))
    except Exception as exc:
        logger.warning("code_predictor: failed to write MTP quant inventory: %s", exc)


'''
if "_dump_mtp_quant_inventory" not in code_text:
    if inventory_helper_anchor not in code_text:
        raise SystemExit(f"Could not patch {code_predictor_path}: inventory helper anchor")
    code_text = code_text.replace(inventory_helper_anchor, inventory_helper, 1)
    changed = True
    print(f"patched {code_predictor_path}: added MTP quant inventory dump helper")
else:
    print(f"already patched {code_predictor_path}: MTP quant inventory dump helper")

old_finalize = '''            if _uses_mtp_quant(self._vllm_config):
                _finalize_mtp_quant_modules(self)

            return loaded
'''
new_finalize = '''            if _uses_mtp_quant(self._vllm_config):
                _finalize_mtp_quant_modules(self)
                _dump_mtp_quant_inventory(self)

            return loaded
'''
code_text, did_replace = replace_once(code_text, old_finalize, new_finalize, "dump MTP quant inventory after finalize")
if did_replace:
    changed = True
    print(f"patched {code_predictor_path}: dump MTP quant inventory after finalize")
elif "_dump_mtp_quant_inventory(self)" in code_text:
    print(f"already patched {code_predictor_path}: dump MTP quant inventory after finalize")
else:
    raise SystemExit(f"Could not patch {code_predictor_path}: dump MTP quant inventory after finalize")

if changed:
    code_predictor_path.write_text(code_text, encoding="utf-8")
PY
