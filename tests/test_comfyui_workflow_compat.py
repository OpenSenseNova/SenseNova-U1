import ast
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = REPO_ROOT / "apps" / "comfyui" / "nodes.py"
WORKFLOWS_PATH = REPO_ROOT / "apps" / "comfyui" / "example_workflows"

LEGACY_LOADER_INPUTS = [
    "model_path",
    "sensenova_u1_src",
    "device",
    "dtype",
    "attn_backend",
    "device_map",
    "max_memory",
    "vram_mode",
    "gguf_checkpoint",
]
FAST_LOADER_INPUTS = [
    "fast_vram_fraction",
    "fast_vram_headroom_gib",
    "fast_activation_reserve_gib",
    "fast_vram_budget_gib",
]


def _loader_class() -> ast.ClassDef:
    tree = ast.parse(NODES_PATH.read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SenseNovaU1LocalLoader")


def _method(loader: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(node for node in loader.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _schema_input_calls(loader: ast.ClassDef) -> list[ast.Call]:
    schema = next(
        node
        for node in ast.walk(_method(loader, "define_schema"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Schema"
    )
    inputs = next(keyword.value for keyword in schema.keywords if keyword.arg == "inputs")
    assert isinstance(inputs, ast.List)
    return [node for node in inputs.elts if isinstance(node, ast.Call)]


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


class ComfyUIWorkflowCompatibilityTest(unittest.TestCase):
    def test_fast_settings_accept_blank_values_from_legacy_workflows(self) -> None:
        calls = _schema_input_calls(_loader_class())
        input_ids = [call.args[0].value for call in calls if call.args]

        self.assertEqual(input_ids[: len(LEGACY_LOADER_INPUTS)], LEGACY_LOADER_INPUTS)
        self.assertEqual(input_ids[-len(FAST_LOADER_INPUTS) :], FAST_LOADER_INPUTS)

        by_id = {call.args[0].value: call for call in calls if call.args}
        for input_id in FAST_LOADER_INPUTS:
            self.assertEqual(ast.unparse(by_id[input_id].func), "io.String.Input")
            self.assertIsInstance(_keyword(by_id[input_id], "default"), ast.expr)
            self.assertIsInstance(_keyword(by_id[input_id], "optional"), ast.Constant)
            self.assertTrue(_keyword(by_id[input_id], "optional").value)
            self.assertIsInstance(_keyword(by_id[input_id], "advanced"), ast.Constant)
            self.assertTrue(_keyword(by_id[input_id], "advanced").value)

    def test_loader_entrypoints_default_all_fast_settings(self) -> None:
        loader = _loader_class()
        expected_defaults = {
            "fast_vram_fraction": "DEFAULT_FAST_VRAM_FRACTION",
            "fast_vram_headroom_gib": "DEFAULT_FAST_VRAM_HEADROOM_GIB",
            "fast_activation_reserve_gib": "DEFAULT_FAST_ACTIVATION_RESERVE_GIB",
            "fast_vram_budget_gib": "0.0",
        }

        for method_name in ("fingerprint_inputs", "execute"):
            method = _method(loader, method_name)
            names = [argument.arg for argument in method.args.args]
            defaults = method.args.defaults
            default_names = names[-len(defaults) :]
            defaults_by_name = dict(zip(default_names, defaults))

            self.assertEqual(names[1 : 1 + len(LEGACY_LOADER_INPUTS)], LEGACY_LOADER_INPUTS)
            for name, expected in expected_defaults.items():
                self.assertIn(name, defaults_by_name)
                self.assertEqual(ast.unparse(defaults_by_name[name]), expected)

    def test_bundled_workflows_keep_the_legacy_loader_widget_prefix(self) -> None:
        found = 0
        for workflow_path in WORKFLOWS_PATH.glob("*.json"):
            workflow = json.loads(workflow_path.read_text())
            for node in workflow.get("nodes", []):
                if node.get("type") != "SenseNovaU1LocalLoader":
                    continue
                found += 1
                values = node.get("widgets_values", [])
                self.assertGreaterEqual(len(values), len(LEGACY_LOADER_INPUTS), workflow_path)
                self.assertIn(
                    len(values),
                    (len(LEGACY_LOADER_INPUTS), len(LEGACY_LOADER_INPUTS) + len(FAST_LOADER_INPUTS)),
                    workflow_path,
                )
        self.assertGreater(found, 0)


if __name__ == "__main__":
    unittest.main()
