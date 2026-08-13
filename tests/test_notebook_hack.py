"""Focused unit and CLI tests for ``notebook_hack.py``."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import importlib.util


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "tools" / "notebook_hack.py"
SPEC = importlib.util.spec_from_file_location("notebook_hack", MODULE_PATH)
assert SPEC and SPEC.loader
notebook_hack = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = notebook_hack
SPEC.loader.exec_module(notebook_hack)


def markdown(source: str | list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str | list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def notebook(*cells: dict) -> dict:
    return {
        "cells": list(cells),
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


class ExistingTransformTests(unittest.TestCase):
    def test_device_map_handles_nested_call_arguments(self) -> None:
        source = 'model = AutoModel.from_pretrained(resolve_model("org/model"))'
        result = notebook_hack.inject_device_map(source)
        self.assertEqual(
            result,
            'model = AutoModel.from_pretrained(resolve_model("org/model"), device_map="cuda")',
        )

    def test_device_map_handles_empty_and_trailing_comma_calls(self) -> None:
        self.assertEqual(
            notebook_hack.inject_device_map("pipeline()"),
            'pipeline(device_map="cuda")',
        )
        self.assertEqual(
            notebook_hack.inject_device_map('pipeline("text-generation",)'),
            'pipeline("text-generation", device_map="cuda",)',
        )

    def test_device_map_handles_multiline_call(self) -> None:
        source = 'model = AutoModel.from_pretrained(\n    "org/model",\n)'
        result = notebook_hack.inject_device_map(source)
        self.assertIn('    device_map="cuda",\n)', result)

    def test_device_map_rewrites_auto_without_double_injection(self) -> None:
        source = 'AutoModel.from_pretrained("x", device_map="auto")'
        result = notebook_hack.inject_device_map(source)
        self.assertEqual(result.count("device_map"), 1)
        self.assertIn('device_map="cuda"', result)

    def test_device_map_respects_explicit_device_and_chained_to(self) -> None:
        explicit = 'pipeline("asr", model="x", device=0)'
        chained = 'AutoModel.from_pretrained("x").to("cuda")'
        self.assertEqual(notebook_hack.inject_device_map(explicit), explicit)
        self.assertEqual(notebook_hack.inject_device_map(chained), chained)

    def test_device_map_ignores_strings_comments_and_definitions(self) -> None:
        source = (
            'example = \'pipeline("text-generation")\'\n'
            '# AutoModel.from_pretrained("x")\n'
            'def pipeline(value):\n    return value\n'
        )
        self.assertEqual(notebook_hack.inject_device_map(source), source)

    def test_malformed_python_is_left_untouched(self) -> None:
        source = 'AutoModel.from_pretrained(resolve_model("x")'
        self.assertEqual(notebook_hack.inject_device_map(source), source)

    def test_diffusers_dtype_is_scoped_to_diffusers(self) -> None:
        diffusers = (
            "from diffusers import DiffusionPipeline\n"
            'pipe = DiffusionPipeline.from_pretrained("x", dtype=torch.bfloat16)'
        )
        transformers = (
            "from transformers import AutoModel\n"
            'model = AutoModel.from_pretrained("x", dtype=torch.bfloat16)'
        )
        self.assertIn(
            "torch_dtype=torch.bfloat16",
            notebook_hack.fix_diffusers_dtype(diffusers),
        )
        self.assertEqual(
            notebook_hack.fix_diffusers_dtype(transformers), transformers
        )

    def test_vram_cleanup_is_injected_only_after_pipeline_reload(self) -> None:
        document = notebook(
            code('pipe = pipeline("text-generation", model="x")'),
            code('model = AutoModel.from_pretrained("x")'),
        )
        patched = notebook_hack.patch_notebook(document)
        first = cell_source(patched["cells"][0])
        second = cell_source(patched["cells"][1])
        self.assertNotIn("empty_cache", first)
        self.assertIn("gc.collect()", second)
        self.assertIn("torch.cuda.empty_cache()", second)

    def test_patch_leaves_device_map_policy_untouched(self) -> None:
        document = notebook(
            code('pipe = pipeline("text-generation", model="org/model")'),
            code(
                'model = AutoModel.from_pretrained('
                '"org/model", device_map="auto")'
            ),
        )

        patched = notebook_hack.patch_notebook(document)

        self.assertNotIn("device_map", cell_source(patched["cells"][0]))
        self.assertIn(
            'device_map="auto"', cell_source(patched["cells"][1])
        )

    def test_kernelspec_and_remote_inference_trim(self) -> None:
        document = notebook(
            code("local = True"),
            markdown("## Serverless Inference"),
            code("remote = True"),
        )
        document["metadata"]["kernelspec"] = {"name": "custom"}
        patched = notebook_hack.patch_notebook(document)
        self.assertEqual(patched["metadata"]["kernelspec"]["name"], "python3")
        self.assertEqual(len(patched["cells"]), 1)


class MirroredModelPageTests(unittest.TestCase):
    def test_rewrites_exact_model_page_line_for_arbitrary_models(self) -> None:
        examples = (
            "Qwen/Qwen3.5-9B",
            "google/gemma-3-1b-it",
            "ibm-granite/granite-speech-4.1-2b",
            "org_with.dots/model_name-v2.0",
        )
        for model_id in examples:
            with self.subTest(model_id=model_id):
                source = f"Title\nModel page: https://hf-mirror.com/{model_id}\nEnd"
                expected = f"Title\nModel page: https://huggingface.co/{model_id}\nEnd"
                self.assertEqual(
                    notebook_hack.fix_mirrored_model_page(source), expected
                )

    def test_accepts_markdown_spacing_and_case_without_changing_them(self) -> None:
        source = "  MODEL   PAGE :   https://hf-mirror.com/org/model  \n"
        expected = "  MODEL   PAGE :   https://huggingface.co/org/model  \n"
        self.assertEqual(notebook_hack.fix_mirrored_model_page(source), expected)

    def test_leaves_canonical_model_page_unchanged(self) -> None:
        source = "Model page: https://huggingface.co/Qwen/Qwen3.5-9B"
        self.assertEqual(notebook_hack.fix_mirrored_model_page(source), source)

    def test_does_not_rewrite_non_model_page_contexts(self) -> None:
        untouched = (
            "Download: https://hf-mirror.com/org/model",
            "See Model page: https://hf-mirror.com/org/model",
            "Model page: [https://hf-mirror.com/org/model](https://hf-mirror.com/org/model)",
            "Model page: https://hf-mirror.com/org/model?download=true",
            "Model page: https://hf-mirror.com/org/model#readme",
            "Model page: https://hf-mirror.com/org/model/blob/main/config.json",
            "Model page: http://hf-mirror.com/org/model",
            "Model page: https://sub.hf-mirror.com/org/model",
            "url = 'https://hf-mirror.com/org/model'",
        )
        for source in untouched:
            with self.subTest(source=source):
                self.assertEqual(
                    notebook_hack.fix_mirrored_model_page(source), source
                )

    def test_patch_changes_only_markdown_model_page_line(self) -> None:
        document = notebook(
            markdown(
                [
                    "## Local Inference on GPU\n",
                    "Model page: https://hf-mirror.com/Qwen/Qwen3.5-9B",
                ]
            ),
            code("url = 'https://hf-mirror.com/Qwen/Qwen3.5-9B'"),
        )
        patched = notebook_hack.patch_notebook(document)
        self.assertIn(
            "Model page: https://huggingface.co/Qwen/Qwen3.5-9B",
            cell_source(patched["cells"][0]),
        )
        self.assertIn("hf-mirror.com", cell_source(patched["cells"][1]))


class NotebookFileInterfaceTests(unittest.TestCase):
    def test_explicit_output_preserves_input(self) -> None:
        original = notebook(
            markdown("Model page: https://hf-mirror.com/org/model"),
            code('model = AutoModel.from_pretrained("org/model")'),
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.ipynb"
            output_path = Path(directory) / "output.ipynb"
            input_path.write_text(json.dumps(original), encoding="utf-8")

            written = notebook_hack.hack_notebook_file(input_path, output_path)

            self.assertEqual(written, output_path)
            self.assertEqual(json.loads(input_path.read_text()), original)
            hacked = json.loads(output_path.read_text())
            self.assertIn("huggingface.co", cell_source(hacked["cells"][0]))
            self.assertNotIn("device_map", cell_source(hacked["cells"][1]))

    def test_default_is_in_place_and_backward_alias_still_works(self) -> None:
        for operation in (
            notebook_hack.hack_notebook_file,
            notebook_hack.patch_notebook_file,
        ):
            with self.subTest(operation=operation.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "input.ipynb"
                path.write_text(
                    json.dumps(
                        notebook(markdown("Model page: https://hf-mirror.com/org/model"))
                    ),
                    encoding="utf-8",
                )
                operation(path)
                self.assertIn("huggingface.co", path.read_text())

    def test_rejects_json_that_is_not_a_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.ipynb"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a notebook"):
                notebook_hack.hack_notebook_file(path)

    def test_cli_supports_in_place_and_explicit_output(self) -> None:
        script = Path(notebook_hack.__file__).resolve()
        source_document = notebook(
            markdown("Model page: https://hf-mirror.com/org/model")
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.ipynb"
            output_path = Path(directory) / "output.ipynb"
            input_path.write_text(json.dumps(source_document), encoding="utf-8")

            subprocess.run(
                [sys.executable, str(script), str(input_path), "--output", str(output_path)],
                check=True,
            )
            self.assertNotIn("huggingface.co", input_path.read_text())
            self.assertIn("huggingface.co", output_path.read_text())

            subprocess.run([sys.executable, str(script), str(input_path)], check=True)
            self.assertIn("huggingface.co", input_path.read_text())


class ExistingHackRegressionTests(unittest.TestCase):
    def test_patch_is_idempotent_for_representative_notebook(self) -> None:
        document = notebook(
            markdown("Model page: https://hf-mirror.com/org/model"),
            code('pipe = pipeline("text-generation", model="org/model")'),
            code('model = AutoModel.from_pretrained("org/model")'),
            markdown("## Remote Inference\nNot executed locally"),
            code("remote_call()"),
        )
        once = notebook_hack.patch_notebook(copy.deepcopy(document))
        twice = notebook_hack.patch_notebook(copy.deepcopy(once))
        self.assertEqual(twice, once)


if __name__ == "__main__":
    unittest.main()
