#!/usr/bin/env python3
"""Portable notebook hacking for Hugging Face one-click notebooks.

This stdlib-only module is suitable for both Radeon Global backend handoff and
the local model-notebook CI runner.  It supports library and command-line use
without importing application code or notebook runtime dependencies.

Library usage::

    from notebook_hack import hack_notebook_file

    # Modify the input notebook in place.
    hack_notebook_file("model.ipynb")

    # Or preserve the input and write a hacked copy elsewhere.
    hack_notebook_file("model.ipynb", "review/model.ipynb")

Command-line usage::

    python notebook_hack.py model.ipynb
    python notebook_hack.py model.ipynb --output review/model.ipynb
"""

import argparse
import io
import json
import re
import tokenize
from pathlib import Path

# Markdown markers that begin the remote/serverless-inference section. AMD GPUs
# run local inference only, so everything from the first matching markdown cell
# onward is dropped (those cells require a real HF inference token).
_REMOTE_INFERENCE_MARKERS = (
    "remote inference",
    "serverless inference",
)

# hf-mirror sometimes rewrites the generated notebook's human-readable model
# page attribution while proxying ``/<namespace>/<model>.ipynb``.  Match only a
# complete Markdown ``Model page:`` line containing a two-segment model page
# URL.  This deliberately does not rewrite code, arbitrary prose, repo file
# URLs, API endpoints, or longer paths under a model repository.
_MODEL_PAGE_MIRROR_RE = re.compile(
    r"(?im)^(?P<prefix>[ \t]*Model[ \t]+page[ \t]*:[ \t]*)"
    r"(?P<url>https://hf-mirror\.com/"
    r"(?P<namespace>[A-Za-z0-9][A-Za-z0-9._-]*)/"
    r"(?P<model>[A-Za-z0-9][A-Za-z0-9._-]*))"
    r"(?P<suffix>[ \t]*)$"
)


def _cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return src


def fix_mirrored_model_page(source: str) -> str:
    """Canonicalize an hf-mirror model-page attribution in Markdown.

    Only a full line shaped as ``Model page:
    https://hf-mirror.com/<namespace>/<model>`` is changed.  The namespace and
    model are intentionally generic so the rule applies to every Hugging Face
    model, while the strict line boundary prevents changes to downloads, code,
    API routes, repository files, query strings, anchors, or incidental prose.
    """

    return _MODEL_PAGE_MIRROR_RE.sub(
        lambda match: (
            f"{match.group('prefix')}https://huggingface.co/"
            f"{match.group('namespace')}/{match.group('model')}"
            f"{match.group('suffix')}"
        ),
        source,
    )


_IGNORED_TOKEN_TYPES = {
    tokenize.COMMENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.NEWLINE,
    tokenize.NL,
}


def _token_offset(line_offsets: list[int], position: tuple[int, int]) -> int:
    """Convert a tokenize ``(row, column)`` position to a string offset."""
    row, column = position
    return line_offsets[row - 1] + column


def _device_map_insertions(source: str) -> list[tuple[int, str]]:
    """Return insertions for target calls, using tokens to pair parentheses.

    A regular expression cannot identify the closing parenthesis of calls such
    as ``from_pretrained(resolve_model_name())``. Tokenization lets us pair the
    outer call's parentheses while ignoring parentheses inside strings and
    comments, without reformatting the rest of the cell.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (IndentationError, tokenize.TokenError):
        # Leave malformed or incomplete cells untouched instead of risking a
        # syntactically invalid rewrite.
        return []

    significant = [
        i for i, token in enumerate(tokens)
        if token.type not in _IGNORED_TOKEN_TYPES
    ]
    significant_position = {token_index: i for i, token_index in enumerate(significant)}

    closing_paren: dict[int, int] = {}
    paren_stack: list[int] = []
    for token_index in significant:
        token = tokens[token_index]
        if token.type != tokenize.OP:
            continue
        if token.string == "(":
            paren_stack.append(token_index)
        elif token.string == ")" and paren_stack:
            closing_paren[paren_stack.pop()] = token_index

    target_opens: set[int] = set()
    for position, token_index in enumerate(significant):
        token = tokens[token_index]
        if token.type != tokenize.NAME or position + 1 >= len(significant):
            continue
        next_index = significant[position + 1]
        if tokens[next_index].string != "(":
            continue
        if token.string == "from_pretrained":
            if position == 0 or tokens[significant[position - 1]].string != ".":
                continue
            target_opens.add(next_index)
        elif token.string == "pipeline":
            previous = tokens[significant[position - 1]].string if position else ""
            if previous not in {"def", "class"}:
                target_opens.add(next_index)

    line_offsets = [0]
    line_offsets.extend(match.end() for match in re.finditer("\n", source))
    insertions: list[tuple[int, str]] = []
    for open_index in target_opens:
        close_index = closing_paren.get(open_index)
        if close_index is None:
            continue

        close_position = significant_position[close_index]
        following = [
            tokens[index].string
            for index in significant[close_position + 1:close_position + 4]
        ]
        if following == [".", "to", "("]:
            continue

        args_start = _token_offset(line_offsets, tokens[open_index].end)
        args_end = _token_offset(line_offsets, tokens[close_index].start)
        args = source[args_start:args_end]
        if re.search(r"\b(?:device_map|device)\s*=", args):
            continue

        trimmed_args = args.rstrip()
        insertion_offset = args_start + len(trimmed_args)
        if not trimmed_args:
            insertion = 'device_map="cuda"'
        elif trimmed_args.endswith(","):
            if "\n" in args:
                last_line = trimmed_args.rsplit("\n", 1)[-1]
                indent = last_line[:len(last_line) - len(last_line.lstrip())]
                insertion = f'\n{indent}device_map="cuda",'
            else:
                insertion = ' device_map="cuda",'
        else:
            insertion = ', device_map="cuda"'
        insertions.append((insertion_offset, insertion))

    return insertions


def inject_device_map(source: str) -> str:
    """Force ``device_map="cuda"`` on model-loading calls.

    Without a device_map the model loads on CPU (peak VRAM stays at baseline)
    and the notebook never exercises the GPU. ``"cuda"`` (not ``"auto"``) is used
    deliberately: ``"auto"`` will offload layers to CPU when the GPU looks full,
    silently reintroducing the CPU-load leak; ``"cuda"`` forces the whole model
    onto the GPU and fails loudly (OOM) instead.

    Two passes: (1) rewrite any existing ``device_map="auto"`` to ``"cuda"``, so
    notebooks that already hardcode ``"auto"`` don't keep the leaky behaviour;
    (2) inject ``device_map="cuda"`` into ``from_pretrained``/``pipeline`` calls
    that have no device_map at all. Chained ``.from_pretrained(...).to(dev)``
    calls are left untouched.
    """
    source = re.sub(r'device_map\s*=\s*(["\'])auto\1', 'device_map="cuda"', source)
    for offset, insertion in sorted(_device_map_insertions(source), reverse=True):
        source = source[:offset] + insertion + source[offset:]
    return source


def _rename_dtype_kwarg(m: "re.Match") -> str:
    full = m.group(0)
    if "torch_dtype" in full:
        return full
    return re.sub(r'(?<![\w.])dtype\s*=', "torch_dtype=", full)


def fix_diffusers_dtype(source: str) -> str:
    """Rewrite ``dtype=`` to ``torch_dtype=`` in diffusers pipeline loads.

    diffusers' ``DiffusionPipeline.from_pretrained`` silently ignores an
    unknown ``dtype=`` kwarg, so the model loads in float32 while the rest of
    the pipeline runs in the requested (e.g. bfloat16) dtype — the forward pass
    then dies with a dtype-mismatch matmul error. diffusers wants
    ``torch_dtype=``. (In transformers 5.x the kwarg was renamed the other way,
    from ``torch_dtype`` to ``dtype``, so this rewrite is scoped to diffusers
    only — applied when the cell imports from ``diffusers`` — to avoid breaking
    transformers loads.) If ``torch_dtype=`` is already present the call is left
    untouched.
    """
    if "diffusers" not in source:
        return source
    return re.sub(r'\.from_pretrained\([^)]*?\)', _rename_dtype_kwarg, source)


_FREE_VRAM_PREAMBLE = (
    "import gc, torch\n"
    "try:\n"
    "    del pipe\n"
    "except NameError:\n"
    "    pass\n"
    "gc.collect()\n"
    "torch.cuda.empty_cache()\n"
    "\n"
)


def free_vram_before_reload(source: str) -> str:
    """Prepend VRAM cleanup to a ``from_pretrained`` cell that follows a
    ``pipeline()`` cell.

    Many HF one-click notebooks load the model twice — once via ``pipeline()``
    and again via ``from_pretrained()`` — leaving both copies resident. For
    large models the second load OOMs on a single GPU. Freeing the ``pipe``
    object first lets the second load reuse the VRAM.
    """
    return _FREE_VRAM_PREAMBLE + source


def patch_notebook(nb: dict) -> dict:
    """Patch a notebook dict in place and return it.

    Applies, in order: kernelspec normalization, canonical model-page links in
    markdown, device_map injection and double-load VRAM cleanup into code cells,
    and the remote-inference trim.
    """
    # Force the python3 kernel so the pod's ipykernel is used.
    nb.setdefault("metadata", {})["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }

    prev_cell_used_pipeline = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            markdown = cell.get("source", "")
            if isinstance(markdown, list):
                markdown = "".join(markdown)
            cell["source"] = [fix_mirrored_model_page(markdown)]
            continue
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        src = inject_device_map(src)
        src = fix_diffusers_dtype(src)
        if (
            prev_cell_used_pipeline
            and "from_pretrained" in src
            and _FREE_VRAM_PREAMBLE not in src
        ):
            src = free_vram_before_reload(src)
        prev_cell_used_pipeline = "pipeline(" in src
        cell["source"] = [src]

    # Trim everything from the first remote/serverless-inference markdown cell
    # onward. Only markdown cells are checked to avoid false positives from code
    # cells that mention these terms in comments or strings.
    cells = nb.get("cells", [])
    cut = len(cells)
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        text = _cell_text(cell).lower()
        if any(marker in text for marker in _REMOTE_INFERENCE_MARKERS):
            cut = i
            break
    if cut < len(cells):
        nb["cells"] = cells[:cut]

    return nb


def hack_notebook_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Hack one notebook file and return the path that was written.

    Args:
        input_path: Existing ``.ipynb`` file to read as UTF-8 JSON.
        output_path: Destination for the hacked notebook.  When omitted, the
            input file is modified in place.  Parent directories are not
            created implicitly, so a misspelled destination fails visibly.

    Returns:
        The destination as a :class:`pathlib.Path`.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        json.JSONDecodeError: If the input is not valid JSON.
        ValueError: If the JSON document is not a notebook object with cells.
    """

    source_path = Path(input_path)
    destination = Path(output_path) if output_path is not None else source_path
    with source_path.open(encoding="utf-8") as f:
        nb = json.load(f)
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        raise ValueError(f"{source_path} is not a notebook JSON document")
    patch_notebook(nb)
    with destination.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return destination


def patch_notebook_file(path: str | Path) -> None:
    """Backward-compatible in-place alias for :func:`hack_notebook_file`."""

    hack_notebook_file(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the Radeon Hugging Face one-click hacks to a notebook."
    )
    parser.add_argument("notebook", help="input .ipynb file")
    parser.add_argument(
        "-o",
        "--output",
        help="write to this file instead of modifying the input in place",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hack_notebook_file(args.notebook, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
