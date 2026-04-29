from __future__ import annotations

import html
import hashlib
import os
import re
import shutil
import threading
import sys
import uuid
import ast
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request

from engine.config import DATA_DIR, DEFAULT_CONFIG, DEFAULT_RP_SCHEMA
from engine import dynamicPPO as dyppo_engine
from engine.dynamicPPO import online_update_dynamic_ppo, train_dynamic_ppo
from engine.selfattention import online_update_self_attention_ppo, train_self_attention_ppo
from engine.storage import append_jsonl, load_json, load_jsonl, save_json, utc_now_iso


app = Flask(__name__, template_folder="templates", static_folder="static")

DOCS_DIR = Path(__file__).resolve().parent / "docs"
DOCS_FILE_ORDER = (
    "CURRENT_ALGORITHMS.md",
    "EFFICIENT_DESIGN.md",
    "DYPPO.md",
    "SELFATTENTION.md",
    "ALGORITHM_COMPARISON.md",
)
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
SUBSCRIPT_IDENT_RE = re.compile(r"\b([A-Za-z]+)((?:_[A-Za-z0-9]+)+)\b")
INLINE_MATH_DOLLAR_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")
INLINE_MATH_PAREN_RE = re.compile(r"\\\((.+?)\\\)")
ALLOWED_VARIABLE_TYPES = ("continuous", "categorical", "ordinal")


def _merge_dict(base: dict, patch: dict) -> dict:
    out = deepcopy(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _doc_anchor_for_name(name: str) -> str:
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return f"doc-{slug or 'item'}"


def _ordered_doc_files() -> list[Path]:
    if not DOCS_DIR.exists():
        return []
    files = {p.name: p for p in DOCS_DIR.glob("*.md") if p.is_file()}
    ordered: list[Path] = []
    for name in DOCS_FILE_ORDER:
        if name in files:
            ordered.append(files.pop(name))
    ordered.extend(sorted(files.values(), key=lambda p: p.name.lower()))
    return ordered


def _doc_link_map(doc_files: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in doc_files:
        anchor = _doc_anchor_for_name(path.name)
        out[path.name] = anchor
        out[str(path)] = anchor
        out[path.resolve().as_posix()] = anchor
    return out


def _render_markdown_inline(text: str, link_map: dict[str, str]) -> str:
    code_tokens: dict[str, str] = {}
    math_tokens: dict[str, str] = {}

    def _inline_math_html(expr: str) -> str:
        tex = html.escape(_pseudo_formula_to_tex(expr), quote=False)
        return f'<span class="doc-inline-math">\\({tex}\\)</span>'

    def _looks_like_inline_math_fragment(text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        if raw.startswith("http://") or raw.startswith("https://"):
            return False
        if re.fullmatch(r"[A-Za-z_]\w*(?:\([^`]*\))?", raw):
            return False
        if "/" in raw and "\\" not in raw and "_" not in raw and "^" not in raw:
            return False
        if "\\" in raw:
            return True
        if "_" in raw or "^" in raw:
            return True
        if re.fullmatch(r"[A-Za-z]+\([^)]*\)", raw) and any(ch in raw for ch in ("_", "^")):
            return True
        return False

    def _replace_code(match: re.Match[str]) -> str:
        content = match.group(1)
        if _looks_like_inline_math_fragment(content):
            token = f"@@MATH{len(math_tokens)}@@"
            math_tokens[token] = _inline_math_html(content)
            return token
        token = f"@@CODE{len(code_tokens)}@@"
        code_tokens[token] = f"<code>{html.escape(content, quote=False)}</code>"
        return token

    def _replace_inline_math(match: re.Match[str]) -> str:
        expr = match.group(1)
        token = f"@@MATH{len(math_tokens)}@@"
        math_tokens[token] = _inline_math_html(expr)
        return token

    text = INLINE_CODE_RE.sub(_replace_code, text)
    text = INLINE_MATH_PAREN_RE.sub(_replace_inline_math, text)
    text = INLINE_MATH_DOLLAR_RE.sub(_replace_inline_math, text)
    escaped = html.escape(text, quote=False)

    def _replace_link(match: re.Match[str]) -> str:
        label = match.group(1)
        raw_href = html.unescape(match.group(2)).strip()
        href = f"#{link_map[raw_href]}" if raw_href in link_map else raw_href
        attrs = ' target="_blank" rel="noopener noreferrer"' if href.startswith("http") else ""
        return f'<a href="{html.escape(href, quote=True)}"{attrs}>{label}</a>'

    escaped = INLINE_LINK_RE.sub(_replace_link, escaped)
    escaped = INLINE_BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    for token, snippet in code_tokens.items():
        escaped = escaped.replace(token, snippet)
    for token, snippet in math_tokens.items():
        escaped = escaped.replace(token, snippet)
    return escaped


def _looks_like_math_line(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    chinese_chars = sum(1 for ch in stripped if "\u4e00" <= ch <= "\u9fff")
    math_markers = ("∈", "Σ", "√", "->", "<=", ">=", "!=", "=", "^")
    if any(marker in stripped for marker in math_markers) and chinese_chars <= 2:
        return True
    if re.fullmatch(r"[A-Za-z0-9_(),.\s+\-/*=^{}\[\]\\|:><]+", stripped) and any(marker in stripped for marker in ("=", "->", "^")):
        return True
    return False


def _pseudo_formula_to_tex(text: str) -> str:
    out = str(text or "").strip()
    out = re.sub(r"\s+", " ", out)
    out = out.replace("...", r"\ldots ")
    out = out.replace("∈", r" \in ")
    out = out.replace("×", r" \times ")
    out = out.replace("->", r" \to ")
    out = out.replace("<=", r" \le ")
    out = out.replace(">=", r" \ge ")
    out = out.replace("!=", r" \ne ")
    out = out.replace("Σ", r"\sum ")
    out = re.sub(r"\bmod\b", r"\\bmod", out)
    out = re.sub(r"\bconcat\(", r"\\operatorname{concat}(", out)
    out = re.sub(r"\bdiag\(", r"\\operatorname{diag}(", out)
    out = re.sub(r"\bsoftmax\(", r"\\operatorname{softmax}(", out)
    out = re.sub(r"\bsigmoid\(", r"\\operatorname{sigmoid}(", out)
    out = re.sub(r"\bsqrt\(([^()]+)\)", r"\\sqrt{\1}", out)
    out = re.sub(r"\bR\^\[([^\]]+)\]", r"\\mathbb{R}^{[\1]}", out)
    out = re.sub(r"(\[[^\]]+\])\^\[([^\]]+)\]", r"\1^{[\2]}", out)

    def _replace_set_dims(match: re.Match[str]) -> str:
        inner = match.group(1)
        dims = match.group(2)
        return rf"\{{{inner}\}}^{{[{dims}]}}"

    out = re.sub(r"\{([^{}]+)\}\^\[([^\]]+)\]", _replace_set_dims, out)

    def _replace_ident(match: re.Match[str]) -> str:
        base = match.group(1)
        suffix = match.group(2).lstrip("_")
        return f"{base}_{{{suffix}}}"

    out = SUBSCRIPT_IDENT_RE.sub(_replace_ident, out)
    return out


def _render_math_block(lines: list[str]) -> str:
    chunks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        stripped = str(line or "").strip()
        if not stripped:
            if current:
                chunks.append(current)
                current = []
            continue
        current.append(stripped)
    if current:
        chunks.append(current)

    rendered: list[str] = []
    for chunk in chunks:
        joined = " ".join(chunk)
        tex = html.escape(_pseudo_formula_to_tex(joined), quote=False)
        rendered.append(f'<div class="doc-math-block">\\[{tex}\\]</div>')
    return "\n".join(rendered)


def _is_formula_fence(lang: str, code_lines: list[str]) -> bool:
    code_nonempty = [line for line in code_lines if str(line).strip()]
    if not code_nonempty:
        return False
    if lang in ("math", "latex"):
        return True
    if lang == "text" and all(_looks_like_math_line(line) for line in code_nonempty):
        return True
    return False


def _is_markdown_block_start(line: str, next_line: str | None = None) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("```"):
        return True
    if re.match(r"#{1,6}\s+", stripped):
        return True
    if re.match(r"^[-*]\s+", stripped):
        return True
    if re.match(r"^\d+\.\s+", stripped):
        return True
    if stripped == "---":
        return True
    if stripped.startswith("|") and next_line and re.match(r"^\s*\|?[-:| ]+\|?\s*$", next_line.strip()):
        return True
    return False


def _render_markdown_table(lines: list[str], start: int, link_map: dict[str, str]) -> tuple[str, int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip().strip("|")
        cells = [cell.strip() for cell in raw.split("|")]
        rows.append(cells)
        i += 1
    if len(rows) < 2:
        return f"<p>{_render_markdown_inline(lines[start].strip(), link_map)}</p>", start + 1
    header = rows[0]
    body = rows[2:]
    header_html = "".join(f"<th>{_render_markdown_inline(cell, link_map)}</th>" for cell in header)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{_render_markdown_inline(cell, link_map)}</td>" for cell in row) + "</tr>"
        for row in body
    )
    table_html = [
        '<div class="doc-table-wrap">',
        '<table class="doc-table">',
        f"<thead><tr>{header_html}</tr></thead>",
        f"<tbody>{body_html}</tbody>",
        "</table>",
        "</div>",
    ]
    return "".join(table_html), i


def _render_markdown_list(lines: list[str], start: int, link_map: dict[str, str]) -> tuple[str, int]:
    def indent_of(text: str) -> int:
        return len(text) - len(text.lstrip(" "))

    def parse(idx: int, base_indent: int) -> tuple[str, int]:
        tag: str | None = None
        items_html: list[str] = []

        while idx < len(lines):
            raw = lines[idx]
            stripped = raw.strip()
            if not stripped:
                idx += 1
                break

            current_indent = indent_of(raw)
            if current_indent < base_indent:
                break
            if current_indent > base_indent:
                break

            ordered_match = re.match(r"^\d+\.\s+(.*)", stripped)
            unordered_match = re.match(r"^[-*]\s+(.*)", stripped)
            if not ordered_match and not unordered_match:
                break

            current_tag = "ol" if ordered_match else "ul"
            if tag is None:
                tag = current_tag
            if current_tag != tag:
                break

            item_text = (ordered_match.group(1) if ordered_match else unordered_match.group(1)).strip()
            idx += 1
            continuation: list[str] = []
            nested_blocks: list[str] = []

            while idx < len(lines):
                nxt_raw = lines[idx]
                nxt_stripped = nxt_raw.strip()
                if not nxt_stripped:
                    idx += 1
                    break

                nxt_indent = indent_of(nxt_raw)
                if nxt_indent < base_indent:
                    break

                nested_ordered = re.match(r"^\d+\.\s+(.*)", nxt_stripped)
                nested_unordered = re.match(r"^[-*]\s+(.*)", nxt_stripped)
                if nested_ordered or nested_unordered:
                    if nxt_indent == base_indent:
                        break
                    if nxt_indent > base_indent:
                        nested_html, idx = parse(idx, nxt_indent)
                        if nested_html:
                            nested_blocks.append(nested_html)
                        continue

                if nxt_indent > base_indent:
                    continuation.append(nxt_stripped)
                    idx += 1
                    continue

                break

            text_html = _render_markdown_inline(" ".join([item_text] + continuation).strip(), link_map)
            items_html.append(f"<li>{text_html}{''.join(nested_blocks)}</li>")

        if not tag:
            return "", idx
        return f"<{tag}>{''.join(items_html)}</{tag}>", idx

    return parse(start, len(lines[start]) - len(lines[start].lstrip(" ")))


def _render_markdown(md_text: str, link_map: dict[str, str]) -> str:
    lines = md_text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        next_line = lines[i + 1] if i + 1 < len(lines) else None

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            i += 1
            code_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            if _is_formula_fence(lang, code_lines):
                blocks.append(_render_math_block(code_lines))
                continue
            lang_attr = f' data-lang="{html.escape(lang, quote=True)}"' if lang else ""
            code_html = html.escape("\n".join(code_lines), quote=False)
            blocks.append(f'<pre class="doc-code"><code{lang_attr}>{code_html}</code></pre>')
            continue

        if re.match(r"#{1,6}\s+", stripped):
            level = min(6, len(stripped) - len(stripped.lstrip("#")))
            content = stripped[level:].strip()
            blocks.append(f"<h{level}>{_render_markdown_inline(content, link_map)}</h{level}>")
            i += 1
            continue

        if stripped == "---":
            blocks.append('<hr class="doc-divider" />')
            i += 1
            continue

        if stripped.startswith("|") and next_line and re.match(r"^\s*\|?[-:| ]+\|?\s*$", next_line.strip()):
            table_html, i = _render_markdown_table(lines, i, link_map)
            blocks.append(table_html)
            continue

        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            list_html, i = _render_markdown_list(lines, i, link_map)
            blocks.append(list_html)
            continue

        paragraph_lines = [stripped]
        i += 1
        while i < len(lines):
            current = lines[i]
            current_stripped = current.strip()
            lookahead = lines[i + 1] if i + 1 < len(lines) else None
            if not current_stripped:
                i += 1
                break
            if _is_markdown_block_start(current, lookahead):
                break
            paragraph_lines.append(current_stripped)
            i += 1
        paragraph_text = " ".join(paragraph_lines)
        if len(paragraph_lines) == 1 and _looks_like_math_line(paragraph_text):
            blocks.append(_render_math_block([paragraph_text]))
        else:
            blocks.append(f"<p>{_render_markdown_inline(paragraph_text, link_map)}</p>")

    return "\n".join(blocks)


def _extract_doc_title(md_text: str, fallback: str) -> str:
    for line in md_text.splitlines():
        stripped = line.strip()
        if re.match(r"^#\s+", stripped):
            return stripped[2:].strip()
    return fallback


def _load_docs_view_data() -> list[dict[str, str]]:
    doc_files = _ordered_doc_files()
    link_map = _doc_link_map(doc_files)
    docs: list[dict[str, str]] = []
    for path in doc_files:
        raw = path.read_text(encoding="utf-8")
        docs.append(
            {
                "name": path.name,
                "anchor": _doc_anchor_for_name(path.name),
                "title": _extract_doc_title(raw, path.stem),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "html": _render_markdown(raw, link_map),
            }
        )
    return docs


def _normalize_runtime_config(raw: dict | None) -> dict:
    merged = _merge_dict(DEFAULT_CONFIG, raw if isinstance(raw, dict) else {})
    # Keep only actively used top-level config keys.
    allowed_top_keys = (
        "dynamic_ppo",
        "self_attention",
        "data_sources",
    )
    cfg = {k: merged.get(k) for k in allowed_top_keys}
    return cfg


def _load_runtime_config() -> dict:
    raw = load_json(paths()["config"], DEFAULT_CONFIG)
    return _normalize_runtime_config(raw if isinstance(raw, dict) else {})


def ensure_app_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths()["sp_design_dir"].mkdir(parents=True, exist_ok=True)
    paths()["sp_design_weights_dir"].mkdir(parents=True, exist_ok=True)
    paths()["respondents_dir"].mkdir(parents=True, exist_ok=True)

    config_path = paths()["config"]
    config = _load_runtime_config()
    save_json(config_path, config)

    for k in ("profile_submissions", "trip_diary_submissions", "sp_submissions"):
        p = paths()[k]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)


COMPUTE_JOBS: dict[str, dict] = {}
COMPUTE_LOCK = threading.Lock()


def paths() -> dict[str, Path]:
    return {
        "config": DATA_DIR / "config.json",
        "profile_submissions": DATA_DIR / "profile_submissions.jsonl",
        "trip_diary_submissions": DATA_DIR / "trip_diary_submissions.jsonl",
        "sp_submissions": DATA_DIR / "sp_submissions.jsonl",
        "design_spec": DATA_DIR / "design_spec_saved.json",
        "sp_design_dir": DATA_DIR / "sp_design",
        "sp_design_weights_dir": DATA_DIR / "sp_design" / "weights",
        "respondents_dir": DATA_DIR / "respondents",
    }


ensure_app_data()


def sanitize_respondent_id(raw: str) -> str:
    rid = "".join(ch for ch in str(raw or "").strip() if ch.isalnum() or ch in ("_", "-"))
    return rid


def sanitize_save_name(name: str) -> str:
    safe = "".join(ch for ch in str(name or "") if ch.isalnum() or ch in ("_", "-", "."))
    return safe.strip("._-")


def normalize_design_type(raw: str | None) -> str:
    t = str(raw or "").strip().lower()
    return t if t in ("efficient", "dyppo", "selfattention") else "efficient"


def _is_numeric_like(value: Any) -> bool:
    try:
        float(str(value).strip())
        return True
    except Exception:
        return False


def _infer_variable_type(var_name: str | None, levels: list[Any] | None) -> str:
    name = str(var_name or "").strip().lower()
    lv = [str(x).strip() for x in (levels or []) if str(x).strip()]
    if not lv:
        return "continuous"
    ordinal_hints = re.compile(r"(level|rank|grade|quality|safety|stress|comfort|crowding|risk|score|count|次数|程度|等级|质量|安全|压力|舒适|拥挤|风险|评分)")
    if not all(_is_numeric_like(x) for x in lv):
        return "categorical"

    nums = [float(x) for x in lv]
    uniq_sorted = sorted(set(nums))
    is_binary = len(uniq_sorted) == 2 and uniq_sorted[0] == 0.0 and uniq_sorted[1] == 1.0
    is_small_integer_scale = (
        2 <= len(uniq_sorted) <= 5
        and all(float(int(n)) == n for n in uniq_sorted)
        and all((uniq_sorted[i] - uniq_sorted[i - 1]) == 1.0 for i in range(1, len(uniq_sorted)))
    )
    if ordinal_hints.search(name) or is_small_integer_scale:
        return "ordinal"
    if is_binary:
        return "ordinal"
    return "continuous"


def normalize_variable_type(raw: str | None, *, var_name: str | None = None, levels: list[Any] | None = None) -> str:
    vt = str(raw or "").strip().lower()
    if vt in ALLOWED_VARIABLE_TYPES:
        return vt
    return _infer_variable_type(var_name, levels)


def normalize_design_spec_variable_types(payload_obj: dict) -> dict:
    out = deepcopy(payload_obj) if isinstance(payload_obj, dict) else {}
    spec = out.get("design_spec", {})
    if not isinstance(spec, dict):
        return out
    alternatives = spec.get("alternatives", [])
    if not isinstance(alternatives, list):
        return out
    for alt in alternatives:
        if not isinstance(alt, dict):
            continue
        variables = alt.get("variables", [])
        if not isinstance(variables, list):
            continue
        for var in variables:
            if not isinstance(var, dict):
                continue
            var_name = str(var.get("name", "")).strip()
            levels = var.get("levels", []) if isinstance(var.get("levels", []), list) else []
            var["variable_type"] = normalize_variable_type(var.get("variable_type"), var_name=var_name, levels=levels)
    return out


def design_spec_file_path(save_name: str) -> Path:
    safe = sanitize_save_name(save_name)
    return paths()["sp_design_dir"] / f"{safe}.json"


def design_weight_file_path(save_name: str) -> Path:
    safe = sanitize_save_name(save_name)
    return paths()["sp_design_weights_dir"] / f"{safe}.pt"


def _resolve_weight_file_path(weight_file: str | None) -> Path | None:
    wf = str(weight_file or "").strip()
    if not wf:
        return None
    p = Path(wf)
    if p.is_absolute():
        return p
    return paths()["sp_design_weights_dir"] / p


def _payload_save_name(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return None
    for k in ("save_name", "design_save_name"):
        v = sanitize_save_name(str(payload.get(k, "")).strip())
        if v:
            return v
    return None


def ensure_design_type_field(rec: dict) -> dict:
    out = dict(rec) if isinstance(rec, dict) else {}
    payload = out.get("payload", {}) if isinstance(out.get("payload", {}), dict) else {}
    if payload:
        out["payload"] = normalize_design_spec_variable_types(payload)
        payload = out["payload"]
    if "type" in out and str(out.get("type", "")).strip().lower() in ("efficient", "dyppo", "selfattention"):
        out["type"] = normalize_design_type(out.get("type"))
        return out
    out["type"] = normalize_design_type(payload.get("design_type", out.get("design_type")))
    return out


def record_design_distribution(
    save_name: str | None,
    *,
    respondent_id: str,
    assignment_id: str,
    tasks: list[dict] | None,
    choices: dict | None,
    source: str,
    submitted_at: str,
    submission_id: str | None = None,
) -> bool:
    safe_name = sanitize_save_name(save_name or "")
    if not safe_name:
        return False
    f = design_spec_file_path(safe_name)
    if not f.exists():
        return False

    rec = ensure_design_type_field(load_json(f, {}))
    if not isinstance(rec, dict):
        return False

    runtime = rec.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    log = runtime.get("distribution_log", [])
    if not isinstance(log, list):
        log = []

    dedupe_key = f"{source}|{submission_id or ''}|{assignment_id}|{respondent_id}"
    if any(str(item.get("dedupe_key", "")) == dedupe_key for item in log if isinstance(item, dict)):
        return True

    tasks = tasks or []
    choices = choices or {}
    entry = {
        "dedupe_key": dedupe_key,
        "submission_id": submission_id,
        "assignment_id": assignment_id,
        "respondent_id": respondent_id,
        "source": source,
        "issued_to_respondent_id": respondent_id,
        "submitted_at": submitted_at,
        "task_ids": [str(t.get("id")) for t in tasks if isinstance(t, dict) and t.get("id")],
        "choices": choices,
    }
    log.append(entry)
    runtime["distribution_log"] = log[-5000:]
    runtime["distribution_count"] = len(runtime["distribution_log"])
    runtime["last_distributed_at"] = submitted_at

    # Lightweight aggregate stats for later allocation control.
    task_counts = runtime.get("task_issue_counts", {})
    if not isinstance(task_counts, dict):
        task_counts = {}
    if str(source) == "api/design/issue":
        for task_id in entry["task_ids"]:
            task_counts[task_id] = int(task_counts.get(task_id, 0)) + 1
    runtime["task_issue_counts"] = task_counts

    choice_counts = runtime.get("choice_counts", {})
    if not isinstance(choice_counts, dict):
        choice_counts = {}
    for task_id, chosen_alt in choices.items():
        key = f"{task_id}::{chosen_alt}"
        choice_counts[key] = int(choice_counts.get(key, 0)) + 1
    runtime["choice_counts"] = choice_counts

    rec["runtime"] = runtime
    save_json(f, rec)
    return True


def validate_sp_design_payload(payload_obj: dict) -> tuple[bool, str]:
    if not isinstance(payload_obj, dict):
        return False, "payload must be an object"
    payload_obj = normalize_design_spec_variable_types(payload_obj)
    design_spec = payload_obj.get("design_spec", {})
    if not isinstance(design_spec, dict):
        return False, "design_spec is required"
    alternatives = design_spec.get("alternatives", [])
    if not isinstance(alternatives, list) or len(alternatives) < 2:
        return False, "至少需要2个选项（alternatives）"

    alt_names = set()
    for idx, alt in enumerate(alternatives, start=1):
        if not isinstance(alt, dict):
            return False, f"选项{idx}格式错误"
        alt_name = sanitize_save_name(str(alt.get("name", "")).strip())
        if not alt_name:
            return False, f"选项{idx}缺少名称"
        if alt_name in alt_names:
            return False, f"选项名称重复：{alt_name}"
        alt_names.add(alt_name)

        variables = alt.get("variables", [])
        if not isinstance(variables, list) or len(variables) < 1:
            return False, f"选项{alt_name}至少需要1个变量"
        for v_idx, var in enumerate(variables, start=1):
            if not isinstance(var, dict):
                return False, f"选项{alt_name}变量{v_idx}格式错误"
            var_name = sanitize_save_name(str(var.get("name", "")).strip())
            if not var_name:
                return False, f"选项{alt_name}变量{v_idx}缺少名称"
            variable_type = normalize_variable_type(var.get("variable_type"), var_name=var_name, levels=var.get("levels", []))
            if variable_type not in ALLOWED_VARIABLE_TYPES:
                return False, f"选项{alt_name}变量{var_name}的变量类型无效"
            levels = var.get("levels", [])
            if not isinstance(levels, list) or len(levels) < 1:
                return False, f"选项{alt_name}变量{var_name}至少需要1个变量水平值"

    design_type = normalize_design_type(payload_obj.get("design_type"))
    sample_size = int(payload_obj.get("sample_size", 0) or 0)
    if sample_size <= 0:
        return False, "sample_size必须大于0"
    n_parameters = int(payload_obj.get("n_parameters", 0) or 0)
    if n_parameters <= 0:
        return False, "n_parameters必须大于0"

    options = payload_obj.get("design_options", {}) if isinstance(payload_obj.get("design_options", {}), dict) else {}
    if design_type == "efficient":
        eff = options.get("efficient", {}) if isinstance(options.get("efficient", {}), dict) else {}
        tpp = int(eff.get("tasks_per_person", 0) or 0)
        if tpp < 1:
            return False, "efficient.tasks_per_person必须>=1"
        row_iters = int(eff.get("row_exchange_iterations", 0) or 0)
        if row_iters < 1:
            return False, "efficient.row_exchange_iterations必须>=1"
    if design_type == "dyppo":
        mo = options.get("dyppo", {}) if isinstance(options.get("dyppo", {}), dict) else {}
        tpr = int(mo.get("tasks_per_round", 0) or 0)
        if tpr < 1:
            return False, "dyppo.tasks_per_round必须>=1"
        eps = float(mo.get("explore_epsilon", 0.0) or 0.0)
        if eps < 0 or eps > 1:
            return False, "dyppo.explore_epsilon必须在[0,1]"
    if design_type == "selfattention":
        so = options.get("selfattention", {}) if isinstance(options.get("selfattention", {}), dict) else {}
        tpr = int(so.get("tasks_per_round", 0) or 0)
        if tpr < 1:
            return False, "selfattention.tasks_per_round必须>=1"
        eps = float(so.get("explore_epsilon", 0.0) or 0.0)
        if eps < 0 or eps > 1:
            return False, "selfattention.explore_epsilon必须在[0,1]"
        win_l = int(so.get("window_length", 0) or 0)
        if win_l < 1:
            return False, "selfattention.window_length/L_hist必须>=1"
        sample_target_dim = int(so.get("sample_target_dim", 0) or 0)
        if sample_target_dim < 1:
            return False, "selfattention.sample_target_dim必须>=1"
    return True, ""


@app.route("/api/design/issue", methods=["POST"])
def api_design_issue():
    payload = request.get_json(silent=True) or {}
    design_save_name = str(payload.get("design_save_name", "")).strip() or None
    respondent_id = sanitize_respondent_id(payload.get("respondent_id", ""))
    assignment_id = str(payload.get("assignment_id", "")).strip() or f"issue_{uuid.uuid4().hex[:12]}"
    issued_at = utc_now_iso()

    if not respondent_id:
        respondent_id = f"R_{uuid.uuid4().hex[:10]}"

    respondent_record = load_respondent_record(respondent_id)
    profile_obj = respondent_record.get("profile", {}) if isinstance(respondent_record, dict) else {}
    if not isinstance(profile_obj, dict) or not profile_obj:
        return jsonify(
            {
                "error": "请先完成并保存当前受访者的RP/Profile信息，再获取SP题组。",
                "requires_profile": True,
                "preview_available": True,
            }
        ), 409

    safe_name = sanitize_save_name(design_save_name or "")
    f = design_spec_file_path(safe_name) if safe_name else None
    rec = load_json(f, {}) if f and f.exists() else {}
    rec = ensure_design_type_field(rec) if isinstance(rec, dict) else {}
    if not safe_name or not f or not f.exists() or not isinstance(rec, dict) or not rec:
        return jsonify({"error": "design_save_name not found"}), 404

    payload_inner = rec.get("payload", {}) if isinstance(rec.get("payload", {}), dict) else {}
    design_type = normalize_design_type(rec.get("type") or payload_inner.get("design_type"))

    tasks = payload.get("tasks", []) or []
    recommendation = rec.get("recommendation", None)
    model_state = {}
    d_error = {"value": None}

    if not tasks:
        if design_type == "dyppo":
            out = _compute_dyppo(
                {**payload_inner, "save_name": safe_name, "design_save_name": safe_name},
                current_respondent=_respondent_context_for_policy(respondent_record),
            )
            tasks = out.get("comb", []) if isinstance(out, dict) else []
            recommendation = out.get("recommendation", recommendation) if isinstance(out, dict) else recommendation
            model_state = out.get("model_state", {}) if isinstance(out, dict) else {}
            d_error = out.get("d_error", d_error) if isinstance(out, dict) else d_error
        elif design_type == "selfattention":
            out = _compute_selfattention(
                {**payload_inner, "save_name": safe_name, "design_save_name": safe_name},
                current_respondent=_respondent_context_for_policy(respondent_record),
            )
            tasks = out.get("comb", []) if isinstance(out, dict) else []
            recommendation = out.get("recommendation", recommendation) if isinstance(out, dict) else recommendation
            model_state = out.get("model_state", {}) if isinstance(out, dict) else {}
            d_error = out.get("d_error", d_error) if isinstance(out, dict) else d_error
        else:
            tasks = rec.get("preview_tasks", []) if isinstance(rec.get("preview_tasks", []), list) else []
            if not tasks and payload_inner:
                out = _compute_efficient(payload_inner)
                tasks = out.get("comb", []) if isinstance(out, dict) else []
                recommendation = out.get("recommendation", recommendation) if isinstance(out, dict) else recommendation
                d_error = out.get("d_error", d_error) if isinstance(out, dict) else d_error

    if not isinstance(tasks, list):
        tasks = []
    if not tasks:
        install_hint = ""
        if isinstance(model_state, dict):
            install_hint = str(model_state.get("install_hint", "") or "").strip()
        msg = install_hint or "当前策略未生成可下发题组。"
        return jsonify({"error": msg, "design_type": design_type, "model_state": model_state, "d_error": d_error}), 409

    ok = record_design_distribution(
        design_save_name,
        respondent_id=respondent_id,
        assignment_id=assignment_id,
        tasks=tasks,
        choices={},
        source="api/design/issue",
        submitted_at=issued_at,
    )
    if not ok:
        return jsonify({"error": "design_save_name not found"}), 404
    return jsonify(
        {
            "ok": True,
            "respondent_id": respondent_id,
            "assignment_id": assignment_id,
            "issued_at": issued_at,
            "design_type": design_type,
            "tasks": tasks,
            "recommendation": recommendation,
            "model_state": model_state,
            "d_error": d_error,
            "policy_version": (model_state or {}).get("policy_version") if isinstance(model_state, dict) else None,
        }
    )


def _tasks_per_questionnaire_from_design(payload_inner: dict, recommendation: dict | None = None, design_type: str | None = None) -> int:
    """读取单份问卷应展示的 SP 题目数，用于预览时避免把全部 block 都塞进页面。"""
    rec_tpp = int((recommendation or {}).get("tasks_per_person", 0) or 0) if isinstance(recommendation, dict) else 0
    if rec_tpp > 0:
        return rec_tpp
    payload_obj = payload_inner if isinstance(payload_inner, dict) else {}
    dtype = normalize_design_type(design_type or payload_obj.get("design_type"))
    options = payload_obj.get("design_options", {}) if isinstance(payload_obj.get("design_options", {}), dict) else {}
    if dtype == "efficient":
        return max(1, int(((options.get("efficient", {}) or {}).get("tasks_per_person", 0) or 0) or 8))
    if dtype == "selfattention":
        return max(1, int(((options.get("selfattention", {}) or {}).get("tasks_per_round", 0) or 0) or 6))
    if dtype == "dyppo":
        return max(1, int(((options.get("dyppo", {}) or {}).get("tasks_per_round", 0) or 0) or 6))
    return 6


def _preview_tasks_for_saved_design(rec: dict, payload_inner: dict, design_type: str, safe_name: str) -> dict:
    """只为页面样式预览生成题组，不读取 respondent、不更新权重、不写 distribution_log。"""
    recommendation = rec.get("recommendation", None) if isinstance(rec, dict) else None
    tasks = rec.get("preview_tasks", []) if isinstance(rec.get("preview_tasks", []), list) else []
    model_state: dict[str, Any] = {
        "preview_only": True,
        "preview_note": "仅用于设计阶段查看题面样式；不绑定 respondent，不记录分发，不更新策略权重。",
    }
    d_error = {"value": None}
    out: dict[str, Any] = {}

    if not tasks and payload_inner:
        preview_payload = deepcopy(payload_inner)
        tpp = _tasks_per_questionnaire_from_design(payload_inner, recommendation, design_type)
        # dyppo/selfattention 的真实 issue 会依赖 respondent 与历史权重；预览只看题面，
        # 因此用 efficient 的可行组合生成器作无副作用 fallback。
        preview_payload["design_type"] = "efficient"
        preview_payload["design_options"] = {
            "efficient": {
                "tasks_per_person": max(1, tpp),
                "row_exchange_iterations": min(
                    40,
                    max(1, int(((payload_inner.get("design_options", {}) or {}).get("efficient", {}) or {}).get("row_exchange_iterations", 20) or 20)),
                ),
            }
        }
        # 预览只需要一份问卷，但先多生成几倍 rows 再切片，避免条件约束过滤后题数不足。
        preview_payload["sample_size"] = max(max(1, tpp) * 3, max(1, tpp))
        preview_payload["target_block_sample"] = max(1, tpp)
        out = _compute_efficient(preview_payload)
        tasks = out.get("comb", []) if isinstance(out, dict) else []
        recommendation = out.get("recommendation", recommendation) if isinstance(out, dict) else recommendation
        d_error = out.get("d_error", d_error) if isinstance(out, dict) else d_error

    tpp = _tasks_per_questionnaire_from_design(payload_inner, recommendation, design_type)
    if isinstance(tasks, list) and tpp > 0:
        tasks = tasks[:tpp]
    return {
        "tasks": tasks if isinstance(tasks, list) else [],
        "recommendation": recommendation,
        "model_state": model_state,
        "d_error": d_error,
        "source_mode": (out.get("mode") if isinstance(out, dict) else None) or "saved_design_preview",
        "design_type": design_type,
        "save_name": safe_name,
    }


@app.route("/api/design/preview", methods=["POST"])
def api_design_preview():
    payload = request.get_json(silent=True) or {}
    design_save_name = str(payload.get("design_save_name", "")).strip() or None
    safe_name = sanitize_save_name(design_save_name or "")
    f = design_spec_file_path(safe_name) if safe_name else None
    rec = load_json(f, {}) if f and f.exists() else {}
    rec = ensure_design_type_field(rec) if isinstance(rec, dict) else {}
    if not safe_name or not f or not f.exists() or not isinstance(rec, dict) or not rec:
        return jsonify({"error": "design_save_name not found"}), 404

    payload_inner = rec.get("payload", {}) if isinstance(rec.get("payload", {}), dict) else {}
    design_type = normalize_design_type(rec.get("type") or payload_inner.get("design_type"))
    preview = _preview_tasks_for_saved_design(rec, payload_inner, design_type, safe_name)
    tasks = preview.get("tasks", [])
    if not tasks:
        return jsonify({"error": "当前设计未生成可预览题组。", "design_type": design_type}), 409

    previewed_at = utc_now_iso()
    return jsonify(
        {
            "ok": True,
            "preview_only": True,
            "respondent_id": "PREVIEW_ONLY",
            "assignment_id": f"preview_{uuid.uuid4().hex[:12]}",
            "previewed_at": previewed_at,
            "issued_at": previewed_at,
            "design_type": design_type,
            "save_name": safe_name,
            "tasks": tasks,
            "recommendation": preview.get("recommendation"),
            "model_state": preview.get("model_state"),
            "d_error": preview.get("d_error"),
            "source_mode": preview.get("source_mode"),
            "policy_version": None,
        }
    )


def respondent_file_path(respondent_id: str) -> Path:
    rid = sanitize_respondent_id(respondent_id)
    return paths()["respondents_dir"] / f"{rid}.json"


def load_respondent_record(respondent_id: str) -> dict:
    f = respondent_file_path(respondent_id)
    return load_json(f, {}) if f.exists() else {}


def save_respondent_section(respondent_id: str, section: str, payload: dict) -> dict:
    rid = sanitize_respondent_id(respondent_id)
    if not rid:
        rid = f"R_{uuid.uuid4().hex[:10]}"
    p = paths()["respondents_dir"]
    p.mkdir(parents=True, exist_ok=True)
    old = load_respondent_record(rid)
    now = utc_now_iso()
    record = {
        "respondent_id": rid,
        "created_at": old.get("created_at", now),
        "updated_at": now,
        "profile": old.get("profile"),
        "trip_diary": old.get("trip_diary"),
        "sp": old.get("sp", {}),
    }
    record[section] = payload
    save_json(respondent_file_path(rid), record)
    return record


def _to_num(v):
    try:
        n = float(v)
        return n if np.isfinite(n) else None
    except Exception:
        return None


def _task_satisfies_conditions(task: dict, conditions: list[str]) -> bool:
    if not conditions:
        return True

    def resolve_expr(expr: str):
        text = str(expr or "").strip()
        if not text:
            return False, 0.0

        plain_num = _to_num(text)
        if plain_num is not None:
            return True, plain_num

        token_values: dict[str, float] = {}

        def repl(match):
            raw = match.group(0)
            a, b = raw.split(".", 1)
            alts = task.get("alternatives", {})
            attrs = alts.get(a, {})
            val = _to_num(attrs.get(b))
            if val is None:
                raise ValueError(f"unknown condition token: {raw}")
            key = f"__cond_{len(token_values)}"
            token_values[key] = float(val)
            return key

        try:
            normalized = re.sub(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b", repl, text)
            node = ast.parse(normalized, mode="eval")
        except Exception:
            return False, 0.0

        def eval_node(n):
            if isinstance(n, ast.Expression):
                return eval_node(n.body)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return float(n.value)
            if isinstance(n, ast.Name) and n.id in token_values:
                return float(token_values[n.id])
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
                v = eval_node(n.operand)
                return v if isinstance(n.op, ast.UAdd) else -v
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left = eval_node(n.left)
                right = eval_node(n.right)
                if isinstance(n.op, ast.Add):
                    return left + right
                if isinstance(n.op, ast.Sub):
                    return left - right
                if isinstance(n.op, ast.Mult):
                    return left * right
                if abs(right) <= 1e-12:
                    raise ZeroDivisionError("condition expression division by zero")
                return left / right
            raise ValueError(f"unsupported condition expression: {ast.dump(n)}")

        try:
            return True, float(eval_node(node))
        except Exception:
            return False, 0.0

    def cmp(left, op, right):
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        return True

    for line in conditions:
        parts = [x.strip() for x in __import__("re").split(r"(>=|<=|>|<)", str(line or "").strip()) if x.strip()]
        if len(parts) < 3 or len(parts) % 2 == 0:
            continue
        ok = True
        for i in range(1, len(parts), 2):
            l_ok, l_val = resolve_expr(parts[i - 1])
            r_ok, r_val = resolve_expr(parts[i + 1])
            if not l_ok or not r_ok or not cmp(l_val, parts[i], r_val):
                ok = False
                break
        if not ok:
            return False
    return True


def _build_param_keys(spec: dict) -> list[str]:
    alts = spec.get("alternatives", []) if isinstance(spec, dict) else []
    base = sanitize_save_name(str(spec.get("asc_base_alternative", "")).strip())
    keys = []
    for alt in alts:
        a = sanitize_save_name(str((alt or {}).get("name", "")).strip())
        if not a:
            continue
        if a != base:
            keys.append(f"{a}.asc")
        for v in (alt or {}).get("variables", []) or []:
            n = sanitize_save_name(str((v or {}).get("name", "")).strip())
            if n:
                keys.append(f"{a}.{n}")
    return keys


def _design_matrix_generic(task: dict, param_keys: list[str]) -> np.ndarray:
    alts = (task or {}).get("alternatives", {}) or {}
    modes = list(alts.keys())
    rows = []
    for m in modes:
        attrs = alts.get(m, {}) or {}
        row = []
        for p in param_keys:
            if p.endswith(".asc"):
                a = p[:-4]
                row.append(1.0 if m == a else 0.0)
            else:
                a, var = p.split(".", 1)
                row.append(float(attrs.get(var, 0.0)) if m == a else 0.0)
        rows.append(row)
    return np.array(rows, dtype=float)


def _bayesian_d_error_generic(tasks: list[dict], spec: dict, beta_defaults: dict, beta_bounds: dict, beta_draws: int = 24, seed: int = 42) -> float:
    param_keys = _build_param_keys(spec)
    if not param_keys:
        return 1e9
    mean_vec = np.array([float(beta_defaults.get(k, 0.0)) for k in param_keys], dtype=float)
    std_vec = []
    for k in param_keys:
        b = beta_bounds.get(k, {}) if isinstance(beta_bounds, dict) else {}
        lo = _to_num((b or {}).get("min"))
        hi = _to_num((b or {}).get("max"))
        if lo is not None and hi is not None and hi > lo:
            std_vec.append(max((hi - lo) / 4.0, 1e-3))
        else:
            std_vec.append(max(abs(float(beta_defaults.get(k, 0.0))) * 0.2, 0.1))
    std_vec = np.array(std_vec, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=mean_vec, scale=std_vec, size=(max(4, beta_draws), len(param_keys)))
    errs = []
    for beta in draws:
        m = np.zeros((len(param_keys), len(param_keys)), dtype=float)
        for t in tasks:
            x = _design_matrix_generic(t, param_keys)
            if x.shape[0] < 2:
                continue
            u = x @ beta
            z = u - np.max(u)
            p = np.exp(z) / np.sum(np.exp(z))
            w = np.diag(p) - np.outer(p, p)
            m += x.T @ w @ x
        m += 1e-6 * np.eye(len(param_keys))
        sign, logdet = np.linalg.slogdet(m)
        if sign <= 0:
            errs.append(1e9)
        else:
            errs.append(float(np.exp((-logdet) / len(param_keys))))
    return float(np.mean(errs)) if errs else 1e9


def _build_tasks_from_spec(spec: dict, rows: int, seed: int = 42) -> list[dict]:
    """依据 design_spec 构造初始题组。

    该函数执行的是带随机扰动的构造式初始化，而不是优化求解。
    对每一行题目、每个备选项、每个变量，都会在对应 levels 中按
    “行号 + 变量序号 + 随机偏移”的方式选择一个取值，以形成可复现
    的初始设计起点。
    """
    rng = np.random.default_rng(seed)
    alts = spec.get("alternatives", []) if isinstance(spec, dict) else []
    tasks = []
    for i in range(rows):
        alternatives = {}
        for alt in alts:
            alt_name = sanitize_save_name(str((alt or {}).get("name", "")).strip())
            if not alt_name:
                continue
            attrs = {}
            vars_list = (alt or {}).get("variables", []) or []
            for v_idx, v in enumerate(vars_list):
                var_name = sanitize_save_name(str((v or {}).get("name", "")).strip())
                levels = (v or {}).get("levels", []) or []
                if not var_name or not levels:
                    continue
                lv = levels[(i + v_idx + int(rng.integers(0, len(levels)))) % len(levels)]
                attrs[var_name] = lv
            alternatives[alt_name] = attrs
        tasks.append({"id": f"comb_{i + 1}", "alternatives": alternatives})
    return tasks


def _design_axes_from_spec(spec: dict) -> list[tuple[str, str, list[Any]]]:
    """将 design_spec 展开为 full-combo 轴定义。

    返回的每个元素表示一个可枚举维度：
    `(alternative_name, variable_name, levels)`。
    """
    axes: list[tuple[str, str, list[Any]]] = []
    alts = spec.get("alternatives", []) if isinstance(spec, dict) else []
    for alt in alts:
        alt_name = sanitize_save_name(str((alt or {}).get("name", "")).strip())
        if not alt_name:
            continue
        for v in (alt or {}).get("variables", []) or []:
            var_name = sanitize_save_name(str((v or {}).get("name", "")).strip())
            levels = list((v or {}).get("levels", []) or [])
            if not var_name or not levels:
                continue
            axes.append((alt_name, var_name, levels))
    return axes


def _full_combo_count(spec: dict) -> int:
    """计算 full factorial 组合总数。"""
    total = 1
    axes = _design_axes_from_spec(spec)
    if not axes:
        return 0
    for _alt_name, _var_name, levels in axes:
        total *= max(1, len(levels))
    return int(total)


def _task_from_axis_choice(axes: list[tuple[str, str, list[Any]]], picked_levels: tuple[Any, ...], idx: int) -> dict:
    """把一组 full-combo 轴取值还原成单个 task。"""
    alternatives: dict[str, dict[str, Any]] = {}
    for (alt_name, var_name, _levels), value in zip(axes, picked_levels):
        alternatives.setdefault(alt_name, {})[var_name] = value
    return {"id": f"comb_{idx + 1}", "alternatives": alternatives}


def _infer_dominance_directions(spec: dict, beta_defaults: dict) -> dict[str, int]:
    """根据先验参数符号推断可比较变量的偏好方向。

    返回:
    - `-1`: 变量越小越好（如 time/cost 的负效用）
    - `+1`: 变量越大越好

    只有当某变量在至少两个备选项中出现，且这些备选项对应先验符号一致时，
    才纳入 dominance 检查；否则跳过，避免误判。
    """
    var_occurs: dict[str, set[str]] = {}
    sign_map: dict[str, set[int]] = {}
    alts = spec.get("alternatives", []) if isinstance(spec, dict) else []
    for alt in alts:
        alt_name = sanitize_save_name(str((alt or {}).get("name", "")).strip())
        if not alt_name:
            continue
        for v in (alt or {}).get("variables", []) or []:
            var_name = sanitize_save_name(str((v or {}).get("name", "")).strip())
            levels = list((v or {}).get("levels", []) or [])
            if not var_name or not levels:
                continue
            var_occurs.setdefault(var_name, set()).add(alt_name)
            coef = _to_num((beta_defaults or {}).get(f"{alt_name}.{var_name}"))
            if coef is None or abs(float(coef)) <= 1e-12:
                continue
            sign_map.setdefault(var_name, set()).add(1 if float(coef) > 0 else -1)
    directions: dict[str, int] = {}
    for var_name, alt_names in var_occurs.items():
        signs = sign_map.get(var_name, set())
        if len(alt_names) < 2 or len(signs) != 1:
            continue
        directions[var_name] = list(signs)[0]
    return directions


def _alt_dominates_generic(a: dict, b: dict, directions: dict[str, int]) -> bool:
    """按推断出的变量方向判断一个备选项是否支配另一个备选项。"""
    shared = [k for k in directions.keys() if k in (a or {}) and k in (b or {})]
    if not shared:
        return False
    strictly_better = False
    for key in shared:
        av = _to_num((a or {}).get(key))
        bv = _to_num((b or {}).get(key))
        if av is None or bv is None:
            return False
        if int(directions.get(key, -1)) < 0:
            if float(av) > float(bv):
                return False
            if float(av) < float(bv):
                strictly_better = True
        else:
            if float(av) < float(bv):
                return False
            if float(av) > float(bv):
                strictly_better = True
    return strictly_better


def _task_has_inferable_dominance(task: dict, directions: dict[str, int]) -> bool:
    """检查 task 中是否存在可由先验方向识别的 dominated alternative。"""
    if not directions:
        return False
    alt_items = list(((task or {}).get("alternatives", {}) or {}).values())
    if len(alt_items) < 2:
        return False
    for i in range(len(alt_items)):
        for j in range(len(alt_items)):
            if i == j:
                continue
            if _alt_dominates_generic(alt_items[i], alt_items[j], directions):
                return True
    return False


def _task_signature(task: dict) -> str:
    alts = (task or {}).get("alternatives", {}) if isinstance(task, dict) else {}
    norm = {}
    for alt, attrs in alts.items():
        norm[str(alt)] = {str(k): attrs.get(k) for k in sorted((attrs or {}).keys())}
    return hashlib.sha1(str(sorted(norm.items())).encode("utf-8")).hexdigest()[:16]


def _default_policy_state() -> dict:
    return {
        "pretrained": False,
        "pretrain_steps": 0,
        "online_updates": 0,
        "response_count": 0,
        "current_d_error": 1.0,
        "level_counts": {},
        "used_task_sigs": [],
        "last_imp_d": 0.0,
    }


def _load_design_policy_state(save_name: str | None) -> tuple[dict, dict | None]:
    safe = sanitize_save_name(save_name or "")
    if not safe:
        return _default_policy_state(), None
    f = design_spec_file_path(safe)
    if not f.exists():
        return _default_policy_state(), None
    rec = load_json(f, {})
    runtime = rec.get("runtime", {}) if isinstance(rec, dict) else {}
    mw = runtime.get("model_weights", {}) if isinstance(runtime, dict) else {}
    ps = {}
    if isinstance(mw, dict):
        weight_file = _resolve_weight_file_path(mw.get("weight_file"))
        if (
            weight_file
            and weight_file.exists()
            and bool(getattr(dyppo_engine, "TORCH_AVAILABLE", False))
            and getattr(dyppo_engine, "torch", None) is not None
        ):
            try:
                obj = dyppo_engine.torch.load(str(weight_file), map_location="cpu")
                ps = obj.get("policy_state", {}) if isinstance(obj, dict) else {}
            except Exception:
                ps = {}
        if not isinstance(ps, dict) or not ps:
            legacy_ps = mw.get("policy_state", {})
            if isinstance(legacy_ps, dict):
                ps = legacy_ps
    if isinstance(ps, dict) and ps:
        out = _default_policy_state()
        out.update(ps)
        return out, rec
    return _default_policy_state(), rec


def _save_design_policy_state(save_name: str | None, policy_state: dict, rec: dict | None = None) -> None:
    safe = sanitize_save_name(save_name or "")
    if not safe:
        return
    f = design_spec_file_path(safe)
    if rec is None:
        rec = load_json(f, {}) if f.exists() else {}
    if not isinstance(rec, dict):
        rec = {}
    runtime = rec.get("runtime", {}) if isinstance(rec.get("runtime", {}), dict) else {}
    updated_at = utc_now_iso()
    if bool(getattr(dyppo_engine, "TORCH_AVAILABLE", False)) and getattr(dyppo_engine, "torch", None) is not None:
        wf = design_weight_file_path(safe)
        package = {
            "save_name": safe,
            "updated_at": updated_at,
            "policy_state": policy_state if isinstance(policy_state, dict) else {},
        }
        dyppo_engine.torch.save(package, str(wf))
        runtime["model_weights"] = {
            "updated_at": updated_at,
            "format": "pt",
            "weight_file": wf.name,
        }
    else:
        runtime["model_weights"] = {
            "updated_at": updated_at,
            "format": "json_fallback_no_torch",
            "policy_state": policy_state if isinstance(policy_state, dict) else {},
        }
    rec["runtime"] = runtime
    save_json(f, rec)


def _spec_id_from_payload(payload: dict) -> str:
    spec = payload.get("design_spec", {}) if isinstance(payload, dict) else {}
    base = {
        "design_type": normalize_design_type(payload.get("design_type")),
        "design_spec": spec,
        "n_parameters": int(payload.get("n_parameters", 0) or 0),
    }
    return hashlib.sha1(str(base).encode("utf-8")).hexdigest()[:20]


def _build_candidate_pool_for_ppo(payload: dict, *, pool_size: int) -> list[dict]:
    spec = payload.get("design_spec", {}) if isinstance(payload, dict) else {}
    conditions = spec.get("conditions", []) if isinstance(spec, dict) else []
    beta_defaults = payload.get("beta_defaults", {}) if isinstance(payload, dict) else {}
    dominance_directions = _infer_dominance_directions(spec, beta_defaults if isinstance(beta_defaults, dict) else {})
    axes = _design_axes_from_spec(spec)
    full_combo_count = _full_combo_count(spec)
    exhaustive_cap = max(int(pool_size or 0), 5000)
    rng = np.random.default_rng(991)

    if axes and full_combo_count > 0 and full_combo_count <= exhaustive_cap:
        combos = list(product(*[levels for _alt_name, _var_name, levels in axes]))
        rng.shuffle(combos)
        base = [_task_from_axis_choice(axes, choice, idx) for idx, choice in enumerate(combos)]
    else:
        rows = max(pool_size * 8, 240)
        base = _build_tasks_from_spec(spec, rows, seed=991)

    out = []
    seen = set()
    for t in base:
        if not _task_satisfies_conditions(t, conditions):
            continue
        if _task_has_inferable_dominance(t, dominance_directions):
            continue
        sig = _task_signature(t)
        if sig in seen:
            continue
        seen.add(sig)
        out.append({**t, "sig": sig})
        if full_combo_count > exhaustive_cap and len(out) >= pool_size:
            break
    return out


def _task_feature_vector(task: dict, *, level_counts: dict, used_sigs: set[str], beta_defaults: dict, spec: dict) -> np.ndarray:
    # f1: utility spread proxy based on prior means.
    keys = _build_param_keys(spec)
    spread = 0.0
    if keys:
        x = _design_matrix_generic(task, keys)
        b = np.array([float(beta_defaults.get(k, 0.0)) for k in keys], dtype=float)
        u = x @ b
        spread = float(np.max(u) - np.min(u)) if u.size else 0.0

    # f2: rarity (encourage under-covered levels)
    rarity_scores = []
    for alt, attrs in (task.get("alternatives", {}) or {}).items():
        for k, v in (attrs or {}).items():
            lk = f"{alt}.{k}.{v}"
            rarity_scores.append(1.0 / (1.0 + float(level_counts.get(lk, 0))))
    rarity = float(np.mean(rarity_scores)) if rarity_scores else 0.0

    # f3: novelty
    sig = str(task.get("sig") or _task_signature(task))
    novelty = 0.0 if sig in used_sigs else 1.0

    # f4: normalized attribute count (shape complexity proxy)
    attr_n = 0
    for _alt, attrs in (task.get("alternatives", {}) or {}).items():
        attr_n += len((attrs or {}).keys())
    complexity = min(float(attr_n) / 20.0, 1.0)

    return np.array([spread, rarity, novelty, complexity], dtype=float)


def _ensure_spec_policy(ppo_state: dict, spec_id: str) -> dict:
    _ = spec_id
    if isinstance(ppo_state, dict):
        out = _default_policy_state()
        out.update(ppo_state)
        return out
    return _default_policy_state()


def _state_vector_from_policy(item: dict) -> np.ndarray:
    d = float(item.get("current_d_error", 1.0) or 1.0)
    n = int(item.get("response_count", 0) or 0)
    imp = float(item.get("last_imp_d", 0.0) or 0.0)
    used = len(item.get("used_task_sigs", []) or [])
    return np.array(
        [
            1.0 / (1.0 + max(d, 1e-6)),
            min(n / 300.0, 1.0),
            max(min((imp + 1.0) / 2.0, 1.0), 0.0),
            min(used / 2000.0, 1.0),
        ],
        dtype=float,
    )


def _pretrain_policy_with_efficient(payload: dict, item: dict, candidates: list[dict]) -> list[dict]:
    mopt = ((payload.get("design_options", {}) or {}).get("dyppo", {}) or {})
    tpr = int(mopt.get("tasks_per_round", 6) or 6)
    payload_eff = dict(payload)
    payload_eff["design_type"] = "efficient"
    payload_eff["design_options"] = {
        "efficient": {
            "tasks_per_person": max(1, tpr),
            "row_exchange_iterations": 30,
        }
    }
    payload_eff["sample_size"] = max(int(payload.get("sample_size", 120) or 120), 80)
    payload_eff["target_block_sample"] = max(int(payload.get("target_block_sample", 80) or 80), 80)
    expert = _compute_efficient(payload_eff)
    expert_tasks = expert.get("comb", []) if isinstance(expert, dict) else []
    expert_sigs = {_task_signature(t) for t in expert_tasks if isinstance(t, dict)}
    if not expert_sigs or not candidates:
        return []

    used_sigs = set(item.get("used_task_sigs", []) or [])
    level_counts = item.get("level_counts", {}) if isinstance(item.get("level_counts", {}), dict) else {}
    actor = np.array(item.get("actor_w", [0.0, 0.0, 0.0, 0.0]), dtype=float)
    lr = float(item.get("lr_actor", 0.03))

    for _ in range(30):
        for t in candidates:
            feat = _task_feature_vector(t, level_counts=level_counts, used_sigs=used_sigs, beta_defaults=payload.get("beta_defaults", {}) or {}, spec=payload.get("design_spec", {}) or {})
            y = 1.0 if str(t.get("sig")) in expert_sigs else 0.0
            z = float(actor @ feat)
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
            actor = actor + lr * (y - p) * feat

    item["actor_w"] = actor.tolist()
    item["pretrained"] = True
    item["pretrain_steps"] = int(item.get("pretrain_steps", 0)) + 1
    return expert_tasks


def _ppo_select_tasks(payload: dict, item: dict, candidates: list[dict], *, tpr: int, eps: float) -> tuple[list[dict], list[list[float]], list[float]]:
    if not candidates:
        return [], [], []
    rng = np.random.default_rng(int(item.get("response_count", 0)) + 2026)
    actor = np.array(item.get("actor_w", [0.0, 0.0, 0.0, 0.0]), dtype=float)
    used_sigs = set(item.get("used_task_sigs", []) or [])
    level_counts = item.get("level_counts", {}) if isinstance(item.get("level_counts", {}), dict) else {}
    beta_defaults = payload.get("beta_defaults", {}) or {}
    spec = payload.get("design_spec", {}) if isinstance(payload, dict) else {}

    remain = list(candidates)
    picked = []
    picked_feat = []
    log_probs = []
    n_pick = min(max(1, tpr), len(remain))

    for _ in range(n_pick):
        feat_list = [
            _task_feature_vector(t, level_counts=level_counts, used_sigs=used_sigs, beta_defaults=beta_defaults, spec=spec)
            for t in remain
        ]
        feat_mat = np.vstack(feat_list)
        logits = feat_mat @ actor
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        if rng.random() < max(0.0, min(1.0, eps)):
            idx = int(rng.integers(0, len(remain)))
        else:
            idx = int(rng.choice(len(remain), p=probs))
        chosen = remain.pop(idx)
        picked.append(chosen)
        picked_feat.append(feat_mat[idx].tolist())
        log_probs.append(float(np.log(max(probs[idx], 1e-12))))
        used_sigs.add(str(chosen.get("sig") or _task_signature(chosen)))
        if not remain:
            break
    return picked, picked_feat, log_probs


def _update_level_counts(level_counts: dict, tasks: list[dict]) -> dict:
    out = dict(level_counts) if isinstance(level_counts, dict) else {}
    for t in tasks:
        for alt, attrs in (t.get("alternatives", {}) or {}).items():
            for k, v in (attrs or {}).items():
                lk = f"{alt}.{k}.{v}"
                out[lk] = int(out.get(lk, 0)) + 1
    return out


def _choice_entropy_bonus(choices: dict) -> float:
    if not choices:
        return 0.0
    vals, counts = np.unique(list(choices.values()), return_counts=True)
    p = counts / np.sum(counts)
    e = -np.sum(p * np.log(np.maximum(p, 1e-12)))
    e_max = np.log(max(len(vals), 1))
    if e_max <= 0:
        return 0.0
    return float(e / e_max)


def _ppo_online_update(payload: dict, item: dict, tasks: list[dict], *, choices: dict | None, feat_override: list[list[float]] | None = None) -> dict:
    if not tasks:
        return {"updated": False}
    beta_defaults = payload.get("beta_defaults", {}) or {}
    beta_bounds = payload.get("beta_bounds", {}) or {}
    spec = payload.get("design_spec", {}) if isinstance(payload, dict) else {}
    old_d = float(item.get("current_d_error", 1.0) or 1.0)
    new_d = _bayesian_d_error_generic(tasks, spec, beta_defaults, beta_bounds, beta_draws=16, seed=31)
    imp_d = (old_d - new_d) / max(old_d, 1e-6)
    entropy_bonus = _choice_entropy_bonus(choices or {})
    reward = float(imp_d) + 0.05 * float(entropy_bonus)

    actor = np.array(item.get("actor_w", [0.0, 0.0, 0.0, 0.0]), dtype=float)
    critic = np.array(item.get("critic_w", [0.0, 0.0, 0.0, 0.0]), dtype=float)
    s = _state_vector_from_policy(item)
    value = float(critic @ s)
    adv = float(np.clip(reward - value, -1.0, 1.0))
    lr_a = float(item.get("lr_actor", 0.03))
    lr_c = float(item.get("lr_critic", 0.05))

    critic = critic + lr_c * adv * s
    if feat_override and len(feat_override) == len(tasks):
        feat_list = [np.array(x, dtype=float) for x in feat_override]
    else:
        used_sigs = set(item.get("used_task_sigs", []) or [])
        level_counts = item.get("level_counts", {}) if isinstance(item.get("level_counts", {}), dict) else {}
        feat_list = [
            _task_feature_vector(t, level_counts=level_counts, used_sigs=used_sigs, beta_defaults=beta_defaults, spec=spec)
            for t in tasks
        ]
    for f in feat_list:
        actor = actor + lr_a * adv * f - 0.001 * actor

    item["actor_w"] = actor.tolist()
    item["critic_w"] = critic.tolist()
    item["current_d_error"] = float(new_d)
    item["last_imp_d"] = float(imp_d)
    item["response_count"] = int(item.get("response_count", 0)) + 1
    item["online_updates"] = int(item.get("online_updates", 0)) + 1
    item["level_counts"] = _update_level_counts(item.get("level_counts", {}), tasks)
    used = list(item.get("used_task_sigs", []) or [])
    used.extend([str(t.get("sig") or _task_signature(t)) for t in tasks])
    item["used_task_sigs"] = used[-5000:]
    return {
        "updated": True,
        "reward": round(float(reward), 6),
        "imp_d": round(float(imp_d), 6),
        "old_d": round(float(old_d), 6),
        "new_d": round(float(new_d), 6),
        "entropy_bonus": round(float(entropy_bonus), 6),
    }


def _compute_efficient(payload: dict) -> dict:
    """按 Efficient Design 流程生成题组。

    当前实现采用两阶段过程：
    1. 先依据 design_spec 构造初始题组；
    2. 再使用单行替换式 row-exchange 局部搜索逐步降低 Bayesian D-error。
    """
    spec = payload.get("design_spec", {}) if isinstance(payload, dict) else {}
    conditions = spec.get("conditions", []) if isinstance(spec, dict) else []
    sample_size = int(payload.get("sample_size", 600))
    target_block_sample = int(payload.get("target_block_sample", 100) or 100)
    tpp = int(((payload.get("design_options", {}) or {}).get("efficient", {}) or {}).get("tasks_per_person", 8) or 8)
    iters = int(((payload.get("design_options", {}) or {}).get("efficient", {}) or {}).get("row_exchange_iterations", 200) or 200)
    blocks = max(1, int(round(sample_size / max(1, target_block_sample))))
    rows = max(1, tpp * blocks)

    tasks = _build_tasks_from_spec(spec, rows, seed=42)
    tasks = [t for t in tasks if _task_satisfies_conditions(t, conditions)]
    if not tasks:
        tasks = _build_tasks_from_spec(spec, max(1, tpp), seed=43)

    beta_defaults = payload.get("beta_defaults", {}) or {}
    beta_bounds = payload.get("beta_bounds", {}) or {}
    best_d = _bayesian_d_error_generic(tasks, spec, beta_defaults, beta_bounds, beta_draws=24, seed=13)
    iter_log = [{"iter": 0, "d_error": round(best_d, 6), "rows": len(tasks)}]

    # 单行替换式 row-exchange 局部搜索：每轮只替换一个 row，且仅在 D-error 严格改善时接受。
    base_tasks = list(tasks)
    for i in range(min(2000, max(1, iters))):
        cand = [dict(x) for x in base_tasks]
        if not cand:
            break
        pos = int(i % len(cand))
        new_task = _build_tasks_from_spec(spec, 1, seed=1000 + i)[0]
        if _task_satisfies_conditions(new_task, conditions):
            cand[pos] = new_task
            d_new = _bayesian_d_error_generic(cand, spec, beta_defaults, beta_bounds, beta_draws=16, seed=100 + i)
            if d_new < best_d:
                base_tasks = cand
                best_d = d_new
        if i % max(1, iters // 10) == 0 or i == iters - 1:
            iter_log.append({"iter": i + 1, "d_error": round(best_d, 6), "rows": len(base_tasks)})

    final_tasks = []
    for idx, t in enumerate(base_tasks):
        final_tasks.append(
            {
                **t,
                "block": (idx // max(1, tpp)) + 1,
                "row_in_block": (idx % max(1, tpp)) + 1,
                "id": f"preview_b{(idx // max(1, tpp)) + 1}_r{(idx % max(1, tpp)) + 1}",
            }
        )

    return {
        "mode": "efficient_compute",
        "recommendation": {
            "tasks_per_person": tpp,
            "blocks": max(1, (len(final_tasks) + tpp - 1) // max(1, tpp)),
            "rows": len(final_tasks),
            "row_exchange_iterations": iters,
        },
        "comb": final_tasks,
        "d_error": {"value": round(best_d, 6), "beta_draws": 24},
        "iteration_log": iter_log,
        "model_state": None,
    }


def _compute_dyppo(payload: dict, current_respondent: dict | None = None) -> dict:
    # dynamicPPO 的动作空间来自 feasible combo pool；
    # efficient design 仅作为专家经验 / 先验参考。
    mopt = ((payload.get("design_options", {}) or {}).get("dyppo", {}) or {})
    tpr = int(mopt.get("tasks_per_round", 6) or 6)
    candidate_pool_target = max(120, tpr * 20)
    candidate_pool = _build_candidate_pool_for_ppo(payload, pool_size=candidate_pool_target)

    payload_eff = dict(payload)
    payload_eff["design_type"] = "efficient"
    payload_eff["design_options"] = {
        "efficient": {
            "tasks_per_person": max(1, tpr),
            "row_exchange_iterations": 30,
        }
    }
    efficient_result = _compute_efficient(payload_eff)

    save_name = _payload_save_name(payload)
    item, rec_for_save = _load_design_policy_state(save_name)
    spec_id = _spec_id_from_payload(payload)
    item = _ensure_spec_policy(item, spec_id)
    cfg = _load_runtime_config()

    trained = train_dynamic_ppo(
        payload=payload,
        policy_state=item,
        candidate_pool=candidate_pool,
        expert_result=efficient_result,
        current_respondent=current_respondent,
        data_dir=DATA_DIR,
        config=cfg if isinstance(cfg, dict) else {},
    )
    trained_policy_state = trained.get("policy_state", item) if isinstance(trained, dict) else item
    _save_design_policy_state(save_name, trained_policy_state, rec_for_save)

    tasks = trained.get("comb", []) if isinstance(trained, dict) else []
    spec = payload.get("design_spec", {}) if isinstance(payload, dict) else {}
    beta_defaults = payload.get("beta_defaults", {}) if isinstance(payload, dict) else {}
    beta_bounds = payload.get("beta_bounds", {}) if isinstance(payload, dict) else {}
    final_d = None
    if tasks:
        final_d = _bayesian_d_error_generic(
            tasks,
            spec,
            beta_defaults if isinstance(beta_defaults, dict) else {},
            beta_bounds if isinstance(beta_bounds, dict) else {},
            beta_draws=24,
            seed=57,
        )
    expert_d = ((efficient_result or {}).get("d_error", {}) or {}).get("value")
    d_err = {
        "value": round(float(final_d), 6) if final_d is not None else None,
        "expert_reference_value": _to_num(expert_d),
    }
    iter_log = trained.get("iteration_log", []) if isinstance(trained, dict) else []
    model_state = trained.get("model_state", {}) if isinstance(trained, dict) else {}
    if isinstance(model_state, dict):
        model_state["action_space_type"] = "feasible_combo_pool"
        model_state["candidate_pool_size"] = len(candidate_pool)
        model_state["candidate_pool_target"] = int(candidate_pool_target)
        model_state["full_combo_count"] = int(_full_combo_count(spec))
        model_state["expert_demo_size"] = len((efficient_result or {}).get("comb", []) or [])
        model_state["expert_role"] = "prior_only"
        model_state["final_d_error"] = d_err.get("value")
        model_state["expert_reference_d_error"] = d_err.get("expert_reference_value")

    return {
        "mode": "dyppo_compute",
        "recommendation": {"tasks_per_person": len(tasks), "blocks": 1, "rows": len(tasks)},
        "comb": tasks,
        "d_error": d_err,
        "iteration_log": iter_log,
        "model_state": model_state,
    }


def _compute_selfattention(payload: dict, current_respondent: dict | None = None) -> dict:
    # SelfAttention 仍可复用 feasible combo pool 作为 warmup / prototype prior，
    # 但最终生成逻辑已经切到并行 question-block generator，而不是直接把 combo pool 当动作空间。
    sopt = ((payload.get("design_options", {}) or {}).get("selfattention", {}) or {})
    tpr = int(sopt.get("tasks_per_round", 6) or 6)
    candidate_pool_target = max(120, tpr * 20)
    candidate_pool = _build_candidate_pool_for_ppo(payload, pool_size=candidate_pool_target)
    payload_eff = dict(payload)
    payload_eff["design_type"] = "efficient"
    payload_eff["design_options"] = {
        "efficient": {
            "tasks_per_person": max(1, tpr),
            "row_exchange_iterations": 30,
        }
    }
    efficient_result = _compute_efficient(payload_eff)

    save_name = _payload_save_name(payload)
    item, rec_for_save = _load_design_policy_state(save_name)
    spec_id = _spec_id_from_payload(payload)
    item = _ensure_spec_policy(item, spec_id)
    cfg = _load_runtime_config()

    history_rows = _load_design_history_rows(save_name)
    trained = train_self_attention_ppo(
        payload=payload,
        policy_state=item,
        candidate_pool=candidate_pool,
        expert_result=efficient_result,
        rows=history_rows,
        current_respondent=current_respondent,
        data_dir=DATA_DIR,
        config=cfg if isinstance(cfg, dict) else {},
    )
    trained_policy_state = trained.get("policy_state", item) if isinstance(trained, dict) else item
    _save_design_policy_state(save_name, trained_policy_state, rec_for_save)

    tasks = trained.get("comb", []) if isinstance(trained, dict) else []
    spec = payload.get("design_spec", {}) if isinstance(payload, dict) else {}
    beta_defaults = payload.get("beta_defaults", {}) if isinstance(payload, dict) else {}
    beta_bounds = payload.get("beta_bounds", {}) if isinstance(payload, dict) else {}
    final_d = None
    if tasks:
        final_d = _bayesian_d_error_generic(
            tasks,
            spec,
            beta_defaults if isinstance(beta_defaults, dict) else {},
            beta_bounds if isinstance(beta_bounds, dict) else {},
            beta_draws=24,
            seed=61,
        )
    expert_d = ((efficient_result or {}).get("d_error", {}) or {}).get("value")
    d_err = {
        "value": round(float(final_d), 6) if final_d is not None else None,
        "expert_reference_value": _to_num(expert_d),
    }
    iter_log = trained.get("iteration_log", []) if isinstance(trained, dict) else []
    model_state = trained.get("model_state", {}) if isinstance(trained, dict) else {}
    sampling_recommendation = trained.get("sampling_recommendation", []) if isinstance(trained, dict) else []
    if isinstance(model_state, dict):
        model_state["action_space_type"] = "parallel_question_block"
        model_state["prototype_pool_size"] = len(candidate_pool)
        model_state["prototype_pool_target"] = int(candidate_pool_target)
        model_state["full_combo_count"] = int(_full_combo_count(spec))
        model_state["expert_demo_size"] = len((efficient_result or {}).get("comb", []) or [])
        model_state["expert_role"] = "prior_only"
        model_state["prototype_role"] = "warmup_reference_only"
        model_state["final_d_error"] = d_err.get("value")
        model_state["expert_reference_d_error"] = d_err.get("expert_reference_value")
        model_state["sampling_recommendation"] = sampling_recommendation

    return {
        "mode": "selfattention_compute",
        "recommendation": {
            "tasks_per_person": len(tasks),
            "blocks": 1,
            "rows": len(tasks),
            "sampling_recommendation": sampling_recommendation,
        },
        "comb": tasks,
        "d_error": d_err,
        "iteration_log": iter_log,
        "model_state": model_state,
        "sampling_recommendation": sampling_recommendation,
    }


def _compute_worker(job_id: str, payload: dict) -> None:
    with COMPUTE_LOCK:
        COMPUTE_JOBS[job_id] = {"status": "running", "progress": 5, "created_at": utc_now_iso()}
    try:
        design_type = normalize_design_type(payload.get("design_type"))
        with COMPUTE_LOCK:
            COMPUTE_JOBS[job_id]["progress"] = 30
        if design_type == "dyppo":
            result = _compute_dyppo(payload)
        elif design_type == "selfattention":
            result = _compute_selfattention(payload)
        else:
            result = _compute_efficient(payload)
        with COMPUTE_LOCK:
            COMPUTE_JOBS[job_id].update({"status": "done", "progress": 100, "result": result, "finished_at": utc_now_iso()})
    except Exception as e:
        with COMPUTE_LOCK:
            COMPUTE_JOBS[job_id].update({"status": "failed", "progress": 100, "error": str(e), "finished_at": utc_now_iso()})


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _parse_point_text(raw: str | None) -> tuple[float, float] | None:
    s = str(raw or "").strip()
    if not s:
        return None
    if s.upper().startswith("POINT(") and s.endswith(")"):
        body = s[s.find("(") + 1 : -1]
        parts = [p.strip() for p in body.split(",")]
        if len(parts) != 2:
            return None
        a = _safe_float(parts[0])
        b = _safe_float(parts[1])
        if a is None or b is None:
            return None
        # The project stores POINT(lat,lng).
        return a, b
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 2:
        a = _safe_float(parts[0])
        b = _safe_float(parts[1])
        if a is not None and b is not None:
            return a, b
    return None


def _normalize_gender(raw: str | None) -> str:
    s = str(raw or "").strip().lower()
    if s in ("男", "male", "m"):
        return "male"
    if s in ("女", "female", "f"):
        return "female"
    return "male"


def _normalize_age_group(raw: str | None) -> str:
    s = str(raw or "").strip()
    if s in ("18-30", "18~30岁", "18~30", "18-30岁"):
        return "18-30"
    if s in ("31-45", "30~45岁", "31~45岁", "30-45岁", "31-45岁"):
        return "31-45"
    if s in ("46-60", "46~60岁", "46-60岁"):
        return "46-60"
    if s in ("60+", "60岁以上", "60以上", "60~", "60+岁"):
        return "60+"
    if s in ("0~6岁", "6~18岁", "6-18岁", "18岁以下"):
        return "18-30"
    return "31-45"


def _normalize_edu(raw: str | None) -> str:
    s = str(raw or "").strip()
    if s in ("博士研究生", "硕士研究生", "本科", "大专", "college_plus"):
        return "college_plus"
    if s in ("高中", "中专", "技校", "high_school"):
        return "high_school"
    if s in ("初中", "小学及以下", "middle_or_below"):
        return "middle_or_below"
    return "college_plus"


def _respondent_context_for_policy(record: dict | None) -> dict:
    """把 respondent 主记录压缩成 dyppo/selfattention 可用的 RP 快照。"""
    rec = record if isinstance(record, dict) else {}
    profile = rec.get("profile", {}) if isinstance(rec.get("profile", {}), dict) else {}
    members = profile.get("household_members", []) if isinstance(profile.get("household_members", []), list) else []
    member = members[0] if members and isinstance(members[0], dict) else {}
    pop_obj = _load_popsim_obj()
    key_format = str(pop_obj.get("key_format", "gender|age_group")).strip() or "gender|age_group"
    dims = [_canonical_dim_name(x) for x in key_format.split("|")]

    attr_segments: list[str] = []
    for d in dims:
        raw = member.get(d)
        if raw is None and d == "education":
            raw = member.get("edu")
        if raw is None:
            raw = profile.get(d)
        if d == "gender":
            attr_segments.append(_normalize_gender(raw))
        elif d == "age_group":
            attr_segments.append(_normalize_age_group(raw))
        elif d == "education":
            attr_segments.append(_normalize_edu(raw))
        else:
            s = str(raw or "").strip()
            attr_segments.append(s or "__missing__")

    zone_id = str(profile.get("zone_id", "") or rec.get("zone_id", "") or "__missing__").strip() or "__missing__"
    return {
        "respondent_id": rec.get("respondent_id") or profile.get("respondent_id"),
        "zone_id": zone_id,
        "attr_segments": attr_segments,
    }


def _load_design_history_rows(design_save_name: str | None, *, exclude_submission_id: str | None = None) -> list[dict]:
    """读取指定 design 的历史 SP 提交记录，供在线奖励计算使用。"""
    safe_name = sanitize_save_name(design_save_name or "")
    if not safe_name:
        return []
    rows = load_jsonl(paths()["sp_submissions"])
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if sanitize_save_name(row.get("design_save_name", "")) != safe_name:
            continue
        if exclude_submission_id and str(row.get("submission_id", "")).strip() == str(exclude_submission_id).strip():
            continue
        out.append(
            {
                "submission_id": row.get("submission_id"),
                "design_save_name": row.get("design_save_name"),
                "respondent_id": row.get("respondent_id"),
                "respondent": _respondent_context_for_policy(load_respondent_record(str(row.get("respondent_id", "") or ""))),
                "tasks": row.get("tasks", []),
                "choices": row.get("choices", {}),
            }
        )
    return out


def _canonical_dim_name(dim: str) -> str:
    d = str(dim or "").strip().lower()
    if d in ("sex",):
        return "gender"
    if d in ("age", "ageband", "age_band"):
        return "age_group"
    if d in ("edu", "education_level"):
        return "education"
    return d


def _parse_age_interval(label: str) -> tuple[float, float] | None:
    s = str(label or "").strip().lower()
    if not s:
        return None
    import re

    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return None
    if ("+" in s) or ("以上" in s):
        lo = float(nums[0])
        return lo, 120.0
    if len(nums) >= 2:
        lo, hi = float(nums[0]), float(nums[1])
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            hi = lo + 1.0
        return lo, hi
    v = float(nums[0])
    return v, v + 1.0


def _normalize_weights(dist: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(0.0, float(v)) for v in dist.values()))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in dist.items() if float(v) > 0}


def _value_to_rp_distribution(dim: str, raw_value: str | None, rp_categories: list[str]) -> dict[str, float]:
    if not rp_categories:
        return {}
    d = _canonical_dim_name(dim)
    raw = str(raw_value or "").strip()
    if not raw:
        return {rp_categories[0]: 1.0}

    # Exact match first.
    for c in rp_categories:
        if raw == c:
            return {c: 1.0}

    if d == "gender":
        g = _normalize_gender(raw)
        if g in rp_categories:
            return {g: 1.0}
        # fallback by substring
        for c in rp_categories:
            if g in str(c).lower():
                return {c: 1.0}
        return {rp_categories[0]: 1.0}

    if d == "age_group":
        src_itv = _parse_age_interval(raw)
        rp_itv = {c: _parse_age_interval(c) for c in rp_categories}
        rp_itv = {k: v for k, v in rp_itv.items() if v is not None}
        if src_itv and rp_itv:
            a0, a1 = src_itv
            hit = {}
            for c, (b0, b1) in rp_itv.items():
                inter = max(0.0, min(a1, b1) - max(a0, b0))
                if inter > 0:
                    hit[c] = inter
            hit = _normalize_weights(hit)
            if hit:
                return hit
            # no overlap: nearest midpoint
            src_mid = (a0 + a1) / 2.0
            best_c = None
            best_d = None
            for c, (b0, b1) in rp_itv.items():
                mid = (b0 + b1) / 2.0
                dd = abs(src_mid - mid)
                if best_d is None or dd < best_d:
                    best_c, best_d = c, dd
            if best_c is not None:
                return {best_c: 1.0}
        return {_normalize_age_group(raw): 1.0} if _normalize_age_group(raw) in rp_categories else {rp_categories[0]: 1.0}

    # Generic fallback: case-insensitive exact/contains
    low = raw.lower()
    for c in rp_categories:
        cl = str(c).lower()
        if low == cl or low in cl or cl in low:
            return {c: 1.0}
    return {rp_categories[0]: 1.0}


def _load_popsim_obj() -> dict:
    cfg = _load_runtime_config()
    ds = cfg.get("data_sources", {}) if isinstance(cfg.get("data_sources", {}), dict) else {}
    pop_rel = str(ds.get("popsim_stats_path", "popSimStats_gz_template.json")).strip() or "popSimStats_gz_template.json"
    pop_path = Path(pop_rel) if Path(pop_rel).is_absolute() else (DATA_DIR / pop_rel)
    pop_obj = load_json(pop_path, {}) if pop_path.exists() else {}
    return pop_obj if isinstance(pop_obj, dict) else {}


def _extract_rp_target_from_popsim(pop_obj: dict, rp_schema: dict) -> tuple[dict, dict, list[str]]:
    default_target = pop_obj.get("default_target", {}) if isinstance(pop_obj.get("default_target", {}), dict) else {}
    zone_targets = pop_obj.get("zone_targets", {}) if isinstance(pop_obj.get("zone_targets", {}), dict) else {}
    if not zone_targets and isinstance(pop_obj.get("zone2_targets", {}), dict):
        zone_targets = pop_obj.get("zone2_targets", {})
    dims = [_canonical_dim_name(x) for x in str(pop_obj.get("key_format", "gender|age_group|edu")).split("|")]

    retained_dims = []
    for d in dims:
        if d in rp_schema and isinstance(rp_schema.get(d), list) and rp_schema.get(d):
            retained_dims.append(d)
    if not retained_dims:
        retained_dims = [d for d in ("gender", "age_group") if d in rp_schema]

    rp_cats_by_dim = {d: [str(x) for x in (rp_schema.get(d) or [])] for d in retained_dims}

    def convert_one(dist_in: dict) -> dict:
        out = {}
        for k, w in (dist_in or {}).items():
            ww = _safe_float(w)
            ww = float(ww) if ww is not None else 0.0
            if ww <= 0:
                continue
            parts = str(k or "").split("|")
            if len(parts) < len(dims):
                parts = parts + ([""] * (len(dims) - len(parts)))
            cur = {"": 1.0}
            for idx, d in enumerate(dims):
                if d not in retained_dims:
                    continue
                rp_cats = rp_cats_by_dim.get(d, [])
                mp = _value_to_rp_distribution(d, parts[idx], rp_cats)
                nxt = {}
                for a, wa in cur.items():
                    for b, wb in mp.items():
                        key = b if not a else f"{a}|{b}"
                        nxt[key] = nxt.get(key, 0.0) + float(wa) * float(wb)
                cur = nxt
            for kk, vv in cur.items():
                out[kk] = out.get(kk, 0.0) + float(ww) * float(vv)
        return out

    default_rp = convert_one(default_target)
    zone_rp = {}
    for zid, zv in (zone_targets or {}).items():
        if isinstance(zv, str) and zv == "__copy_from_default_target__":
            zone_rp[str(zid)] = dict(default_rp)
        elif isinstance(zv, dict):
            zone_rp[str(zid)] = convert_one(zv)
    return default_rp, zone_rp, retained_dims


def _member_to_rp_key_weights(member: dict, retained_dims: list[str], rp_schema: dict) -> dict[str, float]:
    cur = {"": 1.0}
    for d in retained_dims:
        rp_cats = [str(x) for x in (rp_schema.get(d) or [])]
        raw = member.get(d)
        if raw is None and d == "education":
            raw = member.get("edu")
        mp = _value_to_rp_distribution(d, raw, rp_cats)
        nxt = {}
        for a, wa in cur.items():
            for b, wb in mp.items():
                key = b if not a else f"{a}|{b}"
                nxt[key] = nxt.get(key, 0.0) + float(wa) * float(wb)
        cur = nxt
    return cur


def _iter_respondent_records() -> list[dict]:
    out = []
    for rec_file in paths()["respondents_dir"].glob("*.json"):
        rec = load_json(rec_file, {})
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _load_zone_meta() -> dict:
    cfg = _load_runtime_config()
    ds = cfg.get("data_sources", {}) if isinstance(cfg.get("data_sources", {}), dict) else {}
    gj_rel = str(ds.get("zone_geojson_path", "gz_districts_template.geojson")).strip() or "gz_districts_template.geojson"
    gj_path = Path(gj_rel) if Path(gj_rel).is_absolute() else (DATA_DIR / gj_rel)
    gj = load_json(gj_path, {}) if gj_path.exists() else {}
    features = gj.get("features", []) if isinstance(gj, dict) else []
    out = {}
    for f in features if isinstance(features, list) else []:
        props = f.get("properties", {}) if isinstance(f, dict) else {}
        zid = str(
            props.get("zone_id")
            or props.get("zone2_id")
            or props.get("zone")
            or ""
        ).strip()
        if not zid:
            continue
        zname = props.get("zone_name") or props.get("zone2_name") or props.get("name") or zid
        zname_cn = (
            props.get("zone_name_cn")
            or props.get("zone2_name_cn")
            or props.get("name_cn")
            or zname
        )
        out[zid] = {
            "zone_id": zid,
            "zone_name": zname,
            "zone_name_cn": zname_cn,
            "geometry": f.get("geometry"),
        }
    return out


def _zone_bias(obs_counts: dict, target_dist: dict) -> dict:
    keys = set(obs_counts.keys()) | set((target_dist or {}).keys())
    n = float(sum(float(v) for v in obs_counts.values())) if isinstance(obs_counts, dict) else 0.0
    if n <= 0:
        n = 1.0
    l1 = 0.0
    max_abs = 0.0
    by_key = {}
    for k in sorted(keys):
        obs = float(obs_counts.get(k, 0)) / n
        tar = float((target_dist or {}).get(k, 0.0) or 0.0)
        d = obs - tar
        ad = abs(d)
        l1 += ad
        max_abs = max(max_abs, ad)
        by_key[k] = {"observed": round(obs, 4), "target": round(tar, 4), "diff": round(d, 4)}
    return {"l1": round(l1, 4), "tv": round(0.5 * l1, 4), "max_abs_diff": round(max_abs, 4), "by_key": by_key}


def _zone_target_shares(pop_obj: dict, zone_ids: list[str]) -> dict[str, float]:
    """读取或推断 zone 目标占比。

    若 ActivitySim/PopulationSim 配置里没有单独给出 zone 权重，则对所有 zone
    做均匀分配。`zone_targets/zone2_targets` 本身通常是区内条件分布，
    不能直接当作区际人口规模使用。
    """
    ids = [str(z) for z in zone_ids if str(z)]
    if not ids:
        return {}
    for key in ("zone_shares", "zone_share", "zone_target_shares", "zone2_shares", "zone_weights"):
        raw = pop_obj.get(key, {}) if isinstance(pop_obj, dict) else {}
        if isinstance(raw, dict) and raw:
            vals = {z: max(0.0, float(_safe_float(raw.get(z)) or 0.0)) for z in ids}
            total = sum(vals.values())
            if total > 0:
                return {z: vals[z] / total for z in ids}
    share = 1.0 / max(len(ids), 1)
    return {z: share for z in ids}


def _primary_profile_member(profile: dict) -> dict:
    members = profile.get("household_members", []) if isinstance(profile.get("household_members", []), list) else []
    for item in members:
        if isinstance(item, dict) and str(item.get("member_id", "")) == "P1":
            return item
    if members and isinstance(members[0], dict):
        return members[0]
    return {}


def _profile_zone_id(profile: dict) -> str:
    return str(
        profile.get("zone_id")
        or profile.get("home_zone_id")
        or profile.get("residence_zone_id")
        or "UNKNOWN"
    ).strip() or "UNKNOWN"


def _iter_sp_submission_records(recs: list[dict], save_name: str | None) -> list[dict]:
    """列出某个 design 下已经完成 SP 的 respondent 记录。"""
    safe = sanitize_save_name(save_name or "")
    out = []
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        sp = rec.get("sp", {}) if isinstance(rec.get("sp", {}), dict) else {}
        submissions = sp.get("submissions", []) if isinstance(sp.get("submissions", []), list) else []
        for sub in submissions:
            if not isinstance(sub, dict):
                continue
            if safe and sanitize_save_name(str(sub.get("design_save_name", ""))) != safe:
                continue
            out.append({"record": rec, "submission": sub})
    return out


def _build_statistical_sample_plan(
    *,
    recs: list[dict],
    selected: dict | None,
    pop_obj: dict,
    default_target: dict,
    zone_targets: dict,
    retained_dims: list[str],
    rp_schema: dict,
    zone_meta: dict,
) -> dict:
    """基于总体目标分布和当前已完成 SP 样本计算剩余样本建议。

    efficient/dyppo 没有专门的采样 head，所以使用此统计缺口法；
    selfattention 也会保留该统计表，并额外叠加模型输出。
    """
    payload = selected.get("payload", {}) if isinstance(selected, dict) and isinstance(selected.get("payload", {}), dict) else {}
    save_name = str(selected.get("save_name", "")) if isinstance(selected, dict) else ""
    design_type = normalize_design_type(selected.get("type")) if isinstance(selected, dict) else None
    target_sample_size = int(payload.get("sample_size", 0) or 0) if isinstance(payload, dict) else 0
    if target_sample_size <= 0:
        target_sample_size = int(((payload.get("design_options", {}) or {}).get("selfattention", {}) or {}).get("target_sample_size", 200) or 200) if isinstance(payload, dict) else 200

    all_zone_ids = sorted(set(zone_meta.keys()) | set(zone_targets.keys()))
    if not all_zone_ids:
        all_zone_ids = ["UNKNOWN"]
    zone_shares = _zone_target_shares(pop_obj, all_zone_ids)
    submissions = _iter_sp_submission_records(recs, save_name)

    joint_obs: dict[str, float] = {}
    zone_obs: dict[str, float] = {}
    attr_obs: dict[str, float] = {}
    completed = 0
    seen_submission_ids: set[str] = set()
    for item in submissions:
        sub = item.get("submission", {})
        sid = str(sub.get("submission_id") or id(sub))
        # 避免同一个 submission 在 respondent 主文件里被重复写入时重复计数。
        if sid in seen_submission_ids:
            continue
        seen_submission_ids.add(sid)
        rec = item.get("record", {})
        profile = rec.get("profile", {}) if isinstance(rec.get("profile", {}), dict) else {}
        zone_id = _profile_zone_id(profile)
        member = _primary_profile_member(profile)
        key_weights = _member_to_rp_key_weights(member, retained_dims, rp_schema)
        completed += 1
        zone_obs[zone_id] = zone_obs.get(zone_id, 0.0) + 1.0
        for key, wt in key_weights.items():
            attr_obs[key] = attr_obs.get(key, 0.0) + float(wt)
            joint_key = f"{zone_id}|{key}"
            joint_obs[joint_key] = joint_obs.get(joint_key, 0.0) + float(wt)

    # 目标 joint cell：zone_share * 区内 RP 条件分布。
    joint_target: dict[str, float] = {}
    zone_target_n: dict[str, float] = {}
    attr_target: dict[str, float] = {}
    for zid in all_zone_ids:
        zshare = float(zone_shares.get(zid, 0.0))
        zone_target_n[zid] = zshare * target_sample_size
        dist = zone_targets.get(zid, default_target)
        if not isinstance(dist, dict) or not dist:
            dist = default_target
        total = sum(max(0.0, float(v)) for v in dist.values())
        if total <= 0:
            continue
        for key, val in dist.items():
            share = zshare * max(0.0, float(val)) / total
            joint_target[f"{zid}|{key}"] = share
            attr_target[key] = attr_target.get(key, 0.0) + share

    def make_remaining_rows(target_map: dict[str, float], obs_map: dict[str, float], *, level: str, limit: int | None = None) -> list[dict]:
        rows = []
        total_obs = max(1.0, float(completed))
        for cell, target_share in target_map.items():
            target_n = float(target_share) * target_sample_size
            observed_n = float(obs_map.get(cell, 0.0))
            remaining_n = max(0.0, target_n - observed_n)
            gap_share = max(0.0, float(target_share) - observed_n / total_obs)
            zone_id = ""
            rp_cell = str(cell)
            zone_name = ""
            if level == "joint" and "|" in str(cell):
                zone_id, rp_cell = str(cell).split("|", 1)
                zone_name = (zone_meta.get(zone_id, {}) or {}).get("zone_name_cn", zone_id)
            elif level == "zone":
                zone_id = str(cell)
                zone_name = (zone_meta.get(zone_id, {}) or {}).get("zone_name_cn", zone_id)
            rows.append(
                {
                    "level": level,
                    "cell": str(cell),
                    "zone_id": zone_id,
                    "zone_name_cn": zone_name,
                    "rp_cell": rp_cell,
                    "target_share": round(float(target_share), 6),
                    "target_n": round(target_n, 2),
                    "observed_n": round(observed_n, 2),
                    "observed_share": round(observed_n / total_obs, 6),
                    "remaining_n": int(np.ceil(remaining_n)) if remaining_n > 0 else 0,
                    "gap_share": round(gap_share, 6),
                    "priority": round(remaining_n / max(target_sample_size, 1), 6),
                    "source": "statistical_gap",
                }
            )
        rows.sort(key=lambda r: (r["remaining_n"], r["gap_share"]), reverse=True)
        return rows[:limit] if limit else rows

    zone_target_share = {z: float(zone_shares.get(z, 0.0)) for z in all_zone_ids}
    zone_rows = make_remaining_rows(zone_target_share, zone_obs, level="zone")
    attr_rows = make_remaining_rows(attr_target, attr_obs, level="attr")
    joint_rows = make_remaining_rows(joint_target, joint_obs, level="joint", limit=50)

    remaining_total = max(0, int(target_sample_size) - int(completed))
    return {
        "strategy": design_type,
        "design_save_name": save_name,
        "source": "statistical_gap",
        "target_sample_size": int(target_sample_size),
        "completed_sp_submissions": int(completed),
        "remaining_total": int(remaining_total),
        "retained_dims": retained_dims,
        "zone_rows": zone_rows,
        "attr_rows": attr_rows,
        "joint_rows": joint_rows,
        "top_recommendations": joint_rows[:12] if joint_rows else (zone_rows[:12] + attr_rows[:12])[:12],
    }


def _merge_selfattention_sampling_output(plan: dict, selected: dict | None) -> dict:
    """把 selfattention policy_state 中的采样 head 输出叠加到统计计划上。"""
    if not isinstance(selected, dict) or normalize_design_type(selected.get("type")) != "selfattention":
        return plan
    save_name = str(selected.get("save_name", ""))
    policy_state, _rec = _load_design_policy_state(save_name)
    recs = []
    if isinstance(policy_state, dict):
        recs = policy_state.get("last_sampling_recommendation", []) or []
    if not isinstance(recs, list):
        recs = []

    stat_lookup = {}
    for row in (plan.get("zone_rows", []) or []) + (plan.get("attr_rows", []) or []) + (plan.get("joint_rows", []) or []):
        stat_lookup[str(row.get("cell", ""))] = row
        if row.get("zone_id"):
            stat_lookup[f"zone={row.get('zone_id')}"] = row
        if row.get("rp_cell"):
            # respondent_target_head 使用 `gender=female` 这类边际 cell。
            parts = str(row.get("rp_cell", "")).split("|")
            dims = plan.get("retained_dims", []) or []
            for idx, val in enumerate(parts):
                dim = str(dims[idx]) if idx < len(dims) else f"attr_{idx}"
                stat_lookup[f"{dim}={val}"] = row

    model_rows = []
    for item in recs:
        if not isinstance(item, dict):
            continue
        cell = str(item.get("cell", ""))
        base = dict(stat_lookup.get(cell, {}))
        model_rows.append(
            {
                **base,
                "level": base.get("level", "model_cell"),
                "cell": cell,
                "target_share": base.get("target_share"),
                "target_n": base.get("target_n"),
                "observed_n": base.get("observed_n"),
                "remaining_n": int(item.get("needed_n", base.get("remaining_n", 0)) or 0),
                "priority": float(item.get("priority", base.get("priority", 0.0)) or 0.0),
                "source": "selfattention_head",
                "reason": item.get("reason", "respondent_target_head_priority"),
            }
        )
    model_rows.sort(key=lambda r: (float(r.get("priority", 0.0)), int(r.get("remaining_n", 0))), reverse=True)
    plan["selfattention_recommendations"] = model_rows
    if model_rows:
        plan["source"] = "selfattention_head_plus_statistical_gap"
        plan["top_recommendations"] = model_rows[:12]
    return plan


def _choose_design_file(design_type: str | None, design_save_name: str | None) -> dict:
    specs_dir = paths()["sp_design_dir"]
    available = []
    selected = None
    for f in sorted(specs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
        rec = ensure_design_type_field(load_json(f, {}))
        save_name = str(rec.get("save_name", f.stem))
        dtype = normalize_design_type(rec.get("type"))
        available.append({"save_name": save_name, "type": dtype})
        if design_save_name and sanitize_save_name(design_save_name) == sanitize_save_name(save_name):
            selected = rec
    if selected is None:
        picks = [x for x in available if not design_type or x["type"] == design_type]
        if picks:
            target_name = picks[-1]["save_name"]
            f = design_spec_file_path(target_name)
            selected = ensure_design_type_field(load_json(f, {}))
    return {"available": available, "selected": selected}


def _param_keys_for_spec(spec: dict) -> tuple[list[str], list[str], str]:
    alts = spec.get("alternatives", []) if isinstance(spec, dict) else []
    alt_names = [sanitize_save_name(str((a or {}).get("name", "")).strip()) for a in alts]
    alt_names = [x for x in alt_names if x]
    base = sanitize_save_name(str(spec.get("asc_base_alternative", "")).strip()) or (alt_names[0] if alt_names else "")
    keys = []
    for a in alt_names:
        if a != base:
            keys.append(f"{a}.asc")
    for alt in alts:
        a = sanitize_save_name(str((alt or {}).get("name", "")).strip())
        if not a:
            continue
        for v in (alt or {}).get("variables", []) or []:
            n = sanitize_save_name(str((v or {}).get("name", "")).strip())
            if n:
                keys.append(f"{a}.{n}")
    return keys, alt_names, base


def _obs_to_design_rows(task: dict, alt_names: list[str], keys: list[str]) -> np.ndarray:
    alts = (task or {}).get("alternatives", {}) if isinstance(task, dict) else {}
    rows = []
    for a in alt_names:
        attrs = alts.get(a, {}) if isinstance(alts.get(a, {}), dict) else {}
        row = []
        for k in keys:
            if k.endswith(".asc"):
                row.append(1.0 if k == f"{a}.asc" else 0.0)
                continue
            aa, vv = k.split(".", 1)
            row.append(float(attrs.get(vv, 0.0)) if aa == a else 0.0)
        rows.append(row)
    return np.array(rows, dtype=float)


def _estimate_mnl(obs_rows: list[dict], keys: list[str], alt_names: list[str], beta0: np.ndarray, beta_bounds: dict | None = None, epochs: int = 200, lr: float = 0.08, reg: float = 1e-4) -> np.ndarray:
    if not obs_rows or len(keys) == 0:
        return beta0.copy()
    beta = beta0.astype(float).copy()
    n = len(obs_rows)
    for _ in range(max(1, epochs)):
        grad = np.zeros_like(beta)
        for obs in obs_rows:
            x = obs["x"]
            y = int(obs["y"])
            u = x @ beta
            u = u - np.max(u)
            p = np.exp(u)
            p = p / np.sum(p)
            grad += x[y] - (p.reshape(-1, 1) * x).sum(axis=0)
        step = lr * (grad / n - reg * beta)
        beta = beta + step
        if beta_bounds:
            for i, k in enumerate(keys):
                b = beta_bounds.get(k, {}) if isinstance(beta_bounds, dict) else {}
                lo = _safe_float((b or {}).get("min"))
                hi = _safe_float((b or {}).get("max"))
                if lo is not None:
                    beta[i] = max(beta[i], lo)
                if hi is not None:
                    beta[i] = min(beta[i], hi)
        if float(np.linalg.norm(step)) < 1e-6:
            break
    return beta


def _collect_dashboard_summary(design_type: str | None, design_save_name: str | None, batch_size: int) -> dict:
    recs = _iter_respondent_records()
    rp_schema = DEFAULT_RP_SCHEMA
    pop_obj = _load_popsim_obj()
    default_target, zone_targets, retained_dims = _extract_rp_target_from_popsim(pop_obj, rp_schema)
    zone_meta = _load_zone_meta()

    # RP by-zone stats.
    zone_obs_counts = {}
    zone_sample_n = {}
    map_points = []
    rp_total = 0
    trip_total = 0
    sp_total = 0
    for rec in recs:
        rid = str(rec.get("respondent_id", ""))
        profile = rec.get("profile", {}) if isinstance(rec.get("profile", {}), dict) else {}
        trip = rec.get("trip_diary", {}) if isinstance(rec.get("trip_diary", {}), dict) else {}
        sp = rec.get("sp", {}) if isinstance(rec.get("sp", {}), dict) else {}
        if profile:
            rp_total += 1
            members = profile.get("household_members", []) if isinstance(profile.get("household_members", []), list) else []
            m = None
            for item in members:
                if isinstance(item, dict) and str(item.get("member_id", "")) == "P1":
                    m = item
                    break
            if m is None and members and isinstance(members[0], dict):
                m = members[0]
            m = m if isinstance(m, dict) else {}
            key_weights = _member_to_rp_key_weights(m, retained_dims, rp_schema)

            zone_id = str(profile.get("zone_id") or profile.get("home_zone_id") or profile.get("residence_zone_id") or "").strip()
            if not zone_id:
                zone_id = "UNKNOWN"
            zone_obs_counts.setdefault(zone_id, {})
            for key, wt in key_weights.items():
                zone_obs_counts[zone_id][key] = float(zone_obs_counts[zone_id].get(key, 0.0)) + float(wt)
            zone_sample_n[zone_id] = int(zone_sample_n.get(zone_id, 0)) + 1

            home_point = _parse_point_text(profile.get("home_location"))
            if home_point:
                map_points.append(
                    {
                        "type": "home",
                        "respondent_id": rid,
                        "zone_id": zone_id,
                        "purpose": "home",
                        "lat": round(home_point[0], 6),
                        "lng": round(home_point[1], 6),
                        "label": f"{rid} home",
                    }
                )

        if trip:
            trip_total += 1
            for mem in (trip.get("members", []) or []):
                if not isinstance(mem, dict):
                    continue
                for t in (mem.get("trips", []) or []):
                    if not isinstance(t, dict):
                        continue
                    p = _parse_point_text(t.get("destination"))
                    if not p:
                        continue
                    purpose = str(t.get("purpose", "其他")).strip() or "其他"
                    map_points.append(
                        {
                            "type": "destination",
                            "respondent_id": rid,
                            "purpose": purpose,
                            "lat": round(p[0], 6),
                            "lng": round(p[1], 6),
                            "label": f"{rid} {purpose}",
                        }
                    )

        submissions = sp.get("submissions", []) if isinstance(sp.get("submissions", []), list) else []
        if submissions:
            sp_total += 1

    zone_rows = []
    all_zone_ids = set(zone_meta.keys()) | set(zone_obs_counts.keys()) | set(zone_targets.keys())
    for zid in sorted(all_zone_ids):
        target = zone_targets.get(zid, "__copy_from_default_target__")
        if target == "__copy_from_default_target__" or not isinstance(target, dict):
            target = default_target
        obs = zone_obs_counts.get(zid, {})
        bias = _zone_bias(obs, target if isinstance(target, dict) else {})
        meta = zone_meta.get(zid, {"zone_name_cn": zid, "zone_name": zid})
        zone_rows.append(
            {
                "zone_id": zid,
                "zone_name": meta.get("zone_name"),
                "zone_name_cn": meta.get("zone_name_cn", zid),
                "sample_count": int(zone_sample_n.get(zid, 0)),
                "bias_tv": bias["tv"],
                "bias_l1": bias["l1"],
                "bias_max_abs": bias["max_abs_diff"],
                "bias_detail": bias["by_key"],
            }
        )

    # SP design and re-estimation.
    design_select = _choose_design_file(design_type, design_save_name)
    selected = design_select.get("selected") if isinstance(design_select, dict) else None
    sample_collection_plan = _build_statistical_sample_plan(
        recs=recs,
        selected=selected if isinstance(selected, dict) else None,
        pop_obj=pop_obj,
        default_target=default_target if isinstance(default_target, dict) else {},
        zone_targets=zone_targets if isinstance(zone_targets, dict) else {},
        retained_dims=retained_dims,
        rp_schema=rp_schema,
        zone_meta=zone_meta,
    )
    sample_collection_plan = _merge_selfattention_sampling_output(sample_collection_plan, selected if isinstance(selected, dict) else None)
    sp_est = {
        "selected_save_name": None,
        "selected_type": design_type,
        "n_observations": 0,
        "n_submissions": 0,
        "choice_counts": {},
        "choice_share": {},
        "param_keys": [],
        "initial_beta": {},
        "estimated_beta": {},
        "delta_beta": {},
        "batch_history": [],
    }
    if isinstance(selected, dict):
        payload = selected.get("payload", {}) if isinstance(selected.get("payload", {}), dict) else {}
        spec = payload.get("design_spec", {}) if isinstance(payload.get("design_spec", {}), dict) else {}
        beta_defaults = payload.get("beta_defaults", {}) if isinstance(payload.get("beta_defaults", {}), dict) else {}
        beta_bounds = payload.get("beta_bounds", {}) if isinstance(payload.get("beta_bounds", {}), dict) else {}
        save_name = str(selected.get("save_name", ""))
        keys, alt_names, _base = _param_keys_for_spec(spec)
        beta0 = np.array([float(beta_defaults.get(k, 0.0)) for k in keys], dtype=float)

        obs_rows = []
        choice_counts = {}
        n_submissions = 0
        for rec in recs:
            sp = rec.get("sp", {}) if isinstance(rec.get("sp", {}), dict) else {}
            submissions = sp.get("submissions", []) if isinstance(sp.get("submissions", []), list) else []
            for sub in submissions:
                if not isinstance(sub, dict):
                    continue
                if str(sub.get("design_save_name", "")).strip() != save_name:
                    continue
                n_submissions += 1
                tasks = {str((t or {}).get("id")): t for t in (sub.get("tasks", []) or []) if isinstance(t, dict)}
                for task_id, chosen in (sub.get("choices", {}) or {}).items():
                    if chosen not in alt_names:
                        continue
                    t = tasks.get(str(task_id))
                    if not isinstance(t, dict):
                        continue
                    y = alt_names.index(chosen)
                    x = _obs_to_design_rows(t, alt_names, keys)
                    obs_rows.append({"x": x, "y": y})
                    choice_counts[chosen] = int(choice_counts.get(chosen, 0)) + 1

        beta_hat = _estimate_mnl(obs_rows, keys, alt_names, beta0, beta_bounds=beta_bounds, epochs=250, lr=0.08)
        total_choice = max(1, sum(choice_counts.values()))
        choice_share = {k: round(v / total_choice, 4) for k, v in sorted(choice_counts.items(), key=lambda kv: kv[0])}
        init_dict = {k: round(float(beta0[i]), 6) for i, k in enumerate(keys)}
        est_dict = {k: round(float(beta_hat[i]), 6) for i, k in enumerate(keys)}
        delta_dict = {k: round(float(beta_hat[i] - beta0[i]), 6) for i, k in enumerate(keys)}

        batch = max(1, int(batch_size or 1))
        history = []
        if obs_rows:
            cuts = list(range(batch, len(obs_rows) + 1, batch))
            if not cuts or cuts[-1] != len(obs_rows):
                cuts.append(len(obs_rows))
            for c in cuts:
                bh = _estimate_mnl(obs_rows[:c], keys, alt_names, beta0, beta_bounds=beta_bounds, epochs=120, lr=0.09)
                history.append(
                    {
                        "n": c,
                        "beta": {k: round(float(bh[i]), 6) for i, k in enumerate(keys)},
                        "delta": {k: round(float(bh[i] - beta0[i]), 6) for i, k in enumerate(keys)},
                    }
                )

        sp_est = {
            "selected_save_name": save_name,
            "selected_type": normalize_design_type(selected.get("type")),
            "n_observations": len(obs_rows),
            "n_submissions": n_submissions,
            "choice_counts": choice_counts,
            "choice_share": choice_share,
            "param_keys": keys,
            "initial_beta": init_dict,
            "estimated_beta": est_dict,
            "delta_beta": delta_dict,
            "batch_history": history,
        }

    return {
        "updated_at": utc_now_iso(),
        "overview": {
            "respondents_total": len(recs),
            "profile_count": rp_total,
            "trip_diary_count": trip_total,
            "sp_count": sp_total,
        },
        "zone_summary": zone_rows,
        "zone_geojson": {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"zone_id": z["zone_id"], "zone_name_cn": z["zone_name_cn"], "zone_name": z["zone_name"]}, "geometry": z.get("geometry")}
                for z in zone_meta.values()
            ],
        },
        "map_points": map_points,
        "sp_designs": design_select.get("available", []),
        "sp_estimation": sp_est,
        "sample_collection_plan": sample_collection_plan,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/survey/docs")
def survey_docs():
    return render_template("docs.html", docs=_load_docs_view_data())


@app.route("/survey/profile")
def survey_profile():
    return render_template("profile.html")


@app.route("/survey/trip-diary")
def survey_trip_diary():
    return render_template("trip_diary.html")


@app.route("/survey/sp")
def survey_sp():
    return render_template("sp_questionnaire.html")


@app.route("/survey/sp-design")
def survey_sp_design():
    return render_template("sp_design.html")


@app.route("/survey/map")
def survey_map():
    return render_template("web_map.html")


@app.route("/survey/collection-dashboard")
def survey_collection_dashboard():
    return render_template("collection_dashboard.html")


@app.route("/api/dashboard/collection-summary", methods=["GET"])
def api_dashboard_collection_summary():
    design_type_raw = str(request.args.get("design_type", "")).strip().lower()
    if design_type_raw:
        dt = normalize_design_type(design_type_raw)
        design_type = dt if dt in ("efficient", "dyppo", "selfattention") else None
    else:
        design_type = None
    design_save_name = str(request.args.get("design_save_name", "")).strip() or None
    batch_size = int(request.args.get("batch_size", 20) or 20)
    out = _collect_dashboard_summary(design_type, design_save_name, max(1, batch_size))
    return jsonify(out)


@app.route("/api/design/compute", methods=["POST"])
def api_design_compute():
    payload = request.get_json(silent=True) or {}
    ok, err = validate_sp_design_payload(payload if isinstance(payload, dict) else {})
    if not ok:
        return jsonify({"error": err}), 400
    job_id = f"compute_{uuid.uuid4().hex[:12]}"
    with COMPUTE_LOCK:
        COMPUTE_JOBS[job_id] = {"status": "queued", "progress": 0, "created_at": utc_now_iso()}
    th = threading.Thread(target=_compute_worker, args=(job_id, payload), daemon=True)
    th.start()
    return jsonify({"ok": True, "job_id": job_id, "status_url": f"/api/design/compute/{job_id}"})


@app.route("/api/design/compute/<job_id>", methods=["GET"])
def api_design_compute_status(job_id: str):
    with COMPUTE_LOCK:
        job = dict(COMPUTE_JOBS.get(job_id, {}))
    if not job:
        return jsonify({"error": "job not found"}), 404
    if job.get("status") == "done":
        job["result_url"] = f"/api/design/compute/{job_id}/result"
    return jsonify(job)


@app.route("/api/design/compute/<job_id>/result", methods=["GET"])
def api_design_compute_result(job_id: str):
    with COMPUTE_LOCK:
        job = dict(COMPUTE_JOBS.get(job_id, {}))
    if not job:
        return jsonify({"error": "job not found"}), 404
    if job.get("status") != "done":
        return jsonify({"error": "result not ready", "status": job.get("status", "unknown")}), 409
    result = job.get("result", {})
    if not isinstance(result, dict):
        result = {}
    return jsonify(result)


@app.route("/api/design/metrics", methods=["GET"])
def api_design_metrics():
    respondents_total = len(list(paths()["respondents_dir"].glob("*.json")))
    profile_count = 0
    trip_count = 0
    sp_count = 0
    for rec_file in paths()["respondents_dir"].glob("*.json"):
        rec = load_json(rec_file, {})
        if not isinstance(rec, dict):
            continue
        if rec.get("profile"):
            profile_count += 1
        if rec.get("trip_diary"):
            trip_count += 1
        sp_obj = rec.get("sp", {})
        if isinstance(sp_obj, dict) and (sp_obj.get("submissions") or sp_obj.get("last_response")):
            sp_count += 1
    return jsonify(
        {
            "policy_version": 1,
            "response_count": sp_count,
            "current_d_error": None,
            "repr_gap": None,
            "recent_metrics": [],
            "current_design_tasks": [],
            "pending_assignments": 0,
            "respondents_total": respondents_total,
            "profile_count": profile_count,
            "trip_diary_count": trip_count,
            "sp_count": sp_count,
        }
    )


@app.route("/api/config", methods=["GET"])
def api_config():
    config = _load_runtime_config()
    return jsonify(
        {
            "rp_schema": DEFAULT_RP_SCHEMA,
            "policy_version": 1,
            "config": config,
        }
    )


@app.route("/api/survey/profile", methods=["POST"])
def api_save_profile():
    payload = request.get_json(silent=True) or {}
    respondent_id = sanitize_respondent_id(payload.get("respondent_id", ""))
    if not respondent_id:
        respondent_id = f"R_{uuid.uuid4().hex[:10]}"
        payload["respondent_id"] = respondent_id
    submission_id = f"profile_{uuid.uuid4().hex[:12]}"
    save_respondent_section(respondent_id, "profile", payload)
    row = {
        "submission_id": submission_id,
        "respondent_id": respondent_id,
        "saved_at": utc_now_iso(),
        "payload": payload,
    }
    append_jsonl(paths()["profile_submissions"], row)
    return jsonify({"ok": True, "submission_id": submission_id, "respondent_id": respondent_id})


@app.route("/api/survey/trip-diary", methods=["POST"])
def api_save_trip_diary():
    payload = request.get_json(silent=True) or {}
    respondent_id = sanitize_respondent_id(payload.get("respondent_id", ""))
    if not respondent_id:
        respondent_id = f"R_{uuid.uuid4().hex[:10]}"
        payload["respondent_id"] = respondent_id
    submission_id = f"trip_{uuid.uuid4().hex[:12]}"
    save_respondent_section(respondent_id, "trip_diary", payload)
    row = {
        "submission_id": submission_id,
        "respondent_id": respondent_id,
        "saved_at": utc_now_iso(),
        "payload": payload,
    }
    append_jsonl(paths()["trip_diary_submissions"], row)
    return jsonify({"ok": True, "submission_id": submission_id, "respondent_id": respondent_id})


@app.route("/api/survey/sp-submit", methods=["POST"])
def api_save_sp_submit():
    payload = request.get_json(silent=True) or {}
    respondent_id = sanitize_respondent_id(payload.get("respondent_id", ""))
    if not respondent_id:
        respondent_id = f"R_{uuid.uuid4().hex[:10]}"

    assignment_id = str(payload.get("assignment_id", "")).strip() or f"spgen_{uuid.uuid4().hex[:12]}"
    if bool(payload.get("preview_only")) or assignment_id.startswith("preview_"):
        return jsonify({"error": "当前是设计预览题组，不能作为真实SP答案保存。"}), 400
    choices = payload.get("choices", {}) or {}
    tasks = payload.get("tasks", []) or []
    design_save_name = str(payload.get("design_save_name", "")).strip() or None

    task_map = {}
    for t in tasks:
        if isinstance(t, dict) and t.get("id"):
            task_map[str(t.get("id"))] = t

    choice_details = []
    for task_id, chosen_alt in choices.items():
        t = task_map.get(str(task_id), {})
        alternatives = t.get("alternatives", {}) if isinstance(t, dict) else {}
        choice_details.append(
            {
                "task_id": str(task_id),
                "chosen_alternative": chosen_alt,
                "chosen_attributes": alternatives.get(chosen_alt, {}),
                "alternatives": alternatives,
            }
        )

    submission_id = f"sp_{uuid.uuid4().hex[:12]}"
    submitted_at = utc_now_iso()
    submission = {
        "submission_id": submission_id,
        "assignment_id": assignment_id,
        "respondent_id": respondent_id,
        "design_save_name": design_save_name,
        "choices": choices,
        "choice_details": choice_details,
        "tasks": tasks,
        "submitted_at": submitted_at,
    }

    old = load_respondent_record(respondent_id)
    sp_obj = old.get("sp", {}) if isinstance(old, dict) else {}
    submissions = sp_obj.get("submissions", [])
    if not isinstance(submissions, list):
        submissions = []
    submissions.append(submission)
    sp_obj["submissions"] = submissions
    sp_obj["last_response"] = submission
    save_respondent_section(respondent_id, "sp", sp_obj)

    append_jsonl(paths()["sp_submissions"], submission)
    record_design_distribution(
        design_save_name,
        respondent_id=respondent_id,
        assignment_id=assignment_id,
        tasks=tasks,
        choices=choices,
        source="api/survey/sp-submit",
        submitted_at=submitted_at,
        submission_id=submission_id,
    )
    ppo_update = None
    safe_name = sanitize_save_name(design_save_name or "")
    if safe_name:
        f = design_spec_file_path(safe_name)
        rec = load_json(f, {}) if f.exists() else {}
        rec = ensure_design_type_field(rec) if isinstance(rec, dict) else {}
        design_type = normalize_design_type(rec.get("type") or (rec.get("payload", {}) if isinstance(rec.get("payload", {}), dict) else {}).get("design_type"))
        if design_type in ("dyppo", "selfattention"):
            payload_inner = rec.get("payload", {}) if isinstance(rec.get("payload", {}), dict) else {}
            if payload_inner:
                item, rec_for_save = _load_design_policy_state(safe_name)
                spec_id = _spec_id_from_payload(payload_inner)
                item = _ensure_spec_policy(item, spec_id)
                runtime_config = _load_runtime_config()
                historical_rows = _load_design_history_rows(safe_name, exclude_submission_id=submission_id)
                if design_type == "selfattention":
                    ppo_update = online_update_self_attention_ppo(
                        payload=payload_inner,
                        policy_state=item,
                        tasks=tasks,
                        choices=choices,
                        respondent=_respondent_context_for_policy(old),
                        config=runtime_config,
                        historical_rows=historical_rows,
                    )
                else:
                    ppo_update = online_update_dynamic_ppo(
                        payload=payload_inner,
                        policy_state=item,
                        tasks=tasks,
                        choices=choices,
                        respondent=_respondent_context_for_policy(old),
                        historical_rows=historical_rows,
                        config=runtime_config,
                    )
                _save_design_policy_state(safe_name, item, rec_for_save)
    return jsonify({"ok": True, "submission_id": submission_id, "respondent_id": respondent_id, "ppo_update": ppo_update})


@app.route("/api/survey/respondent/<respondent_id>", methods=["GET"])
def api_get_respondent(respondent_id: str):
    rid = sanitize_respondent_id(respondent_id)
    if not rid:
        return jsonify({"error": "invalid respondent_id"}), 400
    f = respondent_file_path(rid)
    if not f.exists():
        return jsonify({"error": "respondent not found"}), 404
    return jsonify(load_json(f, {}))


@app.route("/api/design/spec", methods=["GET", "POST"])
def api_design_spec():
    p = paths()["design_spec"]
    specs_dir = paths()["sp_design_dir"]
    specs_dir.mkdir(parents=True, exist_ok=True)

    def list_spec_files() -> list[Path]:
        return sorted(specs_dir.glob("*.json"), key=lambda x: x.stat().st_mtime)

    def ensure_type_field(rec: dict) -> dict:
        return ensure_design_type_field(rec)

    def migrate_old_aggregate_if_needed() -> None:
        # One-time compatibility migration from old aggregated JSON.
        if list_spec_files():
            return
        old = load_json(p, {})
        if not old:
            return
        if isinstance(old, dict) and "records" in old:
            records = old.get("records", []) or []
        elif isinstance(old, dict):
            records = [old]
        else:
            records = []
        for rec in records:
            rec = ensure_type_field(rec)
            raw_name = str(rec.get("save_name", "")).strip()
            safe = sanitize_save_name(raw_name)
            if not safe:
                continue
            rec["save_name"] = safe
            save_json(specs_dir / f"{safe}.json", rec)

    migrate_old_aggregate_if_needed()

    if request.method == "GET":
        files = list_spec_files()
        if not files:
            return jsonify({})
        requested_name = request.args.get("save_name", "").strip()
        requested_safe = sanitize_save_name(requested_name) if requested_name else ""
        selected_path = None
        if requested_safe:
            candidate = specs_dir / f"{requested_safe}.json"
            if candidate.exists():
                selected_path = candidate
        if selected_path is None:
            selected_path = files[-1]

        selected = ensure_type_field(load_json(selected_path, {}))
        out = dict(selected) if isinstance(selected, dict) else {}
        out["available_save_names"] = [f.stem for f in files]
        out["latest_save_name"] = files[-1].stem
        out["record_count"] = len(files)
        return jsonify(out)

    payload = request.get_json(silent=True) or {}
    save_name = str(payload.get("save_name", "")).strip()
    if not save_name:
        return jsonify({"error": "save_name is required"}), 400
    safe_name = sanitize_save_name(save_name)
    if not safe_name:
        return jsonify({"error": "invalid save_name"}), 400
    source_save_name = sanitize_save_name(str(payload.get("source_save_name", "")).strip())
    inherit_model_weights = bool(payload.get("inherit_model_weights", False))
    payload_inner_raw = payload.get("payload", payload)
    payload_inner = normalize_design_spec_variable_types(payload_inner_raw if isinstance(payload_inner_raw, dict) else {})
    ok, err = validate_sp_design_payload(payload_inner if isinstance(payload_inner, dict) else {})
    if not ok:
        return jsonify({"error": err}), 400

    target = specs_dir / f"{safe_name}.json"
    existing = load_json(target, {}) if target.exists() else {}
    existing_runtime = existing.get("runtime", {}) if isinstance(existing, dict) else {}
    design_type = normalize_design_type(
        payload.get("type")
        or (payload.get("payload", {}) if isinstance(payload.get("payload", {}), dict) else {}).get("design_type")
        or payload.get("design_type")
    )
    copied_from_source = bool(source_save_name and source_save_name != safe_name)

    def empty_runtime() -> dict:
        return {
            "distribution_log": [],
            "distribution_count": 0,
            "task_issue_counts": {},
            "choice_counts": {},
            "last_distributed_at": None,
        }

    if isinstance(existing_runtime, dict) and existing_runtime:
        runtime = existing_runtime
    else:
        runtime = empty_runtime()

    # Copy-as-new semantics: reset distribution runtime; optionally keep dyppo model weights.
    if copied_from_source and not target.exists():
        runtime = empty_runtime()
        if design_type in ("dyppo", "selfattention") and inherit_model_weights:
            src_path = specs_dir / f"{source_save_name}.json"
            src = load_json(src_path, {}) if src_path.exists() else {}
            src_runtime = src.get("runtime", {}) if isinstance(src, dict) else {}
            model_weights = src_runtime.get("model_weights") if isinstance(src_runtime, dict) else None
            if model_weights is not None:
                mw = model_weights if isinstance(model_weights, dict) else {}
                src_wf = _resolve_weight_file_path(mw.get("weight_file"))
                if src_wf and src_wf.exists():
                    dst_wf = design_weight_file_path(safe_name)
                    dst_wf.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_wf, dst_wf)
                    runtime["model_weights"] = {
                        "updated_at": utc_now_iso(),
                        "format": "pt",
                        "weight_file": dst_wf.name,
                        "inherited_from": source_save_name,
                    }
                elif isinstance(mw.get("policy_state"), dict):
                    runtime["model_weights"] = {
                        "updated_at": utc_now_iso(),
                        "format": "json_fallback_copy",
                        "policy_state": mw.get("policy_state", {}),
                        "inherited_from": source_save_name,
                    }

    saved = {
        "save_name": safe_name,
        "type": design_type,
        "saved_at": utc_now_iso(),
        "payload": payload_inner,
        "recommendation": payload.get("recommendation", None),
        "preview_tasks": payload.get("preview_tasks", []),
        "runtime": runtime,
    }
    replaced = target.exists()
    save_json(target, saved)
    count = len(list_spec_files())
    return jsonify(
        {
            "ok": True,
            "saved_at": saved["saved_at"],
            "save_name": safe_name,
            "record_count": count,
            "replaced": replaced,
            "copied_from_source": copied_from_source,
            "source_save_name": source_save_name or None,
            "runtime_reset": bool(copied_from_source and not replaced),
            "model_weights_inherited": bool(copied_from_source and not replaced and design_type in ("dyppo", "selfattention") and inherit_model_weights),
        }
    )


if __name__ == "__main__":
    ensure_app_data()
    host = str(os.environ.get("SP_SURVEY_HOST", "0.0.0.0") or "0.0.0.0").strip() or "0.0.0.0"
    port = int(str(os.environ.get("SP_SURVEY_PORT", "5056") or "5056").strip() or "5056")
    debug_raw = str(os.environ.get("SP_SURVEY_DEBUG", "1") or "1").strip().lower()
    debug = debug_raw not in ("0", "false", "no", "off")
    is_pydev_console = any("pydevconsole.py" in str(arg) for arg in sys.argv)
    is_pycharm_hosted = str(os.environ.get("PYCHARM_HOSTED", "") or "").strip() == "1"
    is_pycharm_runtime = bool(is_pydev_console or is_pycharm_hosted)
    if is_pycharm_runtime:
        # PyCharm 自己已经接管了调试能力；再叠加 Flask debug/reloader 更容易卡在 IDE 运行链路里。
        debug = False
    use_reloader = bool(debug and not is_pycharm_runtime)
    threaded = not is_pycharm_runtime
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader, threaded=threaded)
