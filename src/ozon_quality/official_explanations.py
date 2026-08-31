"""Grounded comments for the official output, with an optional small-LLM rewrite."""

from __future__ import annotations

import gc
import re
from typing import Any, Final

import pandas as pd

from ozon_quality.official_multimodal import TASK_CONTEXT

EVIDENCE: Final[dict[str, tuple[tuple[re.Pattern[str], str], ...]]] = {
    "БАД": (
        (re.compile(r"\bбад\b|биологически\s+активн|dietary\s+supplement", re.I), "прямая маркировка товара как БАД"),
        (re.compile(r"спортивн.{0,12}питан|\bbcaa\b|протеин|аминокислот|l[\s-]?карнитин", re.I), "признаки спортивного питания"),
        (re.compile(r"не\s+(?:является|относится).{0,20}\bбад\b", re.I), "явное указание, что товар не является БАД"),
    ),
    "Легковоспламеняющиеся": (
        (re.compile(r"зажигал|\bспич|огнив|фейервер|пиротех|дымов.{0,12}шаш|цветн.{0,10}дым", re.I), "самостоятельный источник воспламенения или пиротехническое изделие"),
        (re.compile(r"газ.{0,15}(?:баллон|зажигал|горюч)|\bбутан\b|\bпропан\b|\bкеросин\b|\bбензин\b|жидкост.{0,15}розжиг", re.I), "горючее вещество или газ"),
        (re.compile(r"без\s+(?:газа|топлива|жидкости|спичек|угля)|не\s+входит\s+в\s+комплект|поставляется\s+без", re.I), "поставка без горючего содержимого"),
        (re.compile(r"\bмангал|\bгрил|газов.{0,10}плит", re.I), "только конструкция для использования с огнём"),
    ),
}


def deterministic_comment(row: pd.Series | Any, label: int) -> str:
    """Return a truthful fallback comment that is safe for the 50–300 char contract."""
    getter = row.get if hasattr(row, "get") else lambda key, default="": getattr(row, key, default)
    category = str(getter("category", ""))
    text = f"{getter('title', getter('name', ''))} {getter('description', '')}"
    matches = [pattern.search(text) is not None for pattern, _ in EVIDENCE[category]]
    descriptions = [description for _, description in EVIDENCE[category]]
    if category == "БАД":
        if int(label) == 1:
            evidence = descriptions[0] if matches[0] else "прямая маркировка БАД в данных карточки и на изображениях"
            return f"Карточка соответствует правилам категории: обнаружена {evidence}."
        if matches[1]:
            return f"Карточка не соответствует правилам БАД: обнаружены {descriptions[1]}, а обязательная прямая маркировка не подтверждена."
        if matches[2]:
            return f"Карточка не соответствует правилам БАД: в данных есть {descriptions[2]}."
        return "В карточке не подтверждена обязательная прямая маркировка БАД или dietary supplement, поэтому товар не проходит требования категории."
    if int(label) == 1:
        evidence = next(
            (descriptions[index] for index in (0, 1) if matches[index]),
            "источник огня, горючее содержимое или такой предмет в комплекте",
        )
        return f"Карточка соответствует категории: указан {evidence}, то есть товар отвечает критериям легковоспламеняющегося."
    evidence = next(
        (descriptions[index] for index in (2, 3) if matches[index]),
        None,
    )
    if evidence:
        return f"Товар не проходит правила категории: обнаружена {evidence}; самостоятельное горючее содержимое или источник огня не подтверждены."
    return "Название, описание и изображения не подтверждают самостоятельный источник огня, горючее вещество, газ или такой предмет в комплекте."


def _build_prompt(tokenizer: Any, row: pd.Series, label: int) -> str:
    category = str(row["category"])
    verdict = "соответствует категории и получает не бан" if label else "не соответствует категории и получает бан"
    fallback = deterministic_comment(row, label)
    user = (
        f"Правило: {TASK_CONTEXT[category]}\nКатегория: {category}\n"
        f"Название: {str(row.get('title', ''))[:500]}\n"
        f"Описание: {str(row.get('description', ''))[:1400]}\n"
        f"Решение классификатора: товар {verdict}.\n"
        "Напиши одно краткое объяснение по-русски длиной 70–200 символов. Не меняй решение, "
        "не придумывай фактов и не выводи теги. Если явного доказательства в тексте нет, "
        f"используй нейтральную формулировку по названию, описанию и изображениям. Пример: {fallback}"
    )
    messages = [
        {"role": "system", "content": "Ты помощник модератора маркетплейса. Верни только итоговое объяснение без рассуждений."},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _clean_comment(value: str) -> str:
    value = re.sub(r"<think>[\s\S]*?</think>", "", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    return " ".join(value.strip().strip('"').split())


def _contradicts_decision(comment: str, label: int) -> bool:
    lowered = comment.casefold().replace("ё", "е")
    if int(label) == 1:
        return bool(
            re.search(
                r"(?:отрицательн.{0,12}класс|не\s+(?:соответствует|проходит|подтвержден|относится))",
                lowered,
            )
        )
    positive_class = re.search(r"положительн.{0,12}класс", lowered)
    positive_match = re.search(r"(?<!не\s)соответствует\s+(?:категор|правил|критер)", lowered)
    return bool(positive_class or positive_match)


def generate_comments(
    model_path: str,
    frame: pd.DataFrame,
    predictions: list[int] | Any,
    *,
    batch_size: int = 64,
    max_new_tokens: int = 80,
) -> list[str]:
    """Generate concise explanations; invalid generations fall back deterministically."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    labels = [int(value) for value in predictions]
    comments: list[str] = []
    for start in range(0, len(frame), batch_size):
        end = min(start + batch_size, len(frame))
        prompts = [
            _build_prompt(tokenizer, frame.iloc[index], labels[index])
            for index in range(start, end)
        ]
        encoded = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(
            model.device
        )
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        prefix_length = encoded["input_ids"].shape[1]
        for local_index, tokens in enumerate(output):
            comment = _clean_comment(
                tokenizer.decode(tokens[prefix_length:], skip_special_tokens=True)
            )
            fallback = deterministic_comment(frame.iloc[start + local_index], labels[start + local_index])
            if not 50 <= len(comment) <= 300 or _contradicts_decision(
                comment, labels[start + local_index]
            ):
                comment = fallback
            comments.append(comment)
        print(f"generated comments {end}/{len(frame)}", flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return comments
