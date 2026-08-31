#!/usr/bin/env python3
"""Embed rule-derived semantic prototypes in the same space as product cards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from extract_rule_aware_reference import RULE_AWARE_INSTRUCTION

from ozon_quality import official_multimodal

POSITIVE = (
    "Matches sold as a product; a standalone source of open flame.",
    "A refillable cigarette or utility lighter that creates an open flame.",
    "A gas canister containing butane or propane fuel.",
    "A refill bottle containing lighter fluid, kerosene, gasoline, or another flammable liquid.",
    "A camping kit that explicitly includes a filled fuel or gas cylinder.",
    "A torch or burner sold together with a gas canister containing fuel.",
    "Dry fuel tablets, solid alcohol, or fire-starting briquettes sold for ignition.",
    "Charcoal, coal, wood briquettes, or combustible fuel sold for a grill or stove.",
    "Fire starter rolls, paraffin cubes, kindling, or ignition sticks that are consumed by fire.",
    "Fireworks, firecrackers, pyrotechnics, a flare, or a colored smoke bomb.",
    "A candle or wax torch whose main purpose is maintaining an open flame.",
    "A party kit that explicitly includes candles, matches, a lighter, or another flammable item.",
    "A self-heating ration or portable heater kit containing fuel tablets and matches.",
    "A survival kit that explicitly lists matches, dry fuel, or a fire starter among included items.",
    "A combustible gas refill for portable stoves or lighters, with contents present.",
    "Спички или зажигалка как самостоятельный товар и источник открытого огня.",
    "Газовый баллон с бутаном или пропаном; горючее содержимое присутствует.",
    "Жидкость для розжига, керосин, бензин или топливо для зажигалок.",
    "Сухое горючее, парафиновые кубики, растопка или брикеты для разведения огня.",
    "Уголь, дрова или древесные топливные брикеты для мангала и печи.",
    "Газовая горелка в комплекте с наполненным топливным баллоном.",
    "Пиротехника, петарда, фейерверк, сигнальный огонь или цветной дым.",
    "Туристический набор, в который явно входят спички, сухое горючее или газ.",
)

NEGATIVE = (
    "An empty barbecue grill, brazier, fire pit, or metal construction sold without fuel.",
    "A gas stove or cooking appliance sold without a gas cylinder or fuel.",
    "An empty torch or burner attachment that contains no gas or combustible substance.",
    "An appliance with a built-in piezo ignition source; the source is only a component.",
    "A holder, case, cover, stand, adapter, nozzle, or accessory for a lighter or gas cylinder.",
    "A product compatible with gas or fuel, but the gas cylinder or fuel is not included.",
    "A charcoal filter in which activated carbon is only a component of the finished product.",
    "Drawing charcoal, a graphite pencil, pigment, or art material not sold as fuel.",
    "A decorative electric candle or LED light with no open flame or combustible contents.",
    "A fire extinguisher, fire blanket, alarm, or other fire-safety product.",
    "A party decoration or confetti popper with no pyrotechnic or combustible charge.",
    "An empty reusable container intended to be filled later with fuel.",
    "A tool or construction designed to work near fire whose sold contents are not flammable.",
    "A kit where matches, fuel, gas, charcoal, or candles are explicitly not included.",
    "A finished item containing a small combustible material only as a non-sold component.",
    "Пустой мангал, гриль, печь или костровая чаша без угля, газа и топлива.",
    "Газовая плита без баллона; устройство совместимо с газом, но газ не входит в комплект.",
    "Пустая горелка-насадка без горючего содержимого и без баллона в комплекте.",
    "Встроенный пьезоподжиг является компонентом устройства, а не отдельным товаром.",
    "Чехол, держатель, переходник, насадка или аксессуар без топлива и источника огня.",
    "Активированный уголь внутри фильтра или уголь для рисования, не являющийся топливом.",
    "Электрическая декоративная свеча или светильник без открытого огня.",
    "Топливо, газ, спички или уголь явно не входят в комплект товара.",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, examples in ((1, POSITIVE), (0, NEGATIVE)):
        for index, description in enumerate(examples):
            rows.append(
                {
                    "id": f"prototype-{label}-{index}",
                    "name": description,
                    "description": description,
                    "category": "Легковоспламеняющиеся",
                    "label": label,
                    "entity_group": f"prototype-{label}-{index}",
                }
            )
    frame = pd.DataFrame(rows)
    source = output / "prototypes.csv"
    frame.to_csv(source, index=False)
    official_multimodal.REFERENCE_INSTRUCTION = RULE_AWARE_INSTRUCTION
    official_multimodal.extract_reference_embeddings(
        str(source),
        str(output),
        schema=args.schema,
        model_path=args.model,
        mode="text",
        batch_size=32,
    )
    np.save(output / "labels.npy", frame["label"].to_numpy(dtype="int8"))


if __name__ == "__main__":
    main()
