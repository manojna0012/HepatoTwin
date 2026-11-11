"""Read the 2021 international glycaemic index tables into a CSV (Weekly Plan, meeting 11).

Atkinson et al., "International tables of glycemic index and glycemic load values 2021: a systematic
review", AJCN 2021;114(5):1625-1632. The online supplement holds two PDF tables: Supplemental Table 1
(about 2,100 values measured with the ISO 26642:2010 method) and Supplemental Table 2 (values with method
deviations or wide variability). Each entry is read as: food number, item, country, year of test, GI.
The category is the most recent section heading.

Put the two PDFs in data/raw_gi/ as atkinson2021_table1.pdf and atkinson2021_table2.pdf, then:

    python -m src.recipe_scoring.atkinson_gi        (needs pymupdf)
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GI_DIR = ROOT / 'data' / 'raw_gi'
OUTPUT = GI_DIR / 'atkinson2021_gi.csv'
CATEGORIES = ['BAKERY PRODUCTS', 'BEVERAGES', 'BREADS', 'BREAKFAST CEREALS', 'CEREAL GRAINS', 'COOKIES', 'CRACKERS',
              'DAIRY PRODUCTS AND ALTERNATIVES', 'FRUIT AND FRUIT PRODUCTS', 'INFANT FORMULA AND WEANING FOODS',
              'LEGUMES', 'MEAL REPLACEMENT', 'NUTRITIONAL SUPPORT PRODUCTS', 'NUTS', 'PASTA AND NOODLES',
              'SNACK FOODS AND CONFECTIONERY', 'SOUPS', 'SUGARS AND SYRUPS', 'VEGETABLES', 'MIXED MEALS',
              'REGIONAL OR TRADITIONAL FOODS', 'SPORTS']
YEAR = re.compile(r"^(19|20)\d\d\*?$|^NS$")
GI = re.compile(r"^(\d{1,3})(?:\s*±\s*(\d{1,3}))?$")
NUMBER = re.compile(r"^\d{1,4}$")
LEADING_NUMBER = re.compile(r"^(\d{1,4}) (.*)$")


def entries(lines, table):
    """Table rows from the PDF's text lines."""
    rows, category, i = [], None, 0
    while i < len(lines):
        line = lines[i]
        if line.isupper() and any(line.startswith(c) for c in CATEGORIES) and 'PAGE' not in line:
            category = line.title()
        if NUMBER.match(line):
            for k in range(2, 7):
                if i + k + 1 < len(lines) and YEAR.match(lines[i + k]) and GI.match(lines[i + k + 1]):
                    number, item = int(line), ' '.join(lines[i + 1:i + k - 1])
                    m = LEADING_NUMBER.match(item)          # a sub-heading's own number precedes the item
                    if m:
                        number, item = int(m.group(1)), m.group(2)
                    g = GI.match(lines[i + k + 1])
                    rows.append({'table': table, 'number': number, 'item': item, 'country': lines[i + k - 1],
                                 'year': lines[i + k].rstrip('*'), 'gi': int(g.group(1)),
                                 'sem': g.group(2) or '', 'category': category})
                    i += k + 1
                    break
        i += 1
    return rows


def main():
    import pymupdf
    rows = []
    for table in (1, 2):
        doc = pymupdf.open(GI_DIR / f'atkinson2021_table{table}.pdf')
        lines = [l.strip() for page in doc for l in page.get_text().split('\n') if l.strip()]
        rows += entries(lines, table)
    with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} entries ({sum(r["table"] == 1 for r in rows)} from Table 1) -> {OUTPUT}')


if __name__ == '__main__':
    main()
