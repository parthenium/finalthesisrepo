import os
import re
import json
import sys
import subprocess
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Currency / money regex — extended to cover more currencies
# ---------------------------------------------------------------------------

CURRENCY_SYMBOLS = r'(?:EUR|BGN|USD|GBP|RUB|TRY|UAH|RON|HUF|PLN|CZK|CHF|SEK|NOK|DKK|HRK|BAM|RSD|MKD|ALL|MDL|GEL|AMD|AZN|BYR|KZT|TJS|UZS|KGS|TMT)'
CURRENCY_WORDS   = (
    r'(?:Euros?|EUR|BGN|USD|GBP|Russian\s+rubles?|RUB|Turkish\s+liras?|TRY'
    r'|Ukrainian\s+hryvnias?|UAH|Romanian\s+leis?|RON|Hungarian\s+forints?|HUF'
    r'|Polish\s+zlotys?|PLN|Czech\s+korunas?|CZK|Swiss\s+francs?|CHF'
    r'|Bulgarian\s+levs?|Georgian\s+laris?|GEL|Armenian\s+drams?|AMD'
    r'|Azerbaijani\s+manats?|AZN|Belarusian\s+rubles?|BYR|Kazakhstani\s+tenges?|KZT'
    r'|Serbian\s+dinars?|RSD|Albanian\s+leks?|ALL|Moldovan\s+leis?|MDL)'
)

MONEY_RE = re.compile(
    rf'(?:{CURRENCY_SYMBOLS})\s*\d[\d,]*(?:\.\d+)?'
    rf'|\d[\d,]*(?:\.\d+)?\s+(?:{CURRENCY_WORDS})\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Qualifier detection (jointly / each)
# ---------------------------------------------------------------------------

QUALIFIER_RE = re.compile(
    r'\b(jointly(?:\s+and\s+severally)?|each|per\s+applicant|per\s+person'
    r'|for\s+each|to\s+each|apiece)\b',
    re.IGNORECASE
)

def get_qualifier(text, match_start, match_end, window=150):
    lo  = max(0, match_start - window)
    hi  = min(len(text), match_end + window)
    ctx = text[lo:hi].lower()
    if re.search(r'\bjointly\b', ctx):
        return 'jointly'
    if re.search(r'\beach\b|\bper\s+applicant\b|\bper\s+person\b|\bapiece\b', ctx):
        return 'each'
    return None


# ---------------------------------------------------------------------------
# Normalised money entry
# ---------------------------------------------------------------------------

def parse_money_match(raw):
    raw = raw.strip()
    m = re.match(
        rf'^({CURRENCY_SYMBOLS})\s*([\d,]+(?:\.\d+)?)$',
        raw, re.IGNORECASE
    )
    if m:
        return {'raw': raw, 'currency': m.group(1).upper(), 'amount': m.group(2)}
    m = re.match(
        rf'^([\d,]+(?:\.\d+)?)\s+(.+)$',
        raw, re.IGNORECASE
    )
    if m:
        currency_word = m.group(2).strip()
        code = normalise_currency_word(currency_word)
        return {'raw': raw, 'currency': code, 'amount': m.group(1)}
    return {'raw': raw, 'currency': 'UNKNOWN', 'amount': raw}


CURRENCY_WORD_MAP = {
    'euro': 'EUR', 'euros': 'EUR',
    'bulgarian lev': 'BGN', 'bulgarian levs': 'BGN',
    'russian ruble': 'RUB', 'russian rubles': 'RUB',
    'turkish lira': 'TRY', 'turkish liras': 'TRY',
    'ukrainian hryvnia': 'UAH', 'ukrainian hryvnias': 'UAH',
    'romanian lei': 'RON', 'romanian leis': 'RON',
    'hungarian forint': 'HUF', 'hungarian forints': 'HUF',
    'polish zloty': 'PLN', 'polish zlotys': 'PLN',
    'czech koruna': 'CZK', 'czech korunas': 'CZK',
    'swiss franc': 'CHF', 'swiss francs': 'CHF',
    'georgian lari': 'GEL', 'georgian laris': 'GEL',
    'armenian dram': 'AMD', 'armenian drams': 'AMD',
    'azerbaijani manat': 'AZN', 'azerbaijani manats': 'AZN',
    'belarusian ruble': 'BYR', 'belarusian rubles': 'BYR',
    'kazakhstani tenge': 'KZT', 'kazakhstani tenges': 'KZT',
    'serbian dinar': 'RSD', 'serbian dinars': 'RSD',
    'albanian lek': 'ALL', 'albanian leks': 'ALL',
    'moldovan lei': 'MDL', 'moldovan leis': 'MDL',
}

def normalise_currency_word(word):
    key = word.lower().strip()
    for k, v in CURRENCY_WORD_MAP.items():
        if key.startswith(k):
            return v
    return word.upper()[:3]


# ---------------------------------------------------------------------------
# Excluded / section markers
# ---------------------------------------------------------------------------

EXCLUDED_CONTENT_PATTERNS = re.compile(
    r'\b(separate\s+opinion|concurring\s+opinion|dissenting\s+opinion'
    r'|appendix|done\s+in\s+english|pursuant\s+to\s+rule\s+77'
    r'|in\s+accordance\s+with\s+article\s+45)\b',
    re.IGNORECASE
)

LAW_SECTION_MARKERS = re.compile(
    r'^(the\s+law|application\s+of\s+article\s+41|just\s+satisfaction'
    r'|article\s+41|damage|costs\s+and\s+expenses|non.pecuniary'
    r'|pecuniary)',
    re.IGNORECASE
)

SKIP_SECTION_MARKERS = re.compile(
    r'^(the\s+facts?|procedure|relevant\s+(domestic\s+)?law'
    r'|circumstances\s+of\s+the\s+case|introduction)',
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Number-of-applicants extraction
# ---------------------------------------------------------------------------

def extract_num_applicants(full_text):
    full_text = full_text.replace('\xa0', ' ')

    NUMBER_WORDS = {
        'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
        'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,
        'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,
        'eighteen':18,'nineteen':19,'twenty':20
    }

    proc_match = re.search(
        r'\bPROCEDURE\b\s*\n+(.*?)(?=\n\s*2\.|\bTHE\s+FACTS\b|\bTHE\s+LAW\b)',
        full_text, re.IGNORECASE | re.DOTALL
    )
    para1 = proc_match.group(1).strip() if proc_match else full_text[:1000]

    if re.search(r'\bthe applicant\b(?!s)', para1, re.IGNORECASE):
        return 1

    m = re.search(
        r'\bby\s+(one|two|three|four|five|six|seven|eight|nine|ten'
        r'|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen'
        r'|eighteen|nineteen|twenty|\d+)\b'
        r'[^.]{0,80}?\b(nationals?|citizens?|applicants?|persons?)\b',
        para1, re.IGNORECASE
    )
    if m:
        raw = m.group(1).lower()
        return int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw)

    label = re.search(
        u'[\(\u201c\u2018"]the applicants[\)\u201d\u2019"]',
        para1, re.IGNORECASE
    )
    if label:
        names = re.findall(
            r'(?:Mr|Mrs|Ms|Miss|Dr)\.? +[A-Z][^\s,.()\n]{1,30}',
            para1[:label.start()]
        )
        unique = set(n.split()[-1].lower() for n in names)
        if unique:
            return len(unique)

    return None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_docx(docx_path):
    try:
        import docx as docx_lib
        doc = docx_lib.Document(str(docx_path))
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        paragraphs.append(cell.text)
        text = '\n'.join(paragraphs)
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"  python-docx warning on {Path(docx_path).name}: {e}", file=sys.stderr)

    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        result = subprocess.run(
            ['pandoc', '--track-changes=all', '-t', 'plain', str(docx_path)],
            capture_output=True, timeout=60, env=env,
            encoding='utf-8', errors='replace'
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  pandoc warning on {Path(docx_path).name}: {e}", file=sys.stderr)

    try:
        import zipfile
        from xml.etree import ElementTree as ET
        texts = []
        with zipfile.ZipFile(str(docx_path), 'r') as z:
            with z.open('word/document.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
                for elem in root.iter(f'{ns}t'):
                    if elem.text:
                        texts.append(elem.text)
        return ' '.join(texts)
    except Exception as e:
        print(f"  XML fallback failed on {Path(docx_path).name}: {e}", file=sys.stderr)

    return ''


def normalise_text(text):
    text = text.replace('\xa0', ' ')
    text = text.replace('\u200b', '')
    text = text.replace('\u2011', '-')
    text = re.sub(r'[ \t]+', ' ', text)
    return text


# ---------------------------------------------------------------------------
# Section filtering
# ---------------------------------------------------------------------------

def get_law_section_text(full_text):
    lines = full_text.splitlines()
    result_lines = []
    in_skip = True
    for line in lines:
        stripped = line.strip()
        if SKIP_SECTION_MARKERS.match(stripped):
            in_skip = True
            continue
        if LAW_SECTION_MARKERS.match(stripped):
            in_skip = False
        if not in_skip:
            result_lines.append(line)
    return '\n'.join(result_lines)


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def classify_sentence(text):
    t = text.lower()

    if EXCLUDED_CONTENT_PATTERNS.search(t):
        return None, None, []

    money_items = []
    for m in MONEY_RE.finditer(text):
        parsed = parse_money_match(m.group())
        parsed['qualifier'] = get_qualifier(text, m.start(), m.end())
        money_items.append(parsed)

    if not money_items:
        return None, None, []

    claim_kw = bool(re.search(
        r'\b(claimed?|claim|sought|requested?|seeking|seeks'
        r'|applied\s+for|alleged)\b', t))
    award_kw = bool(re.search(
        r'\b(awards?|awarded|grants?|reimburse|entitled\s+to\s+recover'
        r'|pay\s+the\s+applicant|orders?\s+the\s+respondent'
        r'|the\s+court\s+.{0,60}(awards?|grants?)'
        r'|global\s+award|reasonable\s+to\s+award'
        r'|considers\s+it\s+reasonable\s+to\s+award)\b', t, re.IGNORECASE))

    if claim_kw and not award_kw:
        transaction = 'claimed'
    elif award_kw and not claim_kw:
        transaction = 'awarded'
    elif award_kw and claim_kw:
        transaction = 'awarded'
    else:
        return None, None, []

    has_non_pec = bool(re.search(r'non.?pecuniary', t))
    has_pec     = bool(re.search(r'\bpecuniary\b', t))
    has_costs   = bool(re.search(
        r'\b(costs?|expenses?|fees?|legal\s+costs?|translation'
        r'|postage|office\s+supplies?|legal\s+fees?)\b', t))
    has_fine    = bool(re.search(r'\b(fine|penalty|paid\s+the\s+fine)\b', t))
    is_combined = bool(re.search(
        r'(total\s+of'
        r'|in\s+respect\s+of\s+the\s+pecuniary\s+and\s+the\s+non.?pecuniary'
        r'|pecuniary\s+and\s+non.?pecuniary'
        r'|non.?pecuniary\s+and\s+pecuniary'
        r'|any\s+pecuniary\s+and\s+non.?pecuniary'
        r'|pecuniary\s+and\s+non.?pecuniary.*costs)', t))

    has_mixed = has_non_pec and has_pec and not is_combined

    if has_mixed:
        return 'mixed', transaction, money_items
    if is_combined:
        dtype = 'total'
    elif has_non_pec:
        dtype = 'non_pecuniary'
    elif has_pec:
        dtype = 'pecuniary'
    elif has_costs:
        dtype = 'costs'
    elif has_fine:
        dtype = 'fine'
    else:
        dtype = 'total'

    return dtype, transaction, money_items


def split_mixed_items(text, transaction, all_items):
    results = []
    for item in all_items:
        pos = text.find(item['raw'])
        if pos == -1:
            pos = 0
        lo  = max(0, pos - 120)
        ctx = text[lo:pos + len(item['raw'])].lower()

        has_non_pec = bool(re.search(r'non.?pecuniary', ctx))
        has_pec     = bool(re.search(r'\bpecuniary\b', ctx))
        has_costs   = bool(re.search(r'\b(costs?|expenses?|fees?)\b', ctx))
        is_combined = bool(re.search(
            r'total\s+of|pecuniary\s+and\s+non.?pecuniary'
            r'|non.?pecuniary\s+and\s+pecuniary', ctx))

        if is_combined:
            dtype = 'total'
        elif has_non_pec:
            dtype = 'non_pecuniary'
        elif has_pec:
            dtype = 'pecuniary'
        elif has_costs:
            dtype = 'costs'
        else:
            dtype = 'total'

        results.append((dtype, item))
    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def normalise_key(item):
    return re.sub(r'\s+', '', item['raw'].lower())

def dedup_items(lst):
    seen = {}
    out  = []
    for item in lst:
        k = normalise_key(item)
        seen[k] = seen.get(k, 0) + 1
        if seen[k] <= 2:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

BUCKETS = [
    'pecuniary_claimed', 'non_pecuniary_claimed',
    'total_claimed',     'costs_claimed',
    'pecuniary_awarded', 'non_pecuniary_awarded',
    'total_awarded',     'costs_awarded',
    'fine_paid',
]

def extract_damages_from_text(law_text):
    buckets = {k: [] for k in BUCKETS}
    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', law_text)

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            continue

        dtype, trans, items = classify_sentence(sent)
        if not trans or not items:
            continue

        if dtype == 'mixed':
            for sub_dtype, item in split_mixed_items(sent, trans, items):
                key = f"{sub_dtype}_{trans}"
                if key in buckets:
                    buckets[key].append(item)
        else:
            key = f"{dtype}_{'paid' if dtype == 'fine' else trans}"
            if key == 'fine_claimed':
                key = 'fine_paid'
            if key in buckets:
                buckets[key].extend(items)

    for k in buckets:
        buckets[k] = dedup_items(buckets[k])

    return buckets


# ---------------------------------------------------------------------------
# Parse amount string to float
# ---------------------------------------------------------------------------

def parse_amount_float(amount_str):
    """Convert amount string like '300,000' or '16,666.30' to a float."""
    try:
        cleaned = amount_str.replace(',', '')
        return float(cleaned)
    except (ValueError, AttributeError):
        return 0.0


# ---------------------------------------------------------------------------
# Compute lump sums per currency
# ---------------------------------------------------------------------------

def compute_lump_sum(buckets, bucket_keys):
    """
    Sum all monetary amounts across the given bucket keys.
    Groups by currency. Returns a dict like:
      {"EUR": 320068.0, "TRY": 68.0}
    or None if no amounts found.

    For items with qualifier='each' and a known num_applicants, we multiply.
    For items with qualifier='jointly', we count once (already a total).
    Note: num_applicants is passed in separately via a closure; here we
    just sum raw amounts. Callers can post-multiply if needed.
    """
    totals = defaultdict(float)

    for key in bucket_keys:
        for item in buckets.get(key, []):
            currency = item.get('currency', 'UNKNOWN')
            amount   = parse_amount_float(item.get('amount', '0'))
            totals[currency] += amount

    if not totals:
        return None

    # Return as a clean dict, rounded to 2 decimal places
    return {cur: round(val, 2) for cur, val in sorted(totals.items())}


# ---------------------------------------------------------------------------
# Build final lump-sum output
# ---------------------------------------------------------------------------

CLAIMED_BUCKETS = [
    'pecuniary_claimed',
    'non_pecuniary_claimed',
    'total_claimed',
    'costs_claimed',
]

AWARDED_BUCKETS = [
    'pecuniary_awarded',
    'non_pecuniary_awarded',
    'total_awarded',
    'costs_awarded',
    'fine_paid',
]

def build_lump_sums(buckets):
    """
    Returns (total_damage_claimed, total_damage_awarded).
    Each is either None or a dict of {currency: total_amount}.
    """
    claimed = compute_lump_sum(buckets, CLAIMED_BUCKETS)
    awarded = compute_lump_sum(buckets, AWARDED_BUCKETS)
    return claimed, awarded


# ---------------------------------------------------------------------------
# Per-file processor
# ---------------------------------------------------------------------------

def process_docx(docx_path):
    docx_path = Path(docx_path)
    filename  = docx_path.name

    full_text = normalise_text(extract_text_from_docx(docx_path))

    if not full_text.strip():
        print(f"  WARNING: no text extracted from {filename}", file=sys.stderr)
        return {
            'itemid':                docx_path.stem,
            'document':              filename,
            'num_applicants':        None,
            'total_damage_claimed':  None,
            'total_damage_awarded':  None,
        }

    num_applicants = extract_num_applicants(full_text)

    law_text = get_law_section_text(full_text)
    if not law_text.strip():
        law_text = full_text

    buckets                         = extract_damages_from_text(law_text)
    total_damage_claimed, total_damage_awarded = build_lump_sums(buckets)

    return {
        'itemid':               docx_path.stem,
        'document':             filename,
        'num_applicants':       num_applicants,
        'total_damage_claimed': total_damage_claimed,
        'total_damage_awarded': total_damage_awarded,
    }


# ---------------------------------------------------------------------------
# Folder / file processor
# ---------------------------------------------------------------------------

def process_input(input_path):
    input_path = Path(input_path)
    results = []

    if input_path.is_dir():
        docx_files = sorted(input_path.glob('**/*.docx'))
        if not docx_files:
            print(f"No .docx files found in {input_path}", file=sys.stderr)
            return results
        print(f"Found {len(docx_files)} .docx files in {input_path}")
        for f in docx_files:
            print(f"  Processing {f.name}...")
            try:
                results.append(process_docx(f))
            except Exception as e:
                print(f"  ERROR on {f.name}: {e}", file=sys.stderr)
                results.append({
                    'itemid':               f.stem,
                    'document':             f.name,
                    'num_applicants':       None,
                    'total_damage_claimed': None,
                    'total_damage_awarded': None,
                })

    elif input_path.suffix.lower() == '.docx':
        results.append(process_docx(input_path))
    else:
        print("Input must be a .docx file or folder.", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python monetaryextraction_v5_lumpsum.py <file.docx or folder/> [output.json]")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'extracted_damages_lumpsum.json'

    results = process_input(input_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nProcessed {len(results)} file(s) → {output_path}")

    for r in results:
        print(f"\n{'='*60}")
        print(f"  File                 : {r['document']}")
        print(f"  Num applicants       : {r['num_applicants']}")
        print(f"  Total claimed        : {json.dumps(r['total_damage_claimed'])}")
        print(f"  Total awarded        : {json.dumps(r['total_damage_awarded'])}")


if __name__ == '__main__':
    main()