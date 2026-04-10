from pathlib import Path
import requests
import sys
import re

URL = "http://localhost:1234/v1/chat/completions"
MODEL = "local-model"  # remplace si besoin

SYSTEM_PROMPT = (
    "Tu es un correcteur de sous-titres français. "
    "Corrige les fautes d'orthographe, de grammaire, de conjugaison et de ponctuation. "
    "Si la phrase est incorrecte, corrige-la entièrement pour qu'elle soit grammaticalement correcte. "
    "Règle absolue : ne change jamais le sens. "
    "Ne change jamais le temps verbal, même si une autre version semble plus naturelle. "
    "Ne change jamais la personne grammaticale. "
    "Si la phrase est déjà correcte, ne change rien. "
    "Ne reformule pas si ce n'est pas nécessaire. "
    "N'ajoute pas d'information. "
    "Ne supprime pas d'information. "
    "Si une variante plus écrite est utile, ajoute-la à la fin entre ⟦ et ⟧. "
    "La suggestion doit garder exactement le même sens. "
    "N'ajoute pas de suggestion si elle n'est pas utile. "
    "Ne réponds pas comme un assistant. "
    "Ne donne aucune explication. "
    "Ne commente pas. "
    "Conserve STRICTEMENT les tags ASS comme {\\i1}, {\\b1}, etc. "
    "Conserve STRICTEMENT les retours de ligne ASS \\N. "
    "Renvoie uniquement le texte final."
)

BAD_PATTERNS = [
    "veuillez",
    "merci de",
    "fournir le texte",
    "je peux corriger",
    "je corrigerai",
    "balises ass",
    "texte que vous souhaitez",
    "voici",
    "here is",
    "i can help",
    "je peux vous aider",
    "corrige cette ligne",
    "si la phrase est incorrecte",
]

FORBIDDEN_PHRASES = [
    "je vais ",
    "je veux ",
    "je suis en train de ",
    "j'ai ça",
]

def looks_bad(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in BAD_PATTERNS)

def forbidden_output(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in FORBIDDEN_PHRASES)

def suspicious_output(corrected: str) -> bool:
    low = corrected.lower()
    bad_chunks = [
        "qu'me",
        "\\N{",
        "{non",
        "{oui",
        "⟦{",
        "}⟧",
    ]
    return any(x in low for x in bad_chunks)

def extract_ass_tags(text: str):
    return re.findall(r"\{.*?\}", text)

def tags_broken(original: str, corrected: str) -> bool:
    return extract_ass_tags(original) != extract_ass_tags(corrected)

def fix_newlines(text: str) -> str:
    return text.replace("\r", "").replace("\n", "\\N")

def split_suggestion(text: str):
    """
    Sépare la correction principale et la suggestion éventuelle.
    Format attendu : texte⟦suggestion⟧
    """
    m = re.match(r"^(.*?)(?:⟦(.*)⟧)?$", text, flags=re.DOTALL)
    if not m:
        return text, None
    base = (m.group(1) or "").strip()
    suggestion = m.group(2)
    if suggestion is not None:
        suggestion = suggestion.strip()
        if not suggestion:
            suggestion = None
    return base, suggestion

def normalize_for_compare(text: str) -> str:
    text = text.lower()
    text = text.replace("\\N", " ")
    text = re.sub(r"\{.*?\}", "", text)
    text = re.sub(r"⟦.*?⟧", "", text)
    text = re.sub(r"[^\w\sàâäçéèêëîïôöùûüÿœæ'-]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def too_different(original: str, corrected: str) -> bool:
    o = normalize_for_compare(original)
    c = normalize_for_compare(corrected)

    o_words = o.split()
    c_words = c.split()

    if not o_words:
        return False

    ratio = len(c_words) / max(len(o_words), 1)
    if ratio > 1.8 or ratio < 0.55:
        return True

    common = set(o_words) & set(c_words)
    if len(o_words) >= 4 and len(common) < max(1, len(set(o_words)) // 3):
        return True

    return False

def tense_shift_suspect(original: str, corrected: str) -> bool:
    o = normalize_for_compare(original)
    c = normalize_for_compare(corrected)

    suspicious_pairs = [
        ("j ai ", ["je vais ", "je veux ", "je suis en train de ", "je "]),
        ("tu as ", ["tu vas ", "tu veux ", "tu es en train de ", "tu "]),
        ("il a ", ["il va ", "il veut ", "il est en train de ", "il "]),
        ("elle a ", ["elle va ", "elle veut ", "elle est en train de ", "elle "]),
        ("on a ", ["on va ", "on veut ", "on est en train de ", "on "]),
        ("nous avons ", ["nous allons ", "nous voulons ", "nous sommes en train de ", "nous "]),
        ("vous avez ", ["vous allez ", "vous voulez ", "vous êtes en train de ", "vous "]),
        ("ils ont ", ["ils vont ", "ils veulent ", "ils sont en train de ", "ils "]),
        ("elles ont ", ["elles vont ", "elles veulent ", "elles sont en train de ", "elles "]),
    ]

    for old, new_list in suspicious_pairs:
        if old in o:
            for new in new_list:
                if new in c and old not in c:
                    return True

    return False

def pre_fix(text: str) -> str:
    fixes = [
        (r"\bje c\b", "je sais"),
        (r"\bj[' ]?est\b", "j'ai"),
        (r"\bj[' ]?e\b", "j'ai"),
        (r"\bjé\b", "j'ai"),
        (r"\bje est\b", "j'ai"),
        (r"\bsa va\b", "ça va"),
        (r"\bske\b", "ce que"),
        (r"\bjcroi\b", "j'crois"),
        (r"\bquil\b", "qu'il"),
        (r"\btavai qua\b", "t'avais qu'à"),
        (r"\bc koi\b", "c'est quoi"),
        (r"\bta dit\b", "t'as dit"),
    ]

    for pattern, repl in fixes:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    return text

def fix_infinitive_past(text: str) -> str:
    """
    Corrections simples :
    j'ai manger -> j'ai mangé
    il a tomber -> il a tombé (ce n'est pas toujours parfait, donc on reste limité)
    """
    text = re.sub(
        r"\bj['’]ai ([a-zàâäçéèêëîïôöùûüÿœæ-]+)er\b",
        r"j'ai \1é",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bt['’]as ([a-zàâäçéèêëîïôöùûüÿœæ-]+)er\b",
        r"t'as \1é",
        text,
        flags=re.IGNORECASE,
    )
    return text

def post_fix(text: str) -> str:
    text = re.sub(r"\bet les papillon\.\b", "et les papillons.", text, flags=re.IGNORECASE)
    text = re.sub(r"\bqu['’]me\b", "qu'à me", text, flags=re.IGNORECASE)
    return text

def suggestion_looks_ok(base: str, suggestion: str) -> bool:
    if not suggestion:
        return False
    if suggestion == base:
        return False
    if looks_bad(suggestion):
        return False
    if forbidden_output(suggestion):
        return False
    if suspicious_output(suggestion):
        return False
    if suggestion.count("\\N") != base.count("\\N"):
        return False
    if extract_ass_tags(base) != extract_ass_tags(suggestion):
        return False
    return True

def correct_text(text: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Corrige cette ligne de sous-titre en français.\n"
                    "Règles :\n"
                    "- Si la phrase est fausse, corrige-la complètement\n"
                    "- Ne change pas le sens\n"
                    "- Ne change pas le temps verbal\n"
                    "- Ne change pas la personne grammaticale\n"
                    "- Si la phrase est déjà correcte, ne change rien\n"
                    "- Si une suggestion plus écrite est utile, ajoute une seule suggestion entre ⟦ et ⟧\n"
                    "- La suggestion doit garder exactement le même sens\n"
                    "- Garde les tags ASS et les \\N\n"
                    "- Renvoie uniquement le texte final\n\n"
                    f"{text}"
                ),
            },
        ],
        "temperature": 0.0,
    }

    r = requests.post(URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def parse_dialogue(line: str):
    if not line.startswith("Dialogue:"):
        return None

    parts = line.rstrip("\r\n").split(",", 9)
    if len(parts) != 10:
        return None

    return parts[:9], parts[9]

def main():
    if len(sys.argv) < 2:
        print('Usage : py corrige.py "mon_fichier.ass"')
        return

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Fichier introuvable : {input_path}")
        return

    output_path = input_path.with_name(input_path.stem + "_corrige.ass")
    backup_path = input_path.with_name(input_path.stem + "_backup.ass")

    original_text = input_path.read_text(encoding="utf-8-sig")
    backup_path.write_text(original_text, encoding="utf-8")

    lines = original_text.splitlines()
    out = []

    total = 0
    changed = 0
    kept = 0

    for line in lines:
        parsed = parse_dialogue(line)

        if not parsed:
            out.append(line + "\n")
            continue

        prefix, text = parsed
        total += 1

        if not text.strip():
            out.append(line + "\n")
            kept += 1
            continue

        try:
            prefixed = pre_fix(text)
            prefixed = fix_infinitive_past(prefixed)

            full_result = correct_text(prefixed)
            full_result = fix_newlines(full_result)
            full_result = post_fix(full_result)

            if looks_bad(full_result):
                print(f"[SKIP assistant] {text}  ->  {full_result}")
                out.append(",".join(prefix) + "," + text + "\n")
                kept += 1
                continue

            if suspicious_output(full_result):
                print(f"[SKIP suspect] {text}  ->  {full_result}")
                out.append(",".join(prefix) + "," + text + "\n")
                kept += 1
                continue

            base, suggestion = split_suggestion(full_result)

            if forbidden_output(base):
                print(f"[SKIP reformulation] {text}  ->  {base}")
                out.append(",".join(prefix) + "," + text + "\n")
                kept += 1
                continue

            if base.count("\\N") != prefixed.count("\\N"):
                print(f"[SKIP \\N] {text}  ->  {base}")
                out.append(",".join(prefix) + "," + text + "\n")
                kept += 1
                continue

            if tags_broken(prefixed, base):
                print(f"[SKIP tags] {text}  ->  {base}")
                out.append(",".join(prefix) + "," + text + "\n")
                kept += 1
                continue

            if tense_shift_suspect(prefixed, base):
                print(f"[SKIP temps] {text}  ->  {base}")
                out.append(",".join(prefix) + "," + text + "\n")
                kept += 1
                continue

            if too_different(prefixed, base):
                print(f"[SKIP trop différent] {text}  ->  {base}")
                out.append(",".join(prefix) + "," + text + "\n")
                kept += 1
                continue

            final_text = base
            if suggestion_looks_ok(base, suggestion):
                print(f"[SUGG] {base}  ||  {suggestion}")

            out.append(",".join(prefix) + "," + final_text + "\n")

            if final_text != text:
                changed += 1
                print(f"[OK] {text}  ->  {final_text}")
            else:
                kept += 1

        except Exception as e:
            print(f"[ERREUR] {text}  ->  {e}")
            out.append(",".join(prefix) + "," + text + "\n")
            kept += 1

    output_path.write_text("".join(out), encoding="utf-8")

    print()
    print("Terminé.")
    print(f"Backup : {backup_path.name}")
    print(f"Sortie : {output_path.name}")
    print(f"Lignes dialogue : {total}")
    print(f"Lignes modifiées : {changed}")
    print(f"Lignes conservées : {kept}")

if __name__ == "__main__":
    main()
