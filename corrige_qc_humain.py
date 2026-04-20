from pathlib import Path
import requests
import sys
import re
import json

URL = "http://localhost:1234/v1/chat/completions"
MODEL = "local-model"  # remplace si besoin

SYSTEM_PROMPT = (
    "Tu es un relecteur humain de sous-titres français. "
    "Tu ne modifies jamais la ligne d'origine. "
    "Tu dois seulement analyser le texte et produire un diagnostic. "
    "Pour chaque ligne, tu dois répondre en JSON valide avec : "
    "status = RAS | FAUTE_CERTAINE | A_VERIFIER ; "
    "segment = segment précis concerné ou chaîne vide ; "
    "proposal = correction ou reformulation proposée, ou chaîne vide ; "
    "reason = explication courte. "
    "Règles : "
    "ne jamais changer le sens dans proposal ; "
    "ne jamais changer le temps verbal sans nécessité ; "
    "ne jamais changer le tutoiement/vouvoiement ; "
    "ne jamais changer la personne grammaticale ; "
    "si c'est seulement oral/familier mais pas fautif, utiliser A_VERIFIER au lieu de FAUTE_CERTAINE ; "
    "ne jamais modifier les tags ASS ; "
    "ne jamais modifier les \\N ; "
    "réponds uniquement en JSON valide avec ce format exact : "
    '{"status":"RAS","segment":"","proposal":"","reason":""}'
)

def parse_dialogue_line(line: str):
    if not line.startswith("Dialogue:"):
        return None
    parts = line.rstrip("\r\n").split(",", 9)
    if len(parts) != 10:
        return None
    return parts

def strip_tags(text: str):
    return re.sub(r"\{.*?\}", "", text)

def split_ass_segments(text: str):
    parts = re.split(r"(\{.*?\})", text)
    segments = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"\{.*?\}", part):
            segments.append(("tag", part))
        else:
            segments.append(("text", part))
    return segments

def rebuild_from_segments(segments):
    return "".join(value for _, value in segments)

def risky_line(text: str) -> bool:
    if r"\p1" in text or r"\p2" in text or r"\p3" in text:
        return True
    if len(strip_tags(text).strip()) <= 1:
        return True
    return False

def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("\\N", " ")
    text = re.sub(r"\{.*?\}", "", text)
    text = re.sub(r"[^\w\sàâäçéèêëîïôöùûüÿœæ'-]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def similarity_ratio(a: str, b: str) -> float:
    wa = set(normalize(a).split())
    wb = set(normalize(b).split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa | wb), 1)

def normalize_diag_status(status: str) -> str:
    status = (status or "RAS").strip().upper()
    if status not in {"RAS", "FAUTE_CERTAINE", "A_VERIFIER"}:
        return "A_VERIFIER"
    return status

def parse_model_json(raw: str):
    raw = raw.strip()

    def normalize_data(data):
        return {
            "status": normalize_diag_status(str(data.get("status", "RAS"))),
            "segment": str(data.get("segment", "")).strip(),
            "proposal": str(data.get("proposal", "")).strip(),
            "reason": str(data.get("reason", "")).strip(),
        }

    # 1) JSON direct
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "status" in data:
            return normalize_data(data)
    except Exception:
        pass

    # 2) extraction de tous les objets JSON candidats
    candidates = []
    depth = 0
    start = None
    in_string = False
    escape = False

    for i, ch in enumerate(raw):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(raw[start:i + 1])
                    start = None

    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict) and "status" in data:
                return normalize_data(data)
        except Exception:
            continue

    return {
        "status": "A_VERIFIER",
        "segment": "",
        "proposal": "",
        "reason": "réponse modèle non parsée proprement"
    }

def model_reply_is_generic(raw: str) -> bool:
    low = raw.lower()
    bad_snippets = [
        "c'est entendu",
        "je suis prêt à analyser",
        "veuillez me transmettre",
        "veuillez transmettre",
        "le texte à analyser",
        "pour chaque ligne",
        "je produirai",
        "json correspondant",
        "l'utilisateur n'a pas fourni",
    ]
    return any(x in low for x in bad_snippets)

def call_model(text: str):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyse ce texte de sous-titre.\n"
                    "Tu dois seulement diagnostiquer, pas corriger la ligne source.\n"
                    "Utilise :\n"
                    "- FAUTE_CERTAINE si faute claire\n"
                    "- A_VERIFIER si formulation discutable ou trop orale\n"
                    "- RAS si rien à signaler\n\n"
                    f"TEXTE:\n{text}"
                )
            }
        ],
        "temperature": 0.0
    }

    r = requests.post(URL, json=payload, timeout=120)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]

    if not raw or model_reply_is_generic(raw):
        return {
            "status": "A_VERIFIER",
            "segment": "",
            "proposal": "",
            "reason": "réponse modèle générique / hors sujet"
        }

    return parse_model_json(raw)

def clean_proposal_like_source(source_text: str, proposal: str):
    proposal = re.sub(r"\{.*?\}", "", proposal).strip()
    return proposal

def validate_proposal(source_text: str, proposal: str):
    if not proposal:
        return False, "empty"

    if source_text.count("\\N") != proposal.count("\\N"):
        return False, "N"

    if similarity_ratio(source_text, proposal) < 0.30:
        return False, "different"

    return True, "ok"

def analyze_text_segment(segment_text: str):
    if not segment_text.strip():
        return {
            "status": "RAS",
            "segment": "",
            "proposal": "",
            "reason": ""
        }

    return call_model(segment_text)

def merge_segment_diagnostics(text: str):
    segments = split_ass_segments(text)

    statuses = []
    notes = []
    proposals = []

    for kind, value in segments:
        if kind == "tag":
            proposals.append(("tag", value))
            continue

        diag = analyze_text_segment(value)
        status = normalize_diag_status(diag["status"])
        segment = diag["segment"]
        proposal = diag["proposal"]
        reason = diag["reason"]

        statuses.append(status)

        if status != "RAS":
            part = []
            if segment:
                part.append(f"segment: {segment}")
            if reason:
                part.append(f"raison: {reason}")
            notes.append(" | ".join(part) if part else status)

        if proposal:
            proposal = clean_proposal_like_source(value, proposal)
            ok, _ = validate_proposal(value, proposal)
            if ok:
                proposals.append(("text", proposal))
            else:
                proposals.append(("text", value))
        else:
            proposals.append(("text", value))

    if "FAUTE_CERTAINE" in statuses:
        final_status = "FAUTE_CERTAINE"
    elif "A_VERIFIER" in statuses:
        final_status = "A_VERIFIER"
    else:
        final_status = "RAS"

    rebuilt_proposal = rebuild_from_segments(proposals)
    if rebuilt_proposal == text:
        rebuilt_proposal = ""

    note = " || ".join(notes[:4])

    return final_status, rebuilt_proposal, note

def make_comment_line(parts, status, proposal, note):
    comment_parts = parts.copy()
    comment_parts[0] = "Comment: 0"

    if status == "FAUTE_CERTAINE":
        label = "QC-FAUTE"
    elif status == "A_VERIFIER":
        label = "QC-STYLE"
    else:
        label = "QC-RAS"

    text = label
    if proposal:
        text += f": {proposal}"
    if note:
        text += f" [{note}]"

    comment_parts[9] = text
    return ",".join(comment_parts)

def main():
    if len(sys.argv) < 2:
        print('Usage : py qc_comment_only.py "mon_fichier.ass"')
        return

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Fichier introuvable : {input_path}")
        return

    original_text = input_path.read_text(encoding="utf-8-sig")
    backup_path = input_path.with_name(input_path.stem + "_backup.ass")
    output_path = input_path.with_name(input_path.stem + "_qc_comment.ass")

    backup_path.write_text(original_text, encoding="utf-8")

    source_lines = original_text.splitlines()
    output_lines = []

    dialogue_total = sum(1 for line in source_lines if line.startswith("Dialogue:"))
    dialogue_index = 0

    faults = 0
    checks = 0
    ras = 0
    skipped = 0

    for line in source_lines:
        parts = parse_dialogue_line(line)
        if not parts:
            output_lines.append(line)
            continue

        dialogue_index += 1
        text = parts[9]

        print(f"[{dialogue_index}/{dialogue_total}] Analyse : {text[:120]}")

        output_lines.append(line)

        if risky_line(text):
            skipped += 1
            print(f"[{dialogue_index}/{dialogue_total}] SKIP")
            print("  Note : ligne spéciale ou non textuelle")
            continue

        status, proposal, note = merge_segment_diagnostics(text)

        if status == "FAUTE_CERTAINE":
            faults += 1
            comment_line = make_comment_line(parts, status, proposal, note)
            output_lines.append(comment_line)
            print(f"[{dialogue_index}/{dialogue_total}] QC-FAUTE")
            print(f"  Ligne : {text}")
            if proposal:
                print(f"  Proposition : {proposal}")
            if note:
                print(f"  Note : {note}")

        elif status == "A_VERIFIER":
            checks += 1
            comment_line = make_comment_line(parts, status, proposal, note)
            output_lines.append(comment_line)
            print(f"[{dialogue_index}/{dialogue_total}] QC-STYLE")
            print(f"  Ligne : {text}")
            if proposal:
                print(f"  Proposition : {proposal}")
            if note:
                print(f"  Note : {note}")

        else:
            ras += 1
            print(f"[{dialogue_index}/{dialogue_total}] QC-OK")
            print(f"  Ligne : {text}")

    output_path.write_text("\n".join(output_lines), encoding="utf-8")

    print()
    print("Terminé.")
    print(f"Backup : {backup_path.name}")
    print(f"Sortie : {output_path.name}")
    print(f"FAUTE_CERTAINE : {faults}")
    print(f"A_VERIFIER : {checks}")
    print(f"RAS : {ras}")
    print(f"SKIP : {skipped}")

if __name__ == "__main__":
    main()
