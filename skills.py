"""Agent skills: discovery of SKILL.md folders and the prompt similarity check (spec FR1, FR3)."""
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_THRESHOLD = 0.25
SKILL_DIRS = ("skills", "skill", ".claude/skills", ".mini-coding-agent/skills")

STOP_WORDS = set(
    """a an the and or but if then else of to in on at by for with from into onto about as is are was were be been
    being it its this that these those i me my we our you your he she they them their what which who whom how when
    where why can could should would will shall may might must do does did done have has had not no yes so than too
    very just also please use using used make made get let lets want need some any all each more most other such only
    own same s t don now up out over under again further once here there skill""".split()
)


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    body: str
    triggers: list = field(default_factory=list)


# ---------------------------------------------------------------- parsing / discovery
def parse_front_matter(text):
    meta, body = {}, text
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?", text, re.S)
    if not match:
        return meta, body
    body = text[match.end():]
    current = None
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = re.match(r"^\s*-\s+(.*)$", line)
        if item and current:
            meta.setdefault(current, [])
            if isinstance(meta[current], list):
                meta[current].append(item.group(1).strip().strip("'\""))
            continue
        kv = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
        if not kv:
            continue
        key, value = kv.group(1).lower(), kv.group(2).strip()
        current = key
        if value.startswith("[") and value.endswith("]"):
            meta[key] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
        elif value:
            meta[key] = value.strip("'\"")
        else:
            meta[key] = []
    return meta, body


def load_skill(folder):
    folder = Path(folder)
    skill_file = folder / "SKILL.md"
    if not skill_file.is_file():
        skill_file = folder / "skill.md"
        if not skill_file.is_file():
            return None
    meta, body = parse_front_matter(skill_file.read_text(encoding="utf-8", errors="replace"))
    description = meta.get("description")
    if not description or isinstance(description, list):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip() and not p.strip().startswith("#")]
        description = paragraphs[0].replace("\n", " ") if paragraphs else ""
    triggers = meta.get("triggers") or meta.get("keywords") or []
    if isinstance(triggers, str):
        triggers = [t.strip() for t in triggers.split(",") if t.strip()]
    name = meta.get("name") if isinstance(meta.get("name"), str) and meta.get("name") else folder.name
    return Skill(name=name, description=description, path=skill_file, body=body.strip(), triggers=list(triggers))


def discover_skills(workspace_root, extra_dirs=()):
    """Load skills from the standard roots; earlier roots win on name clashes."""
    root = Path(workspace_root)
    roots = [root / d for d in SKILL_DIRS] + [Path(d) for d in extra_dirs]
    skills = {}
    for skill_root in roots:
        if not skill_root.is_dir():
            continue
        for folder in sorted(p for p in skill_root.iterdir() if p.is_dir()):
            skill = load_skill(folder)
            if skill and skill.name not in skills:
                skills[skill.name] = skill
    return list(skills.values())


# ---------------------------------------------------------------- similarity check
def normalize(text):
    return re.sub(r"[_\-./]+", " ", text.lower())


def stem(word):
    for suffix in ("ing", "edly", "ed", "ies", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", normalize(text)) if w not in STOP_WORDS and len(w) > 1]


def cosine(a, b):
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def skill_profile(skill):
    profile = Counter()
    for token in tokenize(skill.name):
        profile[token] += 3
    for trigger in skill.triggers:
        for token in tokenize(trigger):
            profile[token] += 2
    for token in tokenize(skill.description):
        profile[token] += 1
    return profile


def explicit_request(prompt, name):
    n = re.escape(name.lower())
    text = prompt.lower()
    patterns = (rf"(^|\s)/skill\s+{n}\b", rf"(^|\s)\${n}\b", rf"\buse\s+(the\s+)?{n}\s+skill\b")
    return any(re.search(p, text) for p in patterns)


def score_skill(prompt, skill):
    if explicit_request(prompt, skill.name):
        return {"skill": skill.name, "score": 1.0, "method": "explicit", "matched_terms": [], "matched_triggers": []}
    padded = f" {' '.join(normalize(prompt).split())} "
    hits = [t for t in skill.triggers if t.strip() and f" {' '.join(normalize(t).split())} " in padded]
    if hits:
        return {"skill": skill.name, "score": 0.9, "method": "trigger_phrase", "matched_terms": [], "matched_triggers": hits}
    query, profile = Counter(tokenize(prompt)), skill_profile(skill)
    return {
        "skill": skill.name,
        "score": round(cosine(query, profile), 4),
        "method": "cosine",
        "matched_terms": sorted(query.keys() & profile.keys()),
        "matched_triggers": [],
    }


def similarity_check(prompt, skills, threshold=DEFAULT_THRESHOLD):
    """Score every skill against the prompt; return (candidates best-first, selected name or None, reason)."""
    candidates = sorted((score_skill(prompt, s) for s in skills), key=lambda c: (-c["score"], c["skill"]))
    if not candidates:
        return [], None, "no skills available"
    best = candidates[0]
    if best["score"] >= threshold:
        return candidates, best["skill"], f"{best['method']} score {best['score']} >= threshold {threshold}"
    return candidates, None, f"best score {best['score']} ({best['skill']}) < threshold {threshold}"
