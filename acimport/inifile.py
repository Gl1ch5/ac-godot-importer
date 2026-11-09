"""Чтение ini/lut Assetto Corsa (переносы, комментарии, дубли ключей)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Ini:
    sections: dict[str, dict[str, str]] = field(default_factory=dict)

    def get(self, section: str, key: str, default: str | None = None) -> str | None:
        return self.sections.get(section.lower(), {}).get(key.lower(), default)

    def get_float(self, section: str, key: str, default: float = 0.0) -> float:
        raw = self.get(section, key)
        return _to_float(raw, default)

    def get_int(self, section: str, key: str, default: int = 0) -> int:
        raw = self.get(section, key)
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return default

    def get_list(self, section: str, key: str) -> list[float]:
        raw = self.get(section, key) or ""
        out = []
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if part:
                out.append(_to_float(part, 0.0))
        return out

    def has(self, section: str) -> bool:
        return section.lower() in self.sections

    def keys(self, section: str) -> list[str]:
        return list(self.sections.get(section.lower(), {}).keys())


def _to_float(raw, default: float = 0.0) -> float:
    if raw is None:
        return default
    try:
        return float(str(raw).strip().replace(",", "."))
    except ValueError:
        return default


def parse(text: str) -> Ini:
    ini = Ini()
    section = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("//") or line.startswith("#"):
            continue
        if line.startswith("["):
            end = line.find("]")
            if end > 0:
                section = line[1:end].strip().lower()
                ini.sections.setdefault(section, {})
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            ini.sections.setdefault(section, {})[key.strip().lower()] = value.strip()
    return ini


def load(path: str) -> Ini:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse(f.read())


def load_lut(path: str) -> list[tuple[float, float]]:
    """power.lut и прочие .lut: строки «x|y» (иногда через запятую)."""
    points = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";") or line.startswith("//"):
                continue
            sep = "|" if "|" in line else ("," if "," in line else None)
            if sep is None:
                parts = line.split()
                if len(parts) < 2:
                    continue
                x, y = parts[0], parts[1]
            else:
                x, _, y = line.partition(sep)
            try:
                points.append((float(x), float(y)))
            except ValueError:
                continue
    return points