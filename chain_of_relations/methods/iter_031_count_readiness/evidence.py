"""Field-level provenance primitives for gold-free runtime evidence."""
from __future__ import annotations

import re
from dataclasses import dataclass


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, order=True)
class EvidenceRef:
    source_name: str
    source_sha256: str
    field_path: str
    record_key: str = ""
    source_class: str = "runtime"

    @property
    def valid(self) -> bool:
        return bool(
            self.source_name.strip()
            and _SHA256.fullmatch(self.source_sha256)
            and self.field_path.strip()
            and self.source_class in {"runtime", "schema"}
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "field_path": self.field_path,
            "record_key": self.record_key,
            "source_class": self.source_class,
        }
