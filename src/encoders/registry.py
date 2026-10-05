from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class EncoderSpec:
    rank: int
    encoder_id: str
    probing_id: str
    head: str

def encoder_panel():
    source = Path(__file__).resolve().parents[1]/'workers/linear/tokenizers.tsv'
    panel = []
    for line in source.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        rank,requested,probing,head = line.split('\t')
        panel.append(EncoderSpec(int(rank),requested,probing,head))
    if len(panel)!=70 or len({spec.encoder_id for spec in panel})!=70 or [s.rank for s in panel]!=list(range(1,71)):
        raise ValueError('canonical encoder panel is not a unique ordered 70-model registry')
    return tuple(panel)
