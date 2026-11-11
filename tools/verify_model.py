import hashlib
import json
import os
from pathlib import Path

if not __debug__:
    raise SystemExit("Run without -O or PYTHONOPTIMIZE")

root = Path(os.environ["MODEL_DIR"]).resolve()
hashes = [
    "2c1c34004cc8af1d8117f77f4ab3898d9ee6709f0c5c6ca67ac89a8cbe3a7556",
    "25976ebcae74749ce915dd8b5657fff8d06abfcd92122d6f1155047431caec30",
    "b4e9a064c19af0a021fc80505ac9189e55305bd89d6b075796ca2116dc14e7cd",
    "04bc2b31e28bb2fad73e21c7e8656ca4e1b846ccdab1991a9221926f67911702",
    "531f26640e0810c0eb6eaa6578e0dd09fe67e9a1c4c385a18f091a53d2500ba6",
    "e9ddda7a6ab2c7de1840bb9b1c08a9df2aa161ef3c063a744977c4d74dac1d98",
    "a9cb58e5c45c8c857fe3462eb4c38f6fa25b19789de56dd95a0abf92175145ed",
    "90862be20f40e667de050ef5dd78bb13393fb1ac23264552b9040be9a2608f35",
]
sizes = [
    4966078840,
    4932742056,
    4983075328,
    4949521032,
    4932725616,
    4999870800,
    4932725608,
    369118856,
]
small = {
    "added_tokens.json": 8005,
    "config.json": 2631,
    "merges.txt": 1670344,
    "model.safetensors.index.json": 113453,
    "special_tokens_map.json": 39951,
    "tokenizer_config.json": 63546,
    "vocab.json": 3383407,
}
for name, size in small.items():
    p = root / name
    assert p.is_file(), f"Missing: {p}"
    assert p.stat().st_size == size, f"Size mismatch: {p}"
    if p.suffix == ".json":
        json.loads(p.read_text(encoding="utf-8"))
index = json.loads((root / "model.safetensors.index.json").read_text())
names = [f"model-{i:05d}-of-00008.safetensors" for i in range(1, 9)]
assert set(index["weight_map"].values()) == set(names)
assert index["metadata"]["total_size"] == 35065708928
config = json.loads((root / "config.json").read_text())
assert config["model_type"] == "neo_chat"
assert "NEOChatModel" in config["architectures"]
for name, size, want in zip(names, sizes, hashes):
    p = root / name
    assert p.is_file(), f"Missing: {p}"
    assert p.stat().st_size == size, f"Size mismatch: {p}"
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == want, f"SHA256 mismatch: {p}"
    print("SHA256_OK", name, flush=True)
print("MODEL_FILES_CHECK_OK", root)
