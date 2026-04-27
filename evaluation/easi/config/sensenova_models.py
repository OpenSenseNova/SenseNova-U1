"""Local SenseNova-U1 LightLLM endpoints for VLMEvalKit.

Edit this file to tweak endpoint URLs, ports, max_tokens, temperature, retry,
or to add new variants. Changes take effect the next time VLMEvalKit imports
`vlmeval.config` (typically on the next invocation of `run_easi_eval.py`).

Setup mechanism
---------------
`evaluation/easi/scripts/setup.sh` copies this file into the EASI submodule's
VLMEvalKit as `vlmeval/sensenova_models.py`, then applies a one-line patch to
`vlmeval/config.py` that does:

    from .sensenova_models import entries as _sensenova_entries
    supported_VLM.update(_sensenova_entries)

So any edits here just need a re-run of setup.sh (idempotent) to propagate.

Port assignments MUST match `evaluation/easi/scripts/serve.sh`:
    8b-mot -> 8000    (thinking/reasoning variant)
"""

from functools import partial

# This import only resolves once setup.sh has copied this file into
# evaluation/easi/EASI/VLMEvalKit/vlmeval/. Linter warnings in-tree are expected.
from vlmeval.api.gpt import GPT4V  # type: ignore[import-not-found]

entries = {
    "SenseNova-U1-8B-MoT-Local": partial(
        GPT4V,
        model="sensenova-u1-8b-mot",
        api_base="http://localhost:8000/v1/chat/completions",
        key="dummy",
        temperature=0,
        max_tokens=8192,  # thinking mode needs headroom
        retry=10,
        verbose=False,
    ),
}
