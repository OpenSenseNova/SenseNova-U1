# Contributing to SenseNova-U1

Thanks for your interest in improving **SenseNova-U1**! We welcome contributions
from the community — bug reports, feature requests, documentation improvements,
and pull requests are all appreciated.

Please open an issue first to discuss any substantial changes before sending a
pull request, so we can align on scope and design.

## Ways to contribute

- 🐛 **Report bugs** — open an [issue](https://github.com/OpenSenseNova/SenseNova-U1/issues) with a clear reproduction, expected vs. actual behavior, and your environment details.
- 💡 **Propose features or improvements** — describe the use case, motivation, and (optionally) a sketch of the API / UX.
- 📖 **Improve the docs** — typo fixes, clarifications, new examples, and tutorials are all welcome.
- 🧪 **Submit pull requests** — fix bugs, add features, or improve performance.

## Development setup

1. Fork and clone the repository, then follow the [Installation Guide](../docs/installation.md) to set up the environment.
2. Create a feature branch off `main` for your change.
3. Install the pre-commit hook once after cloning so lint / formatting issues
   are caught locally before they fail CI:

   ```bash
   uv pip install pre-commit   # or: pip install pre-commit
   pre-commit install
   pre-commit run --all-files  # optional: check the whole repo now
   ```

4. Make your changes, add or update tests where applicable, and ensure
   `pre-commit run --all-files` passes.
5. Push the branch to your fork and open a pull request against `main`.

## Pull request checklist

- [ ] The PR has a descriptive title and summary explaining the motivation.
- [ ] `pre-commit run --all-files` passes locally.
- [ ] New / updated code is covered by tests or example scripts where reasonable.
- [ ] Documentation (README, `docs/`, or inline docstrings) is updated for user-facing changes.
- [ ] The PR is focused — unrelated changes are split into separate PRs.

## Code of conduct

Please be respectful and constructive in all interactions. We aim to keep this
a welcoming community for contributors of all backgrounds and experience levels.
