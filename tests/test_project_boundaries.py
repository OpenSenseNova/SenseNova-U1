import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def requirement_entries(requirements: Path) -> list[str]:
    return [
        line.strip()
        for line in requirements.read_text().splitlines()
        if line.strip() and not line.startswith(("#", "--"))
    ]


class ProjectBoundariesTest(unittest.TestCase):
    def test_training_is_an_independent_uv_project(self) -> None:
        self.assertTrue((REPO_ROOT / "training" / "pyproject.toml").is_file())
        self.assertTrue((REPO_ROOT / "training" / "uv.lock").is_file())

    def test_each_project_owns_only_its_runtime_dependencies(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as file:
            inference = tomllib.load(file)
        with (REPO_ROOT / "training" / "pyproject.toml").open("rb") as file:
            training = tomllib.load(file)

        def dependency_names(project: dict) -> set[str]:
            return {
                dependency.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].lower()
                for dependency in project["project"]["dependencies"]
            }

        inference_dependencies = dependency_names(inference)
        training_dependencies = dependency_names(training)
        training_only = {"decord", "dill", "einops", "imageio", "opencv-python", "tensorboard", "timm"}

        self.assertTrue(training_only <= training_dependencies)
        self.assertTrue(training_only.isdisjoint(inference_dependencies))
        self.assertNotIn("pre-commit", inference_dependencies)
        self.assertIn("torch==2.8.0", inference["project"]["dependencies"])
        self.assertIn("transformers>=4.57.1,<6", inference["project"]["dependencies"])
        self.assertIn("accelerate>=1.1,<2", inference["project"]["dependencies"])
        self.assertIn("huggingface-hub>=0.34,<2", inference["project"]["dependencies"])
        self.assertIn("safetensors>=0.4.3,<1", inference["project"]["dependencies"])
        self.assertNotIn("tokenizers", inference_dependencies)
        self.assertIn("torch==2.5.1", training["project"]["dependencies"])
        self.assertEqual(training["project"]["optional-dependencies"]["flash"], ["flash-attn>=2.5,<3"])
        self.assertEqual(
            training["project"]["optional-dependencies"]["flash-build"],
            ["ninja", "packaging", "psutil", "wheel"],
        )
        self.assertEqual(inference["tool"]["uv"]["sources"]["torch"]["index"], "pytorch-cu128")
        self.assertEqual(training["tool"]["uv"]["sources"]["torch"]["index"], "pytorch-cu124")
        self.assertNotIn("workspace", inference["tool"]["uv"])
        self.assertNotIn("workspace", training["tool"]["uv"])

    def test_pip_requirements_mirror_direct_pyproject_dependencies(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as file:
            inference = tomllib.load(file)
        with (REPO_ROOT / "training" / "pyproject.toml").open("rb") as file:
            training = tomllib.load(file)

        projects = (
            (REPO_ROOT / "requirements.txt", inference["project"]["dependencies"], "pytorch-cu128"),
            (REPO_ROOT / "training" / "requirements.txt", training["project"]["dependencies"], "pytorch-cu124"),
            (
                REPO_ROOT / "training" / "requirements-flash.txt",
                training["project"]["optional-dependencies"]["flash"],
                None,
            ),
            (
                REPO_ROOT / "training" / "requirements-flash-build.txt",
                training["project"]["optional-dependencies"]["flash-build"],
                None,
            ),
        )

        for requirements, dependencies, cuda_index in projects:
            self.assertTrue(requirements.is_file())
            contents = requirements.read_text()
            self.assertIn("generated from pyproject.toml", contents)
            self.assertNotIn("--hash", contents)
            self.assertNotIn("# via", contents)
            self.assertEqual(requirement_entries(requirements), dependencies)

            if cuda_index is None:
                self.assertNotIn("--extra-index-url", contents)
            else:
                index_url = next(
                    index["url"]
                    for index in (inference if requirements == REPO_ROOT / "requirements.txt" else training)["tool"][
                        "uv"
                    ]["index"]
                    if index["name"] == cuda_index
                )
                self.assertIn(f"--extra-index-url {index_url}", contents)


if __name__ == "__main__":
    unittest.main()
