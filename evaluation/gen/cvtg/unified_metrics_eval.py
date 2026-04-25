#!/usr/bin/env python3

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class UnifiedMetricsEvaluator:
    def __init__(
        self,
        device: str = "auto",
        cache_dir: str = None,
        use_hf_mirror: bool = True,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
    ):
        """Initialize evaluator"""
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = world_size > 1
        self.is_main_process = rank == 0
        self.cuda_device_index = int(
            os.environ.get(
                "TEXTCRAFTER_CUDA_DEVICE_INDEX",
                str(local_rank if self.is_distributed else 0),
            )
        )

        if device == "cpu":
            self.device = "cpu"
        elif torch.cuda.is_available():
            if self.is_distributed:
                self.device = f"cuda:{self.cuda_device_index}"
                torch.cuda.set_device(self.cuda_device_index)
            else:
                self.device = f"cuda:{self.cuda_device_index}"
        else:
            self.device = "cpu"

        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load all required models"""

        # Try to import PaddleOCR
        try:
            use_paddle_gpu = self.device != "cpu"
            if use_paddle_gpu:
                paddle_gpu_id = self.cuda_device_index
                os.environ["FLAGS_selected_gpus"] = str(paddle_gpu_id)
                import paddle

                paddle.device.set_device(f"gpu:{paddle_gpu_id}")
            import difflib

            import Levenshtein
            from paddleocr import PaddleOCR

            self.models["ocr"] = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                use_gpu=use_paddle_gpu,
            )
            self.paddleocr_available = True
        except ImportError:
            self.paddleocr_available = False
            logging.warning("PaddleOCR not available, Word Accuracy and NED will be skipped")

        # Try to import official CLIP
        try:
            import warnings

            import clip
            import sklearn.preprocessing
            from packaging import version
            from sklearn.preprocessing import normalize

            clip_model, clip_preprocess = clip.load("ViT-L/14", device=self.device, jit=False)
            clip_model.eval()
            self.models["clip_official"] = clip_model
            self.models["clip_official_preprocess"] = clip_preprocess
            self.clip_available = True
        except ImportError:
            self.clip_available = False
            logging.warning("Official CLIP not available, CLIPScore will be skipped")

        # Try to import OpenCLIP
        try:
            import open_clip

            # Get cache directory (if set)
            cache_dir = os.environ.get("HF_HOME", None)
            if cache_dir:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-L-14", pretrained="openai", cache_dir=cache_dir
                )
            else:
                model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
            model.to(self.device)
            model.eval()
            self.models["openclip"] = model
            self.models["openclip_preprocess"] = preprocess

            # Load aesthetic predictor
            aesthetic_model = self._load_aesthetic_model()
            if aesthetic_model:
                self.models["aesthetic"] = aesthetic_model
            self.openclip_available = True
        except ImportError:
            self.openclip_available = False
            logging.warning("OpenCLIP not available, Aesthetic will be skipped")

        # Try to import t2v_metrics
        try:
            import t2v_metrics

            # Get cache directory (if set)
            cache_dir = os.environ.get("HF_HOME", None)
            if cache_dir:
                self.models["vqa"] = t2v_metrics.VQAScore(model="clip-flant5-xxl", cache_dir=cache_dir)
            else:
                self.models["vqa"] = t2v_metrics.VQAScore(model="clip-flant5-xxl")
            self.t2v_available = True
        except ImportError:
            self.t2v_available = False
            logging.warning("t2v_metrics not available, VQAScore will be skipped")

    def _load_aesthetic_model(self):
        """Load aesthetic evaluation model"""
        try:
            import torch.nn as nn

            # Use aesthetic model file from this project
            project_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(project_dir, "sa_0_4_vit_l_14_linear.pth")
            if os.path.exists(model_path):
                m = nn.Linear(768, 1)
                s = torch.load(model_path, map_location=self.device)
                m.load_state_dict(s)
                m.eval()
                m.to(self.device)
                return m
        except Exception as e:
            logging.warning(f"Could not load aesthetic model: {e}")
        return None

    def get_ld(self, ls1: str, ls2: str) -> float:
        """Calculate normalized version of Levenshtein distance"""
        if not self.paddleocr_available:
            return 0.0
        import Levenshtein

        edit_dist = Levenshtein.distance(ls1, ls2)
        return 1 - edit_dist / (max(len(ls1), len(ls2)) + 1e-5)

    def extract_words_from_prompt(self, prompt: str) -> List[str]:
        """Extract words within single quotes from prompt"""
        matches = re.findall(r"'(.*?)'", prompt)
        words = []
        for match in matches:
            words.extend(match.lower().split())
        return words

    def compute_ocr_metrics(self, image_path: str, gt_words: List[str]) -> Tuple[int, int, List[float]]:
        """Calculate OCR-related metrics: total words, correct words, edit distance list for each word"""
        if not self.paddleocr_available or "ocr" not in self.models:
            return 0, 0, []

        try:
            result = self.models["ocr"].ocr(image_path, cls=True)
            pred_words = []

            for idx in range(len(result)):
                res = result[idx]
                if res is not None:
                    for line in res:
                        pred_words.extend(line[1][0].lower().split())

            if len(pred_words) == 0:
                pred_words = [""]

            total_words = len(gt_words)
            acc_words = 0
            edit_distances = []

            import difflib

            for gt_word in gt_words:
                if gt_word in pred_words:
                    acc_words += 1

                best_matches = difflib.get_close_matches(gt_word, pred_words, n=1, cutoff=0)
                if best_matches:
                    best_match = best_matches[0]
                    distance = self.get_ld(gt_word, best_match)
                    edit_distances.append(distance)
                else:
                    edit_distances.append(0.0)

            # Return word-level edit distance list without averaging
            return total_words, acc_words, edit_distances

        except Exception as e:
            logging.error(f"OCR processing failed for {image_path}: {e}")
            return len(gt_words), 0, [0.0] * len(gt_words)

    def compute_clip_score_batch(self, image_paths: List[str], texts: List[str]) -> List[float]:
        """Batch compute CLIPScore - matching the batch processing logic of original clipscore.py"""
        if not self.clip_available or "clip_official" not in self.models:
            return [0.0] * len(image_paths)

        try:
            import warnings

            import clip
            import sklearn.preprocessing
            from packaging import version
            from sklearn.preprocessing import normalize

            # Batch process text (add prefix)
            processed_texts = []
            for text in texts:
                prefix = "A photo depicts "
                if not prefix.endswith(" "):
                    prefix += " "
                processed_texts.append(prefix + text)

            # Batch load and preprocess images
            images = []
            for image_path in image_paths:
                try:
                    image = Image.open(image_path)
                    image_input = self.models["clip_official_preprocess"](image).unsqueeze(0)
                    images.append(image_input)
                except Exception as e:
                    logging.warning(f"Failed to load image {image_path}: {e}")
                    # Create a default zero tensor as placeholder
                    images.append(torch.zeros(1, 3, 224, 224))

            # Merge image batches
            images_batch = torch.cat(images, dim=0).to(self.device)

            # Process text batches
            texts_batch = clip.tokenize(processed_texts, truncate=True).to(self.device)

            with torch.no_grad():
                # Extract features
                image_features = self.models["clip_official"].encode_image(images_batch)
                text_features = self.models["clip_official"].encode_text(texts_batch)

                # Convert to numpy for normalization (matching original implementation)
                image_features_np = image_features.cpu().numpy()
                text_features_np = text_features.cpu().numpy()

                # Normalization processing (matching numpy version compatibility logic of original implementation)
                if version.parse(np.__version__) < version.parse("1.21"):
                    image_features_np = sklearn.preprocessing.normalize(image_features_np, axis=1)
                    text_features_np = sklearn.preprocessing.normalize(text_features_np, axis=1)
                else:
                    warnings.warn(
                        "due to a numerical instability, new numpy normalization is slightly different than paper results. "
                        "to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3."
                    )
                    image_features_np = image_features_np / np.sqrt(np.sum(image_features_np**2, axis=1, keepdims=True))
                    text_features_np = text_features_np / np.sqrt(np.sum(text_features_np**2, axis=1, keepdims=True))

                # Calculate CLIPScore (matching original implementation)
                similarities = np.sum(image_features_np * text_features_np, axis=1)
                clip_scores = 2.5 * np.clip(similarities, 0, None)

                return clip_scores.tolist()

        except Exception as e:
            logging.error(f"Batch CLIP score computation failed: {e}")
            return [0.0] * len(image_paths)

    def compute_clip_score(self, image_path: str, text: str) -> float:
        """Calculate CLIPScore - using official CLIP library to match original implementation"""
        if not self.clip_available or "clip_official" not in self.models:
            return 0.0

        try:
            import warnings

            import clip
            import sklearn.preprocessing
            from packaging import version
            from sklearn.preprocessing import normalize

            # Add prefix, consistent with original CLIPScore script
            prefix = "A photo depicts "
            if not prefix.endswith(" "):
                prefix += " "
            full_text = prefix + text

            # Load and preprocess image
            image = Image.open(image_path)
            image_input = self.models["clip_official_preprocess"](image).unsqueeze(0).to(self.device)

            # Process text
            text_input = clip.tokenize([full_text], truncate=True).to(self.device)

            with torch.no_grad():
                # Extract features
                image_features = self.models["clip_official"].encode_image(image_input)
                text_features = self.models["clip_official"].encode_text(text_input)

                # Convert to numpy for normalization (matching original implementation)
                image_features_np = image_features.cpu().numpy()
                text_features_np = text_features.cpu().numpy()

                # Normalization processing (matching numpy version compatibility logic of original implementation)
                if version.parse(np.__version__) < version.parse("1.21"):
                    image_features_np = sklearn.preprocessing.normalize(image_features_np, axis=1)
                    text_features_np = sklearn.preprocessing.normalize(text_features_np, axis=1)
                else:
                    warnings.warn(
                        "due to a numerical instability, new numpy normalization is slightly different than paper results. "
                        "to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3."
                    )
                    image_features_np = image_features_np / np.sqrt(np.sum(image_features_np**2, axis=1, keepdims=True))
                    text_features_np = text_features_np / np.sqrt(np.sum(text_features_np**2, axis=1, keepdims=True))

                # Calculate CLIPScore (matching original implementation)
                similarity = np.sum(image_features_np * text_features_np, axis=1)
                clip_score = 2.5 * np.clip(similarity, 0, None)

            return float(clip_score[0])

        except Exception as e:
            logging.error(f"CLIP score computation failed for {image_path}: {e}")
            return 0.0

    def compute_vqa_score(self, image_path: str, text: str) -> float:
        """Calculate VQAScore"""
        if not self.t2v_available or "vqa" not in self.models:
            return 0.0

        try:
            score = self.models["vqa"](images=[image_path], texts=[text])
            return score.cpu().numpy().mean()
        except Exception as e:
            logging.error(f"VQA score computation failed for {image_path}: {e}")
            return 0.0

    def compute_aesthetic_score(self, image_path: str) -> float:
        """Calculate aesthetic score"""
        if not self.openclip_available or "aesthetic" not in self.models or "openclip" not in self.models:
            return 0.0

        try:
            image = Image.open(image_path)
            image_input = self.models["openclip_preprocess"](image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.models["openclip"].encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                prediction = self.models["aesthetic"](image_features)

            return prediction.cpu().numpy().item()

        except Exception as e:
            logging.error(f"Aesthetic score computation failed for {image_path}: {e}")
            return 0.0

    def _build_area_result(
        self,
        benchmark_type: str,
        area: int,
        total_images: int,
        total_words: int,
        correct_words: int,
        ned_word_data: List[float],
        clip_score_sum: float,
        vqa_score_sum: float,
        aesthetic_score_sum: float,
    ) -> Dict:
        """Build final per-area metrics from aggregated partial statistics."""
        word_acc = correct_words / max(total_words, 1)
        avg_ned = np.mean(ned_word_data) if ned_word_data else 0
        avg_clip = clip_score_sum / max(total_images, 1) if total_images > 0 else 0
        avg_vqa = vqa_score_sum / max(total_images, 1) if total_images > 0 else 0
        avg_aesthetic = aesthetic_score_sum / max(total_images, 1) if total_images > 0 else 0

        return {
            "area": area,
            "benchmark_type": benchmark_type,
            "word_accuracy": word_acc,
            "ned": avg_ned,
            "clipscore": avg_clip,
            "vqascore": avg_vqa,
            "aesthetic_score": avg_aesthetic,
            "total_images": total_images,
            "total_words": total_words,
            "correct_words": correct_words,
            "ned_word_data": ned_word_data,
        }

    def evaluate_single_area_partial(self, benchmark_dir: str, result_dir: str, area: int, benchmark_type: str):
        """Evaluate the rank-local shard of a single area."""
        # Read benchmark JSON file
        prompt_file = os.path.join(benchmark_dir, benchmark_type, f"{area}.json")
        image_dir = os.path.join(result_dir, benchmark_type, str(area))

        if not os.path.exists(prompt_file):
            logging.error(f"Prompt file not found: {prompt_file}")
            return None

        if not os.path.exists(image_dir):
            logging.error(f"Image directory not found: {image_dir}")
            return None

        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
            prompts = {str(item["index"]): item["prompt"] for item in prompt_data.get("data_list", [])}

        # Get image files
        image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))

        results = {
            "ned_word_data": [],
            "clip_score_sum": 0.0,
            "vqa_score_sum": 0.0,
            "aesthetic_score_sum": 0.0,
        }

        total_words = 0
        correct_words = 0

        # Prepare data for batch CLIPScore calculation
        valid_files = []
        valid_prompts = []
        valid_paths = []

        valid_pairs = []
        for img_file in image_files:
            img_id = Path(img_file).stem
            if img_id in prompts:
                valid_pairs.append((img_file, prompts[img_id], os.path.join(image_dir, img_file)))

        rank_pairs = valid_pairs[self.rank :: self.world_size]
        valid_files = [item[0] for item in rank_pairs]
        valid_prompts = [item[1] for item in rank_pairs]
        valid_paths = [item[2] for item in rank_pairs]

        # Batch compute CLIPScore (matching batch processing logic of original script)
        logging.info(
            "Rank %s computing CLIPScore for %s images in %s area %s", self.rank, len(valid_files), benchmark_type, area
        )
        clip_scores = self.compute_clip_score_batch(valid_paths, valid_prompts) if valid_files else []

        # Create CLIPScore mapping
        clip_score_dict = {Path(valid_files[i]).stem: clip_scores[i] for i in range(len(valid_files))}

        progress_bar = self.is_main_process and self.world_size == 1
        for img_file in tqdm(valid_files, desc=f"Processing {benchmark_type} area {area}", disable=not progress_bar):
            img_path = os.path.join(image_dir, img_file)
            img_id = Path(img_file).stem
            prompt = prompts[img_id]
            gt_words = self.extract_words_from_prompt(prompt)

            # Calculate OCR metrics
            t_words, c_words, word_edit_distances = self.compute_ocr_metrics(img_path, gt_words)
            # Use batch computed CLIPScore results
            clip_score = clip_score_dict.get(img_id, 0.0)
            # Compute VQA and Aesthetic per image (consistent with original script)
            vqa_score = self.compute_vqa_score(img_path, prompt)
            aesthetic_score = self.compute_aesthetic_score(img_path)

            total_words += t_words
            correct_words += c_words

            # Collect edit distances at word level (consistent with original script)
            results["ned_word_data"].extend(word_edit_distances)
            results["clip_score_sum"] += clip_score
            results["vqa_score_sum"] += vqa_score
            results["aesthetic_score_sum"] += aesthetic_score

        return {
            "area": area,
            "benchmark_type": benchmark_type,
            "rank": self.rank,
            "total_images": len(valid_files),
            "total_words": total_words,
            "correct_words": correct_words,
            "ned_word_data": results["ned_word_data"],
            "clip_score_sum": results["clip_score_sum"],
            "vqa_score_sum": results["vqa_score_sum"],
            "aesthetic_score_sum": results["aesthetic_score_sum"],
        }

    def evaluate_full_dataset(self, benchmark_dir: str, result_dir: str, output_file: str):
        """Evaluate complete dataset"""
        partial_results = []

        # Iterate through all areas and benchmark types
        for benchmark_type in ["CVTG", "CVTG-Style"]:
            for area in [2, 3, 4, 5]:
                area_result = self.evaluate_single_area_partial(benchmark_dir, result_dir, area, benchmark_type)
                if area_result:
                    partial_results.append(area_result)

        gathered_results = [partial_results]
        if self.is_distributed:
            gathered_results = [None] * self.world_size
            dist.all_gather_object(gathered_results, partial_results)

        if not self.is_main_process:
            return None

        aggregated = {}
        for rank_results in gathered_results:
            for partial in rank_results:
                key = (partial["benchmark_type"], partial["area"])
                if key not in aggregated:
                    aggregated[key] = {
                        "benchmark_type": partial["benchmark_type"],
                        "area": partial["area"],
                        "total_images": 0,
                        "total_words": 0,
                        "correct_words": 0,
                        "ned_word_data": [],
                        "clip_score_sum": 0.0,
                        "vqa_score_sum": 0.0,
                        "aesthetic_score_sum": 0.0,
                    }
                aggregated[key]["total_images"] += partial["total_images"]
                aggregated[key]["total_words"] += partial["total_words"]
                aggregated[key]["correct_words"] += partial["correct_words"]
                aggregated[key]["ned_word_data"].extend(partial["ned_word_data"])
                aggregated[key]["clip_score_sum"] += partial["clip_score_sum"]
                aggregated[key]["vqa_score_sum"] += partial["vqa_score_sum"]
                aggregated[key]["aesthetic_score_sum"] += partial["aesthetic_score_sum"]

        all_results = []
        for benchmark_type in ["CVTG", "CVTG-Style"]:
            for area in [2, 3, 4, 5]:
                key = (benchmark_type, area)
                if key not in aggregated:
                    continue
                stats = aggregated[key]
                area_result = self._build_area_result(
                    benchmark_type=benchmark_type,
                    area=area,
                    total_images=stats["total_images"],
                    total_words=stats["total_words"],
                    correct_words=stats["correct_words"],
                    ned_word_data=stats["ned_word_data"],
                    clip_score_sum=stats["clip_score_sum"],
                    vqa_score_sum=stats["vqa_score_sum"],
                    aesthetic_score_sum=stats["aesthetic_score_sum"],
                )
                all_results.append(area_result)
                print(f"\n=== {benchmark_type} Area {area} Results ===")
                print(f"Word Accuracy: {area_result['word_accuracy']:.4f}")
                print(f"NED: {area_result['ned']:.4f}")
                print(f"CLIPScore: {area_result['clipscore']:.4f}")
                print(f"VQAScore: {area_result['vqascore']:.4f}")
                print(f"Aesthetic Score: {area_result['aesthetic_score']:.4f}")

        # Calculate overall average results (strictly following original script logic)
        if all_results:
            # Word Accuracy: weighted by word count (consistent with original script)
            total_words_all = sum(r["total_words"] for r in all_results)
            correct_words_all = sum(r["correct_words"] for r in all_results)
            overall_word_acc = correct_words_all / max(total_words_all, 1)

            # NED: simple average of all word edit distances (consistent with original script)
            all_word_edit_distances = []
            for r in all_results:
                all_word_edit_distances.extend(r["ned_word_data"])
            overall_ned = np.mean(all_word_edit_distances) if all_word_edit_distances else 0

            # Other metrics: area-weighted average (by actual image count)
            total_images = sum(r["total_images"] for r in all_results)
            overall_clip = (
                sum(r["clipscore"] * r["total_images"] for r in all_results) / max(total_images, 1)
                if total_images > 0
                else 0
            )
            overall_vqa = (
                sum(r["vqascore"] * r["total_images"] for r in all_results) / max(total_images, 1)
                if total_images > 0
                else 0
            )
            overall_aesthetic = (
                sum(r["aesthetic_score"] * r["total_images"] for r in all_results) / max(total_images, 1)
                if total_images > 0
                else 0
            )

            final_results = {
                "overall_results": {
                    "word_accuracy": overall_word_acc,
                    "ned": overall_ned,
                    "clipscore": overall_clip,
                    "vqascore": overall_vqa,
                    "aesthetic_score": overall_aesthetic,
                    "total_images": sum(r["total_images"] for r in all_results),
                    "total_words": total_words_all,
                    "correct_words": correct_words_all,
                },
                "area_results": all_results,
            }

            # Convert numpy types to JSON serializable types
            final_results_converted = convert_numpy_types(final_results)

            with open(output_file, "w") as f:
                json.dump(final_results_converted, f, indent=2)

            print(f"\n=== Overall Results ===")
            print(f"Word Accuracy: {overall_word_acc:.4f}")
            print(f"NED: {overall_ned:.4f}")
            print(f"CLIPScore: {overall_clip:.4f}")
            print(f"VQAScore: {overall_vqa:.4f}")
            print(f"Aesthetic Score: {overall_aesthetic:.4f}")

            return final_results

        return None


def setup_distributed(args) -> Tuple[int, int, int]:
    """Initialize distributed execution from torchrun-style environment variables."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        remap_visible_devices = os.environ.get("TEXTCRAFTER_REMAP_CUDA_VISIBLE_DEVICES", "1") == "1"
        cuda_device_index = local_rank
        if remap_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
            cuda_device_index = 0
        os.environ["TEXTCRAFTER_CUDA_DEVICE_INDEX"] = str(cuda_device_index)
        if args.device == "cpu":
            raise ValueError("Distributed multi-GPU evaluation requires --device auto or --device cuda.")
        if not torch.cuda.is_available():
            raise RuntimeError("WORLD_SIZE > 1 but CUDA is not available.")
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(cuda_device_index)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Tear down the distributed process group when needed."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Unified text-to-image generation evaluation tool")
    parser.add_argument("--benchmark_dir", required=True, help="benchmark directory path")
    parser.add_argument("--result_dir", required=True, help="result image directory path")
    parser.add_argument("--output_file", required=True, help="result output file path")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="computing device")
    parser.add_argument(
        "--cache_dir",
        default="/share/dnk/checkpoint",
        help="HuggingFace model cache directory path (default: /share/dnk/checkpoint)",
    )
    parser.add_argument(
        "--use_hf_mirror", action="store_true", default=True, help="whether to use HuggingFace mirror (default: True)"
    )
    parser.add_argument(
        "--no_hf_mirror", dest="use_hf_mirror", action="store_false", help="do not use HuggingFace mirror"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Set environment variables before initializing evaluator
    if args.use_hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logging.info("Using HuggingFace mirror: https://hf-mirror.com")

    # Set huggingface cache directory
    if args.cache_dir:
        cache_dir = os.path.abspath(args.cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        # Set all relevant cache environment variables
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_HUB_CACHE"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = cache_dir
        os.environ["TORCH_HOME"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        logging.info(f"Set HuggingFace cache directory to: {cache_dir}")

    rank, world_size, local_rank = setup_distributed(args)

    if world_size > 1:
        logging.info(
            "Initialized distributed evaluation: rank=%s local_rank=%s world_size=%s", rank, local_rank, world_size
        )

    try:
        # Initialize evaluator
        evaluator = UnifiedMetricsEvaluator(
            device=args.device,
            cache_dir=args.cache_dir,
            use_hf_mirror=args.use_hf_mirror,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

        # Run evaluation
        evaluator.evaluate_full_dataset(args.benchmark_dir, args.result_dir, args.output_file)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
