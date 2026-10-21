# -*- coding: utf-8 -*-
"""用 yolo26n-cls-porn 模型筛选爬取到的图片。

判定规则与模型卡一致：
    1. top-1 类别必须是目标类别
    2. top-1 置信度 >= 该类别在阈值文件中的阈值
    3. top-1 置信度 - top-2 置信度 >= MODEL_MARGIN（默认 0.4）

用法：
    python classify.py
    python classify.py --target 动漫
    python classify.py --mock          # 不加载模型，仅用于本地测试

输入：
    data/results.json
输出：
    data/matched.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import config
from common import atomic_write_json, now_iso

logger = logging.getLogger("classify")

MOCK_OTHER_CLASSES = ["其他类别A", "其他类别B", "其他类别C", "其他类别D"]


# ---------------------------------------------------------------------------
# 模型文件与类别映射
# ---------------------------------------------------------------------------


def ensure_model_files(model_dir: Path | None = None) -> tuple[Path, Path, Path]:
    """返回 (ONNX 模型, 阈值 JSON, 中英文映射 JSON) 的本地路径。

    model_dir 为空时从 Hugging Face 下载（门控模型，需要本机已登录且有权限）。
    """
    if model_dir is not None:
        model_dir = model_dir.expanduser().resolve()
        onnx_candidates = [model_dir / "best.onnx", model_dir / config.MODEL_ONNX_FILE]
        onnx_path = next((path for path in onnx_candidates if path.is_file()), None)
        if onnx_path is None:
            raise FileNotFoundError(
                f"本地模型目录里找不到 best.onnx 或 {config.MODEL_ONNX_FILE}：{model_dir}"
            )
        thresholds_path = model_dir / config.MODEL_THRESHOLDS_FILE
        zh2en_path = model_dir / config.MODEL_ZH2EN_FILE
        if not thresholds_path.is_file() or not zh2en_path.is_file():
            # 本地只放了模型文件时，类别映射仍从 HF 仓库获取
            logger.info("本地模型目录缺少类别映射文件，尝试从 Hugging Face 获取")
            from huggingface_hub import hf_hub_download

            if not thresholds_path.is_file():
                thresholds_path = Path(
                    hf_hub_download(
                        repo_id=config.MODEL_REPO_ID,
                        filename=config.MODEL_THRESHOLDS_FILE,
                        token=True,
                    )
                )
            if not zh2en_path.is_file():
                zh2en_path = Path(
                    hf_hub_download(
                        repo_id=config.MODEL_REPO_ID,
                        filename=config.MODEL_ZH2EN_FILE,
                        token=True,
                    )
                )
        return onnx_path, thresholds_path, zh2en_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "缺少依赖 huggingface_hub，请先执行：pip install -r requirements.txt"
        ) from exc

    try:
        onnx_path = Path(
            hf_hub_download(
                repo_id=config.MODEL_REPO_ID,
                filename=config.MODEL_ONNX_FILE,
                token=True,
            )
        )
        thresholds_path = Path(
            hf_hub_download(
                repo_id=config.MODEL_REPO_ID,
                filename=config.MODEL_THRESHOLDS_FILE,
                token=True,
            )
        )
        zh2en_path = Path(
            hf_hub_download(
                repo_id=config.MODEL_REPO_ID,
                filename=config.MODEL_ZH2EN_FILE,
                token=True,
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"下载模型文件失败：{config.MODEL_REPO_ID}\n"
            "请确认：1) 已执行 hf auth login；2) 已在模型页面接受访问条款；"
            "3) 网络可访问 huggingface.co。"
        ) from exc

    return onnx_path, thresholds_path, zh2en_path


def load_class_maps(
    thresholds_path: Path, zh2en_path: Path
) -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    """读取阈值和类别中英文映射。"""
    with thresholds_path.open("r", encoding="utf-8") as fp:
        raw_thresholds = json.load(fp)
    thresholds = {str(name): float(value) for name, value in raw_thresholds.items()}

    with zh2en_path.open("r", encoding="utf-8") as fp:
        zh2en = {str(k): str(v) for k, v in json.load(fp).items()}

    en2zh = {value.lower(): key for key, value in zh2en.items()}
    return thresholds, zh2en, en2zh


def resolve_target_class(
    target: str, thresholds: dict[str, float], zh2en: dict[str, str], en2zh: dict[str, str]
) -> str:
    """把用户填写的目标类别（中文或英文）规范化为模型里的中文类名。"""
    text = target.strip()
    if text in thresholds:
        return text
    lowered = text.lower()
    if lowered in en2zh:
        return en2zh[lowered]
    # 再尝试忽略空格、大小写的英文匹配
    normalized = lowered.replace(" ", "")
    for english, chinese in en2zh.items():
        if english.replace(" ", "") == normalized:
            return chinese
    available = "、".join(f"{zh}（{en}）" for zh, en in zh2en.items())
    raise ValueError(f"未知的目标类别「{target}」。可选类别：{available}")


def canonical_class_name(
    name: str, thresholds: dict[str, float], en2zh: dict[str, str]
) -> str:
    """把模型输出的类别名统一成中文。"""
    if name in thresholds:
        return name
    return en2zh.get(name.lower(), name)


# ---------------------------------------------------------------------------
# 分类器
# ---------------------------------------------------------------------------


class YoloClassifier:
    """Ultralytics 分类模型封装。"""

    def __init__(
        self,
        model_path: Path,
        target_zh: str,
        thresholds: dict[str, float],
        *,
        en2zh: dict[str, str],
        margin: float,
        device: str,
        batch_size: int,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "缺少 ultralytics/torch 依赖，请先执行：pip install -r requirements.txt"
            ) from exc

        logging.getLogger("ultralytics").setLevel(logging.WARNING)

        # 必须显式指定 task="classify"，否则 ONNX 模型可能被误判成 detect
        try:
            self.model = YOLO(str(model_path), task="classify")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"加载分类模型失败：{model_path}。"
                "请确认 ultralytics/onnxruntime 已安装且版本足够新。"
            ) from exc
        self.target_zh = target_zh
        self.thresholds = thresholds
        self.en2zh = en2zh
        self.margin = margin
        self.device = device
        self.batch_size = max(1, batch_size)

        model_names_obj = self.model.names
        model_name_values = (
            model_names_obj.values()
            if isinstance(model_names_obj, dict)
            else model_names_obj
        )
        model_names = {str(name) for name in model_name_values}
        model_names = {
            canonical_class_name(name, thresholds, self.en2zh) for name in model_names
        }
        missing = model_names - set(thresholds)
        if missing:
            logger.warning("模型里有类别缺少阈值，将不会被接受：%s", sorted(missing))

    def _result_to_top5(self, result: Any) -> list[tuple[str, float]]:
        probs = result.probs
        if probs is None:
            raise RuntimeError("模型没有返回分类概率")
        ids = [int(index) for index in probs.top5]
        confidences = [float(value) for value in probs.top5conf]
        top5: list[tuple[str, float]] = []
        names = self.model.names
        for index, confidence in zip(ids, confidences):
            if isinstance(names, dict):
                name = names[index] if index in names else names.get(str(index), str(index))
            else:
                name = names[index]
            top5.append(
                (canonical_class_name(str(name), self.thresholds, self.en2zh), confidence)
            )
        # 显式按置信度降序排列，避免不同 ultralytics 版本的 top5 顺序差异
        top5.sort(key=lambda item: item[1], reverse=True)
        return top5

    def _predict_raw(self, paths: list[str]) -> dict[str, Any]:
        """调用模型，返回 {绝对路径: Results}。"""
        results = self.model.predict(
            paths,
            imgsz=config.MODEL_IMG_SIZE,
            device=self.device,
            verbose=False,
        )
        mapping: dict[str, Any] = {}
        for result in results:
            result_path = getattr(result, "path", None)
            if result_path:
                mapping[str(Path(result_path).resolve())] = result
        return mapping

    def predict_paths(self, paths: list[Path]) -> list[dict[str, Any]]:
        """批量推理，返回与输入顺序一致的结果列表。"""
        outputs: list[dict[str, Any] | None] = [None] * len(paths)
        string_paths = [str(path) for path in paths]

        for start in range(0, len(paths), self.batch_size):
            batch_paths = string_paths[start : start + self.batch_size]
            batch_start = start
            logger.info(
                "模型推理：%d/%d", min(batch_start + len(batch_paths), len(paths)), len(paths)
            )
            try:
                mapping = self._predict_raw(batch_paths)
            except Exception as exc:  # noqa: BLE001
                logger.warning("批量推理失败，改为逐张推理：%s", exc)
                mapping = {}
                for path in batch_paths:
                    try:
                        mapping.update(self._predict_raw([path]))
                    except Exception as one_exc:  # noqa: BLE001
                        logger.warning("图片推理失败：%s（%s）", path, one_exc)

            for offset, path in enumerate(batch_paths):
                index = batch_start + offset
                result = mapping.get(str(Path(path).resolve()))
                if result is None:
                    outputs[index] = {"error": "模型未返回该图片的结果"}
                    continue
                try:
                    outputs[index] = {"top5": self._result_to_top5(result)}
                except Exception as exc:  # noqa: BLE001
                    outputs[index] = {"error": f"解析模型结果失败：{exc}"}

        return [output or {"error": "模型未返回结果"} for output in outputs]


class MockClassifier:
    """本地测试用的假分类器，不需要安装 torch/ultralytics。"""

    def __init__(self, target_zh: str, *, margin: float, batch_size: int = 16) -> None:
        self.target_zh = target_zh
        self.margin = margin
        self.batch_size = max(1, batch_size)

    def predict_paths(self, paths: list[Path]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for path in paths:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                number = int(digest[:8], 16)
            except OSError:
                outputs.append({"error": "读取图片失败"})
                continue

            # 约 1/5 的图片命中目标类别，保证测试时能出现结果
            if number % 5 == 0:
                top1 = (self.target_zh, 0.95)
                top2 = (MOCK_OTHER_CLASSES[0], 0.02)
            else:
                top1 = (MOCK_OTHER_CLASSES[number % len(MOCK_OTHER_CLASSES)], 0.80)
                top2 = (MOCK_OTHER_CLASSES[(number + 1) % len(MOCK_OTHER_CLASSES)], 0.05)
            top5 = [top1, top2] + [
                (name, 0.01) for name in MOCK_OTHER_CLASSES if name not in (top1[0], top2[0])
            ]
            outputs.append({"top5": top5[:5]})
        return outputs


# ---------------------------------------------------------------------------
# 判定与结果组装
# ---------------------------------------------------------------------------


def decide(
    top5: list[tuple[str, float]],
    *,
    target_zh: str,
    thresholds: dict[str, float],
    margin: float,
) -> dict[str, Any]:
    """按模型卡规则判定一张图片是否命中目标类别。"""
    top1_name, top1_conf = top5[0]
    top2_name, top2_conf = top5[1] if len(top5) > 1 else ("", 0.0)
    threshold = float(thresholds.get(target_zh, 1.0))

    is_target = top1_name == target_zh
    conf_ok = top1_conf >= threshold
    margin_value = top1_conf - top2_conf
    margin_ok = margin_value >= margin
    matched = bool(is_target and conf_ok and margin_ok)

    return {
        "matched": matched,
        "predicted_class": top1_name,
        "confidence": round(top1_conf, 6),
        "second_class": top2_name,
        "second_confidence": round(top2_conf, 6),
        "margin": round(margin_value, 6),
        "required_threshold": round(threshold, 6),
        "top5": [
            {"class": name, "confidence": round(conf, 6)} for name, conf in top5
        ],
        "checks": {
            "top1_is_target": is_target,
            "confidence_ok": conf_ok,
            "margin_ok": margin_ok,
        },
    }


def image_cache_key(local_path: str) -> str:
    return local_path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def collect_image_jobs(
    results_payload: dict[str, Any], data_dir: Path, max_images: int = 0
) -> list[dict[str, Any]]:
    """把 results.json 里的图片展开成待推理任务列表。"""
    jobs: list[dict[str, Any]] = []
    for article_index, article in enumerate(results_payload.get("articles", [])):
        for image in article.get("images") or []:
            local_path = image.get("local_path")
            if not local_path:
                continue
            absolute_path = data_dir / local_path
            if not absolute_path.is_file():
                logger.warning("图片不存在，跳过：%s", absolute_path)
                continue
            jobs.append(
                {
                    "article_index": article_index,
                    "local_path": local_path,
                    "absolute_path": absolute_path,
                    "source_url": image.get("source_url", ""),
                }
            )
            if max_images and len(jobs) >= max_images:
                return jobs
    return jobs


def load_cache(cache_path: Path, model_key: str) -> dict[str, Any]:
    """读取推理缓存；模型不一致时丢弃旧缓存。"""
    if not cache_path.is_file():
        return {}
    try:
        payload = load_json(cache_path)
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("model_key") != model_key:
        logger.info("推理缓存对应的模型已变化，忽略旧缓存")
        return {}
    entries = payload.get("entries")
    return entries if isinstance(entries, dict) else {}


def save_cache(cache_path: Path, model_key: str, entries: dict[str, Any]) -> None:
    atomic_write_json(
        cache_path,
        {
            "model_key": model_key,
            "updated_at": now_iso(),
            "entries": entries,
        },
    )


def cache_matches(
    cache_entry: dict[str, Any] | None, absolute_path: Path
) -> bool:
    """缓存的大小和修改时间都一致才复用。"""
    if not cache_entry:
        return False
    try:
        stat = absolute_path.stat()
    except OSError:
        return False
    return (
        cache_entry.get("size") == stat.st_size
        and cache_entry.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(cache_entry.get("top5"), list)
        and cache_entry["top5"]
    )


def classify_all(
    jobs: list[dict[str, Any]],
    classifier: YoloClassifier | MockClassifier,
    *,
    cache: dict[str, Any],
    cache_path: Path,
    model_key: str,
    save_cache_enabled: bool,
) -> list[dict[str, Any]]:
    """推理所有图片，带缓存。返回与 jobs 顺序一致的结果列表。"""
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    pending_indices: list[int] = []

    for index, job in enumerate(jobs):
        local_path = job["local_path"]
        cached = cache.get(image_cache_key(local_path))
        if cache_matches(cached, job["absolute_path"]):
            results[index] = {"top5": [tuple(item) for item in cached["top5"]]}
        else:
            pending_indices.append(index)

    logger.info("共 %d 张图片，缓存命中 %d 张，需要推理 %d 张",
                len(jobs), len(jobs) - len(pending_indices), len(pending_indices))

    for start in range(0, len(pending_indices), classifier.batch_size):
        batch_indices = pending_indices[start : start + classifier.batch_size]
        batch_jobs = [jobs[index] for index in batch_indices]
        outputs = classifier.predict_paths(
            [job["absolute_path"] for job in batch_jobs]
        )
        for index, job, output in zip(batch_indices, batch_jobs, outputs):
            results[index] = output
            if "top5" in output:
                try:
                    stat = job["absolute_path"].stat()
                except OSError:
                    continue
                cache[image_cache_key(job["local_path"])] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "top5": [[name, confidence] for name, confidence in output["top5"]],
                }
        if save_cache_enabled:
            save_cache(cache_path, model_key, cache)

    return [result or {"error": "没有推理结果"} for result in results]


def build_matched_payload(
    *,
    results_payload: dict[str, Any],
    jobs: list[dict[str, Any]],
    classified: list[dict[str, Any]],
    target_zh: str,
    target_en: str,
    thresholds: dict[str, float],
    margin: float,
    model_key: str,
) -> dict[str, Any]:
    """把逐图结果按文章组装成最终 JSON。"""
    by_article: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    jobs_by_article: dict[int, list[dict[str, Any]]] = {}
    result_by_path = {
        job["local_path"]: result for job, result in zip(jobs, classified)
    }
    for job in jobs:
        jobs_by_article.setdefault(job["article_index"], []).append(job)
    for job, result in zip(jobs, classified):
        if "error" in result:
            logger.warning("图片分类失败，跳过：%s（%s）", job["local_path"], result["error"])
            continue
        decision = decide(
            result["top5"],
            target_zh=target_zh,
            thresholds=thresholds,
            margin=margin,
        )
        by_article.setdefault(job["article_index"], []).append((job, decision))

    matches: list[dict[str, Any]] = []
    for article_index, items in by_article.items():
        article = results_payload["articles"][article_index]
        matched_items = [(job, decision) for job, decision in items if decision["matched"]]
        if not matched_items:
            continue

        matched_images = []
        for job, decision in sorted(
            matched_items, key=lambda pair: pair[1]["confidence"], reverse=True
        ):
            matched_images.append(
                {
                    "local_path": job["local_path"],
                    "source_url": job["source_url"],
                    **decision,
                }
            )

        matched_paths = {item["local_path"] for item in matched_images}
        decisions_by_path = {job["local_path"]: decision for job, decision in items}
        other_images = []
        for job in jobs_by_article.get(article_index, []):
            if job["local_path"] in matched_paths:
                continue
            decision = decisions_by_path.get(job["local_path"])
            if decision:
                other_images.append(
                    {
                        "local_path": job["local_path"],
                        "source_url": job["source_url"],
                        "predicted_class": decision["predicted_class"],
                        "confidence": decision["confidence"],
                    }
                )
            else:
                # 图片已经下载但推理失败，也放进“其他图片”，避免网站漏图
                error = (result_by_path.get(job["local_path"]) or {}).get(
                    "error", "未分类"
                )
                other_images.append(
                    {
                        "local_path": job["local_path"],
                        "source_url": job["source_url"],
                        "predicted_class": "未分类",
                        "confidence": None,
                        "classification_error": error,
                    }
                )

        matches.append(
            {
                "url": article["url"],
                "title": article.get("title", ""),
                "published_at": article.get("published_at", ""),
                "author": article.get("author", ""),
                "match_count": len(matched_images),
                "matched_image": matched_images[0],
                "matched_images": matched_images,
                "other_images": other_images,
            }
        )

    # 置信度高的文章排在前面
    matches.sort(key=lambda item: item["matched_image"]["confidence"], reverse=True)

    total_images = len(jobs)
    matched_image_total = sum(item["match_count"] for item in matches)
    return {
        "keyword": results_payload.get("keyword", ""),
        "source_url": results_payload.get("source_url", ""),
        "target_class": target_zh,
        "target_class_en": target_en,
        "model": config.MODEL_REPO_ID,
        "model_key": model_key,
        "criteria": {
            "top1_must_equal_target": True,
            "confidence_threshold": round(float(thresholds.get(target_zh, 1.0)), 6),
            "margin": round(margin, 6),
            "margin_definition": "top1_conf - top2_conf",
        },
        "generated_at": now_iso(),
        "stats": {
            "articles": len(results_payload.get("articles", [])),
            "images": total_images,
            "matched_articles": len(matches),
            "matched_images": matched_image_total,
        },
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run(
    *,
    data_dir: Path,
    target: str,
    margin: float,
    device: str,
    batch_size: int,
    model_dir: Path | None,
    mock: bool,
    use_cache: bool,
    max_images: int,
) -> dict[str, Any]:
    results_path = data_dir / config.RESULTS_FILE.name
    if not results_path.is_file():
        raise SystemExit(f"找不到爬取结果：{results_path}\n请先运行：python crawler.py")

    results_payload = load_json(results_path)
    jobs = collect_image_jobs(results_payload, data_dir, max_images=max_images)
    if not jobs:
        raise SystemExit("results.json 里没有可分析的图片，请先检查爬取结果。")

    if mock:
        logger.warning("正在使用 mock 分类器，结果仅用于测试，不代表真实模型输出")
        thresholds = {target: 0.5}
        target_zh = target
        target_en = target
        classifier: YoloClassifier | MockClassifier = MockClassifier(
            target_zh, margin=margin, batch_size=batch_size
        )
        model_key = f"mock:{target_zh}"
    else:
        onnx_path, thresholds_path, zh2en_path = ensure_model_files(model_dir)
        thresholds, zh2en, en2zh = load_class_maps(thresholds_path, zh2en_path)
        target_zh = resolve_target_class(target, thresholds, zh2en, en2zh)
        target_en = zh2en.get(target_zh, target_zh)
        logger.info(
            "模型：%s；目标类别：%s（%s）；阈值：%.4f；margin：%.2f",
            onnx_path,
            target_zh,
            target_en,
            thresholds[target_zh],
            margin,
        )
        classifier = YoloClassifier(
            onnx_path,
            target_zh,
            thresholds,
            en2zh=en2zh,
            margin=margin,
            device=device,
            batch_size=batch_size,
        )
        model_key = f"{config.MODEL_REPO_ID}:{onnx_path}"

    cache_path = data_dir / config.CLASSIFY_CACHE_FILE.name
    cache = load_cache(cache_path, model_key) if use_cache else {}

    classified = classify_all(
        jobs,
        classifier,
        cache=cache,
        cache_path=cache_path,
        model_key=model_key,
        save_cache_enabled=use_cache,
    )

    payload = build_matched_payload(
        results_payload=results_payload,
        jobs=jobs,
        classified=classified,
        target_zh=target_zh,
        target_en=target_en,
        thresholds=thresholds,
        margin=margin,
        model_key=model_key,
    )

    matched_path = data_dir / config.MATCHED_FILE.name
    atomic_write_json(matched_path, payload)
    logger.info(
        "分类完成：文章 %d 篇，图片 %d 张，命中文章 %d 篇，命中图片 %d 张",
        payload["stats"]["articles"],
        payload["stats"]["images"],
        payload["stats"]["matched_articles"],
        payload["stats"]["matched_images"],
    )
    logger.info("新结果文件：%s", matched_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="用分类模型筛选命中目标类别的图片")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help=f"数据目录（默认：{config.DATA_DIR}）",
    )
    parser.add_argument(
        "--target",
        default=config.TARGET_CLASS,
        help=f"目标类别，中文或英文（默认：{config.TARGET_CLASS}）",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=config.MODEL_MARGIN,
        help=f"top1-top2 置信度差值门槛（默认：{config.MODEL_MARGIN}）",
    )
    parser.add_argument(
        "--device",
        default=config.MODEL_DEVICE,
        help=f"推理设备，通常用 cpu；装了 onnxruntime-gpu 才用 0（默认：{config.MODEL_DEVICE}）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.MODEL_BATCH_SIZE,
        help=f"推理批大小（默认：{config.MODEL_BATCH_SIZE}）",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="本地模型目录（含 best.onnx 等），不填则从 Hugging Face 下载",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用假分类器，仅用于测试网站和流程，不加载真实模型",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="不使用推理缓存，强制重新推理",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="最多分析多少张图片，0 表示全部（调试用）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run(
        data_dir=args.data_dir.expanduser().resolve(),
        target=args.target.strip(),
        margin=max(0.0, args.margin),
        device=args.device.strip(),
        batch_size=max(1, args.batch_size),
        model_dir=args.model_dir.expanduser().resolve() if args.model_dir else None,
        mock=args.mock,
        use_cache=not args.no_cache,
        max_images=max(0, args.max_images),
    )


if __name__ == "__main__":
    main()
