import argparse
import json
import re
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field, ValidationError
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


VIDEO_FORMAT_CATEGORIES = [
    "웹예능",
    "광고/CF",
    "제품리뷰",
    "인터뷰",
    "브이로그",
    "정보전달",
    "상황극/콩트",
    "기타",
]


class VLMOutput(BaseModel):
    production_quality: Literal["High", "Medium", "Low"]
    lighting_style: Literal["자연광", "스튜디오조명", "저조도", "혼합"]
    color_mood: Literal["비비드", "중립", "다크"]
    editing_pace: Literal["빠름", "보통", "느림"]
    motion_graphic: Literal["있음", "없음"]
    video_format: Literal[
        "웹예능",
        "광고/CF",
        "제품리뷰",
        "인터뷰",
        "브이로그",
        "정보전달",
        "상황극/콩트",
        "기타",
    ]
    first_3sec: str = Field(min_length=1)
    background_style: Literal["실내", "실외", "스튜디오", "혼합"]
    top_colors: list[str]
    person_ratio: float = Field(ge=0.0, le=1.0)
    face_ratio: float = Field(ge=0.0, le=1.0)
    text_ratio: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=10)


def build_prompt(video_id: str) -> str:
    return f"""
너는 영상 분석 VLM 평가자다.
아래 스키마의 JSON 객체만 출력하라. JSON 외 텍스트 금지.

[목표]
- Gemini와 동일 기준 비교를 위한 분석
- 출력 키는 아래 13개만 허용
- 모든 값은 한국어(단 production_quality는 High/Medium/Low)

[고정 범주]
1) video_format: 반드시 아래 중 1개
{VIDEO_FORMAT_CATEGORIES}

2) production_quality: High / Medium / Low
- High: 광고/방송급 완성도, 고급 조명/색보정/연출
- Medium: 일반적인 상업 영상 품질
- Low: 저예산/간이 촬영, 완성도 낮음

3) color_mood: 비비드 / 중립 / 다크
- 비비드: 채도 높고 색이 선명
- 중립: 자연스럽고 과하지 않음
- 다크: 전반적으로 어둡고 채도 낮음

4) editing_pace: 빠름 / 보통 / 느림
5) lighting_style: 자연광 / 스튜디오조명 / 저조도 / 혼합
6) motion_graphic: 있음 / 없음
7) background_style: 실내 / 실외 / 스튜디오 / 혼합

[수치 항목 정의]
- person_ratio: 전체 입력 프레임 중 사람이 등장하는 프레임 비율 (0.0~1.0)
- face_ratio: 전체 입력 프레임 중 얼굴이 명확히 보이는 프레임 비율 (0.0~1.0)
- text_ratio: 전체 입력 프레임 중 읽을 수 있는 텍스트가 등장하는 프레임 비율 (0.0~1.0)

[출력 스키마]
{{
  "production_quality": "High|Medium|Low",
  "lighting_style": "자연광|스튜디오조명|저조도|혼합",
  "color_mood": "비비드|중립|다크",
  "editing_pace": "빠름|보통|느림",
  "motion_graphic": "있음|없음",
  "video_format": "웹예능|광고/CF|제품리뷰|인터뷰|브이로그|정보전달|상황극/콩트|기타",
  "first_3sec": "첫 3초 특징",
  "background_style": "실내|실외|스튜디오|혼합",
  "top_colors": ["색상1", "색상2", "색상3"],
  "person_ratio": 0.0,
  "face_ratio": 0.0,
  "text_ratio": 0.0,
  "reason": "판단 근거"
}}

video_id: {video_id}
""".strip()


def sample_frames(frame_paths: list[Path], max_frames: int) -> list[Path]:
    if len(frame_paths) <= max_frames:
        return frame_paths
    idx = np.linspace(0, len(frame_paths) - 1, num=max_frames, dtype=int)
    return [frame_paths[i] for i in idx]


def compute_opencv_stats(image_paths: list[Path]) -> dict[str, Any]:
    brightness_list = []
    rgb_list = []

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_list.append(float(gray.mean()))

        small = cv2.resize(frame, (50, 50), interpolation=cv2.INTER_AREA)
        b, g, r = small.mean(axis=(0, 1)).tolist()
        rgb_list.append([r, g, b])

    if not brightness_list or not rgb_list:
        return {
            "avg_brightness": None,
            "avg_rgb": None,
        }

    avg_rgb = np.mean(np.array(rgb_list), axis=0)
    return {
        "avg_brightness": round(float(np.mean(brightness_list)), 4),
        "avg_rgb": [round(float(x), 4) for x in avg_rgb.tolist()],
    }


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("모델 응답에서 JSON 객체를 찾지 못했습니다.")
    return json.loads(match.group(0))


def normalize_output(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)

    video_format_map = {
        "웹 예능": "웹예능",
        "광고": "광고/CF",
        "cf": "광고/CF",
        "제품 리뷰": "제품리뷰",
        "브이 로그": "브이로그",
        "정보 전달": "정보전달",
        "상황극": "상황극/콩트",
        "콩트": "상황극/콩트",
    }

    if isinstance(out.get("video_format"), str):
        vf = out["video_format"].strip()
        out["video_format"] = video_format_map.get(vf.lower(), video_format_map.get(vf, vf))
        if out["video_format"] not in VIDEO_FORMAT_CATEGORIES:
            out["video_format"] = "기타"

    production_map = {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "높음": "High",
        "중간": "Medium",
        "낮음": "Low",
    }
    if isinstance(out.get("production_quality"), str):
        pq = out["production_quality"].strip()
        out["production_quality"] = production_map.get(pq.lower(), production_map.get(pq, pq))

    color_map = {
        "vivid": "비비드",
        "neutral": "중립",
        "dark": "다크",
    }
    if isinstance(out.get("color_mood"), str):
        cm = out["color_mood"].strip()
        out["color_mood"] = color_map.get(cm.lower(), color_map.get(cm, cm))

    pace_map = {"느린": "느림", "중간": "보통", "빠른": "빠름", "slow": "느림", "normal": "보통", "fast": "빠름"}
    if isinstance(out.get("editing_pace"), str):
        ep = out["editing_pace"].strip()
        out["editing_pace"] = pace_map.get(ep.lower(), pace_map.get(ep, ep))

    motion_map = {"yes": "있음", "no": "없음", "유": "있음", "무": "없음"}
    if isinstance(out.get("motion_graphic"), str):
        mg = out["motion_graphic"].strip()
        out["motion_graphic"] = motion_map.get(mg.lower(), motion_map.get(mg, mg))

    bg_map = {
        "indoor": "실내",
        "outdoor": "실외",
        "studio": "스튜디오",
        "mixed": "혼합",
    }
    if isinstance(out.get("background_style"), str):
        bg = out["background_style"].strip()
        out["background_style"] = bg_map.get(bg.lower(), bg_map.get(bg, bg))

    if isinstance(out.get("top_colors"), str):
        split_colors = [c.strip() for c in re.split(r"[,/|]", out["top_colors"]) if c.strip()]
        out["top_colors"] = split_colors[:3]

    for k in ["person_ratio", "face_ratio", "text_ratio"]:
        v = out.get(k)
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                v = 0.0
        if isinstance(v, (int, float)):
            out[k] = round(min(max(float(v), 0.0), 1.0), 4)

    return out


def load_model(model_id: str, use_4bit: bool) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    quant_config = None
    if use_4bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def infer_one(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    video_id: str,
    image_paths: list[Path],
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = build_prompt(video_id)

    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    vision_tokens = "".join(["<|vision_start|><|image_pad|><|vision_end|>" for _ in pil_images])
    full_text = f"{vision_tokens}\n{prompt}"

    inputs = processor(
        text=[full_text],
        images=pil_images,
        return_tensors="pt",
        padding=True,
    )

    model_device = next(model.parameters()).device
    inputs = {k: (v.to(model_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    trimmed_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    parsed = extract_json(output_text)
    normalized = normalize_output(parsed)
    validated = VLMOutput.model_validate(normalized)
    return validated.model_dump()


def find_frame_dirs(frames_root: Path) -> list[Path]:
    return sorted([d for d in frames_root.iterdir() if d.is_dir()])


def gather_images(video_dir: Path) -> list[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    images: list[Path] = []
    for ext in exts:
        images.extend(sorted(video_dir.glob(ext)))
    return sorted(images)


def load_metadata(metadata_csv: Path | None) -> dict[str, dict[str, Any]]:
    if metadata_csv is None:
        return {}
    df = pd.read_csv(metadata_csv)
    if "video_id" not in df.columns:
        raise ValueError("metadata_csv에는 video_id 컬럼이 필요합니다.")

    meta = {}
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        meta[vid] = {
            "channel_title": row.get("channel_title", None),
            "domain": row.get("domain", None),
        }
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen VLM frame-folder analyzer")
    parser.add_argument(
        "--frames-root",
        type=str,
        default="works/Hyeong_Uk/VLM_test/data/frames_for_vlm",
        help="video_id 폴더들이 들어있는 루트 경로",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="works/Hyeong_Uk/VLM_test/results/qwen_vlm_results.csv",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="works/Hyeong_Uk/VLM_test/results/qwen_vlm_results.json",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=None,
        help="(선택) video_id, channel_title, domain 포함 메타 CSV",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen2.5-vl-3b",
        help="결과 비교용 model_name 컬럼 값",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=12,
        help="영상당 VLM 입력 프레임 수 (균등 샘플링)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="GPU 환경에서 4bit 양자화 사용",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames_root = Path(args.frames_root)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not frames_root.exists():
        raise FileNotFoundError(f"frames_root 경로가 없습니다: {frames_root}")

    metadata = load_metadata(Path(args.metadata_csv) if args.metadata_csv else None)

    print(f"[INFO] 모델 로딩: {args.model_id}")
    model, processor = load_model(args.model_id, use_4bit=args.use_4bit)

    video_dirs = find_frame_dirs(frames_root)
    if not video_dirs:
        raise RuntimeError(f"video_id 폴더를 찾지 못했습니다: {frames_root}")

    all_rows = []

    for idx, video_dir in enumerate(video_dirs, start=1):
        video_id = video_dir.name
        print(f"[INFO] ({idx}/{len(video_dirs)}) 분석 중: {video_id}")

        image_paths = gather_images(video_dir)
        if not image_paths:
            print(f"[WARN] 이미지 없음, 건너뜀: {video_id}")
            continue

        sampled = sample_frames(image_paths, args.max_frames)

        try:
            vlm_result = infer_one(
                model=model,
                processor=processor,
                video_id=video_id,
                image_paths=sampled,
                max_new_tokens=args.max_new_tokens,
            )
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            print(f"[ERROR] 스키마 파싱 실패: {video_id} | {e}")
            continue
        except Exception as e:
            print(f"[ERROR] 추론 실패: {video_id} | {e}")
            continue

        opencv_stats = compute_opencv_stats(sampled)
        row = {
            "model_name": args.model_name,
            "video_id": video_id,
            **vlm_result,
            **opencv_stats,
            "channel_title": metadata.get(video_id, {}).get("channel_title"),
            "domain": metadata.get(video_id, {}).get("domain"),
        }
        all_rows.append(row)

    if not all_rows:
        raise RuntimeError("저장할 분석 결과가 없습니다.")

    df = pd.DataFrame(all_rows)
    ordered_cols = [
        "model_name",
        "video_id",
        "production_quality",
        "lighting_style",
        "color_mood",
        "editing_pace",
        "motion_graphic",
        "video_format",
        "first_3sec",
        "background_style",
        "top_colors",
        "person_ratio",
        "face_ratio",
        "text_ratio",
        "reason",
        "avg_brightness",
        "avg_rgb",
        "channel_title",
        "domain",
    ]
    df = df[[c for c in ordered_cols if c in df.columns]]

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    output_json.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] CSV 저장: {output_csv}")
    print(f"[DONE] JSON 저장: {output_json}")


if __name__ == "__main__":
    main()
