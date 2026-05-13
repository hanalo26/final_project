"""학습된 모델로 새 영상 1건의 긍정 댓글 비율을 예측하는 예제.

먼저 longform_comment_analysis.ipynb 를 끝까지 돌려서
models/ 폴더에 longform_comment_model.joblib (best Pipeline) 과
meta.json, best_model.json 이 저장되어 있어야 한다.

실행:
    uv run python predict_longform_comment.py
"""
# TODO: 제목과 영상설명, tag의 개수를 직접 입력하도록 제작
# TODO: 피쳐들 처리 방식은 성과 예측모델과 동일

import bisect
import json
from pathlib import Path
import streamlit as st

import joblib
import numpy as np
import pandas as pd
from google import genai
from google.genai import types


# 1) 추론할 영상 정보. 학습은 ALL 데이터셋(IT + FnB 합본) 한 개로만 했고,
#    domain 은 one-hot 피쳐로 모델에 들어간다.
#
# 학습 데이터 SHAP 으로 본 긍정 댓글 비율 시그널 (참고용):
#   - description 텍스트 임베딩이 압도적 영향 (mean|SHAP| 0.70, 다른 피쳐보다 한 자릿수 큼)
#   - 다음 순위: cls_marketing_purpose > cls_cta_type > cls_is_series > domain
#   - 즉 어떤 마케팅 의도/CTA 의 시리즈물인지가 영상 자체 메타 중에선 가장 큰 차이를 만든다.
INPUT = {
    "domain": "FnB",
    "title": "[화제의 대상 ep.10] 대체당부터 K-분식, 기능성표시식품까지! 신규 브랜드 첫만남 댓글이벤트",
    "description": (
        "이번 주 화제의 대상에서는 대체당부터 K-분식, 기능성표시식품까지 신규 브랜드를 모두 모았습니다. "
        "댓글로 가장 궁금한 제품을 남겨주시면 추첨을 통해 선물을 드리는 이벤트도 진행 중입니다. "
        "구독과 좋아요로 다음 에피소드도 함께 만들어 주세요."
    ),
    "video_length_sec": 420,
    "category_name": "정보형 콘텐츠(튜토리얼)",
    "tags": ["에피소드", "신상", "이벤트", "리뷰", "기획"],
    "caption": True,
    "definition": "hd",
    "embeddable": True,
    "has_paid_product_placement": False,
    "cls_content_type": "에피소드소개",
    "cls_marketing_purpose": "기업이미지",
    "cls_cta_type": "이벤트참여",
    "cls_is_series": True,
    "cls_is_collaboration": False,
}


# 2) 환경/경로
# Jupyter (.ipynb) 같은 인터랙티브 환경에서는 __file__ 이 없어 NameError → cwd 로 fallback.
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
    
GEMINI_MODEL = st.secrets.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = 768
MODELS_DIR = HERE / "models"


# 3) 학습 노트북의 prepare_text 와 동일한 prompt prefix + sentinel 로 텍스트를 만든다.
#    (gemini-embedding-2 는 task_type 파라미터를 지원하지 않아 prefix 로 작업 의도를 표시.)
def make_text(title, desc):
    """학습 노트북의 prepare_text 와 동일한 형식.

    형식: "task: classification | query: TITLE: {title}\\nDESCRIPTION: {desc}".
    desc 가 비면 sentinel "(설명 없음)" 으로 대체. 학습/추론이 같은 prefix·format
    을 공유해야 모델 입력 공간이 어긋나지 않는다.
    """
    title = (title or "").strip()
    desc = (desc or "").strip() or "(설명 없음)"
    return f"task: classification | query: TITLE: {title}\nDESCRIPTION: {desc}"


# 4) 영상 길이 → bucket. 학습 데이터의 length_bucket 컬럼과 동일한 임계값 사용.
_LEN_THRESHOLDS = [30, 180, 480, 900]
_LEN_LABELS = ["ultra_short", "short", "standard", "mid_ads", "long_form"]


def length_bucket(sec):
    return _LEN_LABELS[bisect.bisect_right(_LEN_THRESHOLDS, sec)]


# 5) 입력 dict → 학습 시 structured_feature_cols 와 같은 1행 dict
def build_structured_row(d):
    """학습 시 structured_feature_cols 와 동일한 키/값 형태로 1행을 만든다.

    카테고리 컬럼들은 학습 어휘에 있는 값을 그대로 넣어야 OneHotEncoder 가 매칭한다.
    bool 컬럼들은 학습 시 pandas dtype 이 bool 이라 split_columns 에서 cat 으로 분류되어
    OneHot 으로 'True'/'False' 카테고리가 만들어졌다. 추론에서도 str(bool(...)) 로 보내야
    같은 카테고리에 매칭된다.
    description_missing_flag / tags_missing_flag 는 학습 데이터에서 int (0/1) 이라
    그대로 정수로 보낸다.
    """
    desc = d.get("description") or ""
    tags = d.get("tags") or []
    sec = float(d.get("video_length_sec") or 0)
    return {
        "domain": str(d.get("domain", "FnB")),  # ALL 학습이라 domain 이 one-hot 피쳐
        "description_missing_flag": int(not desc.strip()),
        "tags_missing_flag": int(not tags),
        "tags_count": len(tags),
        "영상길이(초)": sec,
        "caption": str(bool(d.get("caption", False))),
        "category_name": str(d.get("category_name", "unknown")),
        "length_bucket": length_bucket(sec),
        "cls_content_type": str(d.get("cls_content_type", "unknown")),
        "cls_marketing_purpose": str(d.get("cls_marketing_purpose", "unknown")),
        "cls_cta_type": str(d.get("cls_cta_type", "unknown")),
        "cls_is_series": str(bool(d.get("cls_is_series", False))),
        "cls_is_collaboration": str(bool(d.get("cls_is_collaboration", False))),
        "definition": str(d.get("definition", "hd")),
        "embeddable": str(bool(d.get("embeddable", True))),
        "has_paid_product_placement": str(bool(d.get("has_paid_product_placement", False))),
    }


# 6) 메타 + best 모델 정보 로드
meta = json.loads((MODELS_DIR / "meta.json").read_text(encoding="utf-8"))
best_info = json.loads((MODELS_DIR / "best_model.json").read_text(encoding="utf-8"))
best_model_name = best_info["best_model"]

# 7) 임베딩 1건 호출 (gemini-embedding-2 는 단건만 처리되므로 contents 에 단일 string)
client = genai.Client(
    vertexai=True,
    project=st.secrets["GOOGLE_CLOUD_PROJECT"],
    location=st.secrets.get("GOOGLE_CLOUD_REGION", "us-central1"),
)

resp = client.models.embed_content(
    model=GEMINI_MODEL,
    contents=make_text(INPUT.get("title"), INPUT.get("description")),
    config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
)
emb = np.array(resp.embeddings[0].values, dtype=np.float32)

# 8) 학습 시와 같은 컬럼 순서로 1행 DataFrame 구성
struct_row = build_structured_row(INPUT)
emb_row = {col: float(v) for col, v in zip(meta["embedding_feature_cols"], emb)}
full = {**struct_row, **emb_row}
X = pd.DataFrame([{c: full[c] for c in meta["all_feature_cols"]}])

# 9) best Pipeline 로드 + 회귀 예측 (0~1 클리핑)
#    longform_comment_model.joblib 은 학습 끝에 best 모델을 복사 저장한 것.
model = joblib.load(MODELS_DIR / "longform_comment_model.joblib")
pred = float(np.clip(model.predict(X)[0], 0, 1))

print(json.dumps({
    "domain": INPUT["domain"],
    "input_title": INPUT["title"],
    "model": best_model_name,
    "predicted_positive_ratio": round(pred, 4),
    "predicted_positive_ratio_pct": f"{pred * 100:.1f}%",
}, indent=2, ensure_ascii=False))
