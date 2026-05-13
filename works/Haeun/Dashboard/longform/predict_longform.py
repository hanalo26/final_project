"""학습된 모델로 새 영상 1건의 성공 확률을 추론하는 예제.

먼저 longform_analysis_with_embedding.py 를 끝까지 돌려서
models/ 폴더에 모델과 meta.json 이 저장되어 있어야 한다.

실행:
    uv run python predict_longform.py
"""
import bisect
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# 1) 추론할 영상 정보. 학습은 ALL 데이터셋(IT + FnB 합본) 하나로만 했고,
#    domain 자체를 one-hot 피쳐로 모델에 흘려준다. 그래서 domain 도 입력에 들어가야 한다.
#
# 학습 데이터 분석으로 찾은 강한 성공 시그널 조합 (참고용):
#   - category_name='정보형 콘텐츠(튜토리얼)'  -> 성공률 88.6% (n=35)
#   - cls_content_type='브이로그' / '웹예능' / '인터뷰'  -> 60~83%
#   - cls_marketing_purpose='브랜드캠페인' / '기업이미지'  -> 66~73%
#   - cls_cta_type='이벤트참여' / '구독유도'  -> 69~75%
#   - length_bucket='standard'(5~8분)  -> 52.6%, 성공 영상 평균 길이 ~14분
#   - cls_is_series=True 인 시리즈물이 단발 영상보다 평균적으로 좋음
INPUT = {
    "domain": "FnB",
    # TODO: 제목과 영상설명, tags을 직접 입력하도록 제작
    "title": "[화제의 대상 ep.10] 대체당부터 K-분식, 기능성표시식품까지! 신규 브랜드 첫만남 댓글이벤트",
    "description": (
        "이번 주 화제의 대상에서는 대체당부터 K-분식, 기능성표시식품까지 신규 브랜드를 모두 모았습니다. "
        "댓글로 가장 궁금한 제품을 남겨주시면 추첨을 통해 선물을 드리는 이벤트도 진행 중입니다. "
        "구독과 좋아요로 다음 에피소드도 함께 만들어 주세요."
    ),
    "video_length_sec": 420, # TODO: 사용자가 n분 n초로 입력하면 자동으로 초로 변환하는 로직 필요
    "category_name": "정보형 콘텐츠(튜토리얼)",  # TODO: 사용자가 직접 선택할 수 있도록 제작 (슬랙-도하님dm-매핑정보 있음)
    "tags": ["에피소드", "신상", "이벤트", "리뷰", "기획"],
    "caption": True,
    "definition": "hd", # TODO: 사용자에게는 HD와 SD 정의 보고 고를 수 있도록 
    "embeddable": True,
    "has_paid_product_placement": False,
    "cls_content_type": "에피소드소개",
    "cls_marketing_purpose": "기업이미지",  
    "cls_cta_type": "이벤트참여",            
    "cls_is_series": True,
    "cls_is_collaboration": False,
}


# 2) 환경/경로
# Jupyter (.ipynb) 같은 인터랙티브 환경에서는 __file__ 이 없어 NameError -> cwd 로 fallback.
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

GEMINI_MODEL = st.secrets.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = 768
MODELS_DIR = HERE / "models"


# 3) 학습 스크립트와 똑같은 prompt prefix + sentinel 로 텍스트 만들기.
#    (gemini-embedding-2 는 task_type 미지원이라 prompt prefix 방식)
def make_text(title, desc):
    """Gemini 임베딩 한 건 입력 문자열 (학습 스크립트의 prepare_text 와 동일 동작).

    형식: "task: classification | query: TITLE: {title}\\nDESCRIPTION: {desc}".
    desc 가 비면 sentinel "(설명 없음)" 으로 대체. 학습/추론이 같은 prefix·format
    을 공유해야 모델 입력 공간이 어긋나지 않는다.
    """
    title = (title or "").strip()
    desc = (desc or "").strip() or "(설명 없음)"
    return f"task: classification | query: TITLE: {title}\nDESCRIPTION: {desc}"


# 4) 영상 길이 -> bucket. 학습 데이터 생성 시 쓴 임계값과 동일.
#    bisect_right 로 (sec < 30 -> ultra_short, sec < 180 -> short ...) 와 같은 의미.
_LEN_THRESHOLDS = [30, 180, 480, 900]
_LEN_LABELS = ["ultra_short", "short", "standard", "mid_ads", "long_form"]


def length_bucket(sec):
    return _LEN_LABELS[bisect.bisect_right(_LEN_THRESHOLDS, sec)]


# 5) 입력 dict -> 학습 시 structured_feature_cols 와 같은 1행 dict
def build_structured_row(d):
    """입력 dict 를 학습 시 structured_feature_cols 와 같은 키/값 형태의 1행 dict 로 변환.

    bool 컬럼들은 학습 단계에서 'False'/'True' 문자열로 OneHot 되었으므로
    추론에서도 str(bool(...)) 로 보내야 카테고리 매칭이 살아난다.
    카테고리 컬럼들은 학습 어휘에 있는 값을 그대로 넣어야 정보가 보존된다.
    """
    desc = d.get("description") or ""
    tags = d.get("tags") or []
    sec = float(d.get("video_length_sec") or 0)
    # 추론 입력의 타입은 학습 때 본 타입과 같아야 한다 (안 그러면 모델이 입력을 이해 못 함).
    # - missing_flag 두 개: 학습 데이터 CSV 에 0/1 정수로 있어서 정수 그대로.
    # - 나머지 bool 5 개: 학습 때 'True'/'False' 문자열로 변환되어 들어갔어서 같은 문자열로.
    return {
        "domain": str(d.get("domain", "FnB")),  # ALL 학습에 one-hot 으로 들어가는 피쳐
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


# 6) 메타 로드 (학습 때 어떤 컬럼 순서/임베딩 차원이었는지)
meta = json.loads((MODELS_DIR / "meta.json").read_text(encoding="utf-8"))

# 7) 임베딩 1건 호출 (gemini-embedding-2 는 단건만 처리되므로 contents 에 단일 string)
client = genai.Client(
    vertexai=True,
    project=st.secrets['GOOGLE_CLOUD_PROJECT'],
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

# 9) best 모델 로드 + 예측.
#    어떤 모델이 best 인지는 학습 끝에 기록된 best_models.json (PR-AUC 기준 1등) 을 본다.
best_map = json.loads((MODELS_DIR / "best_models.json").read_text(encoding="utf-8"))
best_model_name = best_map["ALL"]  # 학습이 ALL 데이터셋 한 개라 best_map 키도 ALL 한 개뿐.

model = joblib.load(MODELS_DIR / f"ALL__{best_model_name}.joblib")
proba = float(model.predict_proba(X)[0, 1])

print(json.dumps({
    "domain": INPUT["domain"],
    "input_title": INPUT["title"],
    "model": f"ALL__{best_model_name}",
    "predicted_grade": int(proba >= 0.5),
    "success_probability": round(proba, 4),
}, indent=2, ensure_ascii=False))
