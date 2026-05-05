## 파일 구조

### `works/Haeun/analysis/`

#### 탐색 및 시각화 확인용 (`.ipynb`)
- longform_analysis.ipynb - 롱폼 분석 (FnB / IT)
- shortform_analysis.ipynb - 숏폼 분석 (FnB / IT)
- longform_rf.ipynb - 롱폼 랜덤 포레스트
- longform_lr.ipynb - 롱폼 로지스틱 회귀
- shortform_rf.ipynb - 숏폼 랜덤 포레스트
- shortform_lr.ipynb - 숏폼 로지스틱 회귀

#### 스트림릿 연동용 (`.py`)
- longform_analysis.py - 롱폼 분석 함수 모음
- shortform_analysis.py - 숏폼 분석 함수 모음
- rf_analysis.py - 랜덤 포레스트 함수 모음
- lr_analysis.py - 로지스틱 회귀 함수 모음

### 작업 순서
1. notebooks에서 분석 및 시각화 결과 확인
2. 확정된 코드를 .py 함수로 이식
3. 팀 전체 작업 합쳐서 dashboard.py 작성 (팀 공통 작업)