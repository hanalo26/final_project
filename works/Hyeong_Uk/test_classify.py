import os
import pandas as pd
import requests

# 💡 1. 테스트할 Video ID 리스트
video_ids = [
    "pW6OoeiPozA",  # 기존에 AI가 헷갈려 했던 롱폼 (좌우 블랙바)
    "qV42Tc3iBo8",  # 실제 숏츠 영상 예시 1
    "0725IylOZhA",  # 실제 숏츠 영상 예시 2
    "abcdefg1234"   # 잘못된 가짜 ID
]

results = []
print(f"🚀 총 {len(video_ids)}개 영상 URL 리다이렉트 테스트 시작...\n")

# 💡 2. URL 리다이렉트 기반 초고속 분류 로직
for i, vid in enumerate(video_ids):
    url = f"https://www.youtube.com/shorts/{vid}"
    
    try:
        response = requests.get(url, timeout=5)
        final_url = response.url
        
        # 띄어쓰기 등 포맷팅 변수를 제거하여 검색을 정확하게 만듭니다.
        raw_html = response.text.replace(" ", "")
        
        # 💡 핵심 수정: 언어에 의존하지 않고 유튜브 내부의 '재생 불가 상태 코드'를 직접 찾습니다.
        if '"playabilityStatus":{"status":"ERROR"' in raw_html or '"playabilityStatus":{"status":"UNPLAYABLE"' in raw_html:
            verdict = "error"
            reason = "존재하지 않거나, 삭제되었거나, 비공개된 영상입니다."
            
        elif "/watch" in final_url:
            verdict = "longform (롱폼)"
            reason = "유튜브 서버가 /watch 주소로 강제 이동시킴"
            
        elif "/shorts" in final_url:
            verdict = "shorts (숏츠)"
            reason = "유튜브 서버가 /shorts 주소를 그대로 유지함"
            
        else:
            verdict = "unknown"
            reason = f"예상치 못한 URL 반환: {final_url}"

        print(f"[{i+1}/{len(video_ids)}] {vid} -> {verdict}")
        
        results.append({
            "video_id": vid,
            "verdict": verdict,
            "reasoning": reason,
            "final_url": final_url
        })

    except Exception as e:
        print(f"[{i+1}/{len(video_ids)}] {vid} -> ❌ 통신 에러: {e}")
        results.append({
            "video_id": vid,
            "verdict": "error",
            "reasoning": str(e),
            "final_url": "N/A"
        })

# 💡 3. 결과 저장 (기존과 동일하게 현재 폴더에 저장)
print("\n분석 로직 완료. 파일 저장을 시작합니다...")

df = pd.DataFrame(results)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_filename = os.path.join(SCRIPT_DIR, 'youtube_redirect_results.csv')

df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

print(f"✅ 작업 완료! 초고속 분류 결과가 다음 경로에 생성되었습니다:\n -> {SCRIPT_DIR}")