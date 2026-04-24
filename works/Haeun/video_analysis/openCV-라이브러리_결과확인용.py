# ========================================
# OpenCV 정량 분석 성능 테스트
# ========================================
# 목적: video_agent.py에 사용하기 전에 OpenCV가 각 분석 항목을 얼마나 정확하게 추출하는지 확인
# 사용법: uv run <파이썬 코드 파일 경로> <영상파일경로>
# ========================================

import cv2
import argparse
import numpy as np
from pathlib import Path

# ========================================
# 1. argparse
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument('video_path')  # 필수 위치 인자: 테스트할 영상 파일 경로
args = parser.parse_args()

# ========================================
# 2. 영상 열기
# ========================================
# cv2.VideoCapture: 영상 파일을 열어서 프레임 단위로 읽을 수 있는 상태로 만든 객체
# → 아직 재생은 안 했지만 언제든지 프레임을 읽을 준비가 된 상태
cap = cv2.VideoCapture(args.video_path)

if not cap.isOpened():
    print("❌ 영상 열기 실패")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✅ 영상 열기 성공")
print(f" - 총 프레임: {total_frames}")
print(f" - FPS: {fps}")
print(f" - 해상도: {width} x {height}")
print("=" * 60)

# ========================================
# 3. 분석 항목별 테스트
# ========================================

# 분석에 사용할 변수 초기화
frame_count = 0
brightness_sum = 0.0
person_frames = 0
face_frames = 0
text_frames = 0
all_colors = []

# Haar Cascade 로드 (OpenCV 내장 얼굴/인물 감지 모델)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml"
)

while True:
    # cap.read(): 영상에서 프레임을 하나씩 꺼내옴
    # ret   → 프레임을 성공적으로 읽었는지 (True/False)
    # frame → 읽어온 프레임 이미지 데이터
    ret, frame = cap.read()
    if ret == False:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. 평균 밝기 계산 (BGR → 그레이 스케일 변환 후 평균) (색상o -> 흑백으로 변환)
    # <컬러 이미지>
    # ㄴ 각 픽셀이 BGR 세 가지 값으로 구성 → (B:120, G:80, R:200) 이런 식
    # <그레이 스케일>
    # ㄴ 각 픽셀이 밝기 하나의 값으로 구성 → 0 (완전 검정) ~ 255 (완전 흰색)
    brightness = gray.mean()
    brightness_sum += brightness

    # 2. 주요 컬러 추출 (프레임을 1x1로 축소해서 대표 색상 추출)
    small = cv2.resize(frame, (1, 1))
    all_colors.append(small[0][0].tolist())  # BGR값 저장

    # 3. 인물 등장 감지
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    if len(bodies) > 0:
        person_frames += 1

    # 4. 얼굴 등장 감지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        face_frames += 1

    # 5. 텍스트 감지 (엣지 검출로 텍스트 영역 추정)
    edges = cv2.Canny(gray, 100, 200)
    text_pixel_ratio = edges.sum() / (frame.shape[0] * frame.shape[1] * 255)
    if text_pixel_ratio > 0.05:  # 엣지 비율이 5% 이상이면 텍스트 있다고 판단(임의로 결정)
        text_frames += 1

# cap.release(): 영상 파일을 닫아줌
cap.release()

# ========================================
# 4. 결과 출력
# ========================================

avg_brightness = round(brightness_sum / frame_count, 2)

avg_color = [sum(c[i] for c in all_colors) / len(all_colors) for i in range(3)]
blue, green, red = avg_color
if red > blue + 20:
    color_temperature = "따뜻함"
elif blue > red + 20:
    color_temperature = "차가움"
else:
    color_temperature = "중립"

top_colors = all_colors[:3]

print(f"[분석 결과]")
print(f" 총 분석 프레임: {frame_count}")
print('-'*30)
print(f" ☀️ 평균 밝기: {avg_brightness} / 255")
print(f" 🌡️ 색온도: {color_temperature} (R:{red:.1f} G:{green:.1f} B:{blue:.1f})")
print(f" 🖌️ 주요 컬러 (BGR): {top_colors}")
print(f" 🚶 인물 등장 비율: {round(person_frames / frame_count, 2)} ({person_frames}/{frame_count}프레임)")
print(f" 👤 얼굴 등장 비율: {round(face_frames / frame_count, 2)} ({face_frames}/{frame_count}프레임)")
print(f" 📝 텍스트 출현 비율: {round(text_frames / frame_count, 2)} ({text_frames}/{frame_count}프레임)")

# 텍스트 감지 기준값 확인용
print('-'*30)
print(f" 🚶인물 감지 (scaleFactor=1.1, minNeighbors=3)")
print(f"   - 감지된 프레임: {person_frames}개 / 미감지: {frame_count - person_frames}개")
print('-'*30)
print(f" 👤 얼굴 감지 (scaleFactor=1.1, minNeighbors=5)")
print(f"    - 감지된 프레임: {face_frames}개 / 미감지: {frame_count - face_frames}개")
print('-'*30)
print(f" 📝텍스트 감지 (기준값: 0.05)")
print(f"   - 감지된 프레임: {text_frames}개 / 미감지: {frame_count - text_frames}개")


"""
실제 숏츠 영상으로 돌려보면서

이 결과를 보고 기준값을 조정하거나, Gemini에게 옮길 예정
 Gemini에게 부탁하는 것도 좋다고 생각하긴 함

→ 인물/얼굴이 너무 많이 감지되면 minNeighbors 높이기
→ 인물/얼굴이 너무 적게 감지되면 minNeighbors 낮추기

→ 텍스트 있는 프레임이 3%로 나오면 기준을 3%로 낮춰야 함
→ 텍스트 없는 프레임이 6%로 나오면 기준을 높여야 함
"""