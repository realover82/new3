"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5절. 학습 전 데이터 전처리 파이프라인 시각화 (Gradio 웹 인터페이스)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【실행 방법】
  python step05_preprocess_pipeline.py
  → 브라우저에서 http://127.0.0.1:7860 접속

【처리 순서 (10단계)】
  원본(RAW) → 허프변환 리사이즈 → 그레이+샤프닝 → 엣지+ROI
  → 원검출 → 바늘ROI → 바늘마스크 → 바늘엣지 → 직선검출 → 최종 바늘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
import sys
import glob
import io
import contextlib
import tempfile
import gradio as gr

# ──────────────────────────────────────────────
# 기본값 (Gradio 슬라이더로 실시간 조정 가능)
# ──────────────────────────────────────────────
CIRCLE_PARAM2 = 40
HOUGH_SHORT_SIDE = 480
OUTPUT_SIZE = 256
# ──────────────────────────────────────────────


# ══════════════════════════════════════════════
# 단계별 처리 함수
# ══════════════════════════════════════════════

def step1_load_raw(image_path):
    """
    [1단계] 원본 이미지 로드
    - BGR 형식으로 읽어서 RGB로 변환 후 반환
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"  원본 이미지 크기: {bgr.shape[1]}×{bgr.shape[0]} 픽셀")
    return bgr, rgb


def step2_hough_resize(bgr):
    """
    [2단계] 허프변환용 리사이즈
    - 짧은 변 기준으로 HOUGH_SHORT_SIDE 픽셀에 맞게 리사이즈
    - 너무 큰 이미지: 축소 (속도 향상)
    - 너무 작은 이미지: 확대 (검출 정확도 향상)
    """
    h, w = bgr.shape[:2]
    scale = HOUGH_SHORT_SIDE / min(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # 축소 시 INTER_AREA, 확대 시 INTER_LINEAR 사용
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
    direction = "축소" if scale < 1.0 else "확대"
    print(f"  리사이즈 후: {new_w}×{new_h} 픽셀 (비율: {scale:.2f}배, {direction})")
    return resized


def step3_gray_sharpening(bgr_small):
    """
    [3단계] 그레이스케일 변환 + 샤프닝
    - 그레이스케일: 컬러 정보 제거, 처리 속도 향상
    - CLAHE: 명암 대비 향상 (게이지 눈금 선명하게)
    - Unsharp Masking: 경계선 샤프닝 (바늘/원 검출 정확도 향상)
    """
    # 그레이스케일 변환
    gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # → 국소 영역별로 명암 대비를 자동으로 맞춰줌
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Unsharp Masking (샤프닝)
    # → 원본에서 흐린 버전을 빼서 경계를 강조
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    print(f"  그레이스케일 + CLAHE + 샤프닝 완료")
    return gray, sharp


def step4_edge_roi(sharp):
    """
    [4단계] 엣지 검출 + 원형 ROI 마스킹
    - Canny 엣지: 이미지의 경계선만 남김
    - 원형 ROI 마스크: 게이지 중앙부만 남기고 외부 잡음 제거
    """
    h, w = sharp.shape[:2]
    cx, cy = w // 2, h // 2

    # Canny 엣지 검출 (임계값 자동 계산)
    med = np.median(sharp)
    low  = int(max(0,   0.66 * med))
    high = int(min(255, 1.33 * med))
    edges = cv2.Canny(sharp, low, high)

    # 원형 ROI 마스크 생성 (중앙 48% 반경)
    roi_mask = np.zeros_like(sharp)
    roi_r = int(min(h, w) * 0.48)
    cv2.circle(roi_mask, (cx, cy), roi_r, 255, -1)

    # 마스크 적용
    edges_roi = cv2.bitwise_and(edges, roi_mask)

    print(f"  Canny 엣지 (low={low}, high={high}), ROI 반경={roi_r}px")
    return edges, edges_roi, roi_mask


def step5_circle_detection(sharp, edges_roi):
    """
    [5단계] 허프 원검출 (게이지 외곽 원 찾기)
    - HoughCircles: 원형 패턴을 통계적으로 검출
    - 선택 기준: 이미지 중앙에 가장 가까운 원 (가장 큰 원 X)
      → 게이지는 항상 사진 중앙에 위치하기 때문
    반환: (cx, cy, r), 시각화이미지
    """
    h, w = sharp.shape[:2]
    img_cx, img_cy = w // 2, h // 2   # 이미지 중앙점

    # param2를 점진적으로 낮춰가며 원 검출 시도
    best_circle = None
    for p2 in [CIRCLE_PARAM2, CIRCLE_PARAM2 - 10, CIRCLE_PARAM2 - 20, 15]:
        if p2 < 10:
            break
        med = np.median(sharp)
        circles = cv2.HoughCircles(
            sharp,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(min(h, w) * 0.30),
            param1=int(min(255, 1.33 * med)),
            param2=p2,
            minRadius=int(min(h, w) * 0.28),
            maxRadius=int(min(h, w) * 0.49),
        )
        if circles is not None:
            cands = np.round(circles[0]).astype(int)

            # ★ 핵심: 이미지 중앙에서 가장 가까운 원 선택
            # (게이지는 사진 중앙에 위치하므로 중앙에 가까운 원이 정답)
            def center_dist(c):
                return math.hypot(c[0] - img_cx, c[1] - img_cy)

            best_circle = min(cands, key=center_dist)
            print(f"  원 검출 성공 (param2={p2}): "
                  f"후보 {len(cands)}개 중 중앙 기준 선택")
            break

    vis = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    if best_circle is not None:
        cx, cy, r = best_circle
        # 검출된 원이 중앙에서 너무 멀면 경고
        dist = math.hypot(cx - img_cx, cy - img_cy)
        if dist > min(h, w) * 0.20:
            print(f"  ⚠ 경고: 검출 원 중심이 이미지 중앙에서 {dist:.0f}px 떨어짐")
            print(f"    CIRCLE_PARAM2 값을 높여보세요 (현재: {CIRCLE_PARAM2})")

        cv2.circle(vis, (cx, cy), r, (0, 0, 255), 2)      # 빨간 원 (검출됨)
        cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)     # 초록 중심점
        cv2.circle(vis, (img_cx, img_cy), 3, (255, 255, 0), -1)  # 노란 이미지중앙
        print(f"  중심=({cx},{cy}), 반지름={r}px, "
              f"이미지중앙=({img_cx},{img_cy}), 거리={dist:.1f}px")
        return (cx, cy, r), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    else:
        # 검출 완전 실패 → 이미지 중앙을 기본값으로 사용
        cx, cy = img_cx, img_cy
        r = int(min(h, w) * 0.38)
        cv2.circle(vis, (cx, cy), r, (255, 165, 0), 2)    # 주황 (추정)
        cv2.circle(vis, (cx, cy), 4, (0, 200, 255), -1)
        print(f"  ⚠ 원 검출 완전 실패 → 이미지 중앙 사용: ({cx},{cy}), r={r}")
        print(f"    CIRCLE_PARAM2 값을 낮춰보세요 (현재: {CIRCLE_PARAM2} → 20 시도)")
        return (cx, cy, r), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def step6_needle_roi(bgr_small, cx, cy, r):
    """
    [6단계] 바늘 ROI 크롭
    - 검출된 원의 중심과 반지름으로 게이지 영역만 크롭
    - 정사각형으로 자르고 OUTPUT_SIZE로 리사이즈
    """
    h, w = bgr_small.shape[:2]
    pad = 1.15  # 원보다 15% 여유 공간 확보
    S = int(min(max(2 * r * pad, 32), 1.5 * min(h, w)))

    # 중심 기준으로 S×S 크롭
    crop = cv2.getRectSubPix(bgr_small, (S, S), (float(cx), float(cy)))
    resized = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
    gray_crop = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    print(f"  바늘 ROI 크롭: {S}×{S} → {OUTPUT_SIZE}×{OUTPUT_SIZE}px")
    return gray_crop, cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def step7_needle_mask(gray_crop):
    """
    [7단계] 바늘 마스크 생성
    - 이진화(Otsu)로 밝은 배경과 어두운 바늘 분리
    - 모폴로지 연산으로 잡음 제거
    """
    # Otsu 이진화: 임계값 자동 계산
    _, binary = cv2.threshold(gray_crop, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 모폴로지 (잡음 제거)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    print(f"  바늘 마스크 생성 (Otsu 이진화 + 모폴로지)")
    return mask


def step8_needle_edges(gray_crop, mask):
    """
    [8단계] 바늘 영역 엣지 검출
    - 마스크 내부에서만 엣지 검출 → 바늘 윤곽선만 추출
    """
    masked = cv2.bitwise_and(gray_crop, mask)
    edges = cv2.Canny(masked, 30, 100)
    print(f"  바늘 엣지 검출 완료")
    return edges


def step9_line_detection(edges, gray_crop):
    """
    [9단계] 허프 직선 검출 (바늘 방향 찾기)
    - HoughLinesP: 선분 형태의 직선 검출
    - 가장 긴 선분을 바늘로 선택
    반환: (angle_deg, 시각화이미지)
    """
    h, w = edges.shape[:2]
    min_len = int(min(h, w) * 0.15)
    max_gap = int(min(h, w) * 0.10)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )

    vis = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
    cx, cy = w // 2, h // 2
    # 중심 통과 허용 오차: 짧은 변의 20% 이내를 통과해야 바늘 후보
    center_pass_thresh = min(h, w) * 0.20
    best_angle = None
    best_tip_x, best_tip_y = None, None
    best_len = 0
    skipped = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # ★ 핵심 필터: 바늘은 반드시 피벗(이미지 중심) 근처를 통과한다.
            # 눈금선·원호는 중심에서 멀리 떨어져 있으므로 제거된다.
            # 점(x1,y1)-(x2,y2) 직선과 중심(cx,cy) 사이의 수직 거리 계산
            dx = x2 - x1
            dy = y2 - y1
            seg_len = math.hypot(dx, dy)
            if seg_len == 0:
                continue
            perp_dist = abs(dy * cx - dx * cy + x2 * y1 - y2 * x1) / seg_len
            if perp_dist > center_pass_thresh:
                skipped += 1
                cv2.line(vis, (x1, y1), (x2, y2), (80, 80, 80), 1)   # 회색: 무시된 선
                continue

            length = seg_len
            if length > best_len:
                best_len = length

                # 중심에서 더 먼 끝점 = 바늘 끝(tip)
                dist1 = math.hypot(x1 - cx, y1 - cy)
                dist2 = math.hypot(x2 - cx, y2 - cy)
                tip_x, tip_y = (x1, y1) if dist1 > dist2 else (x2, y2)

                # 12시=0°, 시계방향 기준 각도
                theta = math.atan2(cy - tip_y, tip_x - cx)
                best_angle = math.degrees((math.pi / 2 - theta) % (2 * math.pi))
                best_tip_x, best_tip_y = tip_x, tip_y

            cv2.line(vis, (x1, y1), (x2, y2), (0, 180, 0), 1)   # 초록: 후보 선

        if best_angle is not None:
            cv2.circle(vis, (best_tip_x, best_tip_y), 5, (0, 0, 255), -1)
            cv2.arrowedLine(vis, (cx, cy), (best_tip_x, best_tip_y),
                            (255, 100, 0), 2, cv2.LINE_AA, tipLength=0.1)
            print(f"  직선 {len(lines)}개 검출 (중심 통과 필터로 {skipped}개 제거)")
            print(f"  바늘 끝점=({best_tip_x},{best_tip_y}), 각도={best_angle:.1f}°")
        else:
            # 필터가 너무 엄격하면 오차를 넓혀서 재시도
            print(f"  ⚠ 중심 통과 후보 없음 → 오차 범위 확대 후 재시도")
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)
                if length > best_len:
                    best_len = length
                    dist1 = math.hypot(x1 - cx, y1 - cy)
                    dist2 = math.hypot(x2 - cx, y2 - cy)
                    tip_x, tip_y = (x1, y1) if dist1 > dist2 else (x2, y2)
                    theta = math.atan2(cy - tip_y, tip_x - cx)
                    best_angle = math.degrees((math.pi / 2 - theta) % (2 * math.pi))
                    best_tip_x, best_tip_y = tip_x, tip_y
            if best_angle is not None:
                cv2.circle(vis, (best_tip_x, best_tip_y), 5, (0, 165, 255), -1)
                cv2.arrowedLine(vis, (cx, cy), (best_tip_x, best_tip_y),
                                (255, 100, 0), 2, cv2.LINE_AA, tipLength=0.1)
                print(f"  (폴백) 바늘 끝점=({best_tip_x},{best_tip_y}), 각도={best_angle:.1f}°")
    else:
        print(f"  직선 검출 실패")

    return best_angle, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def step10_final_result(gray_crop, best_angle):
    """
    [10단계] 최종 바늘 이미지
    - 검출된 각도로 바늘 화살표를 그려서 결과 시각화
    """
    h, w = gray_crop.shape[:2]
    cx, cy = w // 2, h // 2
    R = int(min(h, w) * 0.42)

    vis = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)

    if best_angle is not None:
        angle = best_angle % 360

        # 12시=0°, 시계방향 → 화면 좌표계 변환
        # 수학 좌표: 오른쪽=0°, 위=90° → 화면: 오른쪽=0°, 아래=90°
        # 게이지: 12시=0°, 시계방향 → sin으로 x, -cos로 y 계산
        tip_x = int(cx + math.sin(math.radians(angle)) * R)
        tip_y = int(cy - math.cos(math.radians(angle)) * R)

        # 바늘 반대쪽 꼬리 (짧게)
        tail_x = int(cx - math.sin(math.radians(angle)) * R * 0.2)
        tail_y = int(cy + math.cos(math.radians(angle)) * R * 0.2)

        # 꼬리 → 끝점 방향으로 화살표 (바늘 끝 방향)
        cv2.arrowedLine(vis, (tail_x, tail_y), (tip_x, tip_y),
                        (0, 220, 80), 3, cv2.LINE_AA, tipLength=0.08)
        cv2.circle(vis, (cx, cy), 5, (0, 220, 80), -1)
        cv2.putText(vis, f"{angle:.1f}deg",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 80), 2, cv2.LINE_AA)
        print(f"  최종 바늘 각도: {angle:.1f}°")

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════
# 파이프라인 전체 실행 + 시각화
# ══════════════════════════════════════════════

def run_pipeline(image_path):
    """전처리 파이프라인 10단계 실행"""

    print(f"\n{'='*50}")
    print(f" 전처리 파이프라인 시작: {os.path.basename(image_path)}")
    print(f"{'='*50}")

    # ── 단계별 실행 ──
    print("\n[1] 원본 이미지 로드")
    bgr_raw, rgb_raw = step1_load_raw(image_path)

    print("\n[2] 허프변환 리사이즈")
    bgr_small = step2_hough_resize(bgr_raw)
    rgb_small = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)

    print("\n[3] 그레이스케일 + 샤프닝")
    gray, sharp = step3_gray_sharpening(bgr_small)

    print("\n[4] 엣지 검출 + ROI 마스킹")
    edges, edges_roi, roi_mask = step4_edge_roi(sharp)

    print("\n[5] 허프 원검출")
    circle_info, circle_vis = step5_circle_detection(sharp, edges_roi)
    cx, cy, r = circle_info

    print("\n[6] 바늘 ROI 크롭")
    gray_crop, rgb_crop = step6_needle_roi(bgr_small, cx, cy, r)

    print("\n[7] 바늘 마스크 생성")
    needle_mask = step7_needle_mask(gray_crop)

    print("\n[8] 바늘 엣지 검출")
    needle_edges = step8_needle_edges(gray_crop, needle_mask)

    print("\n[9] 허프 직선 검출")
    best_angle, line_vis = step9_line_detection(needle_edges, gray_crop)

    print("\n[10] 최종 바늘 이미지 생성")
    final_vis = step10_final_result(gray_crop, best_angle)

    print(f"\n{'='*50}")
    print(" 파이프라인 완료!")
    print(f"{'='*50}\n")

    return {
        "1_raw":        rgb_raw,
        "2_resized":    rgb_small,
        "3_gray_sharp": sharp,
        "4_edges_roi":  edges_roi,
        "5_circle":     circle_vis,
        "6_needle_roi": rgb_crop,
        "7_needle_mask":needle_mask,
        "8_needle_edge":needle_edges,
        "9_lines":      line_vis,
        "10_final":     final_vis,
    }


def setup_korean_font():
    """Windows/Linux 환경에서 한글 폰트 자동 설정"""
    import matplotlib.font_manager as fm
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",          # Windows 맑은 고딕
        "C:/Windows/Fonts/gulim.ttc",           # Windows 굴림
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in candidates:
        if os.path.isfile(fp):
            fm.fontManager.addfont(fp)
            font_name = fm.FontProperties(fname=fp).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    # 한글 폰트 없으면 영문 타이틀로 fallback
    plt.rcParams["font.family"] = "DejaVu Sans"
    return None


def visualize_pipeline(results, image_path):
    """10단계 결과를 2행 5열 그리드로 시각화 — matplotlib Figure 반환"""

    font_name = setup_korean_font()

    if font_name:
        titles = [
            "원본 (RAW)",
            "1. 허프변환 리사이즈",
            "2. 그레이+샤프닝",
            "3. 엣지+ROI",
            "4. 원검출",
            "4b. 바늘 ROI 크롭",
            "5. 바늘 마스크",
            "6. 바늘 엣지",
            "7. 직선검출",
            "8. 최종 바늘",
        ]
        main_title = f"전처리 파이프라인 (10단계)  |  {os.path.basename(image_path)}"
    else:
        titles = [
            "RAW",
            "1. Hough Resize",
            "2. Gray+Sharp",
            "3. Edge+ROI",
            "4. Circle Det.",
            "4b. Needle ROI",
            "5. Needle Mask",
            "6. Needle Edge",
            "7. Line Det.",
            "8. Final Needle",
        ]
        main_title = f"Preprocessing Pipeline (10 Steps)  |  {os.path.basename(image_path)}"

    keys = list(results.keys())

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(main_title, fontsize=13, fontweight="bold",
                 color="white", y=1.005)

    for i, ax in enumerate(axes.flat):
        img = results[keys[i]]
        ax.set_facecolor("#0f0f0f")

        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)

        ax.set_title(titles[i], fontsize=10, pad=5,
                     color="white",
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor="#333355", alpha=0.7))
        ax.axis("off")
        ax.text(0.02, 0.97, str(i + 1), transform=ax.transAxes,
                fontsize=9, color="#aaaaff", va="top", fontweight="bold")

    plt.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════════════════════
# Gradio 처리 함수
# ══════════════════════════════════════════════

def gradio_process(image, circle_param2, hough_short_side, output_size):
    """
    Gradio에서 호출되는 메인 처리 함수.
    image: numpy array (RGB) — Gradio가 자동 변환
    반환: (matplotlib Figure, 로그 텍스트)
    """
    global CIRCLE_PARAM2, HOUGH_SHORT_SIDE, OUTPUT_SIZE

    if image is None:
        return None, "이미지를 업로드해주세요."

    CIRCLE_PARAM2 = int(circle_param2)
    HOUGH_SHORT_SIDE = int(hough_short_side)
    OUTPUT_SIZE = int(output_size)

    # 업로드된 numpy 배열을 임시 파일로 저장 (run_pipeline이 파일 경로를 요구)
    import PIL.Image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        PIL.Image.fromarray(image).save(tmp_path)

    log_buffer = io.StringIO()
    fig = None
    try:
        with contextlib.redirect_stdout(log_buffer):
            results = run_pipeline(tmp_path)
        fig = visualize_pipeline(results, tmp_path)
    except Exception as e:
        log_buffer.write(f"\n[오류] {e}")
    finally:
        os.unlink(tmp_path)

    return fig, log_buffer.getvalue()


# ══════════════════════════════════════════════
# Gradio 인터페이스
# ══════════════════════════════════════════════

def build_interface():
    with gr.Blocks(title="게이지 전처리 파이프라인", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 게이지 전처리 파이프라인 (10단계)\n"
                    "이미지를 업로드하면 전처리 과정을 단계별로 시각화합니다.")

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(
                    type="numpy",
                    label="이미지 업로드 (jpg / png)",
                    height=300,
                )

                with gr.Accordion("파라미터 조정", open=False):
                    param2_slider = gr.Slider(
                        minimum=10, maximum=80, value=CIRCLE_PARAM2, step=1,
                        label="원검출 민감도 (CIRCLE_PARAM2) — 낮을수록 잘 검출",
                    )
                    short_side_slider = gr.Slider(
                        minimum=160, maximum=1024, value=HOUGH_SHORT_SIDE, step=16,
                        label="허프 리사이즈 짧은 변 (px)",
                    )
                    output_size_slider = gr.Slider(
                        minimum=64, maximum=512, value=OUTPUT_SIZE, step=16,
                        label="최종 출력 크기 (px)",
                    )

                run_btn = gr.Button("파이프라인 실행", variant="primary")

            with gr.Column(scale=3):
                fig_output = gr.Plot(label="10단계 파이프라인 결과")

        log_output = gr.Textbox(
            label="처리 로그",
            lines=12,
            max_lines=20,
            interactive=False,
        )

        # 버튼 클릭 시 실행
        run_btn.click(
            fn=gradio_process,
            inputs=[img_input, param2_slider, short_side_slider, output_size_slider],
            outputs=[fig_output, log_output],
        )

        # 이미지 업로드 즉시 자동 실행
        img_input.change(
            fn=gradio_process,
            inputs=[img_input, param2_slider, short_side_slider, output_size_slider],
            outputs=[fig_output, log_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(inbrowser=True)
