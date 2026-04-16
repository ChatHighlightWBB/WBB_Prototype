# services/highlight_service.py
import json
import os
import ffmpeg

# 가중치 설정 (합이 1.0이 되도록)
# 프로토타입이라 채팅 빈도에 가중치 높게 줌
# 본개발에서 KoBERT 감정 분류 추가되면 조정 필요
WEIGHT_CHAT = 0.5      # 채팅 빈도
WEIGHT_AUDIO = 0.3     # 오디오 RMS
WEIGHT_PIXEL = 0.2     # 픽셀 변화량

# 하이라이트 판정 임계값 (0~1 사이, 높을수록 엄격)
HIGHLIGHT_THRESHOLD = 0.3

# 클리핑 버퍼 타임 (앞뒤로 붙이는 여유 시간, 초)
BUFFER_TIME = 3

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize(values: list):
    """
    리스트를 0~1 사이로 정규화.
    최솟값 0, 최댓값 1로 변환.
    """
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [0.0] * len(values)
    return [(v - min_v) / (max_v - min_v) for v in values]

def calculate_weighted_score(analysis_id: str):
    """
    chat_data.json + audio_data.json + pixel_data.json 합쳐서
    초(second)별 가중치 점수 계산.
    """
    session_dir = os.path.join("uploads", analysis_id)
    
    chat_path = os.path.join(session_dir, "chat_data.json")
    audio_path = os.path.join(session_dir, "audio_data.json")
    pixel_path = os.path.join(session_dir, "pixel_data.json")

    # 파일 존재 확인
    for p in [chat_path, audio_path, pixel_path]:
        if not os.path.exists(p):
            print(f"--- [WBB] 파일 없음: {p} ---")
            return None

    chat_data = load_json(chat_path)
    audio_data = load_json(audio_path)
    pixel_data = load_json(pixel_path)

    # 채팅 빈도: 텍스트 길이를 점수로 사용
    chat_scores = [len(item["text"]) for item in chat_data]

    # 오디오 RMS
    audio_scores = [item["rms"] for item in audio_data]

    # 픽셀 변화량
    pixel_scores = [item["pixel_diff"] for item in pixel_data]

    # 길이 맞추기 (가장 짧은 것 기준)
    min_len = min(len(chat_scores), len(audio_scores), len(pixel_scores))
    chat_scores = chat_scores[:min_len]
    audio_scores = audio_scores[:min_len]
    pixel_scores = pixel_scores[:min_len]

    # 정규화
    chat_norm = normalize(chat_scores)
    audio_norm = normalize(audio_scores)
    pixel_norm = normalize(pixel_scores)

    # 가중치 합산
    final_scores = []
    for i in range(min_len):
        score = (
            chat_norm[i] * WEIGHT_CHAT +
            audio_norm[i] * WEIGHT_AUDIO +
            pixel_norm[i] * WEIGHT_PIXEL
        )
        final_scores.append({
            "second": i,
            "score": round(score, 4),
            "chat": round(chat_norm[i], 4),
            "audio": round(audio_norm[i], 4),
            "pixel": round(pixel_norm[i], 4)
        })

    # 결과 저장
    score_path = os.path.join(session_dir, "highlight_scores.json")
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(final_scores, f, ensure_ascii=False, indent=4)

    print(f"--- [WBB] 점수 계산 완료: {score_path} ---")
    return final_scores

def detect_highlight_segments(final_scores: list):
    """
    점수가 임계값 이상인 구간을 하이라이트로 판정.
    연속된 구간은 하나로 합침.
    """
    segments = []
    in_highlight = False
    start = 0

    for item in final_scores:
        second = item["second"]
        score = item["score"]

        if score >= HIGHLIGHT_THRESHOLD and not in_highlight:
            # 하이라이트 시작
            start = max(0, second - BUFFER_TIME)
            in_highlight = True

        elif score < HIGHLIGHT_THRESHOLD and in_highlight:
            # 하이라이트 끝
            end = second + BUFFER_TIME
            segments.append({"start": start, "end": end})
            in_highlight = False

    # 마지막 구간 처리
    if in_highlight:
        end = final_scores[-1]["second"] + BUFFER_TIME
        segments.append({"start": start, "end": end})

    print(f"--- [WBB] 하이라이트 구간 {len(segments)}개 탐지 ---")
    for seg in segments:
        print(f"    {seg['start']}초 ~ {seg['end']}초")

    return segments

def clip_highlights(analysis_id: str, video_path: str, segments: list):
    """
    FFmpeg Stream Copy로 하이라이트 구간 추출 후 병합.
    재인코딩 없이 빠르게 처리.
    """
    session_dir = os.path.join("uploads", analysis_id)
    clips_dir = os.path.join(session_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    clip_paths = []

    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        clip_path = os.path.join(clips_dir, f"clip_{i:03d}.mp4")

        print(f"--- [WBB] 클립 {i+1} 추출 중: {start}초 ~ {end}초 ---")

        try:
            (
                ffmpeg
                .input(video_path, ss=start, t=duration)
                # Stream Copy: 재인코딩 없이 추출 (빠름, 화질 손실 없음)
                .output(clip_path, c="copy")
                .overwrite_output()
                .run(quiet=True)
            )
            clip_paths.append(clip_path)
            print(f"    저장 완료: {clip_path}")

        except Exception as e:
            print(f"    클립 추출 실패: {str(e)}")

    if not clip_paths:
        print("--- [WBB] 추출된 클립 없음 ---")
        return None

    # 클립 병합
    output_path = os.path.join(session_dir, "highlight.mp4")
    print(f"--- [WBB] 클립 병합 시작 ({len(clip_paths)}개) ---")

    try:
        # 클립 목록 파일 생성 (FFmpeg concat 방식)
        concat_path = os.path.join(clips_dir, "concat.txt")
        with open(concat_path, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{os.path.abspath(cp)}'\n")

        (
            ffmpeg
            .input(concat_path, format="concat", safe=0)
            .output(output_path, c="copy")
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"--- [WBB] 하이라이트 영상 생성 완료: {output_path} ---")
        return output_path

    except Exception as e:
        print(f"--- [WBB] 병합 실패: {str(e)} ---")
        return None

def run_highlight_pipeline(analysis_id: str, video_path: str):
    """
    전체 하이라이트 생성 파이프라인 실행.
    1. 점수 계산
    2. 구간 탐지
    3. 클리핑 + 병합
    """
    print(f"--- [WBB] 하이라이트 파이프라인 시작 ---")

    # 1. 점수 계산
    final_scores = calculate_weighted_score(analysis_id)
    if not final_scores:
        return None

    # 2. 구간 탐지
    segments = detect_highlight_segments(final_scores)
    if not segments:
        print("--- [WBB] 탐지된 하이라이트 없음. 임계값 낮추는 것 고려 ---")
        return None

    # 3. 클리핑 + 병합
    output_path = clip_highlights(analysis_id, video_path, segments)
    return output_path