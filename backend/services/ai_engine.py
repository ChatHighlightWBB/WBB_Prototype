# services/ai_engine.py
import os
import json
import easyocr

# 최초 실행 시 모델 자동 다운로드
reader = easyocr.Reader(['ko', 'en'], gpu=True)

def run_ocr_on_frames(analysis_id: str):
    print(f"--- [WBB] OCR 엔진 초기화 (EasyOCR Mode) ---")

    processed_dir = f"uploads/{analysis_id}/processed"
    if not os.path.exists(processed_dir):
        print(f"--- [WBB] processed 폴더 없음: {processed_dir} ---")
        return None
        
    frame_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.jpg')])
    
    if not frame_files:
        print("--- [WBB] 분석할 프레임 없음 ---")
        return None
    
    ocr_results = []
    print(f"--- [WBB] 분석 시작 ({len(frame_files)}장) ---")

    for frame_file in frame_files:
        img_path = os.path.join(processed_dir, frame_file)
        
        try:
            # EasyOCR 인식 (결과: [[bbox, text, confidence], ...])
            result = reader.readtext(img_path)
            
            texts = []
            raw = []
            for (bbox, text, confidence) in result:
                if confidence >= 0.3:
                    texts.append(text)
                raw.append((text, round(confidence, 2)))
            
            full_text = " ".join(texts)
            ocr_results.append({
                "frame": frame_file,
                "text": full_text,
                "raw": raw
            })
            print(f"[{frame_file}] 인식 완료: {full_text[:30]}...")
            
        except Exception as e:
            print(f"[{frame_file}] OCR 실패: {str(e)[:50]}")
            ocr_results.append({
                "frame": frame_file,
                "text": "",
                "raw": []
            })

    result_path = f"uploads/{analysis_id}/chat_data.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=4)

    print(f"--- [WBB] 결과 저장 완료: {result_path} ---")
    return result_path