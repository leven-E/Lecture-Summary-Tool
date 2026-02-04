# @title 👑 講義サマリー統合版 (Master v1.2 - Multimodal)
# @markdown ### ⚙️ 実行モード
# @markdown - `Fast_Gemini`: **【推奨】** 爆速。音声とスライド画像を同時に解析。
# @markdown - `Quality_Gemini`: 高精度。Geminiによる二段階処理（文字起こし → 要約）。GPU不要。

OPERATION_MODE = "Fast_Gemini" # @param ["Fast_Gemini", "Quality_Gemini"]
OUTPUT_STYLE = "\u8B1B\u7FA9\u30CE\u30FC\u30C8\u98A8" # @param ["講義ノート風", "レポート記事風"]

# @markdown ---
# @markdown ### 🔐 認証設定 (Secrets優先)
MANUAL_API_KEY = "" # @param {type:"string"}
MANUAL_USER = "team" # @param {type:"string"}
MANUAL_PASS = "pass123" # @param {type:"string"}

!apt-get install -y ffmpeg
!pip install -q -U google-generativeai gradio opencv-python

import os, cv2, time, shutil, glob, subprocess, numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
import google.generativeai as genai
from google.colab import userdata, drive
import gradio as gr

# --- 1. 認証 ---
try:
    API_KEY = userdata.get('GEMINI_API_KEY') or MANUAL_API_KEY
    APP_USER = userdata.get('APP_USER') or MANUAL_USER
    APP_PASS = userdata.get('APP_PASS') or MANUAL_PASS
    genai.configure(api_key=API_KEY)
except: raise ValueError("❌ APIキーを設定してください")

# --- 2. 環境準備 ---
print("📂 Google Driveを確認中...")
try: drive.mount('/content/drive', force_remount=True)
except: pass

TARGET_FOLDER = "/content/drive/MyDrive/Lecture_Videos"
BASE_DIR = Path("/content/lecture_workspace")
SLIDES_DIR = BASE_DIR / "slides"
JST = timezone(timedelta(hours=9), 'JST')

if not os.path.exists(TARGET_FOLDER): os.makedirs(TARGET_FOLDER, exist_ok=True)

# --- 3. プロンプト定義 ---

PROMPTS = {
    "transcribe": """
あなたはプロの文字起こし担当です。提供された講義音声の内容を、一言一句漏らさず正確にテキスト化してください。
タイムスタンプなどは不要です。純粋な発言内容のみを出力してください。
""",
    "note_style": """
あなたは優秀な学生です。提供された「文字起こしテキスト」と「講義スライド（画像）」を組み合わせて、後から見返した時に最も学習効率が高い「講義ノート」を作成してください。

### 構成案
1. **講義テーマ・目的**: この講義で何を学ぶのか
2. **スライドごとの詳細解説**: 
   - 提供された各スライド画像の内容と、それに対応する文字起こし部分を紐付けて解説。
   - 専門用語の定義、講師が強調していたポイントを逃さず記述。
3. **重要ポイントまとめ**: 試験に出そうな、または実務で重要な箇所の抜粋
4. **Q&A・補足**: 講義中にあった質問や、文脈から推測される補足情報

単なる要約ではなく、文脈を重視した「読み物としての質の高さ」を意識してください。
""",
    "report_style": """
あなたは敏腕のテックライターです。提供された「文字起こしテキスト」と「講義スライド（画像）」をベースに、講義に参加できなかった人でも内容が深く理解できる、体系的な「レポート記事」を作成してください。

### 執筆ガイドライン
- **タイトル**: 読者の目を引く、内容を凝縮したタイトル
- **導入**: 講義の背景と、なぜこのテーマが重要なのか
- **本論**: 各トピックを論理的な章立て（見出し）で構成。スライド画像の内容を適宜「図説」として文中に引用する形式で記述。
- **結論**: 全体の総括と、今後の展望や学びの活用方法
- **トーン**: 客観的でありつつ、読者が知的好奇心を刺激されるような文体

細切れの情報を統合し、一つの完成された記事として構成してください。
""",
    "fast_mode": """
提供された講義音声と、抽出されたスライド画像から、以下の形式で詳細な講義サマリーを作成してください。

1. **全体サマリー**: 講義の要旨（300文字程度）
2. **スライド別解説**: 各画像の内容と、そこで語られている重要な解説を紐付けて記述
3. **重要キーワード**: 出現した専門用語とその解説

単なる箇条書きではなく、スライドと音声の文脈を統合した解説文として記述してください。
出力形式は「{style}」に従ってください。
"""
}

# --- 4. コアロジック ---

def setup_dirs():
    if BASE_DIR.exists(): shutil.rmtree(BASE_DIR)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)

def extract_slides_stable(video_path):
    """重複排除を強化したスライド抽出"""
    setup_dirs()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    slides = []
    last_saved_time = -999
    
    SIMILARITY_THRESHOLD = 0.94
    MIN_GAP_SEC = 10

    print("🎬 スライド抽出中...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        curr_sec = curr_frame / fps
        
        if int(curr_frame) % int(fps) == 0:
            if curr_sec - last_saved_time < MIN_GAP_SEC: continue
            
            h, w, _ = frame.shape
            roi = frame[0:int(h*0.7), 0:w]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            save_this = False
            if not slides: save_this = True
            else:
                sim = cv2.compareHist(slides[-1]['hist'], hist, cv2.HISTCMP_CORREL)
                if sim < SIMILARITY_THRESHOLD: save_this = True
            
            if save_this:
                ts_str = f"{int(curr_sec//60):02d}m{int(curr_sec%60):02d}s"
                fname = f"slide_{ts_str}.jpg"
                p = str(SLIDES_DIR/fname)
                cv2.imwrite(p, frame)
                slides.append({"time": curr_sec, "hist": hist, "filename": fname, "path": p})
                last_saved_time = curr_sec
    cap.release()
    print(f"✅ {len(slides)}枚のスライドを抽出しました")
    return slides

def upload_to_gemini(path):
    f = genai.upload_file(path)
    while f.state.name == "PROCESSING":
        time.sleep(2)
        f = genai.get_file(f.name)
    return f

def process_fast_gemini(audio_path, slide_paths, style):
    print("⚡ Fast_Geminiモード実行中 (Multimodal)...")
    model = genai.GenerativeModel("gemini-1.5-pro")
    
    # ファイルアップロード
    audio_file = upload_to_gemini(audio_path)
    image_files = [upload_to_gemini(p) for p in slide_paths]
    
    prompt = PROMPTS["fast_mode"].format(style=style)
    content = [audio_file] + image_files + [prompt]
    
    res = model.generate_content(content)
    return res.text

def process_quality_gemini(audio_path, slide_paths, style):
    print("💎 Quality_Geminiモード実行中 (2-Step)...")
    model = genai.GenerativeModel("gemini-1.5-pro")
    
    # Step 1: 文字起こし
    print("  Step 1: 文字起こしを生成中...")
    audio_file = upload_to_gemini(audio_path)
    t_res = model.generate_content([audio_file, PROMPTS["transcribe"]])
    transcription = t_res.text
    
    # Step 2: 要約・整形
    print(f"  Step 2: {style}に整形中...")
    image_files = [upload_to_gemini(p) for p in slide_paths]
    
    base_prompt = PROMPTS["note_style"] if style == "講義ノート風" else PROMPTS["report_style"]
    final_prompt = f"{base_prompt}\n\n### 提供された文字起こしテキスト\n{transcription}"
    
    content = image_files + [final_prompt]
    res = model.generate_content(content)
    
    # 文字起こしログも保存
    with open(BASE_DIR/"transcription.txt", "w") as f: f.write(transcription)
    
    return res.text

def analyze(filename, style):
    if not filename or "(動画" in filename: return "⚠️ 未選択", [], None
    video_path = os.path.join(TARGET_FOLDER, filename)
    
    # 1. スライド抽出
    slides = extract_slides_stable(video_path)
    slide_paths = [s['path'] for s in slides]
    
    # 2. 音声抽出
    print(f"🔊 音声変換中...")
    audio_path = str(BASE_DIR/"audio.mp3")
    subprocess.run(["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-q:a", "4", "-y", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. モード別処理
    if OPERATION_MODE == "Quality_Gemini":
        result_text = process_quality_gemini(audio_path, slide_paths, style)
    else:
        result_text = process_fast_gemini(audio_path, slide_paths, style)

    # 4. 保存 & アーカイブ
    with open(BASE_DIR/"summary.md", "w") as m: m.write(result_text)
    
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    style_en = "Note" if style == "講義ノート風" else "Report"
    zip_name = f"{style_en}_{ts}"
    zip_p = f"/content/{zip_name}"
    shutil.make_archive(zip_p, 'zip', BASE_DIR)
    
    final_zip_name = f"{zip_name}_{filename}.zip"
    shutil.copy(f"{zip_p}.zip", os.path.join(TARGET_FOLDER, final_zip_name))
    
    return f"✅ 完了 ({style}): Driveに保存しました", [s['path'] for s in slides], f"{zip_p}.zip"

# --- 5. GUI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"## 👑 講義サマリー統合版 Master v1.2 (Multimodal)")
    with gr.Row():
        files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(TARGET_FOLDER, "*.*")) if f.lower().endswith(('.mp4','.mov','.m4a','.mp3'))])
        dd = gr.Dropdown(label="動画選択", choices=files, value=files[0] if files else None)
        style_dd = gr.Radio(label="出力形式", choices=["講義ノート風", "レポート記事風"], value="講義ノート風")
        btn_run = gr.Button("▶ 解析開始", variant="primary")
    with gr.Row():
        out_t = gr.Markdown(); out_f = gr.File(label="ダウンロード")
    out_g = gr.Gallery(label="抽出スライド", columns=4)
    
    btn_run.click(analyze, inputs=[dd, style_dd], outputs=[out_t, out_g, out_f])

demo.launch(share=True, auth=(APP_USER, APP_PASS))
