import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import json
from huggingface_hub import hf_hub_download
from transformers import ViTForImageClassification
import os

# ---------------------------
# 페이지 설정
# ---------------------------
st.set_page_config(
    page_title="음식 이미지 분류기",
    page_icon="🍱",
    layout="centered"
)

# ---------------------------
# Glassmorphism CSS
# ---------------------------
st.markdown("""
<style>
/* 전체 배경 (컬러풀 그라데이션) */
.stApp {
    background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 30%, #fad0c4 70%, #fbc2eb 100%);
    color: #fff;
    font-family: 'Segoe UI', sans-serif;
    position: relative;
    overflow: hidden;
}

/* 블러 원형 장식 */
.stApp::before {
    content: "";
    position: absolute;
    width: 400px;
    height: 400px;
    top: -100px;
    left: -100px;
    background: radial-gradient(circle, rgba(255,255,255,0.4), transparent 70%);
    filter: blur(100px);
    border-radius: 50%;
}
.stApp::after {
    content: "";
    position: absolute;
    width: 500px;
    height: 500px;
    bottom: -150px;
    right: -150px;
    background: radial-gradient(circle, rgba(255,255,255,0.3), transparent 70%);
    filter: blur(120px);
    border-radius: 50%;
}

/* Glassmorphism 카드 */
.glass-card {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    box-shadow: 0 4px 30px rgba(0,0,0,0.1);
    padding: 1.5rem;
    margin: 1rem 0;
}

/* 제목 */
.main-title {
    font-size: 2.5rem;
    color: #fff;
    text-align: center;
    font-weight: bold;
    margin-bottom: 1.5rem;
    text-shadow: 0 3px 8px rgba(0,0,0,0.2);
}

/* 경고 박스 */
.warning-box {
    color: #fff;
    font-size: 1rem;
}

/* 업로더 */
.stFileUploader > div {
    background: rgba(255,255,255,0.2) !important;
    border-radius: 20px !important;
    border: 1px dashed rgba(255,255,255,0.4) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    padding: 2rem !important;
    margin: 1rem 0 2rem 0 !important;
    text-align: center !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
}

/* 버튼 */
.stFileUploader button {
    background: rgba(255,255,255,0.2) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.4) !important;
    border-radius: 20px !important;
    padding: 0.6rem 1.5rem !important;
    backdrop-filter: blur(8px) !important;
    box-shadow: 0 3px 8px rgba(0,0,0,0.15) !important;
    transition: all 0.2s ease-in-out;
}
.stFileUploader button:hover {
    background: rgba(255,255,255,0.3) !important;
    transform: translateY(-2px);
}

/* 음식 목록 */
.food-grid {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    padding: 1.5rem;
    margin: 1rem 0;
    text-align: center;
}
.food-item {
    display: inline-block;
    margin: 6px;
    padding: 8px 14px;
    border-radius: 12px;
    background: rgba(255,255,255,0.25);
    border: 1px solid rgba(255,255,255,0.4);
    backdrop-filter: blur(10px);
    font-size: 0.9rem;
    color: #fff;
}

/* 결과 카드 */
.result-success {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 1rem;
    font-size: 1.2rem;
    font-weight: bold;
    text-align: center;
    color: #fff;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.food-card {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(15px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.food-card h4 {
    color: #fff;
    margin-bottom: 1rem;
}
.food-card p, .food-card strong {
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 레이아웃 순서
# ---------------------------

# 1) 제목
st.markdown('<div class="main-title">🍱 음식 이미지 분류기<br><small style="font-size: 0.7em;">(ViT-B16 Demo)</small></div>', unsafe_allow_html=True)

# 2) 경고 박스
st.markdown("""
<div class="glass-card warning-box">
⚠️ <strong>데모 버전 안내</strong><br>
이 앱은 인공지능을 활용한 음식 이미지 분류기입니다.<br>
총 50가지 음식을 분류할 수 있으며,<br>
칼로리 / 영양소 / 추천 페어링 / 음악 / 영화 정보를 제공합니다.
</div>
""", unsafe_allow_html=True)

# 3) 업로더
st.markdown('<div class="upload-title">📸 이미지 업로드</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"], label_visibility="collapsed")

# 4) 음식 목록
st.markdown('<div class="subtitle">🍣 분류 가능한 50가지 음식</div>', unsafe_allow_html=True)

food_emojis = {
    'baklava': '🥧', 'beef_tartare': '🥩', 'beignets': '🍩', 'bibimbap': '🍚', 'bread_pudding': '🍞',
    'breakfast_burrito': '🌯', 'bruschetta': '🍞', 'cannoli': '🧁', 'caprese_salad': '🥗', 'ceviche': '🍤',
    'cheesecake': '🍰', 'chicken_curry': '🍛', 'chicken_wings': '🍗', 'churros': '🥨', 'clam_chowder': '🍲',
    'club_sandwich': '🥪', 'crab_cakes': '🦀', 'creme_brulee': '🍮', 'deviled_eggs': '🥚', 'dumplings': '🥟',
    'escargots': '🐌', 'falafel': '🧆', 'fried_calamari': '🦑', 'garlic_bread': '🍞', 'gnocchi': '🍝',
    'grilled_salmon': '🐟', 'guacamole': '🥑', 'gyoza': '🥟', 'hamburger': '🍔', 'hot_dog': '🌭',
    'hummus': '🫘', 'ice_cream': '🍦', 'macaroni_and_cheese': '🧀', 'macarons': '🍪', 'nachos': '🌮',
    'onion_rings': '🧅', 'oysters': '🦪', 'peking_duck': '🦆', 'pho': '🍜', 'pizza': '🍕',
    'ramen': '🍜', 'risotto': '🍚', 'shrimp_and_grits': '🍤', 'spaghetti_carbonara': '🍝', 'steak': '🥩',
    'sushi': '🍣', 'tacos': '🌮', 'takoyaki': '🐙', 'tiramisu': '🍰', 'waffles': '🧇'
}

food_list_html = '<div class="food-grid">'
for food, emoji in food_emojis.items():
    display_name = food.replace('_', ' ').title()
    food_list_html += f'<span class="food-item">{emoji} {display_name}</span>'
food_list_html += '</div>'
st.markdown(food_list_html, unsafe_allow_html=True)

# ---------------------------
# JSON 불러오기
# ---------------------------
with open("food_info.json", "r", encoding="utf-8") as f:
    food_info = json.load(f)
classes = list(food_info.keys())

# ---------------------------
# 모델 불러오기
# ---------------------------
@st.cache_resource
def load_model():
    repo_id = "eNtangedAI/my_foodie_classifier_demo"
    filename = "vit_best.pth"
    token = os.getenv("HF_TOKEN")
    try:
        weight_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    except Exception as e:
        st.error(f"❌ Hugging Face Hub에서 모델 다운로드 실패: {e}")
        st.stop()
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(classes),
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------------------
# 추론 실행
# ---------------------------
if uploaded_file is not None:
    input_img = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(input_img, caption="📷 업로드된 이미지", use_container_width=True)

    x = transform(input_img).unsqueeze(0)
    with st.spinner('🔍 이미지 분석 중...'):
        with torch.no_grad():
            outputs = model(x).logits
            pred = outputs.argmax(1).item()

    result = classes[pred]
    result_emoji = food_emojis.get(result, '🍽️')
    display_result = result.replace('_', ' ').title()

    st.markdown(f'<div class="result-success">{result_emoji} 예측 결과: {display_result}</div>', unsafe_allow_html=True)

    info = food_info.get(result, None)
    if info:
        st.markdown("""
        <div class="food-card">
            <h4>🍽️ 음식 정보 카드</h4>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        **🔥 칼로리:** {info['칼로리']}  
        **💪 주요 영양소:** {info['주요 영양소']}  
        **📝 설명:** {info['설명']}  
        **🥂 추천 페어링:** {", ".join(info['추천 페어링'])}  
        **🎵 추천 음악:** {info['추천 음악']}  
        **🎬 추천 영화:** {info['추천 영화']}
        """)

        st.markdown("</div>", unsafe_allow_html=True)
