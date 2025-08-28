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
# 네오모피즘 CSS 스타일링
# ---------------------------
st.markdown("""
<style>
/* =============================
   Global Neumorphism Style
   ============================= */

/* 전체 배경 */
.stApp {
    background: #EDEDED;  /* 밝은 회색 */
    color: #333;
    font-family: 'Segoe UI', sans-serif;
}

/* 메인 타이틀 */
.main-title {
    font-size: 2.2rem;
    color: #9B7EBD;
    text-align: center;
    margin-bottom: 1rem;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
}

/* 서브 타이틀 */
.subtitle, .upload-title {
    font-size: 1.1rem;
    color: #8A7CA8;
    text-align: center;
    margin: 1.5rem 0 1rem 0;
}

/* 네오모피즘 카드 공통 */
.neu-card {
    background: #EDEDED;
    border-radius: 20px;
    box-shadow: 8px 8px 16px #cacaca,
                -8px -8px 16px #ffffff;
    padding: 1.5rem;
    margin: 1rem 0 2rem 0;
}

/* 음식 목록 박스 */
.food-grid {
    background: #EDEDED;
    border-radius: 20px;
    padding: 1.5rem;
    margin: 0.5rem 0 2rem 0;
    box-shadow: inset 6px 6px 12px #cacaca,
                inset -6px -6px 12px #ffffff;
    text-align: center;
}
.food-item {
    display: inline-block;
    margin: 6px;
    padding: 6px 12px;
    border-radius: 12px;
    background: #EDEDED;
    box-shadow: inset 4px 4px 8px #cacaca,
                inset -4px -4px 8px #ffffff;
    font-size: 0.9rem;
    color: #555;
}

/* 경고 안내 박스 */
.warning-box {
    background: #EDEDED;
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0 2rem 0;
    font-size: 0.9rem;
    color: #444;
    box-shadow: 8px 8px 16px #cacaca,
                -8px -8px 16px #ffffff;
}
.warning-box strong {
    color: #8B4513;
    font-weight: bold;
}

/* 결과 카드 */
.result-success {
    background: #EDEDED;
    border-radius: 20px;
    padding: 1rem;
    font-size: 1.2rem;
    font-weight: bold;
    text-align: center;
    color: #4A7C59;
    box-shadow: inset 4px 4px 8px #cacaca,
                inset -4px -4px 8px #ffffff;
}

/* 음식 정보 카드 */
.food-card {
    background: #EDEDED;
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 8px 8px 16px #cacaca,
                -8px -8px 16px #ffffff;
}
.food-card h4 {
    color: #9B7EBD;
    margin-bottom: 1rem;
}
.food-card p, .food-card strong {
    color: #333;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* 파일 업로더 */
.stFileUploader > div {
    background: #EDEDED !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    margin: 1rem 0 2rem 0 !important;
    text-align: center !important;
    box-shadow: 8px 8px 16px #cacaca,
                -8px -8px 16px #ffffff !important;
    transition: all 0.3s ease !important;
}
.stFileUploader > div:hover {
    transform: scale(1.02);
}

/* 업로더 버튼 */
.stFileUploader button {
    background: #EDEDED !important;
    border-radius: 20px !important;
    padding: 0.6rem 1.5rem !important;
    color: #444 !important;
    border: none !important;
    box-shadow: 6px 6px 12px #cacaca,
                -6px -6px 12px #ffffff !important;
    transition: all 0.2s ease-in-out;
}
.stFileUploader button:hover {
    transform: translateY(-2px);
    box-shadow: 6px 6px 14px #b0b0b0,
                -6px -6px 14px #ffffff !important;
}

/* 업로더 안내 텍스트 */
.stFileUploader > div::before {
    content: "🍽️ 음식 이미지를 업로드하거나 드래그해주세요! 🤗";
    display: block;
    margin-bottom: 1rem;
    color: #666;
    font-size: 1rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 제목 & 안내
# ---------------------------
st.markdown('<div class="main-title">🍱 음식 이미지 분류기<br><small style="font-size: 0.7em; color: #999;">(ViT-B16 Demo)</small></div>', unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
    <strong>⚠️ 데모 버전 안내</strong><br>
    <span>이 앱은 인공지능을 활용한 음식 이미지 분류기입니다. 음식 사진을 업로드하면 ViT(Vision Transformer) 모델이 자동으로 음식의 종류를 식별하고, 해당 음식에 대한 상세 정보(칼로리, 영양소, 추천 페어링, 음악, 영화 등)를 카드 형태로 제공합니다. 총 50가지 다양한 음식을 분류할 수 있습니다.</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# 50가지 음식 리스트 표시
# ---------------------------
st.markdown('<div class="subtitle">분류 가능한 50가지 음식</div>', unsafe_allow_html=True)

# 음식 이모지 매핑
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

# 음식 목록 생성
food_list_html = '<div class="food-grid">'
foods = list(food_emojis.keys())
for food in foods:
    emoji = food_emojis.get(food, '🍽️')
    display_name = food.replace('_', ' ').title()
    food_list_html += f'<span class="food-item">{emoji} {display_name}</span>'
food_list_html += '</div>'
st.markdown(food_list_html, unsafe_allow_html=True)

# ---------------------------
# 파일 업로드
# ---------------------------
st.markdown('<div class="upload-title">📸 이미지 업로드</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

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
