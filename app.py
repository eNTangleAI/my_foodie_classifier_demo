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
# Neumorphism CSS (업로더 통합)
# ---------------------------
st.markdown("""
<style>
/* 전체 배경 */
.stApp {
    background: #ECECEC;
    color: #333;
    font-family: 'Segoe UI', sans-serif;
}

/* 공통 Neumorphism 카드 */
.neu-card {
    background: #ECECEC;
    border-radius: 20px;
    box-shadow: 8px 8px 16px #c5c5c5,
                -8px -8px 16px #ffffff;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* 제목 */
.main-title {
    font-size: 2.2rem;
    color: #4A4A4A;
    text-align: center;
    font-weight: bold;
    margin-bottom: 1.5rem;
}

/* 경고 박스 */
.warning-box {
    background: #ECECEC;
    border-radius: 20px;
    box-shadow: inset 6px 6px 12px #c5c5c5,
                inset -6px -6px 12px #ffffff;
    padding: 1.5rem;
    margin: 1rem 0;
    font-size: 1rem;
    color: #555;
}

/* ---------------------------
   파일 업로더 Neumorphism 스타일
--------------------------- */
div[data-testid="stFileUploaderDropzone"] {
    background: #ECECEC !important;
    border-radius: 20px !important;
    border: none !important;
    padding: 2rem !important;
    box-shadow: inset 6px 6px 12px #c5c5c5,
                inset -6px -6px 12px #ffffff !important;
    color: #333 !important;
}
div[data-testid="stFileUploaderDropzone"] label {
    color: #333 !important;
}
div[data-testid="stFileUploaderDropzone"] p {
    color: #555 !important;
    font-size: 0.9rem !important;
}

/* 업로더 버튼 */
div[data-testid="stFileUploaderDropzone"] button {
    background: #ECECEC !important;
    color: #333 !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    border: none !important;
    font-weight: 500 !important;
    box-shadow: 6px 6px 12px #c5c5c5,
                -6px -6px 12px #ffffff !important;
    transition: all 0.2s ease-in-out;
}
div[data-testid="stFileUploaderDropzone"] button:hover {
    box-shadow: inset 6px 6px 12px #c5c5c5,
                inset -6px -6px 12px #ffffff !important;
}

/* 음식 목록 */
.food-grid {
    background: #ECECEC;
    border-radius: 20px;
    box-shadow: inset 6px 6px 12px #c5c5c5,
                inset -6px -6px 12px #ffffff;
    padding: 1.5rem;
    margin: 1rem 0;
    text-align: center;
}
.food-item {
    display: inline-block;
    margin: 6px;
    padding: 6px 12px;
    border-radius: 12px;
    background: #ECECEC;
    box-shadow: inset 4px 4px 8px #c5c5c5,
                inset -4px -4px 8px #ffffff;
    font-size: 0.9rem;
    color: #333;
}

/* 결과 카드 */
.result-success {
    background: #ECECEC;
    border-radius: 20px;
    padding: 1rem;
    font-size: 1.2rem;
    font-weight: bold;
    text-align: center;
    color: #333;
    box-shadow: inset 6px 6px 12px #c5c5c5,
                inset -6px -6px 12px #ffffff;
}
.food-card {
    background: #ECECEC;
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 6px 6px 12px #c5c5c5,
                -6px -6px 12px #ffffff;
}
.food-card h4 {
    color: #4A4A4A;
    margin-bottom: 1rem;
}
.food-card p, .food-card strong {
    color: #333;
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
<div class="warning-box">
⚠️ <strong>데모 버전 안내</strong><br>
이 앱은 인공지능을 활용한 음식 이미지 분류기입니다.<br>
총 50가지 음식을 분류할 수 있으며,<br>
칼로리 / 영양소 / 추천 페어링 / 음악 / 영화 정보를 제공합니다.
</div>
""", unsafe_allow_html=True)

# 3) 업로더
uploaded_file = st.file_uploader("📸 이미지 업로드", type=["jpg","png","jpeg"])

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

# ---------------------------
# 업로드된 이미지 + 추론 결과
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

# ---------------------------
# 음식 목록 (마지막)
# ---------------------------
st.markdown('<div class="subtitle">🍣 분류 가능한 50가지 음식</div>', unsafe_allow_html=True)

food_list_html = '<div class="food-grid">'
for food, emoji in food_emojis.items():
    display_name = food.replace('_', ' ').title()
    food_list_html += f'<span class="food-item">{emoji} {display_name}</span>'
food_list_html += '</div>'
st.markdown(food_list_html, unsafe_allow_html=True)
