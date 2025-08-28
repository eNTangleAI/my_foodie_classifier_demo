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
# 개선된 네오모피즘 CSS 스타일링
# ---------------------------
st.markdown("""
<style>
/* 전체 배경 */
.stApp {
    background: linear-gradient(135deg, #f0f0f3 0%, #e8e8ec 100%);
    color: #333;
    font-family: 'Segoe UI', sans-serif;
}

/* 메인 타이틀 */
.main-title {
    font-size: 2.2rem;
    background: linear-gradient(135deg, #9B7EBD, #B89DDB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
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

/* 경고 안내 박스 - 볼록한 효과 */
.warning-box {
    background: linear-gradient(145deg, #f5f5f8, #e6e6ea);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0 2rem 0;
    font-size: 0.9rem;
    color: #444;
    box-shadow: 12px 12px 24px rgba(181, 181, 181, 0.4),
                -12px -12px 24px rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.3);
}
.warning-box strong {
    color: #8B4513;
    font-weight: bold;
}

/* 파일 업로더 - 볼록한 효과 + 컬러 */
.stFileUploader > div {
    background: linear-gradient(145deg, #f2f2f5, #e8e8ec) !important;
    border-radius: 25px !important;
    padding: 2.5rem !important;
    margin: 1rem 0 2rem 0 !important;
    text-align: center !important;
    box-shadow: 15px 15px 30px rgba(163, 163, 163, 0.3),
                -15px -15px 30px rgba(255, 255, 255, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    transition: all 0.4s ease !important;
}

/* 드래그 오버 상태 - 은은한 컬러 추가 */
.stFileUploader > div:hover {
    background: linear-gradient(145deg, 
                rgba(155, 126, 189, 0.08), 
                rgba(184, 157, 219, 0.05)) !important;
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 18px 18px 35px rgba(155, 126, 189, 0.2),
                -18px -18px 35px rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(155, 126, 189, 0.2) !important;
}

/* 업로더 버튼 - 볼록한 효과 */
.stFileUploader button {
    background: linear-gradient(145deg, #f0f0f3, #e3e3e7) !important;
    border-radius: 20px !important;
    padding: 0.7rem 1.8rem !important;
    color: #555 !important;
    border: none !important;
    box-shadow: 8px 8px 16px rgba(163, 163, 163, 0.3),
                -8px -8px 16px rgba(255, 255, 255, 0.8) !important;
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
}
.stFileUploader button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 10px 10px 20px rgba(163, 163, 163, 0.4),
                -10px -10px 20px rgba(255, 255, 255, 0.9) !important;
    background: linear-gradient(145deg, #f2f2f5, #e5e5e9) !important;
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

/* 음식 목록 박스 - 볼록한 효과 + 은은한 컬러 */
.food-grid {
    background: linear-gradient(145deg, #f3f3f6, #e7e7eb);
    border-radius: 25px;
    padding: 2rem;
    margin: 0.5rem 0 2rem 0;
    text-align: center;
    box-shadow: 15px 15px 30px rgba(163, 163, 163, 0.3),
                -15px -15px 30px rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

.food-item {
    display: inline-block;
    margin: 6px;
    padding: 8px 14px;
    border-radius: 15px;
    background: linear-gradient(145deg, #f0f0f3, #e4e4e8);
    box-shadow: 6px 6px 12px rgba(163, 163, 163, 0.25),
                -6px -6px 12px rgba(255, 255, 255, 0.8);
    font-size: 0.9rem;
    color: #555;
    transition: all 0.3s ease;
    border: 1px solid rgba(255, 255, 255, 0.4);
}

.food-item:hover {
    transform: translateY(-1px);
    box-shadow: 8px 8px 16px rgba(163, 163, 163, 0.3),
                -8px -8px 16px rgba(255, 255, 255, 0.9);
    background: linear-gradient(145deg, 
                rgba(155, 126, 189, 0.05), 
                rgba(184, 157, 219, 0.03));
}

/* 결과 카드 - 볼록한 효과 + 성공 컬러 */
.result-success {
    background: linear-gradient(145deg, 
                rgba(240, 248, 240, 0.9), 
                rgba(230, 245, 230, 0.9));
    border-radius: 20px;
    padding: 1.2rem;
    font-size: 1.2rem;
    font-weight: bold;
    text-align: center;
    color: #4A7C59;
    box-shadow: 12px 12px 24px rgba(163, 163, 163, 0.3),
                -12px -12px 24px rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(208, 224, 208, 0.5);
    margin: 1rem 0;
}

/* 음식 정보 카드 - 볼록한 효과 + 은은한 컬러 */
.food-card {
    background: linear-gradient(145deg, 
                rgba(250, 250, 253, 0.95), 
                rgba(245, 245, 248, 0.95));
    border-radius: 25px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 15px 15px 30px rgba(163, 163, 163, 0.25),
                -15px -15px 30px rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.5);
}

.food-card h4 {
    color: #9B7EBD !important;
    margin-bottom: 1.5rem !important;
    font-size: 1.15rem !important;
    text-shadow: 1px 1px 2px rgba(155, 126, 189, 0.1);
}

.food-card p {
    color: #2E2E2E !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    margin-bottom: 0.8rem !important;
}

.food-card strong {
    color: #1A1A1A !important;
    font-weight: 600 !important;
}

/* 스피너 스타일 개선 */
.stSpinner {
    color: #9B7EBD !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 제목 & 안내
# ---------------------------
st.markdown('<div class="main-title">🍱 음식 이미지 분류기<br><small style="font-size: 0.7em; color: #999;">(ViT-B16 Demo)</small></div>', unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
    <strong style="color: #8B4513;">⚠️ 데모 버전 안내</strong><br>
    <span style="color: #555; font-size: 0.9rem;">이 앱은 인공지능을 활용한 음식 이미지 분류기입니다. 음식 사진을 업로드하면 ViT(Vision Transformer) 모델이 자동으로 음식의 종류를 식별하고, 해당 음식에 대한 상세 정보(칼로리, 영양소, 추천 페어링, 음악, 영화 등)를 카드 형태로 제공합니다. 휴대폰으로 찍은 음식 사진이나 인터넷에서 다운받은 이미지 모두 사용 가능하며, 총 50가지 다양한 음식을 정확하게 분류할 수 있습니다.</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# 파일 업로드
# ---------------------------
st.markdown('<div class="upload-title">📸 이미지 업로드</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

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
