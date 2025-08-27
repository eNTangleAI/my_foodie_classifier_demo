import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import json
from huggingface_hub import hf_hub_download
from transformers import ViTForImageClassification
import os

# ---------------------------
# 페이지 설정 및 스타일링
# ---------------------------
st.set_page_config(
    page_title="음식 이미지 분류기",
    page_icon="🍱",
    layout="centered"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 2rem;
    }
    .upload-title {
        font-size: 1.1rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 2rem;
        font-weight: normal;
    }
    .food-grid {
        background: linear-gradient(135deg, #FFF5F5 0%, #F0F8FF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #FFE4E1;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .food-item {
        display: inline-block;
        margin: 3px 8px;
        padding: 4px 8px;
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #FFB6C1;
        font-size: 0.85rem;
        color: #666;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .warning-box {
        background: linear-gradient(135deg, #FFF8DC 0%, #F5F5DC 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #FFE4B5;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .warning-box strong {
        color: #8B4513 !important;
        font-size: 1rem;
        font-weight: bold;
    }
    .warning-box span {
        color: #555 !important;
        font-size: 0.9rem !important;
        line-height: 1.4;
    }
    .result-success {
        background: linear-gradient(135deg, #E8F5E8 0%, #F0FFF0 100%);
        padding: 1rem;
        border-radius: 15px;
        border: 2px solid #98FB98;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        color: #228B22;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .food-card {
        background: linear-gradient(135deg, #FFF0F5 0%, #F8F8FF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #DDA0DD;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    /* 파일 업로더 스타일링 */
    .stFileUploader > div {
        background: linear-gradient(135deg, #F0F8FF 0%, #E6E6FA 100%);
        border: 2px solid #B0C4DE;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stFileUploader > div > div {
        border: none !important;
        background: transparent !important;
    }
    .stFileUploader > div > div > div {
        color: #666 !important;
        font-size: 1rem !important;
    }
    .stFileUploader label {
        display: none !important;
    }
    /* Browse files 버튼 스타일링 */
    .stFileUploader button {
        background: linear-gradient(135deg, #FF9A9E 0%, #FECFEF 100%) !important;
        color: #666 !important;
        border: 2px solid #FFB6C1 !important;
        border-radius: 25px !important;
        padding: 0.5rem 1.5rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 8px rgba(255, 182, 193, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stFileUploader button:hover {
        background: linear-gradient(135deg, #FECFEF 0%, #FF9A9E 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(255, 182, 193, 0.4) !important;
    }
    /* 드래그 앤 드롭 텍스트 */
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
    <strong style="color: #8B4513;">⚠️ 데모 버전 안내</strong><br>
    <span style="color: #555; font-size: 0.9rem;">업로드한 이미지를 분류하여 음식 이름과 정보를 카드 형태로 보여줍니다.<br>
    모델 가중치는 Hugging Face Hub에서 자동으로 다운로드됩니다.</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# 50가지 음식 리스트 표시
# ---------------------------
st.markdown('<div class="subtitle">🎯 분류 가능한 50가지 음식</div>', unsafe_allow_html=True)

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
foods = [
    'baklava', 'beef_tartare', 'beignets', 'bibimbap', 'bread_pudding',
    'breakfast_burrito', 'bruschetta', 'cannoli', 'caprese_salad', 'ceviche',
    'cheesecake', 'chicken_curry', 'chicken_wings', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'deviled_eggs', 'dumplings',
    'escargots', 'falafel', 'fried_calamari', 'garlic_bread', 'gnocchi',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_dog',
    'hummus', 'ice_cream', 'macaroni_and_cheese', 'macarons', 'nachos',
    'onion_rings', 'oysters', 'peking_duck', 'pho', 'pizza',
    'ramen', 'risotto', 'shrimp_and_grits', 'spaghetti_carbonara', 'steak',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'waffles'
]

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
# 모델 불러오기 (Hugging Face Hub에서 가중치 다운로드)
# ---------------------------
@st.cache_resource
def load_model():
    repo_id = "eNtangedAI/my_foodie_classifier_demo"   # 정확한 Hugging Face repo 이름
    filename = "vit_best.pth"                          # Hub에 올라간 weight 파일명
    # Streamlit Secret에서 HF_TOKEN 가져오기
    token = os.getenv("HF_TOKEN")
    # Hub에서 다운로드
    try:
        weight_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    except Exception as e:
        st.error(f"❌ Hugging Face Hub에서 모델 다운로드 실패: {e}")
        st.stop()
        
    # ViT 모델 구조 정의
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(classes),
        ignore_mismatched_sizes=True
    )
    # 학습된 가중치 불러오기
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
    
    # 이미지를 중앙 정렬로 표시
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
    
    # 결과 표시
    st.markdown(f'<div class="result-success">{result_emoji} 예측 결과: {display_result}</div>', unsafe_allow_html=True)
    
    info = food_info.get(result, None)
    if info:
        st.markdown("""
        <div class="food-card">
            <h4 style="color: #8A2BE2; margin-bottom: 1rem;">🍽️ 음식 정보 카드</h4>
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
