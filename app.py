import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import json
import requests
from io import BytesIO
from duckduckgo_search import DDGS  # pip install duckduckgo-search
from huggingface_hub import hf_hub_download
from transformers import ViTForImageClassification

# ---------------------------
# 제목 & 안내
# ---------------------------
st.title("🍱 음식 이미지 분류기 (ViT-B16 Demo + 검색 지원)")

st.markdown("""
⚠️ **데모 버전 안내**  
- 업로드한 이미지를 분류하거나,  
- 음식 이름을 검색해서 나온 이미지를 선택할 수 있습니다.  

모델 가중치는 Hugging Face Hub에서 자동으로 다운로드됩니다.
""")

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
    repo_id = "eNtangledAI/my_foodie_classifier_demo"   # 정확한 Hugging Face repo 이름
    filename = "vit_best.pth"                           # Hub에 올라간 weight 파일명

    # Hub에서 다운로드
   try:
        weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
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
# 파일 업로드 or 검색
# ---------------------------
tab1, tab2 = st.tabs(["📂 파일 업로드", "🔍 음식 검색"])

uploaded_file = None
selected_image = None

with tab1:
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

with tab2:
    query = st.text_input("검색할 음식 이름을 입력하세요 (예: ramen, pizza)")
    if query:
        with DDGS() as ddgs:
            results = [r for r in ddgs.images(query, max_results=4)]
        urls = [r["image"] for r in results]
        if urls:
            st.write("🔎 검색된 이미지 (클릭하여 선택):")
            cols = st.columns(len(urls))
            for i, url in enumerate(urls):
                try:
                    response = requests.get(url, timeout=5)
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    with cols[i]:
                        if st.button(f"선택 {i+1}"):
                            selected_image = img
                        st.image(img, use_container_width=True)
                except:
                    continue

# ---------------------------
# 입력 이미지 결정
# ---------------------------
input_img = None
if uploaded_file is not None:
    input_img = Image.open(uploaded_file).convert("RGB")
elif selected_image is not None:
    input_img = selected_image

# ---------------------------
# 추론 실행
# ---------------------------
if input_img is not None:
    st.image(input_img, caption="선택된 이미지", use_container_width=True)
    x = transform(input_img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x).logits
        pred = outputs.argmax(1).item()

    result = classes[pred]
    st.success(f"예측 결과: **{result}**")

    info = food_info.get(result, None)
    if info:
        with st.expander("🍽️ 음식 카드 펼쳐보기"):
            st.markdown(f"""
            **칼로리:** {info['칼로리']}  
            **주요 영양소:** {info['주요 영양소']}  
            **설명:** {info['설명']}  

            🥂 **추천 페어링:** {", ".join(info['추천 페어링'])}  
            🎵 **추천 음악:** {info['추천 음악']}  
            🎬 **추천 영화:** {info['추천 영화']}  
            """)
