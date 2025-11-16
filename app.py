# # ...existing code...
# import streamlit as st
# import torch
# import timm
# from PIL import Image
# from torchvision import transforms
# import torch.nn.functional as F

# # =============================
# # ⚙️ Cấu hình ban đầu
# # =============================
# st.set_page_config(page_title="Image Similarity with DINOv3", layout="wide")
# st.title("🧠 So sánh độ tương đồng giữa hai ảnh (DINOv3)")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# st.write(f"**Thiết bị sử dụng:** {device}")
# @st.cache_resource
# def load_model():
#     model_name="vit_huge_plus_patch16_dinov3"
#     # model_name = "vit_large_patch14_dinov3.lvd142m"
#     st.info(f"Đang tải mô hình {model_name}...")
#     model = timm.create_model(model_name, pretrained=True)
#     model.eval()
#     model.to(device)
#     return model

# model = load_model()

# # =============================
# # 🧩 Hàm tiền xử lý và lấy embedding
# # =============================
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # đảm bảo chia hết cho 16
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=(0.5, 0.5, 0.5),
#         std=(0.5, 0.5, 0.5)
#     )
# ])

# def get_embedding(img: Image.Image):
#     x = transform(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         feats = model.forward_features(x)
#     emb = feats.mean(dim=1).squeeze()  # (C,)
#     return emb.cpu()

# # =============================
# # 🖼️ Giao diện upload ảnh
# # =============================
# col1, col2 = st.columns(2)
# with col1:
#     img1_file = st.file_uploader("📂 Ảnh thứ nhất", type=["jpg", "jpeg", "png"])
# with col2:
#     img2_file = st.file_uploader("📂 Ảnh thứ hai", type=["jpg", "jpeg", "png"])

# # =============================
# # 🔍 Xử lý & hiển thị kết quả
# # =============================
# if img1_file and img2_file:
#     img1 = Image.open(img1_file).convert("RGB")
#     img2 = Image.open(img2_file).convert("RGB")

#     col1.image(img1, caption="Ảnh 1", width='stretch')
#     col2.image(img2, caption="Ảnh 2", width='stretch')

#     with st.spinner("Đang tính độ tương đồng..."):
#         emb1 = get_embedding(img1)
#         emb2 = get_embedding(img2)
#         similarity = F.cosine_similarity(emb1, emb2, dim=0).item()

#     st.success(f"🔹 **Độ tương đồng (Cosine Similarity): {similarity:.4f}**")

#     if similarity > 0.85:
#         st.markdown("✅ Hai ảnh **rất giống nhau**")
#     elif similarity > 0.5:
#         st.markdown("🟨 Hai ảnh **có nét tương đồng**")
#     else:
#         st.markdown("❌ Hai ảnh **khác nhau rõ rệt**")

# else:

#     st.info("👆 Hãy tải lên hai ảnh để so sánh.")



import streamlit as st
import base64
from openai import OpenAI

# DeepInfra OpenAI client
client = OpenAI(
    api_key="XDE02cttBlH48cdGoArXCTNRNPWoMlnt",
    base_url="https://api.deepinfra.com/v1/openai",
)

st.title("DeepInfra OCR-1B - Extract Markdown Information")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Convert image → base64
    img_bytes = uploaded_file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    if st.button("🔍 Extract Markdown Info"):
        with st.spinner("Processing..."):

            response = client.chat.completions.create(
                model="hoangngochongo3/OCR-3B",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Chuyển sang dạng text markdown với các phần tiêu đề, danh sách, bảng và đoạn văn bản từ hình ảnh được cung cấp bên dưới."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ]
            )

            extracted_markdown = response.choices[0].message.content

        st.subheader("📄 Extracted Markdown:")
        st.markdown(extracted_markdown)
#--trust_remote_code --tokenizer-mode=auto
