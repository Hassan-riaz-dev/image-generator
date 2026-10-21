
import io
import streamlit as st
from groq import Groq
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom styling ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101124, #191a35);
    color: #ffffff;
}
.block-container {
    max-width: 1050px;
    padding-top: 2rem;
}
.hero {
    text-align: center;
    padding: 24px 12px;
}
.hero h1 {
    font-size: 2.7rem;
    font-weight: 800;
    color: #b9a5ff;
}
.hero p {
    color: #d0cde7;
    font-size: 1.05rem;
}
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    min-height: 46px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>🎨 AI Image Generator</h1>
    <p>Turn your imagination into beautiful AI-generated images.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Image Settings")

    style = st.selectbox(
        "Art Style",
        [
            "Photorealistic",
            "Digital Art",
            "Anime",
            "Oil Painting",
            "3D Render",
            "Watercolor",
            "Fantasy Art",
            "Cinematic",
            "Minimalist",
        ],
    )

    aspect_ratio = st.selectbox(
        "Image Shape",
        ["Square (1:1)", "Portrait (2:3)", "Landscape (3:2)"],
    )

    enhance_prompt = st.checkbox(
        "Enhance prompt with Groq",
        value=True,
    )

    st.caption("Image generation requires a Hugging Face token.")
    st.caption("Groq is used for optional prompt enhancement.")

# ---------- Main input ----------
st.subheader("✨ Describe Your Image")

prompt = st.text_area(
    "Enter your image prompt",
    placeholder=(
        "Example: A futuristic city at night, glowing neon lights, "
        "rainy streets, cinematic atmosphere..."
    ),
    height=130,
    max_chars=1500,
)

negative_prompt = st.text_input(
    "Things to avoid (optional)",
    placeholder="Blurry, low quality, distorted objects...",
)

generate = st.button(
    "🚀 Generate Image",
    type="primary",
    use_container_width=True,
)

def get_secret(name):
    try:
        value = st.secrets.get(name, "")
        return str(value).strip() if value else ""
    except Exception:
        return ""

def improve_prompt(user_prompt, art_style, groq_key):
    """Use Groq to turn a basic prompt into a detailed image prompt."""
    client = Groq(api_key=groq_key)

    response = client.chat.completions.create(
        
model="openai/gpt-oss-120b"

        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert text-to-image prompt engineer. "
                    "Convert the user's idea into one detailed, visual "
                    "prompt suitable for an image generation model. "
                    "Include composition, lighting, colors, atmosphere, "
                    "and artistic details where appropriate. "
                    "Preserve the user's original idea. "
                    "Return only the final image prompt."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original idea: {user_prompt}\n"
                    f"Art style: {art_style}\n"
                    "Create a detailed image-generation prompt."
                ),
            },
        ],
        temperature=0.7,
        max_tokens=300,
    )

    result = response.choices[0].message.content
    if not result or not result.strip():
        raise ValueError("Groq returned an empty prompt.")
    return result.strip()

# ---------- Generate image ----------
if generate:
    if not prompt.strip():
        st.warning("Please enter a description for your image.")
        st.stop()

    hf_token = get_secret("HF_TOKEN")
    groq_key = get_secret("GROQ_API_KEY")

    if not hf_token:
        st.error(
            "HF_TOKEN is missing. Add your Hugging Face token "
            "in Streamlit app Settings → Secrets."
        )
        st.stop()

    width, height = {
        "Square (1:1)": (1024, 1024),
        "Portrait (2:3)": (768, 1152),
        "Landscape (3:2)": (1152, 768),
    }[aspect_ratio]

    final_prompt = f"{prompt.strip()}, {style} art style"

    progress = st.empty()

    try:
        if enhance_prompt:
            if not groq_key:
                st.warning(
                    "GROQ_API_KEY is missing. Using your original "
                    "prompt instead. Add the key to enable enhancement."
                )
            else:
                with st.spinner("✨ Improving your prompt with Groq..."):
                    try:
                        final_prompt = improve_prompt(
                            prompt.strip(), style, groq_key
                        )
                    except Exception as exc:
                        st.warning(
                            "Prompt enhancement failed. "
                            "Using the original prompt. "
                            f"Details: {str(exc)[:250]}"
                        )
                        final_prompt = f"{prompt.strip()}, {style} art style"

        image_prompt = (
            f"{final_prompt}. "
            f"Visual style: {style}. "
            "High quality, carefully composed image."
        )

        if negative_prompt.strip():
            image_prompt += (
                f" Avoid these elements: {negative_prompt.strip()}."
            )

        with st.spinner(
            "🎨 Creating your image... This may take a little while."
        ):
            client = InferenceClient(
                provider="auto",
                api_key=hf_token,
                timeout=180,
            )

            image = client.text_to_image(
                prompt=image_prompt,
                model="black-forest-labs/FLUX.1-schnell",
                width=width,
                height=height,
            )

            image_buffer = io.BytesIO()
            image.save(image_buffer, format="PNG")
            image_bytes = image_buffer.getvalue()

        st.success("Your image is ready!")

        st.subheader("🖼️ Generated Image")
        st.image(image_bytes, use_container_width=True)

        st.download_button(
            label="⬇️ Download Image (PNG)",
            data=image_bytes,
            file_name="ai_generated_image.png",
            mime="image/png",
            use_container_width=True,
        )

        with st.expander("View the prompt used"):
            st.write(image_prompt)

    except Exception as exc:
        st.error("Image generation failed.")
        st.markdown(
            "Check your Hugging Face token, model availability, "
            "provider access, credits, and network connection."
        )
        st.code(str(exc)[:1500])

st.divider()
st.caption("AI Image Generator • Built with Streamlit, Groq & Hugging Face")
