
import io
import streamlit as st
from groq import Groq
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101124, #191a35);
    color: white;
}
.block-container {
    max-width: 1050px;
    padding-top: 2rem;
}
.hero {
    text-align: center;
    padding: 20px;
}
.hero h1 {
    font-size: 2.7rem;
    color: #b9a5ff;
}
.hero p {
    color: #d0cde7;
}
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    min-height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🎨 AI Image Generator</h1>
    <p>Turn your imagination into beautiful AI images.</p>
</div>
""", unsafe_allow_html=True)

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
            "Minimalist"
        ]
    )

    aspect_ratio = st.selectbox(
        "Image Shape",
        ["Square (1:1)", "Portrait (2:3)", "Landscape (3:2)"]
    )

    enhance_prompt = st.checkbox(
        "Enhance prompt with Groq",
        value=True
    )

    st.caption("Requires Groq and Hugging Face API keys.")

prompt = st.text_area(
    "✨ Describe your image",
    placeholder="A futuristic city at night with neon lights...",
    height=130,
    max_chars=1500
)

negative_prompt = st.text_input(
    "Things to avoid (optional)",
    placeholder="Blurry, low quality, distorted objects..."
)

generate = st.button(
    "🚀 Generate Image",
    type="primary",
    use_container_width=True
)


def get_secret(name):
    try:
        value = st.secrets.get(name, "")
        return str(value).strip() if value else ""
    except Exception:
        return ""


def improve_prompt(user_prompt, art_style, groq_key):
    client = Groq(api_key=groq_key)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert text-to-image prompt engineer. "
                    "Improve the user's idea into one detailed visual "
                    "prompt. Include composition, lighting, colors, "
                    "atmosphere, and artistic details. Preserve the "
                    "original idea. Return only the final image prompt."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Image idea: {user_prompt}\n"
                    f"Art style: {art_style}\n"
                    "Write a detailed image-generation prompt."
                )
            }
        ],
        temperature=0.7,
        max_completion_tokens=500
    )

    result = response.choices[0].message.content

    if not result or not result.strip():
        raise ValueError("Groq returned an empty prompt.")

    return result.strip()


if generate:
    if not prompt.strip():
        st.warning("Please enter an image description.")
        st.stop()

    groq_key = get_secret("GROQ_API_KEY")
    hf_token = get_secret("HF_TOKEN")

    if not hf_token:
        st.error(
            "HF_TOKEN missing. Add your Hugging Face token "
            "in Streamlit Settings → Secrets."
        )
        st.stop()

    dimensions = {
        "Square (1:1)": (1024, 1024),
        "Portrait (2:3)": (768, 1152),
        "Landscape (3:2)": (1152, 768)
    }

    width, height = dimensions[aspect_ratio]

    final_prompt = f"{prompt.strip()}, {style} art style"

    if enhance_prompt:
        if not groq_key:
            st.warning(
                "GROQ_API_KEY is missing. Using the original prompt."
            )
        else:
            try:
                with st.spinner("✨ Improving prompt with Groq..."):
                    final_prompt = improve_prompt(
                        prompt.strip(),
                        style,
                        groq_key
                    )
            except Exception as exc:
                st.warning(
                    "Prompt enhancement failed. Using original prompt. "
                    f"Details: {str(exc)[:300]}"
                )
                final_prompt = f"{prompt.strip()}, {style} art style"

    image_prompt = (
        f"{final_prompt}. "
        f"Art style: {style}. High quality, carefully composed."
    )

    if negative_prompt.strip():
        image_prompt += (
            f" Avoid these elements: {negative_prompt.strip()}."
        )

    try:
        with st.spinner("🎨 Generating your image..."):
            client = InferenceClient(
                provider="auto",
                api_key=hf_token,
                timeout=180
            )

            image = client.text_to_image(
                prompt=image_prompt,
                model="stabilityai/stable-diffusion-3-medium-diffusers",
                width=width,
                height=height
            )

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        st.success("Your image is ready!")

        st.subheader("🖼️ Generated Image")
        st.image(image_bytes, use_container_width=True)

        st.download_button(
            "⬇️ Download Image (PNG)",
            data=image_bytes,
            file_name="ai_generated_image.png",
            mime="image/png",
            use_container_width=True
        )

        with st.expander("View the final prompt"):
            st.write(image_prompt)

    except Exception as exc:
        st.error("Image generation failed.")
        st.write(
            "Check the Hugging Face token, model access, "
            "inference provider availability, and usage limits."
        )
        st.code(str(exc)[:1500])

st.divider()
st.caption("AI Image Generator | Streamlit + Groq + Hugging Face")
