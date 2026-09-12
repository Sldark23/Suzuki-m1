import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

MODEL = "Raivatv24/suzuki-m1-gpt"

try:
    revision = HfApi().model_info(MODEL).sha
except Exception:
    revision = None

if st.session_state.get("last_rev") != revision:
    st.cache_resource.clear()
    st.session_state["last_rev"] = revision


@st.cache_resource
def carregar():
    tk = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    return tk, model


tk, model = carregar()

st.set_page_config(page_title="Suzuki-m1", page_icon=":brain:")
st.title("Suzuki-m1 — GPT treinado do zero em português")
rev = revision[:8] if revision else "?"
st.caption(f"Modelo ~23M parâmetros, pré-treinado do zero sobre o corpus Suzuki-m1. Versão no Hub: `{rev}`.")

prompt = st.text_area("Começo do texto", "Era uma vez,", height=100)

col1, col2, col3 = st.columns(3)
max_tokens = col1.slider("Máx. tokens novos", 10, 300, 100, 10)
temperature = col2.slider("Temperatura", 0.1, 1.5, 0.8, 0.1)
top_k = col3.slider("Top-K", 1, 100, 50, 1)

if st.button("Gerar", type="primary"):
    with st.spinner("Gerando..."):
        ids = tk.encode(prompt, return_tensors="pt")
        try:
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    top_k=top_k,
                    temperature=temperature,
                    pad_token_id=tk.pad_token_id,
                    eos_token_id=tk.eos_token_id,
                )
        except RuntimeError as e:
            st.error(f"Falha na amostragem: {e}\nTentando geração gulosa...")
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tk.pad_token_id,
                    eos_token_id=tk.eos_token_id,
                )
        st.markdown(tk.decode(out[0].tolist()))