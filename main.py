import os
import io
import re
import pandas as pd
import streamlit as st
from datetime import datetime
import xlsxwriter
from dotenv import load_dotenv
from funcs import main  # função que retorna um DataFrame

# =========================
# Configuração
# =========================
load_dotenv(".env.txt")
CAPTIV8_API_KEY = os.getenv("CAPTIV8_API_KEY")
APIFY_KEY = os.getenv("APIFY_KEY")

st.set_page_config(page_title="Análise de Influenciadores • Instagram", page_icon="📊", layout="wide")
st.title("📊 Análise de Influenciadores")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("### Configurações")
    st.markdown("- Informe um ou mais @usernames do Instagram (um por linha ou separados por vírgula).")
    st.divider()

    user_input = st.text_area("Usernames do Instagram", placeholder="fulano", height=120)
    btn = st.button("Analisar", type="primary")

# =========================
# Funções auxiliares
# =========================
def parse_usernames(txt: str):
    items = [x.strip() for x in (txt or "").replace("\n", ",").replace("\r", ",").replace("\t", ",").split(",")]
    clean, seen, out = [], set(), []
    for it in items:
        if not it:
            continue
        it = it.replace("@", "").strip()
        if it:
            clean.append(it)
    for u in clean:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def df_html(df: pd.DataFrame) -> str:
    """Renderização HTML preservando <br> nas células."""
    return df.to_html(escape=False, index=False)

def df_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Converte <br> em quebras de linha para Excel."""
    df2 = df.copy()
    # aplica apenas em colunas de texto/objeto
    obj_cols = df2.select_dtypes(include=["object"]).columns
    df2[obj_cols] = df2[obj_cols].replace({r"<br\s*/?>": "\n"}, regex=True)
    return df2

# =========================
# Execução quando clicar em Analisar
# =========================
if btn:
    usernames = parse_usernames(user_input)
    if not usernames:
        st.warning("Informe ao menos um username.")
        st.stop()
    if not CAPTIV8_API_KEY:
        st.error("Sem CAPTIV8_API_KEY; não é possível consultar a API.")
        st.stop()

    try:
        with st.spinner("🔎 Coletando e analisando dados dos influenciadores..."):
            df = main(usernames)
        # guarda no estado para persistir após o clique do download (que causa rerun)
        st.session_state["resultado_df"] = df
    except Exception as e:
        st.error(f"Ocorreu um erro durante a análise: {e}")

# =========================
# Renderização persistente + Download
# =========================
if "resultado_df" in st.session_state:
    df = st.session_state["resultado_df"]

    if not df.empty:
        st.subheader("Resultados da Análise")
        st.markdown(df_html(df), unsafe_allow_html=True)

        # --- Excel com quebras de linha ---
        buffer = io.BytesIO()
        df_xlsx = df_for_excel(df)

        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            sheet_name = "Análise Influenciadores"
            df_xlsx.to_excel(writer, index=False, sheet_name=sheet_name)

            workbook  = writer.book
            worksheet = writer.sheets[sheet_name]

            # formato com quebra de linha automática
            wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

            # aplica wrap em todas as colunas e ajusta uma largura padrão
            for col_idx, col in enumerate(df_xlsx.columns):
                worksheet.set_column(col_idx, col_idx, 30, wrap_fmt)
        
        dataehora = datetime.now().strftime("%Y%m%d_%H%M%S")

        st.download_button(
            label="📥 Baixar resultados em XLSX",
            data=buffer.getvalue(),
            file_name=f"analise_influenciadores_{dataehora}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:

        st.warning("Nenhum dado foi retornado pela análise.")
