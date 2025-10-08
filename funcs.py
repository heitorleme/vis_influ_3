import pandas as pd
import numpy as np
import streamlit as st
import json
import io
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import requests
import traceback
from dotenv import load_dotenv
import os
from apify_client import ApifyClient
import math

load_dotenv(".env.txt")
CAPTIV8_API_KEY = os.getenv("CAPTIV8_API_KEY")
APIFY_KEY = os.getenv("APIFY_KEY")
headers = {"X-API-Key": CAPTIV8_API_KEY}

interests_translation = {
    "Arts, Culture & Society": "Arte, Cultura e Sociedade",
    "Arts & Entertainment": "Artes e Entretenimento",
    "Pets": "Animais de Estimação",
    "Travel": "Viagens",
    "Family & Parenting": "Família e Paternidade",
    "Food & Drink": "Comer e Beber",
    "Beauty, Style & Fashion": "Beleza, Estilo e Moda",
    "Business & World Affairs": "Negócios e Atualidades",
    "Comedy": "Comédia",
    "Culinary & Food": "Culinária e Gastronomia",
    "Digital Influencer": "Influenciador Digital",
    "DIY": "Faça Você Mesmo (DIY)",
    "Education, Science & Technology": "Educação, Ciência e Tecnologia",
    "Entertainment": "Entretenimento",
    "Style & Fashion": "Moda",
    "Family": "Família",
    "Gaming": "Jogos Digitais",
    "Health & Fitness": "Saúde e Fitness",
    "Home & Garden": "Casa e Jardim",
    "Lifestyle & Blogs": "Estilo de Vida e Blogs",
    "Music": "Música",
    "Pets & Animals": "Animais de Estimação",
    "Product Reviewing": "Análise de Produtos",
    "Social Issues, Nonprofits & Activism": "Questões Sociais e Ativismo",
    "Sport & Athletics": "Esportes e Atividades Físicas",
    "Toys & Collectables": "Brinquedos e Colecionáveis",
    "Travelling & Outdoors": "Viagens e Atividades ao Ar Livre",
    "TV, Film & Animation": "TV, Cinema e Animação",
    "Vehicles & Transportation": "Veículos e Transporte",
    "Sports": "Esportes",
    "Automotive": "Carros"
}

dados_escolaridade = pd.read_excel("educacao_por_cidade.xlsx")
dados_classes_sociais = pd.read_excel("classes_sociais_por_cidade.xlsx")

# Funções para extrair posts recentes - calcular dispersão
def extrair_posts_recentes(influencers: list, apify_key: str = APIFY_KEY, results_limit: int = 10):
    """
    Retorna:
      { influencer: {"likes": [...], "comments": [...]} }
    Usando os campos ownerUsername, likesCount, commentsCount do dataset do Apify.
    """
    urls = [f"https://www.instagram.com/{u}" for u in influencers]

    client = ApifyClient(apify_key)
    run_input = {
        "addParentData": False,
        "directUrls": urls,
        "enhanceUserSearchWithFacebookPage": False,
        "isUserReelFeedURL": False,
        "isUserTaggedFeedURL": False,
        "resultsLimit": results_limit,
        "resultsType": "posts",
    }
    run = client.actor("shu8hvrXbJbY3Eb9W").call(run_input=run_input)

    # inicia dict já com todos os influenciadores
    posts_dict = {u: {"likes": [], "comments": []} for u in influencers}

    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        user = item.get("ownerUsername")
        if not user:
            continue  # sem username, ignora

        likes = item.get("likesCount", 0) or 0
        comments = item.get("commentsCount", 0) or 0

        # garante chaves
        if user not in posts_dict:
            posts_dict[user] = {"likes": [], "comments": []}

        posts_dict[user]["likes"].append(int(likes))
        posts_dict[user]["comments"].append(int(comments))

    # garante listas existentes (mesmo vazias) para todos solicitados
    for u in influencers:
        posts_dict.setdefault(u, {"likes": [], "comments": []})

    return posts_dict

def calcular_dispersao_posts(posts_dict: dict):
    """
    posts_dict: { influencer: {"likes": [...], "comments": [...]} }
    Retorna: (dispersao_influencers: dict, df_dispersao: DataFrame)
    """
    dispersao_influencers = {}

    for influencer, payload in posts_dict.items():
        likes = [int(x) for x in (payload.get("likes") or [])]
        comments = [int(x) for x in (payload.get("comments") or [])]

        z_l = z_c = 0.0
        if len(likes) > 0 and np.sum(likes) > 0:
            ml = float(np.mean(likes))
            sl = float(np.std(likes))
            z_l = (sl / ml) * 100.0 if ml != 0 else 0.0

        if len(comments) > 0 and np.sum(comments) > 0:
            mc = float(np.mean(comments))
            sc = float(np.std(comments))
            z_c = (sc / mc) * 100.0 if mc != 0 else 0.0

        # mesma regra do seu código
        dispersao = round((z_c + z_l) / 2.0, 0) if z_l > 0 else round(z_c, 0)
        dispersao_influencers[influencer] = int(dispersao)

    df_dispersao = (
        pd.DataFrame.from_dict(dispersao_influencers, orient="index")
        .reset_index()
        .rename(columns={"index": "Influencer", 0: "Dispersão"})
        .astype({"Dispersão": int})
    )

    return dispersao_influencers, df_dispersao

# Funções para buscas de dados a partir do Instagram do influenciador
def buscar_slug(influencer):
    url = "https://public-api.captiv8.io/discovery/creators/search/account/"
    params = {"social_account": influencer, "social_network":"instagram"}
    response = requests.get(url, headers=headers, params=params)
    response_json = response.json()
    return response_json["data"]["creator"]["slug"]

def buscar_dados_brutos(slug):
    url = f"https://public-api.captiv8.io/discovery/creators/{slug}"
    response = requests.get(url, headers=headers)
    response_json = response.json()
    response_json = response_json["data"]
    return response_json

# Criar funções para extrair dados sem necessidade de processamento
def extrair_nome(dados_brutos):
    return dados_brutos["name"]

# Extrair médias de views para os formatos (Reels e Feed)
def extrair_views(dados_brutos):
    media_views_post = "N/A"
    media_views_reel = "N/A"
    for account in dados_brutos["accounts"]:
        if account.get("connector") == "instagram":
            media_views_post = account.get("impressions_est", {}).get("fi", "N/A")
            media_views_reel = account.get("impressions_est", {}).get("vv", "N/A")

            # Formata com separador de milhares se for número
            if isinstance(media_views_post, (int, float)):
                media_views_post = f"{media_views_post:,}".replace(",", ".")  # separador de milhar em estilo brasileiro
            else:
                media_views_post = "N/A"

            if isinstance(media_views_reel, (int, float)):
                media_views_reel = f"{media_views_reel:,}".replace(",", ".")
            else:
                media_views_reel = "N/A"

            return f"{media_views_post}"
    return "Não há dados disponíveis"

# Extrair e traduzir interesses da audiência
def extrair_interesses_da_audiencia(dados_brutos, interests_translation=interests_translation):
    """
    Extrai, normaliza e traduz os interesses da audiência.
    Retorna uma string formatada como:
    'Amigos, Família e Relacionamentos (34,46%), Televisão e Filmes (24,83%), ...'
    """

    # Garante que a chave exista
    interesses_lista = dados_brutos.get("audience_interests", [])
    if not interesses_lista:
        return "Sem dados de interesses disponíveis."

    # Calcula o total
    total_valores = sum(float(item.get("value", 0)) for item in interesses_lista)
    if total_valores == 0:
        return "Sem dados de interesses disponíveis."

    # Normaliza e aplica tradução
    interesses_normalizados = []
    for item in interesses_lista:
        label_en = item.get("label", "")
        valor = float(item.get("value", 0))
        percentual = (valor / total_valores) * 100
        label_pt = interests_translation.get(label_en, label_en)  # traduz se possível
        interesses_normalizados.append((label_pt, percentual))

    # Ordena por percentual decrescente
    interesses_normalizados.sort(key=lambda x: x[1], reverse=True)

    # Pega top 5
    top5 = interesses_normalizados[:5]

    # Formata como string no estilo desejado
    resultado_formatado = ", <br>".join(
        [f"{label} ({percentual:.2f}%)" for label, percentual in top5]
    ).replace(".", ",")  # vírgula como separador decimal

    return resultado_formatado

# Extrair tópicos abordados pelo Influ
def extrair_topicos_influ(dados_brutos, interests_translation=interests_translation):
    lista_topicos = []
    topicos_brutos = dados_brutos.get("content_topics", [])

    for i in topicos_brutos:
        valor = i.get("value", "").strip()
        if valor:
            lista_topicos.append(valor)

    # traduzir se houver dicionário de tradução
    if interests_translation:
        lista_topicos_traduzidos = [
            interests_translation.get(t, t) for t in lista_topicos
        ]
    else:
        lista_topicos_traduzidos = lista_topicos

    return ",  <br>".join(lista_topicos_traduzidos)

# Criar DataFrame com as localizações da audiência
def consolidar_geo_audiencia(dados_brutos, country_filter="BR"):
    """
    Consolida cidades a partir de dados_brutos['data']['geo']['network']['All']['cities'],
    soma os pesos por cidade e normaliza por influenciador.
    """
    # Tenta identificar o influenciador
    influencer = dados_brutos["accounts"][0]["username"]

    # Pega o bloco de cidades
    try:
        cities_root = dados_brutos["geo"]["network"]["All"]["cities"]
    except KeyError:
        # Sem dados de geo
        return pd.DataFrame(columns=["influencer", "Cidade", "country.code", "weight", "normalized_weight"])

    # Preferir a lista do país (ex.: 'BR'); senão cair para 'WORLD'
    cities_list = cities_root.get(country_filter) or cities_root.get("WORLD") or []

    # Transforma em DataFrame
    df = pd.DataFrame(cities_list)
    if df.empty:
        return pd.DataFrame(columns=["influencer", "Cidade", "country.code", "weight", "normalized_weight"])

    # Renomeia/seleciona campos para ficar compatível com o código base
    # (no JSON: 'name' é a cidade; usaremos 'followers' como peso, se não houver usa 'followers_count' ou 'value')
    df = df.rename(columns={"name": "Cidade"})
    if "followers" in df.columns:
        weight_series = df["followers"]
    elif "followers_count" in df.columns:
        weight_series = df["followers_count"]
    else:
        weight_series = df["value"]  # fallback em proporção

    df["weight"] = pd.to_numeric(weight_series, errors="coerce").fillna(0.0)
    df["influencer"] = influencer
    df["country.code"] = country_filter if cities_root.get(country_filter) else None

    # Limpeza leve no nome da cidade
    df["Cidade"] = df["Cidade"].astype(str).str.strip()

    # Agregar por influenciador + cidade (caso haja duplicatas)
    df_sum = (
        df.groupby(["influencer", "Cidade", "country.code"], as_index=False, dropna=False)["weight"]
          .sum()
    )

    # Normalizar por influenciador
    total_weight = df_sum.groupby("influencer")["weight"].transform("sum")
    df_sum["normalized_weight"] = np.where(total_weight > 0, df_sum["weight"] / total_weight, 0.0)

    # Ordena por maior participação
    df_sum = df_sum.sort_values(["influencer", "normalized_weight"], ascending=[True, False]).reset_index(drop=True)

    # Colunas finais no mesmo espírito do base
    return df_sum[["influencer", "Cidade", "country.code", "weight", "normalized_weight"]]

######################## Calcular escolaridade e classes sociais ########################

# ==========================================================
# Helpers
# ==========================================================
def _coerce_frac(x: pd.Series | list | float | int) -> pd.Series | float:
    """
    Garante frações em 0–1 (se vier em %, divide por 100).
    Aceita escalar, lista ou Series.
    """
    if isinstance(x, (list, tuple)):
        x = pd.to_numeric(pd.Series(x), errors="coerce").fillna(0.0)
        return x.where(x <= 1, x / 100.0).tolist()
    if isinstance(x, (int, float)):
        return x / 100.0 if x > 1 else float(x)
    # Series
    s = pd.to_numeric(x, errors="coerce").fillna(0.0)
    return s.where(s <= 1, s / 100.0)

def _extrair_demografia(dados_brutos) -> pd.DataFrame:
    """
    Extrai demografia por idade/gênero.
    - Captiv8: usa ageGender (instagram > All) e reescala vetores para somarem a maleTotal/femaleTotal.
    - IMAI: audience_followers.data.audience_genders_per_age
    Entrada pode ser {influencer: json} ou um único json.
    Retorna colunas: ['influencer','code','male','female'] (frações do total 0–1).
    """
    def _nome_influencer(b):
        try:
            for a in b.get("accounts") or []:
                if a.get("connector") == "instagram" and a.get("username"):
                    return a["username"]
            return b.get("slug") or b.get("name") or "influencer"
        except Exception:
            return "influencer"

    # Normaliza entrada
    if isinstance(dados_brutos, dict) and (
        "ageGender" in dados_brutos or "geo" in dados_brutos or "accounts" in dados_brutos
    ):
        dados_norm = {_nome_influencer(dados_brutos): dados_brutos}
    elif isinstance(dados_brutos, dict):
        dados_norm = dados_brutos
    else:
        return pd.DataFrame(columns=["influencer", "code", "male", "female"])

    linhas = []
    # buckets padrão Captiv8 (6)
    buckets_default = ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"]

    for influencer, bruto in dados_norm.items():
        # --------- IMAI ----------
        try:
            items = bruto["audience_followers"]["data"]["audience_genders_per_age"]
            df_tmp = pd.json_normalize(items)
            if {"code", "male", "female"}.issubset(df_tmp.columns):
                df_tmp["male"] = _coerce_frac(df_tmp["male"])
                df_tmp["female"] = _coerce_frac(df_tmp["female"])
                df_tmp["influencer"] = influencer
                linhas.append(df_tmp[["influencer", "code", "male", "female"]])
                continue
        except Exception:
            pass

        # --------- Captiv8 ----------
        try:
            age_gender = bruto.get("ageGender") or bruto.get("data", {}).get("ageGender")
            if not age_gender:
                raise KeyError("ageGender não encontrado")

            # escolher bloco preferencial
            bloco = next((b for b in age_gender if b.get("network") == "instagram"), None)
            if bloco is None:
                bloco = next((b for b in age_gender if str(b.get("network")).lower() in ("all","*","overall")), age_gender[0])

            male_arr = bloco.get("male", []) or []
            female_arr = bloco.get("female", []) or []

            # converte para frações (0–1)
            male_arr = _coerce_frac(male_arr)
            female_arr = _coerce_frac(female_arr)

            # Totais reportados pelo Captiv8 (frações do total)
            male_total = _coerce_frac(bloco.get("maleTotal", np.nansum(male_arr)))
            female_total = _coerce_frac(bloco.get("femaleTotal", np.nansum(female_arr)))

            # Reescala vetores para "conservar massa":
            # - se os vetores estiverem normalizados por sexo (somam 1 cada),
            #   multiplicamos por male_total/female_total para virarem frações do TOTAL.
            sum_m = float(np.nansum(male_arr))  or 0.0
            sum_f = float(np.nansum(female_arr)) or 0.0
            if sum_m > 0:
                male_arr = [v * (male_total / sum_m) for v in male_arr]
            if sum_f > 0:
                female_arr = [v * (female_total / sum_f) for v in female_arr]

            # Garante mesmo comprimento e aplica buckets padrão
            n = min(len(buckets_default), len(male_arr), len(female_arr))
            if n == 0:
                continue

            df_blk = pd.DataFrame({
                "influencer": influencer,
                "code": buckets_default[:n],
                "male": male_arr[:n],
                "female": female_arr[:n],
            })
            linhas.append(df_blk[["influencer", "code", "male", "female"]])
        except Exception:
            continue

    if not linhas:
        return pd.DataFrame(columns=["influencer", "code", "male", "female"])

    df_demo = pd.concat(linhas, ignore_index=True)
    # higiene final
    df_demo["male"]   = pd.to_numeric(df_demo["male"], errors="coerce").fillna(0.0)
    df_demo["female"] = pd.to_numeric(df_demo["female"], errors="coerce").fillna(0.0)
    return df_demo

# ==========================================================
# 1) CLASSES SOCIAIS POR INFLUENCIADOR
# ==========================================================
def estimar_classes_sociais_por_influenciador(
    df_cidades: pd.DataFrame,
    dados_classes_sociais: pd.DataFrame,
):
    """
    Replica a lógica do código base para estimar a distribuição de classes sociais
    por influenciador a partir do DF de cidades.

    Parâmetros
    ----------
    df_cidades : DataFrame
        Colunas mínimas: ["influencer", "Cidade", "weight"].
    dados_classes_sociais : DataFrame
        Colunas: ["Cidade", "Classes D e E", "Classe C", "Classe B", "Classe A"] (em %).

    Retorna
    -------
    result_classes : DataFrame
        Distribuição final por influenciador (proporções 0-1) + coluna 'distribuicao_formatada'.
    classes_sociais_dict : dict
        {influencer: "Classes D e E: x%, ..."} para uso direto.
    """
    # Cópias seguras
    df_classes_influ = df_cidades.copy()
    base_classes = dados_classes_sociais.copy()

    # Higiene
    if "Unnamed: 0" in base_classes.columns:
        base_classes = base_classes.drop(columns=["Unnamed: 0"])
    df_classes_influ["weight"] = pd.to_numeric(df_classes_influ["weight"], errors="coerce").fillna(0.0)

    # Normalização por influenciador e cidade (exatamente como no código base)
    total_weight_por_influencer = df_classes_influ.groupby("influencer")["weight"].transform("sum")
    total_weight_por_cidade = df_classes_influ.groupby(["influencer", "Cidade"])["weight"].transform("sum")
    df_classes_influ["normalized_weight"] = np.divide(
        total_weight_por_cidade, total_weight_por_influencer, out=np.zeros_like(total_weight_por_cidade, dtype=float), where=total_weight_por_influencer.ne(0)
    )

    # Merge com a base de classes
    df_merged = pd.merge(df_classes_influ, base_classes, on=["Cidade"], how="inner")

    # Recalcular normalized_weight (para re-normalizar após o merge)
    df_merged["total_weight"] = df_merged.groupby("influencer")["normalized_weight"].transform("sum")
    df_merged["normalized_weight"] = np.divide(
        df_merged["normalized_weight"], df_merged["total_weight"],
        out=np.zeros_like(df_merged["normalized_weight"], dtype=float),
        where=df_merged["total_weight"].ne(0)
    )
    df_merged.drop(columns="total_weight", inplace=True)

    # Contribuição ponderada
    df_merged["normalized_classe_de"] = df_merged["normalized_weight"] * df_merged["Classes D e E"]
    df_merged["normalized_classe_c"]  = df_merged["normalized_weight"] * df_merged["Classe C"]
    df_merged["normalized_classe_b"]  = df_merged["normalized_weight"] * df_merged["Classe B"]
    df_merged["normalized_classe_a"]  = df_merged["normalized_weight"] * df_merged["Classe A"]

    # Agrega por influenciador
    result = df_merged.groupby("influencer")[[
        "normalized_classe_de", "normalized_classe_c", "normalized_classe_b", "normalized_classe_a"
    ]].sum().round(2)

    # Renomeia e converte de % para proporção
    result.columns = ["Classes D e E", "Classe C", "Classe B", "Classe A"]
    result[["Classes D e E", "Classe C", "Classe B", "Classe A"]] = \
        result[["Classes D e E", "Classe C", "Classe B", "Classe A"]] / 100.0

    # String formatada
    result["distribuicao_formatada"] = result.apply(
        lambda row: (
            f"Classes D e E: {row['Classes D e E']:.2%}, <br>"
            f"Classe C: {row['Classe C']:.2%},  <br>"
            f"Classe B: {row['Classe B']:.2%},  <br>"
            f"Classe A: {row['Classe A']:.2%}"
        ),
        axis=1
    ).str.replace('.', ',', regex=False)

    result = result.reset_index()
    classes_sociais_dict = result.set_index('influencer')['distribuicao_formatada'].to_dict()
    return result, classes_sociais_dict


# ==========================================================
# 2) ESCOLARIDADE POR INFLUENCIADOR
# ==========================================================
from scipy.stats import norm

def _pick_col_case_insensitive(df: pd.DataFrame, names):
    """Acha coluna por nomes (case-insensitive). Lança erro se não encontrar."""
    lowmap = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        key = str(n).strip().lower()
        if key in lowmap:
            return lowmap[key]
    raise KeyError(f"Coluna não encontrada. Procurei por: {names}. Colunas: {list(df.columns)}")

def estimar_escolaridade_por_influenciador(
    df_cidades: pd.DataFrame,
    dados_brutos,                     # dict {influencer: json} OU json único
    dados_escolaridade: pd.DataFrame, # colunas: Cidade, Grupo Etário, Female, Male (anos)
    std_dev: float = 3.0,
    debug: bool = False
):
    # 1) Demografia por idade/gênero (frações 0–1) a partir do JSON
    df_demo = _extrair_demografia(dados_brutos)  # colunas: influencer, code, male, female
    if df_demo.empty:
        cols = ["Influencer", "< 5 anos", "5-9 anos", "9-12 anos", "> 12 anos", "distribuicao_formatada"]
        return pd.DataFrame(columns=cols), {}

    df_demo = df_demo.rename(columns={"code": "Grupo Etário",
                                      "male": "prop_male",
                                      "female": "prop_female"})

    # 2) Normaliza peso de cidade por influenciador (soma = 1 por influencer)
    df_cid = df_cidades.copy()
    df_cid["weight"] = pd.to_numeric(df_cid["weight"], errors="coerce").fillna(0.0)
    total_w_inf = df_cid.groupby("influencer")["weight"].transform("sum")
    total_w_city = df_cid.groupby(["influencer", "Cidade"])["weight"].transform("sum")
    df_cid["weight_norm"] = np.divide(
        total_w_city, total_w_inf,
        out=np.zeros_like(total_w_city, dtype=float),
        where=total_w_inf.ne(0)
    )

    # 3) Cruza Cidades x Demografia
    df_unido = pd.merge(df_cid, df_demo, on="influencer", how="inner")

    # 4) Merge com a base de educação (anos médios) — usando colunas com nomes exatos/case-insensitive
    base = dados_escolaridade.copy()
    # mapear nomes garantidos
    col_cid  = _pick_col_case_insensitive(base, ["Cidade"])
    col_age  = _pick_col_case_insensitive(base, ["Grupo Etário", "Grupo Etario", "Faixa Etária", "Faixa Etaria"])
    col_f_y  = _pick_col_case_insensitive(base, ["Female"])
    col_m_y  = _pick_col_case_insensitive(base, ["Male"])

    base = base.rename(columns={col_cid:"Cidade", col_age:"Grupo Etário", col_f_y:"Female_anos", col_m_y:"Male_anos"})
    # coagir para float (suporta vírgula decimal)
    base["Female_anos"] = pd.to_numeric(base["Female_anos"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    base["Male_anos"]   = pd.to_numeric(base["Male_anos"].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # deduplicar por segurança (média se houver entradas repetidas)
    base = (base.groupby(["Cidade", "Grupo Etário"], as_index=False)[["Female_anos","Male_anos"]].mean())

    df_unido_edu = df_unido.merge(base, on=["Cidade", "Grupo Etário"], how="inner")  # <-- INNER: só usa pares existentes

    # 5) Pesos efetivos por sexo (proporção da audiência * peso normalizado da cidade)
    df_unido_edu["w_f"] = df_unido_edu["prop_female"] * df_unido_edu["weight_norm"]
    df_unido_edu["w_m"] = df_unido_edu["prop_male"]   * df_unido_edu["weight_norm"]

    # 6) Contribuições ponderadas (anos * peso)
    df_unido_edu["part_f"] = df_unido_edu["w_f"] * df_unido_edu["Female_anos"]
    df_unido_edu["part_m"] = df_unido_edu["w_m"] * df_unido_edu["Male_anos"]

    # 7) Média ponderada de anos por influenciador = soma(partes) / soma(pesos)
    num = df_unido_edu.groupby("influencer")[["part_f","part_m"]].sum().sum(axis=1)
    den = df_unido_edu.groupby("influencer")[["w_f","w_m"]].sum().sum(axis=1).replace(0, np.nan)
    mean_years = (num / den).fillna(0.0)

    if debug:
        print("\n--- DEBUG Escolaridade ---")
        print(df_unido_edu[["influencer","Cidade","Grupo Etário","Female_anos","Male_anos","w_f","w_m","part_f","part_m"]].head(12))
        print("mean_years:\n", mean_years)
        print("--------------------------\n")

    # 8) Converte média em distribuição nas 4 faixas via Normal padrão (std_dev, sem truncagem)
    rows = []
    for inf, mean in mean_years.items():
        mean = float(mean)
        sd = float(std_dev)
        p_lt5  = norm.cdf(5,  mean, sd)
        p_5_9  = norm.cdf(9,  mean, sd) - norm.cdf(5,  mean, sd)
        p_9_12 = norm.cdf(12, mean, sd) - norm.cdf(9,  mean, sd)
        p_gt12 = 1 - norm.cdf(12, mean, sd)

        # renormaliza por higiene numérica: soma = 1
        s = p_lt5 + p_5_9 + p_9_12 + p_gt12
        if s > 0:
            p_lt5, p_5_9, p_9_12, p_gt12 = (p_lt5/s, p_5_9/s, p_9_12/s, p_gt12/s)

        rows.append({
            "Influencer": inf,
            "< 5 anos":  p_lt5,
            "5-9 anos":  p_5_9,
            "9-12 anos": p_9_12,
            "> 12 anos": p_gt12
        })

    result_edu = pd.DataFrame(rows)

    # 9) Formatação amigável
    result_edu["distribuicao_formatada"] = result_edu.apply(
        lambda r: (
            f"< 5 anos: {r['< 5 anos']:.2%},  <br>"
            f"5-9 anos: {r['5-9 anos']:.2%},  <br>"
            f"9-12 anos: {r['9-12 anos']:.2%},  <br>"
            f"> 12 anos: {r['> 12 anos']:.2%}"
        ),
        axis=1
    ).str.replace('.', ',', regex=False)

    escolaridade_dict = result_edu.set_index("Influencer")["distribuicao_formatada"].to_dict()
    return result_edu, escolaridade_dict

######################## Calcular score da audiência ########################

def audience_score(data: dict) -> float:
    """Calcula um Audience Score (0–100) simplificado a partir de dados brutos."""

    # === Engajamento ===
    avg_eng_rate = float(data.get("avg_eng_rate") or 0)
    avg_views_per_post = float(data.get("avg_views_per_post") or 0)
    eng_score = min(100, (avg_eng_rate / 10) * 100 + min(avg_views_per_post / 50000 * 10, 10))

    # === Tamanho ===
    followers = float(data.get("followers") or 0)
    size_score = 0 if followers <= 0 else min(100, (math.log10(followers) / 7) * 100)

    # === Crescimento ===
    prev_followers = float(data.get("prev_followers") or 0)
    if followers > 0 and prev_followers > 0:
        growth_rate = (followers - prev_followers) / prev_followers * 100
        growth_score = min(100, max(0, (growth_rate / 50) * 100))
    else:
        growth_score = 0

    # === Qualidade demográfica ===
    # País BR principal
    br_share = 0
    try:
        for c in data["geo"]["network"]["All"]["regions"]["WORLD"]:
            if c[0] == "BR":
                br_share = float(c[1])
                break
    except Exception:
        pass
    if br_share >= 0.9:
        country_pts = 40
    elif br_share >= 0.75:
        country_pts = 20
    else:
        country_pts = 0

    # Cidades principais
    try:
        cities = data["geo"]["network"]["All"]["cities"]["WORLD"]
        top3 = sum(
            c["value"] for c in cities
            if c["name"] in ["São Paulo", "Rio de Janeiro", "Belo Horizonte"]
        )
    except Exception:
        top3 = 0
    if top3 >= 0.5:
        city_pts = 30
    elif top3 >= 0.3:
        city_pts = 20
    elif top3 >= 0.15:
        city_pts = 10
    else:
        city_pts = 0

    # Gênero
    female, male = None, None
    try:
        for g in data["ageGender"]:
            if g["network"] == "All":
                female = float(g.get("femaleTotal"))
                male = float(g.get("maleTotal"))
                break
    except Exception:
        pass
    if female and male:
        if 40 <= female <= 60 and 40 <= male <= 60:
            gender_pts = 30
        else:
            dev = (abs(female - 50) + abs(male - 50)) / 2
            gender_pts = max(0, 30 - min(30, dev * 1.5))
    else:
        gender_pts = 0
    demo_score = min(100, country_pts + city_pts + gender_pts)

    # === Autenticidade ===
    verified = bool(data.get("profile_verified"))
    has_ads = bool(data.get("has_ads_content"))
    healthy_accounts = 0
    for acc in data.get("accounts", []):
        rate = acc.get("post_avg_eng_rate") or acc.get("eng_rate_per_post") or 0
        if float(rate) >= 3:
            healthy_accounts += 1
    multi_pts = 70 if healthy_accounts >= 2 else 35 if healthy_accounts == 1 else 0
    auth_score = min(100, (20 if verified else 0) + (0 if has_ads else 10) + multi_pts)

    # === Score Final ===
    score = (
        0.35 * eng_score +
        0.25 * size_score +
        0.15 * growth_score +
        0.15 * demo_score +
        0.10 * auth_score
    )
    return round(score, 0)

def main(influencers: list):
    dispersao_por_influ = calcular_dispersao_posts(extrair_posts_recentes(influencers))
    dispersoes = {}
    for influencer in influencers:
        dispersoes[influencer] = dispersao_por_influ[0][influencer]
    
    dados_influencer = {}
    for influencer in influencers:
        slug = buscar_slug(influencer)
        dados = buscar_dados_brutos(slug)
        dados_influencer[influencer] = dados
    
    nomes_influs = {}
    for i in influencers:
        nomes_influs[i] = extrair_nome(dados_influencer[i])

    educacao_por_influ = {}
    classes_sociais_por_influ = {}

    for influencer in influencers:
        geografia = consolidar_geo_audiencia(dados_influencer[influencer])
        classes = estimar_classes_sociais_por_influenciador(geografia, dados_classes_sociais)[1].values()
        educacao = estimar_escolaridade_por_influenciador(geografia, dados_influencer[influencer], dados_escolaridade)[1].values()

        # Converter para string (ex: "Classe A, Classe B, Classe C")
        classes_sociais_por_influ[influencer] = ",  <br>".join(map(str, classes))
        educacao_por_influ[influencer] = ",  <br>".join(map(str, educacao))
    
    scores_audiencia = {}
    for influencer in influencers:
        scores_audiencia[influencer] = audience_score(dados_influencer[influencer])
    
    topicos_influs = {}
    for influencer in influencers:
        topicos_influs[influencer] = extrair_topicos_influ(dados_influencer[influencer])
    
    interesses_audiencia = {}
    for influencer in influencers:
        interesses_audiencia[influencer] = extrair_interesses_da_audiencia(dados_influencer[influencer])
    
    alcances_audiencia = {}
    for influencer in influencers:
        alcances_audiencia[influencer] = extrair_views(dados_influencer[influencer])

        # Montar dataframe final
    df = pd.DataFrame({
        "Username do influenciador": influencers,
        "Nome do influenciador": [nomes_influs[i] for i in influencers],
        "Score da audiência": [scores_audiencia[i] for i in influencers],
        "Dispersão de interações": [dispersoes[i] for i in influencers],
        "Alcance médio esperado": [alcances_audiencia[i] for i in influencers],
        "Tópicos do Influenciador": [topicos_influs[i] for i in influencers],
        "Interesses da audiência": [interesses_audiencia[i] for i in influencers],
        "Classes sociais": [classes_sociais_por_influ[i] for i in influencers],
        "Escolaridade": [educacao_por_influ[i] for i in influencers],
    })

    return df
