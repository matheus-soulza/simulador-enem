import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import unicodedata
import re
import difflib

# ====== UI / página ======
st.set_page_config(page_title="Simulador ENEM", page_icon="🎯", layout="wide")

# ====== CSS (highlight das legendas) ======
st.markdown("""
<style>
.legend-title {
  font-weight: 700;
  padding: 4px 8px;
  border-left: 6px solid #ffae00;
  background: linear-gradient(90deg, rgba(255,174,0,0.18), rgba(255,174,0,0.05));
  border-radius: 6px;
  display: inline-block;
}
.legend-box {
  margin-top: 6px;
  margin-bottom: 12px;
  padding: 10px 12px;
  background: rgba(255,174,0,0.08);
  border: 1px solid rgba(255,174,0,0.22);
  border-radius: 8px;
}
.legend-box ul {
  margin: 0;
  padding-left: 20px;
}
</style>
""", unsafe_allow_html=True)

# ====== caminhos ======
MODEL_PATH = "enem_lgbm.pkl"
FEATURES_PATH = "enem_features.json"

# ====== artefatos ======
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    try:
        feats = list(model.booster_.feature_name())
    except Exception:
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            feats = json.load(f)
    return model, feats

model, FEATURE_LIST = load_artifacts()

# ====== normalização de nomes ======
def normalize(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

ACTUAL_MAP = {normalize(c): c for c in FEATURE_LIST}
_missing_cols = []
_suggestions = {}

def resolve_col(col_name: str):
    norm = normalize(col_name)
    if norm in ACTUAL_MAP:
        return ACTUAL_MAP[norm]
    close = difflib.get_close_matches(norm, ACTUAL_MAP.keys(), n=3, cutoff=0.6)
    if close:
        _suggestions[col_name] = [ACTUAL_MAP[c] for c in close]
    return None

# ====== listas ======
UF_LIST = [
    "AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT",
    "PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"
]

# Escolaridade (E1..E7)
INSTR_LIST = [
    "E1 — Nunca estudou.",
    "E2 — Não completou a 4ª série/5º ano do Ensino Fundamental.",
    "E3 — Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano.",
    "E4 — Completou a 8ª série/9º ano, mas não completou o Ensino Médio.",
    "E5 — Completou o Ensino Médio, mas não completou a Faculdade.",
    "E6 — Completou a Faculdade, mas não completou a Pós-graduação.",
    "E7 — Completou a Pós-graduação."
]
INSTR_MAP = {txt: i+1 for i, txt in enumerate(INSTR_LIST)}

# Profissão (G1..G5)
PROF_LIST = [
    "G1 — Agricultura/extrativismo/boia-fria etc.",
    "G2 — Serviços gerais/comércio/auxiliares/operacionais.",
    "G3 — Indústria/ofícios/operadores/motoristas.",
    "G4 — Técnicos/professores (básico)/supervisão/autônomos/pequenos empresários.",
    "G5 — Profissionais de nível superior/alta gestão/empresários médios e grandes."
]
PROF_MAP = {txt: i+1 for i, txt in enumerate(PROF_LIST)}

RENDA_LIST = [
    "Nenhuma Renda","Até 1 Salário Mínimo","De 1 a 1,5 Salários Mínimos",
    "De 1,5 a 2 Salários Mínimos","De 2 a 2,5 Salários Mínimos","De 2,5 a 3 Salários Mínimos",
    "De 3 a 4 Salários Mínimos","De 4 a 5 Salários Mínimos","De 5 a 6 Salários Mínimos",
    "De 6 a 7 Salários Mínimos","De 7 a 8 Salários Mínimos","De 8 a 9 Salários Mínimos",
    "De 9 a 10 Salários Mínimos","De 10 a 12 Salários Mínimos","De 12 a 15 Salários Mínimos",
    "De 15 a 20 Salários Mínimos","Acima de 20 Salários Mínimos"
]
RENDA_MAP = {txt: i+1 for i, txt in enumerate(RENDA_LIST)}

FAIXA_ETARIA = [
    "Menor de 17 anos","17 anos","18 anos","19 anos","20 anos","21 anos","22 anos",
    "23 anos","24 anos","25 anos","26–30 anos","31–35 anos","36–40 anos","41–45 anos",
    "46–50 anos","51–55 anos","56–60 anos","61–65 anos","66–70 anos","Maior de 70 anos"
]
FAIXA_MAP = {txt: i+1 for i, txt in enumerate(FAIXA_ETARIA)}

# ====== helpers ======
def empty_row(feature_list):
    return pd.DataFrame([0]*len(feature_list), index=feature_list).T

def set_if_exists(row, col, val):
    real = resolve_col(col)
    if real is not None and real in row.columns:
        row[real] = val
    else:
        _missing_cols.append(col)

def map_nao_sei(selection: str, mapping: dict):
    if selection == "Não sei":
        return 0, 1
    return int(mapping[selection]), 0

def make_input_row(inputs: dict, feature_list):
    row = empty_row(feature_list)
    set_if_exists(row, "TP_FAIXA_ETARIA", inputs["TP_FAIXA_ETARIA"])
    set_if_exists(row, "Qtd Residentes", inputs["Qtd_Residentes"])
    set_if_exists(row, "Renda Familiar_ord", inputs["Renda_Familiar_ord"])
    set_if_exists(row, "Instrução do pai_ord", inputs["Instrucao_pai_ord"])
    set_if_exists(row, "Instrução da mãe_ord", inputs["Instrucao_mae_ord"])
    set_if_exists(row, "Profissão do pai_ord", inputs["Prof_pai_ord"])
    set_if_exists(row, "Profissão da mãe_ord", inputs["Prof_mae_ord"])
    set_if_exists(row, "Instrução do pai_nao_sei", inputs["Instrucao_pai_ns"])
    set_if_exists(row, "Instrução da mãe_nao_sei", inputs["Instrucao_mae_ns"])
    set_if_exists(row, "Profissão do pai_nao_sei", inputs["Prof_pai_ns"])
    set_if_exists(row, "Profissão da mãe_nao_sei", inputs["Prof_mae_ns"])
    for s in ["F","M"]:
        set_if_exists(row, f"TP_SEXO_{s}", 1 if inputs["TP_SEXO"]==s else 0)
    for k in ["0.0","1.0","2.0","3.0","4.0"]:
        set_if_exists(row, f"TP_ESTADO_CIVIL_{k}", 1 if inputs["TP_ESTADO_CIVIL"]==k else 0)
    for k in ["0","1","2","3","4","5","6"]:
        set_if_exists(row, f"TP_COR_RACA_{k}", 1 if inputs["TP_COR_RACA"]==k else 0)
    for k in ["0","1","2","3","4"]:
        set_if_exists(row, f"TP_NACIONALIDADE_{k}", 1 if inputs["TP_NACIONALIDADE"]==k else 0)
    for k in ["1","2","4"]:
        set_if_exists(row, f"TP_ST_CONCLUSAO_{k}", 1 if inputs["TP_ST_CONCLUSAO"]==k else 0)
    for k in ["1","2","3"]:
        set_if_exists(row, f"TP_ESCOLA_{k}", 1 if inputs["TP_ESCOLA"]==k else 0)
    set_if_exists(row, f"SG_UF_PROVA_{inputs['UF']}", 1)
    for k in ["0","1"]:
        set_if_exists(row, f"TP_LINGUA_{k}", 1 if inputs["TP_LINGUA"]==k else 0)
    set_if_exists(row, "Acesso à internet_A", inputs["Internet_A"])
    set_if_exists(row, "Acesso à internet_B", inputs["Internet_B"])
    return row[feature_list]

# ====== UI principal ======
st.title("🎯 Simulador de Nota Estimada do ENEM")

with st.form("form"):
    st.subheader("Preencha suas informações")
    c1, c2 = st.columns(2)

    with c1:
        fx = st.selectbox("Faixa etária", FAIXA_ETARIA, index=1)
        qtd = st.number_input("Quantidade de residentes no domicílio", 1, 20, 3)
        renda_txt = st.selectbox("Renda familiar (categoria)", RENDA_LIST, index=5)
        sexo = st.radio("Sexo", ["F","M"], horizontal=True)
        escola = st.radio("Tipo de escola", ["1","2","3"], index=1, captions=["Não respondeu","Pública","Privada"])
        uf = st.selectbox("UF da prova", UF_LIST, index=24)

    with c2:
        lingua = st.radio("Língua estrangeira feita", ["0","1"], index=0, captions=["Inglês","Espanhol"])
        est_civil = st.selectbox(
            "Estado civil", ["0.0","1.0","2.0","3.0","4.0"], index=0,
            format_func=lambda x: {
                "0.0":"Não informado","1.0":"Solteiro(a)","2.0":"Casado/Companheiro",
                "3.0":"Divorciado(a)","4.0":"Viúvo(a)"
            }[x]
        )
        raca = st.selectbox(
            "Cor/Raça", ["0","1","2","3","4","5","6"], index=3,
            format_func=lambda x: {
                "0":"Não declarado","1":"Branco","2":"Preto","3":"Pardo",
                "4":"Amarelo","5":"Indígena","6":"Sem informação"
            }[x]
        )
        nac = st.selectbox(
            "Nacionalidade", ["0","1","2","3","4"], index=1,
            format_func=lambda x: {
                "0":"Não informada","1":"Brasileiro","2":"Naturalizado",
                "3":"Estrangeiro","4":"Brasileiro nascido no exterior"
            }[x]
        )
        st_conc = st.selectbox(
            "Situação de conclusão do EM", ["1","2","4"], index=1,
            format_func=lambda x: {"1":"Já concluiu","2":"Conclui este ano","4":"Conclui depois deste ano"}[x]
        )

    # ====== Escolaridade dos pais ======
    instr_pai_sel = st.selectbox("Instrução do pai", ["Não sei"] + INSTR_LIST, index=0)
    st.markdown('<div class="legend-title">Legenda — Escolaridade</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend-box"><ul>' + "".join([f"<li>{x}</li>" for x in INSTR_LIST]) + "</ul></div>", unsafe_allow_html=True)

    instr_mae_sel = st.selectbox("Instrução da mãe", ["Não sei"] + INSTR_LIST, index=0)
    st.markdown('<div class="legend-title">Legenda — Escolaridade</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend-box"><ul>' + "".join([f"<li>{x}</li>" for x in INSTR_LIST]) + "</ul></div>", unsafe_allow_html=True)

    # ====== Profissão dos pais ======
    prof_pai_sel  = st.selectbox("Profissão do pai (grupo)", ["Não sei"] + PROF_LIST, index=0)
    st.markdown('<div class="legend-title">Legenda — Profissões</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend-box"><ul>' + "".join([f"<li>{x}</li>" for x in PROF_LIST]) + "</ul></div>", unsafe_allow_html=True)

    prof_mae_sel  = st.selectbox("Profissão da mãe (grupo)", ["Não sei"] + PROF_LIST, index=0)
    st.markdown('<div class="legend-title">Legenda — Profissões</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend-box"><ul>' + "".join([f"<li>{x}</li>" for x in PROF_LIST]) + "</ul></div>", unsafe_allow_html=True)

    st.markdown("**Acesso à internet**")
    internet = st.radio("Acesso à internet no domicílio", ["Tem acesso","Não tem acesso"], horizontal=True)

    submitted = st.form_submit_button("Calcular nota estimada")

if submitted:
    Instrucao_pai_ord, Instrucao_pai_ns = map_nao_sei(instr_pai_sel, INSTR_MAP)
    Instrucao_mae_ord, Instrucao_mae_ns = map_nao_sei(instr_mae_sel, INSTR_MAP)
    Prof_pai_ord, Prof_pai_ns = map_nao_sei(prof_pai_sel, PROF_MAP)
    Prof_mae_ord, Prof_mae_ns = map_nao_sei(prof_mae_sel, PROF_MAP)

    inputs = dict(
        TP_FAIXA_ETARIA = FAIXA_MAP[fx],
        Qtd_Residentes = int(qtd),
        Renda_Familiar_ord = RENDA_MAP[renda_txt],
        TP_SEXO = sexo,
        TP_ESCOLA = escola,
        UF = uf,
        TP_LINGUA = "0" if lingua=="0" else "1",
        TP_ESTADO_CIVIL = est_civil,
        TP_COR_RACA = raca,
        TP_NACIONALIDADE = nac,
        TP_ST_CONCLUSAO = st_conc,
        Instrucao_pai_ord = Instrucao_pai_ord,
        Instrucao_mae_ord = Instrucao_mae_ord,
        Prof_pai_ord = Prof_pai_ord,
        Prof_mae_ord = Prof_mae_ord,
        Instrucao_pai_ns = Instrucao_pai_ns,
        Instrucao_mae_ns = Instrucao_mae_ns,
        Prof_pai_ns = Prof_pai_ns,
        Prof_mae_ns = Prof_mae_ns,
        Internet_A = 1 if internet=="Tem acesso" else 0,
        Internet_B = 1 if internet=="Não tem acesso" else 0,
    )

    row = make_input_row(inputs, FEATURE_LIST)

    if _missing_cols:
        st.warning("Algumas colunas não foram localizadas: " + ", ".join(sorted(set(_missing_cols))))
        if _suggestions:
            st.info("Sugestões de colunas parecidas:")
            st.json(_suggestions)

    yhat = float(model.predict(row)[0])
    st.success(f"🎯 Nota estimada: **{yhat:.1f}**")
    st.caption("Estimativa baseada nas suas respostas. Não representa garantia de resultado.")

    st.subheader("📦 Dados enviados para o modelo (dict)")
    st.json(inputs)
    st.subheader("📊 Vetor de features (1 linha)")
    st.dataframe(row)
