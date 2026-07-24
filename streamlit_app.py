# =============================================================================
# SIMULADOR CAUSAL DE POLÍTICA PÚBLICA · Streamlit + Streamlit Community Cloud
# Tesis de Maestría · SNI y Transición Energética Baja en Carbono
# Autor: Angel A. Ruiz Muñiz — UASLP / SECIHTI
#
# Migración desde Gradio 4.x a Streamlit 1.38+ conservando:
#   · Motor causal v11 S3b (θ heterogéneos líder / seguidor por motor)
#   · Panel FE + Driscoll-Kraay (BW=4) + DML Secuencial + Bootstrap N=2000
#   · Canal de mediación GIDE → Patentes → ShareLC (γ=0.1077)
#   · Comparativo multi-país (11 economías del panel)
#   · Análisis sexenal México (Calderón, Peña Nieto, AMLO, Sheinbaum)
#   · Exportación a PDF de 3–4 páginas
#
# Referencias metodológicas centrales:
#   Driscoll & Kraay (1998); Chernozhukov et al. (2018);
#   Preacher & Hayes (2008); Abramovitz (1986); Unruh (2000).
# =============================================================================

import os
import io
import tempfile
import unicodedata
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

import streamlit as st

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN GLOBAL DE STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Simulador Causal · Innovación y Transición Energética",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": (
            "**Simulador Causal de Política Pública**\n\n"
            "Innovación Renovable y Transición Energética Baja en Carbono.\n\n"
            "Tesis de Maestría — Angel A. Ramírez Martínez\n"
            "UASLP / SECIHTI"
        ),
    },
)


# =============================================================================
# A. PARÁMETROS CAUSALES v11 — HETEROGENEIDAD LÍDER / SEGUIDOR
# =============================================================================
LIDERES_M1 = {"china", "japon", "estados_unidos", "corea_del_sur",
              "alemania", "francia"}
LIDERES_M2 = {"francia", "brasil", "canada", "dinamarca",
              "chile", "alemania"}

THETA_STK = {
    "M1": {
        "GIDE":    dict(seg= 1.1614, se_seg=0.6979, adj=-1.3923, se_adj=0.7863,
                        sig_seg="*",    sig_adj="*"   ),
        "Credito": dict(seg=-0.8285, se_seg=0.2990, adj= 0.8816, se_adj=0.3304,
                        sig_seg="***",  sig_adj="***" ),
        "IED":     dict(seg= 0.0439, se_seg=0.0387, adj=-0.1551, se_adj=0.0755,
                        sig_seg="n.s.", sig_adj="**"  ),
        "PagosPI": dict(seg=-0.0553, se_seg=0.2154, adj=-0.0086, se_adj=0.1373,
                        sig_seg="n.s.", sig_adj="n.s."),
    },
    "M2": {
        "GIDE":    dict(seg=-0.1678, se_seg=0.2621, adj= 0.2607, se_adj=0.7724,
                        sig_seg="n.s.", sig_adj="n.s."),
        "Credito": dict(seg=-0.0811, se_seg=0.2245, adj=-1.0000, se_adj=0.4043,
                        sig_seg="n.s.", sig_adj="**"  ),
        "IED":     dict(seg= 0.1884, se_seg=0.1057, adj=-0.2040, se_adj=0.0889,
                        sig_seg="*",    sig_adj="**"  ),
        "PagosPI": dict(seg=-0.1988, se_seg=0.0752, adj= 0.0868, se_adj=0.0614,
                        sig_seg="***",  sig_adj="n.s."),
    },
}

THETA_SHK = {
    "M1": {
        "GIDE":    dict(theta=-0.0558, se=0.726, sig="n.s."),
        "Credito": dict(theta=-0.9539, se=0.436, sig="*"   ),
        "IED":     dict(theta=-0.0612, se=0.020, sig="***" ),
        "PagosPI": dict(theta=-0.3032, se=0.148, sig="*"   ),
    },
    "M2": {
        "GIDE":    dict(theta=-1.9203, se=0.702, sig="**"  ),
        "Credito": dict(theta=-0.8894, se=0.439, sig="*"   ),
        "IED":     dict(theta=-0.0742, se=0.021, sig="***" ),
        "PagosPI": dict(theta=-0.3477, se=0.155, sig="**"  ),
    },
}

GAMMA_PPER_LC = 0.1077
SE_PPER_LC    = 0.0513

PALANCAS = ["GIDE", "Credito", "IED", "PagosPI"]
NOM_PAL  = {
    "GIDE":    "GIDE (% PIB)",
    "Credito": "Crédito Privado (% PIB)",
    "IED":     "IED (% PIB)",
    "PagosPI": "Pagos Propiedad Intelectual",
}
SIG_COL = {"***": "#d63031", "**": "#e17055", "*": "#e6b800", "n.s.": "#b2bec3"}

N_BOOT     = 2000
DRIFT_WIN  = 5
OPCIONES_H = [1, 3, 5, 10]


def get_theta_stk(pais: str, motor: str) -> Tuple[Dict, bool]:
    lid_set  = LIDERES_M1 if motor == "M1" else LIDERES_M2
    es_lider = pais in lid_set
    out = {}
    for p, h in THETA_STK[motor].items():
        if es_lider:
            theta = h["seg"] + h["adj"]
            se    = np.sqrt(h["se_seg"] ** 2 + h["se_adj"] ** 2)
        else:
            theta, se = h["seg"], h["se_seg"]
        out[p] = (float(theta), float(se))
    return out, es_lider


# =============================================================================
# B. CARGA DE DATOS Y DETECCIÓN DE COLUMNAS
# =============================================================================
def limpiar(txt):
    if not isinstance(txt, str):
        return txt
    txt = unicodedata.normalize("NFKD", txt).encode("ASCII", "ignore").decode()
    txt = txt.lower().strip()
    for ch in " ,().-/%$":
        txt = txt.replace(ch, "_")
    while "__" in txt:
        txt = txt.replace("__", "_")
    return txt.strip("_")


def _col(df, *kws, req=True):
    for c in df.columns:
        if all(k in c for k in kws):
            return c
    if req:
        raise KeyError(f"Columna no encontrada: {kws}")
    return None


def logit(p, eps=0.001):
    p = np.clip(np.asarray(p, dtype=float) / 100.0, eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit_pct(x):
    return 100.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@st.cache_data
def cargar_datos():
    """Carga y prepara el panel — cacheado por Streamlit."""
    candidates = ["data.csv", "./data.csv", "/mount/src/data.csv"]
    for _p in candidates:
        if os.path.exists(_p):
            df = pd.read_csv(_p)
            break
    else:
        raise FileNotFoundError("No se encontró data.csv en la raíz.")
    df.columns = [limpiar(c) for c in df.columns]
    df["pais"] = df["pais"].apply(limpiar)
    _year_c = [c for c in df.columns
               if c.lower() in ("ano", "año", "anio", "year", "yr")]
    if _year_c:
        df = df.rename(columns={_year_c[0]: "año"})
    df = df.sort_values(["pais", "año"]).reset_index(drop=True)
    return df


df_raw = cargar_datos()

COL_PPER  = _col(df_raw, "patentes", "renovables")
COL_SHARE = _col(df_raw, "share", "low_carbon")
COL_GIDE  = _col(df_raw, "gasto", "investigacion")
COL_CRED  = _col(df_raw, "credito", "privado")
COL_IED   = _col(df_raw, "inversion_extranjera")
COL_PI    = _col(df_raw, "cargos", "propiedad_intelectual")
COL_INV   = _col(df_raw, "investigadores", "millon")
COL_EXP   = _col(df_raw, "exportaciones", "alta_tecnologia")
COL_ART   = _col(df_raw, "articulos", "publicaciones")


def _find_pib(df):
    EXCL = {"gasto", "gide", "credito", "ied", "pago", "inversion",
            "porcentaje", "porc", "share", "capita", "deuda", "bajo"}
    cands = [c for c in df.columns
             if ("pib" in c or "gdp" in c or "producto_interno" in c)
             and not any(x in c for x in EXCL)]
    if not cands:
        return None
    return max(cands, key=lambda c: df[c].abs().median())


def _find_elec(df):
    KWS = [("electricidad", "total"), ("total", "electricidad"),
           ("generacion", "total"), ("produccion", "electricidad"),
           ("kwh",), ("twh",), ("energia", "total")]
    EXCL = {"share", "low_carbon", "renovable", "limpia", "fossil"}
    for kws in KWS:
        for c in df.columns:
            if all(k in c for k in kws) and not any(x in c for x in EXCL):
                return c
    return None


COL_PIB   = _find_pib(df_raw)
COL_TELEC = _find_elec(df_raw)

PAL_COLS  = {"GIDE": COL_GIDE, "Credito": COL_CRED,
             "IED": COL_IED, "PagosPI": COL_PI}
ESTRUCT_W = {"Investigadores": COL_INV,
             "Export. Alta Tec.": COL_EXP,
             "Artículos Cient.": COL_ART}
NOM_W_UND = {"Investigadores": "inv./M hab.",
             "Export. Alta Tec.": "% exp.",
             "Artículos Cient.": "art./año"}

NOMBRES_PAIS = {
    "alemania": "Alemania", "brasil": "Brasil", "canada": "Canadá",
    "chile": "Chile", "china": "China", "corea_del_sur": "Corea del Sur",
    "dinamarca": "Dinamarca", "estados_unidos": "EE.UU.",
    "francia": "Francia", "japon": "Japón", "mexico": "México",
}

PAISES_ESTUDIO = set(NOMBRES_PAIS.keys())
PAISES_DISP    = sorted(p for p in df_raw["pais"].unique() if p in PAISES_ESTUDIO)


# =============================================================================
# C. ELASTICIDADES β(W|D) — SPILLOVERS ESTRUCTURALES
# =============================================================================
@st.cache_data
def calcular_beta_wd():
    BETA: Dict[str, Dict[str, float]] = {}
    for _nw, _cw in ESTRUCT_W.items():
        BETA[_nw] = {}
        if not _cw or _cw not in df_raw.columns:
            continue
        _wlog  = np.log1p(df_raw[_cw].clip(lower=0))
        _wmean = df_raw.groupby("pais")[_cw].transform(
            lambda x: np.log1p(x.clip(lower=0)).mean())
        _wdm = _wlog - _wmean
        for _pk, _cd in PAL_COLS.items():
            if not _cd or _cd not in df_raw.columns:
                continue
            if _cd == COL_IED:
                _dlog  = np.sign(df_raw[_cd]) * np.log1p(np.abs(df_raw[_cd]))
                _dmean = df_raw.groupby("pais")[_cd].transform(
                    lambda x: (np.sign(x) * np.log1p(np.abs(x))).mean())
            else:
                _dlog  = np.log1p(df_raw[_cd].clip(lower=0))
                _dmean = df_raw.groupby("pais")[_cd].transform(
                    lambda x: np.log1p(x.clip(lower=0)).mean())
            _ddm = _dlog - _dmean
            _vd  = _ddm.var()
            BETA[_nw][_pk] = float(_ddm.cov(_wdm) / _vd) if _vd > 1e-10 else 0.0
    return BETA


BETA_W_D = calcular_beta_wd()


# =============================================================================
# D. MOTOR DE SIMULACIÓN
# =============================================================================
def drift_hist(serie_log, window=DRIFT_WIN):
    v = np.asarray(serie_log, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.mean(np.diff(v[-window:]))) if len(v) >= 2 else 0.0


def pct2dlog(pct):
    return np.log(max(1.0 + pct / 100.0, 0.001))


ABS_STK_CFG = {
    "GIDE":    {"min":    0.0,   "max":  8.0,      "step": 0.05,
                "unit": "% PIB",     "scale": 1.0},
    "Credito": {"min":    0.0,   "max": 200.0,     "step": 1.0,
                "unit": "% PIB",     "scale": 1.0},
    "IED":     {"min":   -5.0,   "max":  15.0,     "step": 0.1,
                "unit": "% PIB",     "scale": 1.0},
    "PagosPI": {"min":    0.0,   "max": 100_000.0, "step": 100.0,
                "unit": "mill. USD", "scale": 1e6},
}
ABS_SHK_CFG = {
    "GIDE":    {"min":   -2.0,   "max":  5.0,      "step": 0.05,
                "unit": "Δ pp PIB",    "scale": 1.0},
    "Credito": {"min":  -30.0,   "max":  50.0,     "step": 1.0,
                "unit": "Δ pp PIB",    "scale": 1.0},
    "IED":     {"min":   -3.0,   "max":  10.0,     "step": 0.1,
                "unit": "Δ pp PIB",    "scale": 1.0},
    "PagosPI": {"min":  -5000.0, "max":  10000.0,  "step": 100.0,
                "unit": "Δ mill. USD", "scale": 1e6},
}


def _get_base_vals(pais_key):
    df_p = df_raw[df_raw["pais"] == pais_key].sort_values("año")
    if len(df_p) == 0:
        return {p: 0.0 for p in PALANCAS}
    ult = df_p.iloc[-1]
    return {
        p: float(ult[PAL_COLS[p]])
        if (PAL_COLS[p] and PAL_COLS[p] in ult.index
            and not pd.isna(ult.get(PAL_COLS[p])))
        else 0.0
        for p in PALANCAS
    }


def _dls_exact(pal, v_new, v_base):
    if pal == "IED":
        f = lambda v: np.sign(v) * np.log1p(abs(v))
        return float(f(v_new) - f(v_base))
    return float(np.log1p(max(v_new, 0)) - np.log1p(max(v_base, 0)))


def _stk_to_eff_pct(pal, slider_val, v_base_raw):
    v_new = slider_val * ABS_STK_CFG[pal]["scale"]
    dls   = _dls_exact(pal, v_new, v_base_raw)
    return float((np.exp(dls) - 1) * 100)


def _shk_to_eff_pct(pal, delta_slider, v_base_raw):
    delta_raw = delta_slider * ABS_SHK_CFG[pal]["scale"]
    v_new     = v_base_raw + delta_raw
    dls       = _dls_exact(pal, v_new, v_base_raw)
    return float((np.exp(dls) - 1) * 100)


def simular(df_p, d_stk, d_shk, n_years, pais_key, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    lp_hist  = np.log1p(df_p[COL_PPER].clip(lower=0).values)
    llc_hist = logit(df_p[COL_SHARE].values)
    lp0 = float(lp_hist[-1]); ll0 = float(llc_hist[-1])
    dr_m1 = drift_hist(lp_hist); dr_m2 = drift_hist(llc_hist)

    th_m1, es_l1 = get_theta_stk(pais_key, "M1")
    th_m2, es_l2 = get_theta_stk(pais_key, "M2")

    dls  = {p: pct2dlog(d_stk[p]) for p in PALANCAS}
    dlsh = {p: pct2dlog(d_shk[p]) for p in PALANCAS}

    def _run(t1, t2, gam):
        lp_i = np.zeros(n_years + 1); lp_s = np.zeros(n_years + 1)
        ll_i = np.zeros(n_years + 1); ll_s = np.zeros(n_years + 1)
        lp_i[0] = lp_s[0] = lp0
        ll_i[0] = ll_s[0] = ll0
        for t in range(1, n_years + 1):
            is_t1 = (t == 1)
            pol1 = sum(
                t1[p] * dls[p]
                + (THETA_SHK["M1"][p]["theta"] * dlsh[p] if is_t1 else 0.0)
                for p in PALANCAS
            )
            lp_i[t] = lp_i[t - 1] + dr_m1
            lp_s[t] = lp_s[t - 1] + dr_m1 + pol1
            pol2 = sum(
                t2[p] * dls[p]
                + (THETA_SHK["M2"][p]["theta"] * dlsh[p] if is_t1 else 0.0)
                for p in PALANCAS
            )
            ll_i[t] = ll_i[t - 1] + dr_m2
            ll_s[t] = ll_s[t - 1] + dr_m2 + pol2 + gam * pol1
        return lp_i[1:], lp_s[1:], ll_i[1:], ll_s[1:]

    t1n = {p: th_m1[p][0] for p in PALANCAS}
    t2n = {p: th_m2[p][0] for p in PALANCAS}
    lp_i, lp_s, ll_i, ll_s = _run(t1n, t2n, GAMMA_PPER_LC)

    _mu1s = np.array([THETA_STK["M1"][p]["seg"]    for p in PALANCAS])
    _se1s = np.array([THETA_STK["M1"][p]["se_seg"] for p in PALANCAS])
    _mu2s = np.array([THETA_STK["M2"][p]["seg"]    for p in PALANCAS])
    _se2s = np.array([THETA_STK["M2"][p]["se_seg"] for p in PALANCAS])
    th1_b = rng.normal(_mu1s, _se1s, (n_boot, len(PALANCAS)))
    th2_b = rng.normal(_mu2s, _se2s, (n_boot, len(PALANCAS)))
    if es_l1:
        th1_b += rng.normal(
            [THETA_STK["M1"][p]["adj"]    for p in PALANCAS],
            [THETA_STK["M1"][p]["se_adj"] for p in PALANCAS],
            (n_boot, len(PALANCAS))
        )
    if es_l2:
        th2_b += rng.normal(
            [THETA_STK["M2"][p]["adj"]    for p in PALANCAS],
            [THETA_STK["M2"][p]["se_adj"] for p in PALANCAS],
            (n_boot, len(PALANCAS))
        )
    gb_b = rng.normal(GAMMA_PPER_LC, SE_PPER_LC, n_boot)
    blp = np.zeros((n_boot, n_years)); bll = np.zeros((n_boot, n_years))
    for b in range(n_boot):
        _, as_, _, bs = _run(
            dict(zip(PALANCAS, th1_b[b])),
            dict(zip(PALANCAS, th2_b[b])),
            gb_b[b],
        )
        blp[b] = as_; bll[b] = bs

    blp_lvl = np.expm1(blp); bll_pct = inv_logit_pct(bll)
    ps_lo = np.percentile(blp_lvl, 2.5,  axis=0)
    ps_hi = np.percentile(blp_lvl, 97.5, axis=0)
    ls_lo = np.percentile(bll_pct, 2.5,  axis=0)
    ls_hi = np.percentile(bll_pct, 97.5, axis=0)

    ult = df_p.iloc[-1]
    spillovers = {}
    for nom_w, col_w in ESTRUCT_W.items():
        vb = float(ult[col_w]) if col_w and col_w in ult.index \
                                  and not pd.isna(ult[col_w]) else np.nan
        dlog_w = sum(BETA_W_D.get(nom_w, {}).get(p, 0.0) * dls[p]
                     for p in PALANCAS)
        pct_w = float(np.expm1(dlog_w)) * 100.0
        spillovers[nom_w] = {
            "base":   vb,
            "nuevo":  vb * (1.0 + pct_w / 100.0) if not np.isnan(vb) else np.nan,
            "pct":    pct_w,
            "unidad": NOM_W_UND.get(nom_w, ""),
        }

    return dict(
        lp_i=lp_i, lp_s=lp_s, ll_i=ll_i, ll_s=ll_s,
        ps_lo=ps_lo, ps_hi=ps_hi, ls_lo=ls_lo, ls_hi=ls_hi,
        blp_lvl=blp_lvl, bll_pct=bll_pct,
        dr_m1=dr_m1, dr_m2=dr_m2,
        spillovers=spillovers,
        th_m1=th_m1, th_m2=th_m2,
        es_l1=es_l1, es_l2=es_l2,
    )


def top_drivers(d_stk, d_shk, n_years, th_m1, th_m2):
    out = {}
    for motor in ("M1", "M2"):
        th_s = th_m1 if motor == "M1" else th_m2
        out[motor] = {}
        for p in PALANCAS:
            ds  = pct2dlog(d_stk[p])
            dsh = pct2dlog(d_shk[p])
            stk = th_s[p][0] * ds * n_years
            shk = THETA_SHK[motor][p]["theta"] * dsh
            fb  = (GAMMA_PPER_LC * th_m1[p][0] * ds * n_years
                   if motor == "M2" else 0.0)
            out[motor][p] = {"stock": stk, "shock": shk,
                             "feedback": fb, "total": stk + shk + fb}
    return out


# =============================================================================
# E. ESTILO MATPLOTLIB
# =============================================================================
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlepad":     10,
    "figure.facecolor":  "#fafafa",
    "axes.facecolor":    "#fafafa",
    "grid.color":        "#e0e0e0",
    "xtick.color":       "#444",
    "ytick.color":       "#444",
    "text.color":        "#2d3436",
})

C_HIST  = "#2d3436"; C_INER  = "#7f8c8d"; C_SCEN  = "#00897b"
C_CI    = "#80cbc4"; C_STK   = "#00897b"; C_SHK   = "#e67e22"
C_FB    = "#8e44ad"; C_TOT   = "#1a252f"
C_POS   = "#27ae60"; C_POS_S = "#82e0aa"
C_NEG   = "#e74c3c"; C_NEG_S = "#f1948a"


# =============================================================================
# F. FIGURA DE ESCENARIO INDIVIDUAL
# =============================================================================
def fig_escenario(pais_key, motor_vis, n_years, d_stk_abs, d_shk_abs):
    df_p = df_raw[df_raw["pais"] == pais_key].sort_values("año").copy()
    if len(df_p) < 3:
        return None, {"error": f"Datos insuficientes para {pais_key}"}

    ano_b  = int(df_p["año"].iloc[-1])
    f_anos = list(range(ano_b + 1, ano_b + n_years + 1))
    anos_h = df_p["año"].values

    bv = _get_base_vals(pais_key)
    d_stk = {p: _stk_to_eff_pct(p, d_stk_abs[p], bv[p]) for p in PALANCAS}
    d_shk = {p: _shk_to_eff_pct(p, d_shk_abs[p], bv[p]) for p in PALANCAS}

    any_c = any(
        abs(d_stk_abs[p] * ABS_STK_CFG[p]["scale"] - bv[p])
        > 1e-6 * max(abs(bv[p]), 1)
        or abs(d_shk_abs[p]) > 1e-6
        for p in PALANCAS
    )

    res = simular(df_p, d_stk, d_shk, n_years, pais_key)
    td  = top_drivers(d_stk, d_shk, n_years, res["th_m1"], res["th_m2"]) if any_c else None

    pper_h = np.expm1(np.log1p(df_p[COL_PPER].clip(lower=0).values))
    lc_h   = df_p[COL_SHARE].values
    pper_i = np.expm1(res["lp_i"]);   pper_s = np.expm1(res["lp_s"])
    lc_i   = inv_logit_pct(res["ll_i"]); lc_s = inv_logit_pct(res["ll_s"])

    dif_pp = ((pper_s[-1] - pper_i[-1]) / max(pper_i[-1], 0.1)) * 100.0
    dif_lc = lc_s[-1] - lc_i[-1]

    sign_stab_m1 = sign_stab_m2 = None
    if any_c:
        pp_dif_b = ((res["blp_lvl"][:, -1] - pper_i[-1])
                    / max(pper_i[-1], 0.1)) * 100.0
        lc_dif_b = res["bll_pct"][:, -1] - lc_i[-1]
        sign_stab_m1 = float((pp_dif_b >= 0).mean() * 100
                             if dif_pp >= 0 else (pp_dif_b < 0).mean() * 100)
        sign_stab_m2 = float((lc_dif_b >= 0).mean() * 100
                             if dif_lc >= 0 else (lc_dif_b < 0).mean() * 100)

    n_rows = 3 if motor_vis == "AMBOS" else 2
    fig = plt.figure(figsize=(15.5, 4.9 * n_rows), facecolor="#fafafa")
    gspec = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.55, wspace=0.32)
    reg_lbl = (f'M1: {"Líder" if res["es_l1"] else "Seguidor"}  |  '
               f'M2: {"Líder" if res["es_l2"] else "Seguidor"}')
    pais_disp = NOMBRES_PAIS.get(pais_key, pais_key.title())
    fig.suptitle(
        f'{pais_disp}  ·  Horizonte {n_years} año{"s" if n_years > 1 else ""} '
        f'({ano_b + 1}–{ano_b + n_years})  ·  {reg_lbl}',
        fontsize=13, fontweight="bold", y=1.005, color="#1b3a4b"
    )

    def _ax_traj(row, hist, iner, scen, s_lo, s_hi, ylab, titulo, dif_str, motor):
        ax  = fig.add_subplot(gspec[row, :2])
        axd = fig.add_subplot(gspec[row, 2])
        xp     = [ano_b] + f_anos
        i_plot = np.insert(iner, 0, hist[-1])
        s_plot = np.insert(scen, 0, hist[-1])
        sl_p   = np.insert(s_lo, 0, hist[-1])
        sh_p   = np.insert(s_hi, 0, hist[-1])

        ax.plot(anos_h, hist, color=C_HIST, lw=1.8, marker=".", ms=5,
                alpha=0.85, label="Histórico real", zorder=5)
        ax.axvline(ano_b + 0.5, color="#9ca3af", ls=":", lw=1.1, zorder=3)
        ax.plot(xp, i_plot, "--", color=C_INER, lw=2.0,
                label=f"Sin política (tendencia {DRIFT_WIN}a)", zorder=4)
        if any_c:
            ax.plot(xp, s_plot, "-o", color=C_SCEN, lw=2.5, ms=4.5,
                    label="Escenario simulado", zorder=6)
            ax.fill_between(xp, sl_p, sh_p, color=C_CI, alpha=0.25,
                            label="IC 95% Bootstrap", zorder=5)
            _dif = dif_pp if motor == "M1" else dif_lc
            col_ann = C_SCEN if _dif >= 0 else C_NEG
            ax.annotate(dif_str, xy=(xp[-1], s_plot[-1]),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=9, color=col_ann, fontweight="bold",
                        va="center")

        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(titulo, fontsize=11, fontweight="bold", color="#1b3a4b")
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.85)
        ax.grid(True, ls=":", alpha=0.5)
        ax.tick_params(labelsize=9)
        all_x = list(anos_h) + xp
        ax.set_xticks(sorted(set(int(y) for y in all_x)))
        ax.tick_params(axis="x", rotation=45)

        if motor == "M1" and any_c:
            _h_max = float(np.nanmax(hist)) if len(hist) > 0 else 1.0
            _s_max = float(np.nanmax(np.concatenate([scen, s_hi])))
            _ratio = _s_max / max(_h_max, 1.0)
            if _ratio > 50:
                _lin = max(_h_max * 10, 1.0)
                ax.set_yscale("symlog", linthresh=_lin)
                ax.set_ylabel(f"{ylab} (escala log)", fontsize=10, color="#7c3aed")
                ax.yaxis.label.set_color("#7c3aed")
                ax.tick_params(axis="y", colors="#7c3aed")
                ax.text(0.99, 0.02,
                        f"⚠ Escala log — proyección {_ratio:.0f}× el histórico",
                        transform=ax.transAxes, fontsize=8, color="#7c3aed",
                        ha="right", va="bottom", style="italic",
                        bbox=dict(facecolor="#f5f3ff", alpha=0.8,
                                  edgecolor="#7c3aed", boxstyle="round,pad=0.3"))

        if any_c and td:
            data  = td[motor]
            items = sorted(data.items(), key=lambda x: abs(x[1]["total"]), reverse=True)
            noms  = [NOM_PAL[p][:17] for p, _ in items]
            vstk  = [v["stock"]    for _, v in items]
            vshk  = [v["shock"]    for _, v in items]
            vfb   = [v["feedback"] for _, v in items]
            vtot  = [v["total"]    for _, v in items]
            x = np.arange(len(PALANCAS)); w = 0.26
            axd.bar(x - w, vstk, w, label="Sostenido\n(acumulado)",
                    color=C_STK, alpha=0.85, edgecolor="white", linewidth=0.5)
            axd.bar(x, vshk, w, label="Impulso\n(año 1)",
                    color=C_SHK, alpha=0.85, edgecolor="white", linewidth=0.5)
            if motor == "M2":
                axd.bar(x + w, vfb, w, label="Indirecto\n(GIDE→pat.→LC)",
                        color=C_FB, alpha=0.75, edgecolor="white", linewidth=0.5)
            axd.scatter(x, vtot, color=C_TOT, s=55, zorder=6,
                        label="Efecto total", marker="D",
                        edgecolors="white", linewidths=0.5)
            axd.axhline(0, color="#374151", lw=0.8, ls="--", alpha=0.7)
            axd.set_xticks(x)
            axd.set_xticklabels(noms, fontsize=7.5, rotation=22, ha="right")
            axd.set_ylabel(f"Impacto acumulado (t+{n_years})", fontsize=8.5)
            axd.set_title(f"¿Qué palanca impulsa más?\n({motor} · t+{n_years})",
                          fontsize=9, fontweight="bold", color="#1b3a4b")
            axd.legend(fontsize=7, loc="upper right", framealpha=0.85)
            axd.grid(True, ls=":", alpha=0.45, axis="y")
        else:
            axd.text(0.5, 0.5,
                     "Mueve algún slider fuera\ndel baseline para ver\nlos principales impulsores",
                     ha="center", va="center", transform=axd.transAxes,
                     fontsize=10, color="#9ca3af")
            axd.set_xticks([]); axd.set_yticks([])

    if motor_vis in ("M1", "AMBOS"):
        _ax_traj(0, pper_h, pper_i, pper_s, res["ps_lo"], res["ps_hi"],
                 "Patentes renovables / año", "Motor 1 — Patentes Renovables",
                 f"Δ {dif_pp:+.1f}% vs inercia", "M1")

    if motor_vis in ("M2", "AMBOS"):
        row2 = 1 if motor_vis == "AMBOS" else 0
        _ax_traj(row2, lc_h, lc_i, lc_s, res["ls_lo"], res["ls_hi"],
                 "Share Low-Carbon (%)", "Motor 2 — Share Low-Carbon",
                 f"Δ {dif_lc:+.2f} pp vs inercia", "M2")

    if motor_vis == "AMBOS":
        ax_sp  = fig.add_subplot(gspec[2, :2])
        ax_ctx = fig.add_subplot(gspec[2, 2])
        any_stk = any(abs(d_stk[p]) > 0.01 for p in PALANCAS)

        if any_c and any_stk:
            sp = res["spillovers"]
            noms_w = list(sp.keys())
            pcts_w = [sp[k]["pct"] for k in noms_w]
            cols_sp = [C_SCEN if pv >= 0 else C_NEG for pv in pcts_w]
            ax_sp.barh(noms_w, pcts_w, color=cols_sp, alpha=0.82,
                       edgecolor="white", height=0.5)
            ax_sp.axvline(0, color="#374151", lw=0.8, ls="--", alpha=0.7)
            for i, (k, pct) in enumerate(zip(noms_w, pcts_w)):
                ax_sp.text(pct + (0.25 if pct >= 0 else -0.25), i,
                           f"{pct:+.1f}%", va="center", fontsize=10,
                           color="#1b3a4b", fontweight="bold")
            ax_sp.set_title(
                "Efectos colaterales en el ecosistema de innovación\n"
                "(correlación histórica — informativo, no causal)",
                fontsize=10, fontweight="bold", color="#1b3a4b"
            )
            ax_sp.set_xlabel("Cambio estimado (%)", fontsize=9)
            ax_sp.grid(True, ls=":", alpha=0.4, axis="x")
        elif any_c and not any_stk:
            ax_sp.text(0.5, 0.5,
                       "⚡ Solo impulsos puntuales activos.\n"
                       "Los efectos colaterales sostenidos requieren\n"
                       "cambios estructurales (sliders verdes).",
                       ha="center", va="center", transform=ax_sp.transAxes,
                       fontsize=11, color="#d97706", fontweight="bold")
            ax_sp.set_xticks([]); ax_sp.set_yticks([])
        else:
            ax_sp.text(0.5, 0.5,
                       "Activa sliders sostenidos (verde)\n"
                       "para ver efectos colaterales",
                       ha="center", va="center", transform=ax_sp.transAxes,
                       fontsize=10, color="#9ca3af")
            ax_sp.set_xticks([]); ax_sp.set_yticks([])

        ctx_ok = False
        anos_int = [int(y) for y in anos_h]
        tick_step = max(1, len(anos_int) // 6)
        if COL_PIB and COL_PIB in df_p.columns:
            pib_v = df_p[COL_PIB].ffill().values
            pib_med = float(np.nanmedian(pib_v))
            if pib_med > 1e12:
                pib_plot, pib_lbl = pib_v / 1e12, "PIB (billones USD)"
            elif pib_med > 1e9:
                pib_plot, pib_lbl = pib_v / 1e9, "PIB (miles mill. USD)"
            elif pib_med > 1e6:
                pib_plot, pib_lbl = pib_v / 1e6, "PIB (millones USD)"
            else:
                pib_plot, pib_lbl = pib_v, "PIB"
            ax_ctx.plot(anos_int, pib_plot, "o-", color="#1e6091",
                        lw=1.8, ms=4, label=pib_lbl)
            ax_ctx.set_ylabel(pib_lbl, fontsize=8, color="#1e6091")
            ax_ctx.tick_params(axis="y", labelcolor="#1e6091", labelsize=7)
            ctx_ok = True
        if COL_TELEC and COL_TELEC in df_p.columns:
            tel_v = df_p[COL_TELEC].ffill().values
            tel_med = float(np.nanmedian(tel_v))
            if tel_med > 1e6:
                tel_plot, tel_lbl = tel_v / 1e6, "Elec. Total (TWh)"
            elif tel_med > 1e3:
                tel_plot, tel_lbl = tel_v / 1e3, "Elec. Total (GWh)"
            else:
                tel_plot, tel_lbl = tel_v, "Elec. Total (MWh)"
            ax2t = ax_ctx.twinx()
            ax2t.plot(anos_int, tel_plot, "s--", color="#b45309",
                      lw=1.5, ms=3.5, alpha=0.85, label=tel_lbl)
            ax2t.set_ylabel(tel_lbl, fontsize=7.5, color="#b45309")
            ax2t.tick_params(axis="y", labelcolor="#b45309", labelsize=7)
            l1, lb1 = ax_ctx.get_legend_handles_labels()
            l2, lb2 = ax2t.get_legend_handles_labels()
            ax_ctx.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="upper left")
            ctx_ok = True
        if ctx_ok:
            ax_ctx.set_title(
                "Contexto macroeconómico\n(variables de control — fijas)",
                fontsize=9, fontweight="bold", color="#1b3a4b"
            )
            ax_ctx.set_xticks(anos_int[::tick_step])
            ax_ctx.tick_params(axis="x", labelsize=7, rotation=45)
            ax_ctx.grid(True, ls=":", alpha=0.35)
            if not COL_TELEC:
                ax_ctx.legend(fontsize=7.5, loc="upper left")
        else:
            ax_ctx.text(0.5, 0.5, "Variables de contexto\nno disponibles",
                        ha="center", va="center", transform=ax_ctx.transAxes,
                        fontsize=8, color="#9ca3af")
            ax_ctx.set_xticks([]); ax_ctx.set_yticks([])

    plt.tight_layout()

    meta = dict(
        pais_key=pais_key, pais_disp=pais_disp, n_years=n_years, ano_b=ano_b,
        dif_pp=dif_pp, dif_lc=dif_lc,
        es_l1=res["es_l1"], es_l2=res["es_l2"],
        sign_stab_m1=sign_stab_m1, sign_stab_m2=sign_stab_m2,
        any_c=any_c, res=res, td=td,
        d_stk=d_stk_abs, d_shk=d_shk_abs, bv=bv,
        pper_i_last=float(pper_i[-1]), pper_s_last=float(pper_s[-1]),
        lc_i_last=float(lc_i[-1]),     lc_s_last=float(lc_s[-1]),
    )
    return fig, meta


# =============================================================================
# G. FIGURA COMPARATIVA MULTI-PAÍS
# =============================================================================
def fig_comparativo(pais_key_ref, n_years, d_stk_abs, d_shk_abs):
    bv_sel = _get_base_vals(pais_key_ref)
    any_c = any(
        abs(d_stk_abs[p] * ABS_STK_CFG[p]["scale"] - bv_sel[p])
        > 1e-6 * max(abs(bv_sel[p]), 1)
        or abs(d_shk_abs[p]) > 1e-6
        for p in PALANCAS
    )
    if not any_c:
        return None, "Activa al menos un slider antes de generar el comparativo."

    resultados = []
    for pk in PAISES_DISP:
        dfp = df_raw[df_raw["pais"] == pk].copy().sort_values("año")
        if len(dfp) < 3:
            continue
        bv_pk = _get_base_vals(pk)
        d_stk = {p: _stk_to_eff_pct(p, d_stk_abs[p], bv_pk[p]) for p in PALANCAS}
        d_shk = {p: _shk_to_eff_pct(p, d_shk_abs[p], bv_pk[p]) for p in PALANCAS}
        r = simular(dfp, d_stk, d_shk, n_years, pk, n_boot=300, seed=7)
        pper_i = np.expm1(r["lp_i"]); pper_s = np.expm1(r["lp_s"])
        lc_i   = inv_logit_pct(r["ll_i"]); lc_s = inv_logit_pct(r["ll_s"])
        dif_pp_abs = pper_s[-1] - pper_i[-1]
        dif_lc     = lc_s[-1] - lc_i[-1]
        pper_base  = float(dfp[COL_PPER].clip(lower=0).iloc[-1])
        lc_base    = float(dfp[COL_SHARE].iloc[-1])
        pp_abs_b = r["blp_lvl"][:, -1] - pper_i[-1]
        lc_b     = r["bll_pct"][:, -1] - lc_i[-1]
        resultados.append(dict(
            pais=pk, nombre=NOMBRES_PAIS.get(pk, pk.title()),
            dif_pp_abs=dif_pp_abs, dif_lc=dif_lc,
            pper_base=pper_base, lc_base=lc_base,
            es_l1=r["es_l1"], es_l2=r["es_l2"],
            pp_abs_lo=float(np.percentile(pp_abs_b, 2.5)),
            pp_abs_hi=float(np.percentile(pp_abs_b, 97.5)),
            lc_lo=float(np.percentile(lc_b, 2.5)),
            lc_hi=float(np.percentile(lc_b, 97.5)),
        ))

    df_all = pd.DataFrame(resultados)
    df_m1 = df_all.sort_values("dif_pp_abs")
    df_m2 = df_all.sort_values("dif_lc")

    fig = plt.figure(figsize=(15.5, 13), facecolor="#fafafa")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32,
                           height_ratios=[1, 0.85])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    fig.suptitle(
        f'Comparativo Multi-País  ·  Horizonte {n_years} año{"s" if n_years > 1 else ""}\n'
        f"Misma intervención aplicada simultáneamente a los 11 países del panel",
        fontsize=13, fontweight="bold", color="#1b3a4b", y=1.01
    )

    def _barh(ax, df_s, col_val, col_lo, col_hi, lid_set, titulo, xlbl,
              fmt="{:+.0f}", anno_suffix=""):
        noms = df_s["nombre"].values
        vals = df_s[col_val].values
        lo_e = np.abs(vals - df_s[col_lo].values)
        hi_e = np.abs(df_s[col_hi].values - vals)
        colors = [
            C_POS   if (v >= 0 and r.pais in lid_set) else
            C_POS_S if (v >= 0)                        else
            C_NEG   if r.pais in lid_set               else C_NEG_S
            for v, r in zip(vals, df_s.itertuples())
        ]
        y = np.arange(len(noms))
        ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="white",
                height=0.62, linewidth=0.4)
        ax.errorbar(vals, y, xerr=[lo_e, hi_e], fmt="none",
                    color="#374151", capsize=3.5, lw=1.1, alpha=0.6)
        ax.set_yticks(y); ax.set_yticklabels(noms, fontsize=10)
        ax.axvline(0, color="#374151", lw=0.9, ls="--", alpha=0.6)
        data_range = (vals.max() - vals.min()) if vals.max() != vals.min() else 1.0
        off = data_range * 0.025
        for i, (v, row) in enumerate(zip(vals, df_s.itertuples())):
            badge = "L" if row.pais in lid_set else "S"
            txt = fmt.format(v) + anno_suffix
            ax.text(v + (off if v >= 0 else -off), i,
                    f"[{badge}] {txt}", va="center", fontsize=8.5,
                    color="#1b3a4b", fontweight="bold",
                    ha="left" if v >= 0 else "right")
        ax.set_title(titulo, fontsize=11, fontweight="bold", color="#1b3a4b")
        ax.set_xlabel(xlbl, fontsize=10)
        ax.grid(True, ls=":", alpha=0.4, axis="x")
        ax.legend(handles=[
            mpatches.Patch(color=C_POS,   label="Líder ↑"),
            mpatches.Patch(color=C_POS_S, label="Seguidor ↑"),
            mpatches.Patch(color=C_NEG,   label="Líder ↓"),
            mpatches.Patch(color=C_NEG_S, label="Seguidor ↓"),
        ], fontsize=8, loc="lower right", framealpha=0.85)
        ax.spines["left"].set_visible(True)

    _barh(ax1, df_m1, "dif_pp_abs", "pp_abs_lo", "pp_abs_hi", LIDERES_M1,
          f"Patentes Renovables Adicionales\nt+{n_years} (número absoluto)",
          "Patentes adicionales vs inercia",
          fmt="{:+.0f}", anno_suffix=" pat.")
    ax1.text(0.02, 0.01,
             "* Valores negativos: la intervención reduce patentes vs inercia (θ_LíderM1 < 0)",
             transform=ax1.transAxes, fontsize=7.0, color="#6b7280",
             style="italic", va="bottom")
    _barh(ax2, df_m2, "dif_lc", "lc_lo", "lc_hi", LIDERES_M2,
          f"Cambio en Share de Energía Limpia\nt+{n_years} (puntos porcentuales)",
          "Δ Share LC (pp)",
          fmt="{:+.2f}", anno_suffix=" pp")

    _log_pats = [np.log1p(max(r["pper_base"], 0)) for _, r in df_all.iterrows()]
    _lc_bases = [r["lc_base"]                    for _, r in df_all.iterrows()]
    _med_lp   = float(np.median(_log_pats))
    _med_lc   = float(np.median(_lc_bases))
    _xlim_l, _xlim_r = min(_log_pats) - 0.3, max(_log_pats) + 0.4
    _ylim_b, _ylim_t = max(0, min(_lc_bases) - 5), min(100, max(_lc_bases) + 5)

    ax3.fill_between([_med_lp, _xlim_r], [_med_lc, _med_lc], [_ylim_t, _ylim_t],
                     color="#d1fae5", alpha=0.18, zorder=0)
    ax3.fill_between([_xlim_l, _med_lp], [_med_lc, _med_lc], [_ylim_t, _ylim_t],
                     color="#dbeafe", alpha=0.18, zorder=0)
    ax3.fill_between([_med_lp, _xlim_r], [_ylim_b, _ylim_b], [_med_lc, _med_lc],
                     color="#fef3c7", alpha=0.28, zorder=0)
    ax3.fill_between([_xlim_l, _med_lp], [_ylim_b, _ylim_b], [_med_lc, _med_lc],
                     color="#fee2e2", alpha=0.15, zorder=0)
    ax3.axvline(_med_lp, color="#9ca3af", ls="--", lw=0.9, alpha=0.7, zorder=1)
    ax3.axhline(_med_lc, color="#9ca3af", ls="--", lw=0.9, alpha=0.7, zorder=1)

    for _, row in df_all.iterrows():
        el1  = row["pais"] in LIDERES_M1
        el2  = row["pais"] in LIDERES_M2
        both = el1 and el2
        lp = np.log1p(max(row["pper_base"], 0))
        lc = row["lc_base"]
        if both:
            c_dot, marker, sz, zz = "#b45309", "*", 320, 7
        elif el1:
            c_dot, marker, sz, zz = "#1e6091", "D", 180, 6
        elif el2:
            c_dot, marker, sz, zz = "#065f46", "s", 180, 6
        else:
            c_dot, marker, sz, zz = "#6b7280", "o", 130, 5
        ax3.scatter(lp, lc, s=sz, c=c_dot, marker=marker,
                    alpha=0.88, edgecolors="white", linewidths=1.4, zorder=zz)

    _label_offsets = {
        "china": (0.08, 4.0), "estados_unidos": (0.08, 4.0),
        "japon": (-0.12, 4.5), "corea_del_sur": (0.08, -5.5),
        "alemania": (0.08, 4.0), "francia": (0.08, -5.5),
        "canada": (0.08, 4.0), "brasil": (0.08, -5.5),
        "dinamarca": (0.08, -5.5), "mexico": (0.08, -5.5),
        "chile": (0.08, 4.0),
    }
    for _, row in df_all.iterrows():
        lp = np.log1p(max(row["pper_base"], 0))
        lc = row["lc_base"]
        pk = row["pais"]; nom = row["nombre"]
        dx, dy = _label_offsets.get(pk, (0.08, 4.0))
        _lbl3 = f"{nom}\n({int(row['pper_base'])} pat. | {lc:.0f}% LC)"
        ax3.annotate(_lbl3, xy=(lp, lc), xytext=(lp + dx, lc + dy),
                     textcoords="data", fontsize=7.8, color="#1b3a4b",
                     fontweight="500",
                     ha="left" if dx > 0 else "right", va="center",
                     arrowprops=dict(arrowstyle="-", color="#9ca3af",
                                     lw=0.6, alpha=0.55, shrinkA=4, shrinkB=2))

    ax3.set_xlim(_xlim_l, _xlim_r); ax3.set_ylim(_ylim_b, _ylim_t)
    ax3.set_xlabel("Capacidad de Innovación Renovable  (log de patentes base)",
                   fontsize=10.5)
    ax3.set_ylabel("Participación de Energía Limpia  (%)", fontsize=10.5)
    ax3.set_title(
        "Mapa de Posicionamiento Histórico — H4: Paradoja del Líder\n"
        "Alta innovación no implica alta descarbonización",
        fontsize=12, fontweight="bold", color="#1b3a4b"
    )
    ax3.grid(True, ls=":", alpha=0.30)
    ax3.tick_params(labelsize=9)
    _xr = _xlim_r - _xlim_l; _yr = _ylim_t - _ylim_b
    _qt = dict(fontsize=8, style="italic", ha="center", va="center", zorder=2)
    ax3.text(_xlim_l + _xr * 0.78, _ylim_b + _yr * 0.90,
             "Alta innovación\nAlta descarbonización", color="#065f46", **_qt)
    ax3.text(_xlim_l + _xr * 0.22, _ylim_b + _yr * 0.90,
             "Baja innovación\nAlta descarbonización", color="#1e6091", **_qt)
    ax3.text(_xlim_l + _xr * 0.78, _ylim_b + _yr * 0.10,
             "Alta innovación\nBaja descarbonización\n(Paradoja del Líder)",
             color="#92400e", **_qt)
    ax3.text(_xlim_l + _xr * 0.22, _ylim_b + _yr * 0.10,
             "Baja innovación\nBaja descarbonización", color="#991b1b", **_qt)
    ax3.legend(handles=[
        mpatches.Patch(color="#b45309", label="★ Líder M1 + M2"),
        mpatches.Patch(color="#1e6091", label="◆ Líder Innovación (M1)"),
        mpatches.Patch(color="#065f46", label="■ Líder Descarbonización (M2)"),
        mpatches.Patch(color="#6b7280", label="● Seguidor ambos"),
    ], fontsize=8.5, loc="upper right", framealpha=0.92, frameon=True,
        title="Régimen motor-específico", title_fontsize=8,
        edgecolor="#d1d5db")

    plt.tight_layout()
    return fig, df_all


# =============================================================================
# H. EXPORTACIÓN A PDF
# =============================================================================
def generar_pdf_bytes(meta, fig_scen, fig_comp=None) -> bytes:
    """Genera el PDF y retorna los bytes (adaptado para Streamlit)."""
    if fig_scen is None:
        return None
    pais_key  = meta.get("pais_key", "pais")
    pais_disp = meta.get("pais_disp", pais_key)
    n_years   = meta.get("n_years", 5)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig_scen, bbox_inches="tight",
                    facecolor="#fafafa", dpi=150)
        if fig_comp is not None:
            pdf.savefig(fig_comp, bbox_inches="tight",
                        facecolor="#fafafa", dpi=150)

        fig_txt = plt.figure(figsize=(11, 8.5), facecolor="white")
        ax_txt  = fig_txt.add_axes([0.06, 0.05, 0.88, 0.90])
        ax_txt.axis("off")

        dif_pp = meta.get("dif_pp", 0.0); dif_lc = meta.get("dif_lc", 0.0)
        es_l1  = meta.get("es_l1", False); es_l2 = meta.get("es_l2", False)
        ano_b  = meta.get("ano_b", 0)

        pal_lines = []
        for p in PALANCAS:
            cs = ABS_STK_CFG[p]; ch = ABS_SHK_CFG[p]
            v_bas = meta["bv"][p] / cs["scale"]
            v_sld = meta["d_stk"][p]; v_shk = meta["d_shk"][p]
            delta = v_sld - v_bas
            if abs(delta) > 0.01 or abs(v_shk) > 0.01:
                pal_lines.append(
                    f"  {NOM_PAL[p]:<32} Nivel: {v_sld:.2f} {cs['unit']:<11}"
                    f"Δ vs base: {delta:+.2f}    Impulso: {v_shk:+.2f} {ch['unit']}"
                )
        pal_txt = "\n".join(pal_lines) if pal_lines \
                  else "  (Sin intervenciones activas)"

        summary = f"""
SIMULADOR CAUSAL DE POLÍTICA PÚBLICA · REPORTE DE ESCENARIO
{'═' * 72}

País:      {pais_disp}
Horizonte: {n_years} año{'s' if n_years > 1 else ''}  ({ano_b + 1}–{ano_b + n_years})
Régimen:   M1 {'Líder' if es_l1 else 'Seguidor'} (Innovación)  |  M2 {'Líder' if es_l2 else 'Seguidor'} (Descarbonización)

INTERVENCIONES APLICADAS:
{pal_txt}

RESULTADOS PRINCIPALES:
  Innovación Renovable (M1):   {dif_pp:+.1f}% vs inercia
  Energía Limpia / Share LC:   {dif_lc:+.2f} pp vs inercia

MODELO:  Panel FE + Driscoll-Kraay (BW=4) + DML Secuencial
         θ heterogéneos v11 S3b — Clasificación motor-específica (mediana)
         Bootstrap paramétrico N=2,000 — Seed=42 — IC 95%

REFERENCIAS METODOLÓGICAS CENTRALES:
  Driscoll & Kraay (1998)   —  SE robusta a dependencia seccional cruzada
  Chernozhukov et al. (2018)—  Double/Debiased Machine Learning
  Preacher & Hayes (2008)   —  Mediación por bootstrap paramétrico
  Abramovitz (1986)         —  Convergencia condicional (catch-up)
  Unruh (2000)              —  Carbon lock-in en trayectorias tecnológicas
  Fagerberg & Srholec (2008)—  Capacidad tecnológica y desarrollo
  Mazzucato & Penna (2016)  —  Financiamiento público de misión

HALLAZGOS CENTRALES DE LA TESIS (v11):
  H1  Catch-Up GIDE (M1-SEG):   θ = +1.161*    (Abramovitz 1986 confirmado)
  H2  Dirty Finance (M1-Cred):  θ_SEG = -0.829***    θ_LID ≈ 0
  H3  Green Crowding-Out (M2):  θ_LID = -1.081**    [LOO: -0.972 a -1.093]
  H4  Paradoja del Líder:       Alemania y Francia únicos líderes en ambos motores
  H5  IED Saturación (M2-SEG):  θ = +0.188*    θ_LID ≈ 0

{'─' * 72}
Generado el {datetime.now().strftime('%d de %B de %Y, %H:%M')}
Simulador Causal — Streamlit Community Cloud
Tesis de Maestría · SNI y Transición Energética Baja en Carbono
Autor: Angel A. Ramírez Martínez  ·  UASLP / SECIHTI
"""
        ax_txt.text(0.0, 1.0, summary, transform=ax_txt.transAxes,
                    fontsize=8.5, va="top", ha="left",
                    fontfamily="monospace", color="#1b3a4b")
        fig_txt.suptitle(
            f"Reporte de Escenario — {pais_disp}  ·  {n_years} años",
            fontsize=12, fontweight="bold", color="#1b3a4b", y=0.98
        )
        pdf.savefig(fig_txt, bbox_inches="tight", facecolor="white", dpi=120)
        plt.close(fig_txt)

        d = pdf.infodict()
        d["Title"]    = f"Simulador Causal — {pais_disp} {n_years}a"
        d["Author"]   = "Tesis Maestría — SNI y Transición Energética"
        d["Subject"]  = "Política energética · Panel DK + DML · θ heterogéneos"
        d["Keywords"] = "DML, renovables, crowding-out, catch-up, mediación"

    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# I. ESCENARIOS SEXENALES MÉXICO
# =============================================================================
SEXENIOS_MX = [
    {
        "id": "calderon",
        "nombre": "Felipe Calderón Hinojosa",
        "periodo": "2006–2012",
        "color_hex": "#1e40af", "color_bg": "#dbeafe",
        "contexto_html": (
            "Sexenio marcado por la <b>Ley para el Aprovechamiento de Energías "
            "Renovables y el Financiamiento de la Transición Energética "
            "(LAERFTE)</b>, publicada en el DOF el 28 de noviembre de 2008, y "
            "por el <b>Programa Especial de Cambio Climático (PECC) 2009–2012</b>. "
            "El Programa Especial de Ciencia, Tecnología e Innovación (PECITI) "
            "2008–2012 fijó como meta explícita alcanzar el <b>1 % del PIB en "
            "GIDE</b>; México cerró el sexenio en 0.40 %."
        ),
        "escenarios": [
            {"id": "cal_contrafactual_1pct", "tipo": "contrafactual",
             "titulo": "Contrafactual: cumplimiento de meta 1 % GIDE/PIB",
             "subtitulo": "Meta declarada en el PECITI 2008–2012 (nunca alcanzada)",
             "descripcion": "Sostiene GIDE = 1.00 % del PIB (meta oficial del PECITI) junto con las variables observadas al cierre de 2012 en Crédito, IED y Pagos PI.",
             "valores_stk": {"GIDE": 1.00, "Credito": 24.93, "IED": 1.45, "PagosPI": 3648.0},
             "valores_shk": {"GIDE": 0.0, "Credito": 0.0, "IED": 0.0, "PagosPI": 0.0},
             "horizonte": 5},
            {"id": "cal_prospectivo", "tipo": "prospectivo",
             "titulo": "Prospectivo desde 2012 (statu quo Calderón)",
             "subtitulo": "Valores observados al cierre del sexenio",
             "descripcion": "Fija los valores registrados en 2012: GIDE 0.40 %, Crédito 24.9 %, IED 1.45 %, Pagos PI 3 648 mill. USD. Sin choques.",
             "valores_stk": {"GIDE": 0.40, "Credito": 24.93, "IED": 1.45, "PagosPI": 3648.0},
             "valores_shk": {"GIDE": 0.0, "Credito": 0.0, "IED": 0.0, "PagosPI": 0.0},
             "horizonte": 5},
        ],
    },
    {
        "id": "penanieto",
        "nombre": "Enrique Peña Nieto",
        "periodo": "2012–2018",
        "color_hex": "#065f46", "color_bg": "#d1fae5",
        "contexto_html": (
            "Sexenio de la <b>Reforma Energética constitucional</b> (DOF, 20 de "
            "diciembre de 2013) y sus leyes secundarias — Ley de Hidrocarburos "
            "y Ley de la Industria Eléctrica, ambas publicadas en el DOF el 11 "
            "de agosto de 2014. La <b>Ley de Transición Energética (LTE)</b>, "
            "DOF 24 de diciembre de 2015, fijó metas de 25 % de energías "
            "limpias al 2018, 30 % al 2021 y 35 % al 2024. Se realizaron tres "
            "subastas eléctricas de largo plazo (2015–2017) que atrajeron "
            "inversión renovable a estados como San Luis Potosí. "
            "Paradójicamente, el GIDE cayó de 0.40 % (2012) a 0.30 % (2018)."
        ),
        "escenarios": [
            {"id": "pn_contrafactual_reforma_id", "tipo": "contrafactual",
             "titulo": "Contrafactual: Reforma Energética con I+D acompañante",
             "subtitulo": "GIDE al 1 % + IED al 4 % del PIB en clima post-subastas",
             "descripcion": "Combina el nivel observado de IED en 2018 (elevada por la Reforma Energética) con un GIDE hipotético al 1 % del PIB — la meta reafirmada en el Programa Sectorial CONACYT y en el PECITI 2014–2018.",
             "valores_stk": {"GIDE": 1.00, "Credito": 35.00, "IED": 4.00, "PagosPI": 5074.0},
             "valores_shk": {"GIDE": 0.0, "Credito": 0.0, "IED": 0.0, "PagosPI": 0.0},
             "horizonte": 6},
            {"id": "pn_prospectivo", "tipo": "prospectivo",
             "titulo": "Prospectivo desde 2018 (statu quo Peña Nieto)",
             "subtitulo": "Valores observados al cierre del sexenio",
             "descripcion": "Fija los valores registrados en 2018: GIDE 0.30 %, Crédito 33.6 %, IED 3.01 %, Pagos PI 5 074 mill. USD.",
             "valores_stk": {"GIDE": 0.30, "Credito": 33.62, "IED": 3.01, "PagosPI": 5074.0},
             "valores_shk": {"GIDE": 0.0, "Credito": 0.0, "IED": 0.0, "PagosPI": 0.0},
             "horizonte": 6},
        ],
    },
    {
        "id": "amlo",
        "nombre": "Andrés Manuel López Obrador",
        "periodo": "2018–2024",
        "color_hex": "#7c2d12", "color_bg": "#fed7aa",
        "contexto_html": (
            "Sexenio caracterizado por la <b>reversión regulatoria</b>: "
            "cancelación de la cuarta subasta eléctrica de largo plazo (2019), "
            "reforma a la Ley de la Industria Eléctrica publicada en el DOF el "
            "9 de marzo de 2021 — cuyos artículos fueron parcialmente "
            "invalidados por la Suprema Corte en abril de 2024 — y rescate "
            "financiero de la CFE. En materia científica, la <b>Ley General de "
            "Humanidades, Ciencias, Tecnologías e Innovación (LGMHCTI)</b>, "
            "DOF 8 de mayo de 2023, sustituyó el marco anterior; el GIDE cayó "
            "de 0.30 % (2018) a 0.25 % (2024) — el nivel más bajo en dos décadas."
        ),
        "escenarios": [
            {"id": "amlo_contrafactual", "tipo": "contrafactual",
             "titulo": "Contrafactual: continuidad de la Reforma + LGMHCTI con presupuesto real",
             "subtitulo": "GIDE al 0.50 % + Crédito y IED sin fuga por reversión",
             "descripcion": "Simula el escenario en que la Reforma Energética hubiera continuado y la LGMHCTI se hubiera dotado del presupuesto para revertir la caída del GIDE.",
             "valores_stk": {"GIDE": 0.50, "Credito": 40.00, "IED": 3.50, "PagosPI": 6431.0},
             "valores_shk": {"GIDE": 0.0, "Credito": 0.0, "IED": 0.0, "PagosPI": 0.0},
             "horizonte": 6},
            {"id": "amlo_prospectivo", "tipo": "prospectivo",
             "titulo": "Prospectivo desde 2024 (statu quo AMLO)",
             "subtitulo": "Valores observados al cierre del sexenio",
             "descripcion": "Fija los valores registrados en 2024: GIDE 0.25 %, Crédito 34.7 %, IED 2.45 %, Pagos PI 6 431 mill. USD.",
             "valores_stk": {"GIDE": 0.25, "Credito": 34.65, "IED": 2.45, "PagosPI": 6431.0},
             "valores_shk": {"GIDE": 0.0, "Credito": 0.0, "IED": 0.0, "PagosPI": 0.0},
             "horizonte": 6},
        ],
    },
    {
        "id": "sheinbaum",
        "nombre": "Claudia Sheinbaum Pardo",
        "periodo": "2024–2030 (prospectivo)",
        "color_hex": "#7c3aed", "color_bg": "#ede9fe",
        "contexto_html": (
            "Sexenio en curso. Marco jurídico: <b>reforma constitucional a los "
            "artículos 27 y 28</b> (DOF 31 de octubre de 2024) que reclasifica "
            "a CFE y Pemex como empresas públicas del Estado; nueva <b>Ley del "
            "Sector Eléctrico (LSE)</b> y Ley de la Empresa Pública del Estado "
            "CFE, ambas DOF 18 de marzo de 2025. El decreto de creación de la "
            "<b>Secretaría de Ciencia, Humanidades, Tecnología e Innovación "
            "(SECIHTI)</b> se publicó en el DOF el 28 de noviembre de 2024. "
            "El <b>Plan México</b>, anunciado el 13 de enero de 2025, declara "
            "como meta alcanzar <b>1.5 % del PIB en GIDE al 2030</b> y "
            "consolidar 45 % de generación eléctrica limpia al mismo horizonte."
        ),
        "escenarios": [
            {"id": "sh_plan_mexico", "tipo": "contrafactual",
             "titulo": "Plan México ambicioso",
             "subtitulo": "Cumplimiento pleno de la meta 1.5 % GIDE/PIB al 2030",
             "descripcion": "Sostiene GIDE en 1.5 % del PIB (meta declarada del Plan México) y eleva el Crédito privado y la IED en línea con un shock institucional favorable.",
             "valores_stk": {"GIDE": 1.50, "Credito": 45.00, "IED": 3.00, "PagosPI": 7000.0},
             "valores_shk": {"GIDE": 0.0, "Credito": 0.0, "IED": 0.0, "PagosPI": 0.0},
             "horizonte": 6},
            {"id": "sh_secihti_intermedio", "tipo": "prospectivo",
             "titulo": "Recuperación SECIHTI (intermedio)",
             "subtitulo": "Trayectoria intermedia entre statu quo y meta declarada",
             "descripcion": "Escenario realista: GIDE recupera terreno hasta 0.50 % del PIB, con estabilización de Crédito e IED.",
             "valores_stk": {"GIDE": 0.50, "Credito": 40.00, "IED": 2.80, "PagosPI": 6500.0},
             "valores_shk": {"GIDE": 0.0, "Credito": 0.0, "IED": 0.0, "PagosPI": 0.0},
             "horizonte": 6},
            {"id": "sh_statu_quo", "tipo": "prospectivo",
             "titulo": "Statu quo (continuidad AMLO)",
             "subtitulo": "Sin desvíos del régimen 2024",
             "descripcion": "Escenario pesimista: los valores observados en 2024 se mantienen sin modificaciones durante todo el sexenio.",
             "valores_stk": {"GIDE": 0.25, "Credito": 34.65, "IED": 2.45, "PagosPI": 6431.0},
             "valores_shk": {"GIDE": 0.0, "Credito": 0.0, "IED": 0.0, "PagosPI": 0.0},
             "horizonte": 6},
        ],
    },
]


def find_escenario(esc_id):
    for sx in SEXENIOS_MX:
        for esc in sx["escenarios"]:
            if esc["id"] == esc_id:
                return sx, esc
    return None, None


# =============================================================================
# J. CSS PERSONALIZADO
# =============================================================================
CUSTOM_CSS = """
<style>
/* Contenedor principal */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* Header hero */
.sim-hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b3a4b 50%, #1e6091 100%);
    padding: 24px 32px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.28);
    color: #e8f4f8;
}
.sim-hero h2 {
    font-family: Georgia, serif;
    margin: 0 0 8px 0;
    font-size: 24px;
    letter-spacing: 0.3px;
    color: #e8f4f8;
    font-weight: 700;
}
.sim-hero p {
    margin: 0;
    font-size: 12px;
    line-height: 1.65;
    color: #a8d8ea;
}

/* Cards KPI */
.kpi-card {
    padding: 16px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin: 8px 0;
}

/* Sliders styling */
.stSlider [data-baseweb="slider"] > div > div > div {
    background-color: #00897b !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: #f1f5f9;
    padding: 8px 18px;
    border-radius: 8px 8px 0 0;
    color: #1e3a5f;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1e6091 !important;
    color: white !important;
    font-weight: 700;
}

/* Botones */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}

/* Container box para panel de controles */
.control-box {
    background: #f8fafc;
    padding: 14px 16px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    margin-bottom: 8px;
}

/* Footer */
.footer-sim {
    text-align: center;
    font-size: 11px;
    color: #94a3b8;
    padding: 20px 0 8px 0;
    border-top: 1px solid #e2e8f0;
    margin-top: 30px;
}
.footer-sim a {
    color: #1e6091;
    text-decoration: none;
    font-weight: 500;
}
.footer-sim a:hover {
    text-decoration: underline;
}
</style>
"""


# =============================================================================
# K. FUNCIONES DE UI — INFORMES HTML
# =============================================================================
def html_regimen(pais_key):
    _, el1 = get_theta_stk(pais_key, "M1")
    _, el2 = get_theta_stk(pais_key, "M2")
    h1g = THETA_STK["M1"]["GIDE"]; h2c = THETA_STK["M2"]["Credito"]
    th1 = (h1g["seg"] + h1g["adj"]) if el1 else h1g["seg"]
    th2 = (h2c["seg"] + h2c["adj"]) if el2 else h2c["seg"]
    r1c = ("#065f46", "#d1fae5", "🏆 Líder") if el1 else \
          ("#1e40af", "#dbeafe", "📈 Seguidor")
    r2c = ("#065f46", "#d1fae5", "🏆 Líder") if el2 else \
          ("#1e40af", "#dbeafe", "📈 Seguidor")
    paises_dobles = LIDERES_M1 & LIDERES_M2
    nota = (" <span style='color:#b45309;font-size:10px;font-weight:bold;'>"
            "★ Líder en ambos motores</span>") if pais_key in paises_dobles else ""
    return f"""
<div style="display:flex;gap:10px;margin:8px 0;flex-wrap:wrap;">
  <div style="background:{r1c[1]};border-left:4px solid {r1c[0]};
       padding:10px 14px;border-radius:6px;flex:1;min-width:200px;">
    <b style="color:{r1c[0]};font-size:12.5px;">M1 · Innovación: {r1c[2]}{nota}</b><br>
    <span style="font-size:11px;color:#374151;">
      θ_GIDE aplicado = <b>{th1:+.4f}</b>
      {'(SEG+ADJ)' if el1 else '(SEG)'}
    </span>
  </div>
  <div style="background:{r2c[1]};border-left:4px solid {r2c[0]};
       padding:10px 14px;border-radius:6px;flex:1;min-width:200px;">
    <b style="color:{r2c[0]};font-size:12.5px;">M2 · Descarbonización: {r2c[2]}</b><br>
    <span style="font-size:11px;color:#374151;">
      θ_Crédito aplicado = <b>{th2:+.4f}</b>
      {'(SEG+ADJ)' if el2 else '(SEG)'}
    </span>
  </div>
</div>"""


def html_baseline(pais_key):
    bv = _get_base_vals(pais_key)
    items = []
    for p in PALANCAS:
        cfg = ABS_STK_CFG[p]
        v_bas = bv[p] / cfg["scale"]
        lbl = (f"{v_bas:,.0f} {cfg['unit']}" if cfg["unit"] == "mill. USD"
               else f"{v_bas:.2f} {cfg['unit']}")
        items.append(
            f"<div style='flex:1;min-width:120px;background:#f8fafc;"
            f"border-left:3px solid #1e6091;padding:6px 10px;border-radius:4px;'>"
            f"<div style='font-size:9.5px;color:#64748b;text-transform:uppercase;"
            f"letter-spacing:.5px;font-weight:600;'>{NOM_PAL[p]}</div>"
            f"<div style='font-size:13.5px;font-weight:700;color:#1b3a4b;"
            f"margin-top:2px;'>{lbl}</div>"
            f"<div style='font-size:9px;color:#9ca3af;margin-top:1px;'>"
            f"último dato disponible</div></div>"
        )
    return ("<div style='display:flex;gap:8px;margin:8px 0 4px 0;flex-wrap:wrap;'>"
            + "".join(items) + "</div>")


def _sign_card(pct, label):
    if pct is None:
        return ""
    col = "#065f46" if pct >= 85 else "#92400e" if pct >= 70 else "#991b1b"
    bg  = "#d1fae5" if pct >= 85 else "#fef3c7" if pct >= 70 else "#fee2e2"
    interp = "Alta certeza" if pct >= 85 else "Moderada" if pct >= 70 else "Baja"
    return (
        f"<div style='flex:1;min-width:170px;background:{bg};padding:12px;"
        f"border-radius:8px;border-left:5px solid {col};'>"
        f"<div style='font-size:10.5px;color:#4b5563;text-transform:uppercase;"
        f"letter-spacing:.5px;font-weight:600;'>Certeza direccional · {label}</div>"
        f"<div style='font-size:24px;font-weight:bold;color:{col};margin:4px 0;'>"
        f"{pct:.0f}%</div>"
        f"<div style='font-size:10px;color:#6b7280;'>{interp} — simulaciones que "
        f"confirman la dirección del efecto</div>"
        f"<div style='font-size:9px;color:#9ca3af;margin-top:2px;'>"
        f"Bootstrap N={N_BOOT:,} · complemento al IC 95%</div></div>"
    )


def html_informe(meta) -> str:
    if meta.get("error"):
        return (f"<div style='color:#991b1b;padding:12px;background:#fee2e2;"
                f"border-radius:6px;'>{meta['error']}</div>")
    if not meta["any_c"]:
        return (
            "<div style='padding:14px;background:#fff8e1;border-left:4px solid #f9a825;"
            "border-radius:6px;color:#92400e;'>"
            "<b>Sin intervención activa.</b> Ajusta al menos un slider fuera del "
            "baseline (sostenido = nivel objetivo distinto al valor actual, o "
            "impulso ≠ 0) para generar un escenario contrafactual."
            "</div>"
        )

    dif_pp = meta["dif_pp"]; dif_lc = meta["dif_lc"]
    ic_pp_lo = np.percentile(
        ((meta["res"]["blp_lvl"][:, -1] - meta["pper_i_last"]) /
         max(meta["pper_i_last"], 0.1)) * 100.0, 2.5)
    ic_pp_hi = np.percentile(
        ((meta["res"]["blp_lvl"][:, -1] - meta["pper_i_last"]) /
         max(meta["pper_i_last"], 0.1)) * 100.0, 97.5)
    ic_lc_lo = np.percentile(meta["res"]["bll_pct"][:, -1] - meta["lc_i_last"], 2.5)
    ic_lc_hi = np.percentile(meta["res"]["bll_pct"][:, -1] - meta["lc_i_last"], 97.5)

    kpi_m1_col = "#065f46" if dif_pp >= 0 else "#991b1b"
    kpi_m1_bg  = "#d1fae5" if dif_pp >= 0 else "#fee2e2"
    kpi_m2_col = "#065f46" if dif_lc >= 0 else "#991b1b"
    kpi_m2_bg  = "#d1fae5" if dif_lc >= 0 else "#fee2e2"

    kpi_html = f"""
<div style="display:flex;gap:10px;margin:10px 0;flex-wrap:wrap;">
  <div style='flex:1;min-width:200px;background:{kpi_m1_bg};padding:14px;
       border-radius:8px;border-left:5px solid {kpi_m1_col};'>
    <div style='font-size:10.5px;color:#4b5563;text-transform:uppercase;
         letter-spacing:.5px;font-weight:600;'>M1 · Patentes Renovables (t+{meta['n_years']})</div>
    <div style='font-size:26px;font-weight:bold;color:{kpi_m1_col};margin:4px 0;'>
      {dif_pp:+.1f}%</div>
    <div style='font-size:10.5px;color:#6b7280;'>vs inercia · IC 95%: [{ic_pp_lo:+.1f}%, {ic_pp_hi:+.1f}%]</div>
  </div>
  <div style='flex:1;min-width:200px;background:{kpi_m2_bg};padding:14px;
       border-radius:8px;border-left:5px solid {kpi_m2_col};'>
    <div style='font-size:10.5px;color:#4b5563;text-transform:uppercase;
         letter-spacing:.5px;font-weight:600;'>M2 · Share Low-Carbon (t+{meta['n_years']})</div>
    <div style='font-size:26px;font-weight:bold;color:{kpi_m2_col};margin:4px 0;'>
      {dif_lc:+.2f} pp</div>
    <div style='font-size:10.5px;color:#6b7280;'>vs inercia · IC 95%: [{ic_lc_lo:+.2f}, {ic_lc_hi:+.2f}]</div>
  </div>
</div>"""

    sign_html = (
        "<div style='display:flex;gap:10px;margin:6px 0 10px 0;flex-wrap:wrap;'>"
        + _sign_card(meta["sign_stab_m1"], "M1 — Patentes")
        + _sign_card(meta["sign_stab_m2"], "M2 — Share LC")
        + "</div>"
    )

    rows_html = ""
    for motor, th_d in [("M1", meta["res"]["th_m1"]), ("M2", meta["res"]["th_m2"])]:
        td_m = meta["td"][motor] if meta["td"] else {}
        es_lid = meta["es_l1"] if motor == "M1" else meta["es_l2"]
        for p in PALANCAS:
            total = td_m.get(p, {}).get("total", 0)
            _sig_perm = (THETA_STK[motor][p]["sig_adj"]
                         if es_lid else THETA_STK[motor][p]["sig_seg"])
            cs = ABS_STK_CFG[p]; ch = ABS_SHK_CFG[p]
            v_sld = meta["d_stk"][p]; v_shk = meta["d_shk"][p]
            v_bas = meta["bv"][p] / cs["scale"]
            delta = v_sld - v_bas
            reg = "🏆 L" if es_lid else "📈 S"
            m_lbl = "M1 Patentes" if motor == "M1" else "M2 Energía"
            imp_txt = (f"{v_shk:+.2f} {ch['unit']}"
                       if abs(v_shk) > 1e-6 else "—")
            rows_html += f"""
<tr>
  <td>{m_lbl}</td><td>{NOM_PAL[p]}</td>
  <td style='text-align:center;'>{reg}</td>
  <td>{th_d[p][0]:+.4f} ({_sig_perm})</td>
  <td>{THETA_SHK[motor][p]['theta']:+.4f} ({THETA_SHK[motor][p]['sig']})</td>
  <td>{v_sld:.2f} {cs['unit']}</td>
  <td>{delta:+.2f}</td><td>{imp_txt}</td>
  <td style='font-weight:bold;color:{"#065f46" if total>=0 else "#991b1b"};'>{total:+.4f}</td>
</tr>"""
    tabla_html = f"""
<style>
  .tbl-sim{{border-collapse:collapse;width:100%;font-size:11.5px;
           font-family:"Segoe UI",Arial,sans-serif;margin:8px 0;}}
  .tbl-sim th{{background:#1b3a4b;color:#f0f9ff;padding:8px 10px;
              text-align:left;font-weight:600;font-size:11px;
              letter-spacing:0.3px;border-right:1px solid #2d4a5e;}}
  .tbl-sim th:last-child{{border-right:none;}}
  .tbl-sim td{{padding:7px 10px;border-bottom:1px solid #f1f5f9;color:#1f2937;
              border-right:1px solid #f1f5f9;}}
  .tbl-sim tr:nth-child(odd) td{{background:#f8fafc;}}
</style>
<h4 style="color:#1b3a4b;margin:14px 0 6px;font-size:13px;
    border-bottom:1px solid #e5e7eb;padding-bottom:5px;">
  📊 Parámetros aplicados por palanca
</h4>
<table class="tbl-sim">
  <tr>
    <th>Motor</th><th>Palanca</th><th>Régimen</th>
    <th>θ sostenido</th><th>θ impulso</th>
    <th>Nivel objetivo</th><th>Δ vs base</th>
    <th>Impulso año 1</th><th>Efecto t+{meta['n_years']}</th>
  </tr>
  {rows_html}
</table>
<div style="font-size:10px;color:#9ca3af;margin-top:6px;line-height:1.6;">
  🏆 L = país líder en ese motor · 📈 S = seguidor ·
  <b>Nivel objetivo</b>: valor absoluto fijado durante toda la proyección ·
  <b>Δ vs base</b>: diferencia respecto al último dato observado del país ·
  Solo M2-Crédito es confirmatorio tras corrección múltiple (p_BH=0.0008***);
  M1-GIDE es sugestivo (p_BH=0.074*).
</div>
"""
    return kpi_html + sign_html + tabla_html


def render_escenario_html(sx, esc):
    tipo_lbl = "Contrafactual" if esc["tipo"] == "contrafactual" else "Prospectivo"
    tipo_col = "#7c3aed" if esc["tipo"] == "contrafactual" else "#0891b2"
    stk = esc["valores_stk"]
    return (
        f"<div style='padding:10px 12px;background:white;border:1px solid #e5e7eb;"
        f"border-radius:6px;margin-bottom:6px;'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:flex-start;gap:8px;flex-wrap:wrap;'>"
        f"<div style='flex:1;min-width:280px;'>"
        f"<b style='font-size:12.5px;color:#1b3a4b;'>{esc['titulo']}</b><br>"
        f"<span style='font-size:10.5px;color:#6b7280;font-style:italic;'>"
        f"{esc['subtitulo']}</span></div>"
        f"<span style='background:{tipo_col};color:white;padding:3px 9px;"
        f"border-radius:10px;font-size:9.5px;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:.4px;height:fit-content;'>"
        f"{tipo_lbl}</span></div>"
        f"<div style='display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;'>"
        f"<span style='background:#f0f7ff;color:#1e3a8a;padding:3px 8px;"
        f"border-radius:4px;font-size:10.5px;'><b>GIDE</b> {stk['GIDE']:.2f} % PIB</span>"
        f"<span style='background:#f0f7ff;color:#1e3a8a;padding:3px 8px;"
        f"border-radius:4px;font-size:10.5px;'><b>Crédito</b> {stk['Credito']:.2f} % PIB</span>"
        f"<span style='background:#f0f7ff;color:#1e3a8a;padding:3px 8px;"
        f"border-radius:4px;font-size:10.5px;'><b>IED</b> {stk['IED']:.2f} % PIB</span>"
        f"<span style='background:#f0f7ff;color:#1e3a8a;padding:3px 8px;"
        f"border-radius:4px;font-size:10.5px;'><b>PagosPI</b> {stk['PagosPI']:,.0f} M USD</span>"
        f"<span style='background:#fef3c7;color:#92400e;padding:3px 8px;"
        f"border-radius:4px;font-size:10.5px;'>Horizonte <b>{esc['horizonte']} años</b></span>"
        f"</div></div>"
    )


# =============================================================================
# L. GESTIÓN DE STATE Y APLICACIÓN DE ESCENARIOS SEXENALES
# =============================================================================
def init_state():
    """Inicializa session_state con valores por defecto (México, baseline)."""
    if "pais_key" not in st.session_state:
        st.session_state.pais_key = "mexico"
    if "motor_vis" not in st.session_state:
        st.session_state.motor_vis = "AMBOS"
    if "horizonte" not in st.session_state:
        st.session_state.horizonte = 5

    bv = _get_base_vals(st.session_state.pais_key)
    for p in PALANCAS:
        cs = ABS_STK_CFG[p]; ch = ABS_SHK_CFG[p]
        key_stk = f"stk_{p}"; key_shk = f"shk_{p}"
        if key_stk not in st.session_state:
            v_default = float(np.clip(bv[p] / cs["scale"], cs["min"], cs["max"]))
            st.session_state[key_stk] = v_default
        if key_shk not in st.session_state:
            st.session_state[key_shk] = 0.0

    if "last_fig_scen" not in st.session_state:
        st.session_state.last_fig_scen = None
    if "last_meta" not in st.session_state:
        st.session_state.last_meta = None
    if "last_fig_comp" not in st.session_state:
        st.session_state.last_fig_comp = None
    if "sexenio_notif" not in st.session_state:
        st.session_state.sexenio_notif = None
    if "sexenio_result" not in st.session_state:
        st.session_state.sexenio_result = None


def apply_reset_baseline():
    """Resetea sliders al baseline del país actual."""
    bv = _get_base_vals(st.session_state.pais_key)
    for p in PALANCAS:
        cs = ABS_STK_CFG[p]
        st.session_state[f"stk_{p}"] = float(np.clip(
            bv[p] / cs["scale"], cs["min"], cs["max"]
        ))
        st.session_state[f"shk_{p}"] = 0.0


def apply_scenario(esc_id):
    """Precarga los sliders con los valores del escenario sexenal."""
    sx, esc = find_escenario(esc_id)
    if esc is None:
        return
    st.session_state.pais_key  = "mexico"
    st.session_state.motor_vis = "AMBOS"
    st.session_state.horizonte = esc["horizonte"]
    for p in PALANCAS:
        cs = ABS_STK_CFG[p]
        st.session_state[f"stk_{p}"] = float(np.clip(
            esc["valores_stk"][p], cs["min"], cs["max"]
        ))
        st.session_state[f"shk_{p}"] = float(esc["valores_shk"][p])
    st.session_state.sexenio_notif = (
        f"✅ Escenario **{esc['titulo']}** ({sx['nombre']}, {sx['periodo']}) "
        f"aplicado a la pestaña **🎯 Escenario individual**. "
        f"Ve a esa pestaña para revisar y calcular. "
        f"Horizonte fijado en **{esc['horizonte']} años**, país en **México**."
    )


def calculate_scenario_sexenal(esc_id):
    """Ejecuta el cálculo del escenario sexenal directamente."""
    sx, esc = find_escenario(esc_id)
    if esc is None:
        return
    stk = esc["valores_stk"]; shk = esc["valores_shk"]
    n_years = int(esc["horizonte"])
    fig, meta = fig_escenario("mexico", "AMBOS", n_years, stk, shk)
    if fig is None:
        st.session_state.sexenio_result = {"error": "Error al generar figura"}
        return
    st.session_state.last_fig_scen = fig
    st.session_state.last_meta = meta
    st.session_state.sexenio_result = {
        "fig": fig, "meta": meta, "sx": sx, "esc": esc,
    }


# =============================================================================
# M. HEADER Y ESTRUCTURA PRINCIPAL
# =============================================================================
init_state()

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="sim-hero">
  <h2>🌿 Simulador Causal de Política Pública · Innovación y Transición Energética</h2>
  <p>
    θ heterogéneos v11 (líder / seguidor por motor) ·
    Panel FE + Driscoll–Kraay + DML Secuencial ·
    Canal GIDE → Patentes → ShareLC (Preacher &amp; Hayes 2008) ·
    Bootstrap IC 95% · Comparativo 11 países · Análisis sexenal México · PDF export
    <br><em style="opacity:0.75;">Tesis de Maestría · UASLP / SECIHTI · Angel A. Ramírez Martínez</em>
  </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Escenario individual",
    "🌍 Comparativo multi-país",
    "🇲🇽 México · Análisis sexenal",
    "📚 Metodología y referencias",
])


# =============================================================================
# TAB 1 — ESCENARIO INDIVIDUAL
# =============================================================================
with tab1:
    col_ctrl, col_lever = st.columns([2, 3], gap="medium")

    with col_ctrl:
        st.markdown("### Selección de país y régimen")

        pais_opts = [(NOMBRES_PAIS.get(p, p), p) for p in PAISES_DISP]
        pais_names = [x[0] for x in pais_opts]
        pais_keys  = [x[1] for x in pais_opts]
        try:
            _pais_idx = pais_keys.index(st.session_state.pais_key)
        except ValueError:
            _pais_idx = pais_keys.index("mexico")
        pais_sel_name = st.selectbox(
            "País", pais_names, index=_pais_idx, key="_pais_widget"
        )
        pais_sel_key = pais_keys[pais_names.index(pais_sel_name)]
        if pais_sel_key != st.session_state.pais_key:
            st.session_state.pais_key = pais_sel_key
            apply_reset_baseline()
            st.rerun()

        motor_opts_lbl = {
            "M1": "Patentes Renovables (M1)",
            "M2": "Share Low-Carbon % (M2)",
            "AMBOS": "Ambos motores",
        }
        motor_keys = list(motor_opts_lbl.keys())
        _mot_idx = motor_keys.index(st.session_state.motor_vis)
        motor_sel = st.radio(
            "Visualizar motor",
            options=motor_keys,
            format_func=lambda k: motor_opts_lbl[k],
            index=_mot_idx, key="_motor_widget",
        )
        st.session_state.motor_vis = motor_sel

        horiz_opts = OPCIONES_H
        _h_idx = horiz_opts.index(st.session_state.horizonte) \
                 if st.session_state.horizonte in horiz_opts else 2
        horiz_sel = st.radio(
            "Horizonte de proyección",
            options=horiz_opts,
            format_func=lambda h: f"{h} año{'s' if h != 1 else ''}",
            index=_h_idx, key="_horiz_widget", horizontal=True,
        )
        st.session_state.horizonte = horiz_sel

        if horiz_sel == 10:
            st.warning(
                "⚠️ **Horizonte 10 años:** proyección fuera de muestra "
                "histórica (T=17). Los θ pueden no ser estables. Interpretar "
                "como escenario ilustrativo, no como pronóstico."
            )

        st.markdown(html_regimen(st.session_state.pais_key),
                    unsafe_allow_html=True)

        st.markdown(
            "<b style='font-size:13px;color:#1b3a4b;'>📊 Valores base del país "
            "(último dato disponible):</b>",
            unsafe_allow_html=True
        )
        st.markdown(html_baseline(st.session_state.pais_key),
                    unsafe_allow_html=True)

        if st.session_state.sexenio_notif:
            st.info(st.session_state.sexenio_notif)
            if st.button("Descartar aviso", key="dismiss_notif"):
                st.session_state.sexenio_notif = None
                st.rerun()

    with col_lever:
        st.markdown(
            """<div style="background:#f0f7ff;border-left:4px solid #1e6091;
            padding:10px 14px;border-radius:5px;font-size:12px;line-height:1.55;">
            <b style="color:#1b3a4b;">🎛️ Cómo leer los sliders</b><br>
            <b style="color:#065f46;">● SOSTENIDO</b> — nivel objetivo absoluto
            de la variable, se mantiene durante todo el horizonte. El slider
            parte del último valor observado del país.<br>
            <b style="color:#92400e;">● IMPULSO</b> — delta puntual solo en año 1.
            <br><span style="font-size:11px;color:#6b7280;">
            Los θ bajo cada slider son coeficientes causales del modelo (v11 S3b).
            Solo M2-Crédito es confirmatorio tras corrección múltiple
            (p<sub>BH</sub>=0.0008***).</span>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("### 🎛️ Palancas de política")

        for p in PALANCAS:
            cs = ABS_STK_CFG[p]; ch = ABS_SHK_CFG[p]
            h1 = THETA_STK["M1"][p]; h2 = THETA_STK["M2"][p]
            sh1 = THETA_SHK["M1"][p]; sh2 = THETA_SHK["M2"][p]

            col_stk, col_shk = st.columns(2, gap="small")

            with col_stk:
                st.markdown(
                    f"<div style='padding:2px 0;font-size:11.5px;'>"
                    f"<b style='color:#065f46;'>↑ SOSTENIDO · {NOM_PAL[p]}</b>"
                    f"&nbsp;<span style='background:#d1fae5;color:#065f46;"
                    f"padding:1px 8px;border-radius:10px;font-size:9.5px;"
                    f"font-weight:700;'>NIVEL · {cs['unit']}</span>"
                    f"<br><span style='font-size:10px;color:#4b5563;'>"
                    f"M1: SEG={h1['seg']:+.3f}"
                    f"<span style='color:{SIG_COL[h1['sig_seg']]};font-weight:bold;'> {h1['sig_seg']}</span>"
                    f" | LID={h1['seg']+h1['adj']:+.3f}"
                    f"<span style='color:{SIG_COL[h1['sig_adj']]};font-weight:bold;'> {h1['sig_adj']}</span>"
                    f"&emsp;M2: SEG={h2['seg']:+.3f}"
                    f"<span style='color:{SIG_COL[h2['sig_seg']]};font-weight:bold;'> {h2['sig_seg']}</span>"
                    f" | LID={h2['seg']+h2['adj']:+.3f}"
                    f"<span style='color:{SIG_COL[h2['sig_adj']]};font-weight:bold;'> {h2['sig_adj']}</span>"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
                st.slider(
                    label=f"stk_{p}",
                    min_value=float(cs["min"]),
                    max_value=float(cs["max"]),
                    step=float(cs["step"]),
                    key=f"stk_{p}",
                    label_visibility="collapsed",
                )

            with col_shk:
                st.markdown(
                    f"<div style='padding:2px 0;font-size:11.5px;'>"
                    f"<b style='color:#92400e;'>⚡ IMPULSO · {NOM_PAL[p]}</b>"
                    f"&nbsp;<span style='background:#fef3c7;color:#92400e;"
                    f"padding:1px 8px;border-radius:10px;font-size:9.5px;"
                    f"font-weight:700;'>DELTA AÑO 1 · {ch['unit']}</span>"
                    f"<br><span style='font-size:10px;color:#4b5563;'>"
                    f"M1 θ={sh1['theta']:+.4f}"
                    f"<span style='color:{SIG_COL[sh1['sig']]};font-weight:bold;'> {sh1['sig']}</span>"
                    f"&emsp;M2 θ={sh2['theta']:+.4f}"
                    f"<span style='color:{SIG_COL[sh2['sig']]};font-weight:bold;'> {sh2['sig']}</span>"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
                st.slider(
                    label=f"shk_{p}",
                    min_value=float(ch["min"]),
                    max_value=float(ch["max"]),
                    step=float(ch["step"]),
                    key=f"shk_{p}",
                    label_visibility="collapsed",
                )

        col_a, col_b = st.columns(2)
        if col_a.button("↺ Resetear al baseline",
                        use_container_width=True, key="btn_reset_1"):
            apply_reset_baseline()
            st.rerun()

        calcular = col_b.button("▶ CALCULAR ESCENARIO", type="primary",
                                use_container_width=True, key="btn_calc_1")

        if calcular:
            with st.spinner("Ejecutando bootstrap paramétrico N=2000 réplicas..."):
                d_stk = {p: st.session_state[f"stk_{p}"] for p in PALANCAS}
                d_shk = {p: st.session_state[f"shk_{p}"] for p in PALANCAS}
                fig, meta = fig_escenario(
                    st.session_state.pais_key,
                    st.session_state.motor_vis,
                    st.session_state.horizonte,
                    d_stk, d_shk,
                )
                st.session_state.last_fig_scen = fig
                st.session_state.last_meta = meta

    st.markdown("---")
    st.markdown("### 📈 Resultados del escenario")

    if st.session_state.last_fig_scen is not None \
            and st.session_state.last_meta is not None:
        st.pyplot(st.session_state.last_fig_scen, clear_figure=False)
        st.markdown(html_informe(st.session_state.last_meta),
                    unsafe_allow_html=True)

        st.markdown("### 📄 Exportar reporte")
        col_pdf, col_msg = st.columns([1, 3])

        if col_pdf.button("Generar PDF", type="primary",
                          use_container_width=True, key="gen_pdf_1"):
            with st.spinner("Compilando PDF (2–4 páginas)..."):
                pdf_bytes = generar_pdf_bytes(
                    st.session_state.last_meta,
                    st.session_state.last_fig_scen,
                    st.session_state.last_fig_comp,
                )
                st.session_state.last_pdf = pdf_bytes

        if "last_pdf" in st.session_state and st.session_state.get("last_pdf"):
            n_pags = 1 + (1 if st.session_state.last_fig_comp is not None else 0) + 1
            col_msg.success(f"✅ PDF de {n_pags} páginas listo para descarga.")
            _fname = (f"simulador_causal_"
                      f"{st.session_state.last_meta['pais_key']}_"
                      f"{st.session_state.last_meta['n_years']}a_"
                      f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
            st.download_button(
                label="⬇ Descargar PDF",
                data=st.session_state.last_pdf,
                file_name=_fname,
                mime="application/pdf",
                use_container_width=True,
                key="dl_pdf_1",
            )
    else:
        st.info(
            "Configura la intervención con los sliders y presiona "
            "**▶ CALCULAR ESCENARIO** para ver la trayectoria proyectada."
        )


# =============================================================================
# TAB 2 — COMPARATIVO MULTI-PAÍS
# =============================================================================
with tab2:
    st.markdown("### Aplicación simultánea a los 11 países del panel")
    st.markdown(
        "Los sliders configurados en la pestaña **Escenario individual** se "
        "aplican como intervención uniforme sobre las 11 economías del estudio, "
        "respetando el régimen líder/seguidor de cada país por motor. Al final "
        "se traza el mapa histórico de la **Paradoja del Líder** (H4)."
    )

    col_c, col_help = st.columns([1, 2])

    if col_c.button("🌍 GENERAR COMPARATIVO", type="primary",
                    use_container_width=True, key="btn_comp"):
        with st.spinner(
            "Ejecutando 11 países × bootstrap n=300 réplicas c/u (~15–25 s)..."
        ):
            d_stk = {p: st.session_state[f"stk_{p}"] for p in PALANCAS}
            d_shk = {p: st.session_state[f"shk_{p}"] for p in PALANCAS}
            fig_c, res_c = fig_comparativo(
                st.session_state.pais_key,
                st.session_state.horizonte,
                d_stk, d_shk,
            )
            if fig_c is None:
                col_help.warning(f"⚠️ {res_c}")
                st.session_state.last_fig_comp = None
            else:
                st.session_state.last_fig_comp = fig_c
                st.session_state.last_comp_res = res_c

    col_help.info(
        "Requiere haber movido al menos un slider fuera del baseline en la "
        "pestaña **Escenario individual**. El cálculo demora ~15–25 s."
    )

    if st.session_state.last_fig_comp is not None:
        st.pyplot(st.session_state.last_fig_comp, clear_figure=False)
        st.markdown(
            """<div style='font-size:11px;color:#4b5563;padding:10px 12px;
            background:#f0f9ff;border-left:4px solid #1e6091;border-radius:5px;
            margin-top:6px;'>
            <b>Nota metodológica.</b> El comparativo aplica exactamente la
            misma intervención a los 11 países del panel, respetando el régimen
            líder/seguidor de cada uno por motor. El Panel 3 traza el mapa
            histórico (log-patentes vs % Share LC) y visualiza la
            <b>H4 Paradoja del Líder</b>: alta innovación no implica alta
            descarbonización. Solo Alemania y Francia lideran en ambos motores.
            IC 95% con bootstrap de 300 réplicas por país.</div>""",
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 3 — MÉXICO · ANÁLISIS SEXENAL
# =============================================================================
with tab3:
    st.markdown("""
<div style="padding:14px 18px;background:linear-gradient(135deg,#065f46 0%,#047857 100%);
     border-radius:10px;color:#ecfdf5;margin-bottom:14px;">
  <h3 style="margin:0 0 4px;font-family:Georgia,serif;font-size:18px;">
    🇲🇽 Escenarios sexenales — Innovación y transición energética en México
  </h3>
  <p style="margin:0;font-size:11.5px;line-height:1.55;color:#a7f3d0;">
    Simulación de escenarios de política pública anclados en marcos jurídicos
    verificables y metas oficiales declaradas de cada sexenio (2006–2030).
    Cada escenario proyecta hacia adelante desde el último dato del panel
    (2024) — <b>no reconstruye historia hacia atrás</b>.
  </p>
</div>
""", unsafe_allow_html=True)

    st.warning(
        "**Advertencia metodológica.** Los \"contrafactuales sexenales\" "
        "simulan la aplicación de la orientación de política del sexenio X "
        "al horizonte proyectivo actual, usando los coeficientes θ v11 estimados "
        "sobre el panel 2007–2024. Un contrafactual retrospectivo estricto "
        "(¿qué habría pasado si Calderón hubiera cumplido el 1 % GIDE en "
        "2007–2012?) requeriría re-estimar el modelo con datos alterados y "
        "no es alcanzable desde esta interfaz."
    )

    for sx in SEXENIOS_MX:
        with st.expander(
            f"**{sx['nombre']}**  ·  {sx['periodo']}",
            expanded=(sx["id"] == "sheinbaum"),
        ):
            st.markdown(
                f"<div style='padding:12px 14px;background:{sx['color_bg']};"
                f"border-left:4px solid {sx['color_hex']};border-radius:6px;"
                f"font-size:11.5px;color:#1f2937;line-height:1.6;"
                f"margin-bottom:12px;'>{sx['contexto_html']}</div>",
                unsafe_allow_html=True,
            )
            for esc in sx["escenarios"]:
                st.markdown(render_escenario_html(sx, esc), unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                if col_a.button(
                    "📋 Aplicar en pestaña Escenario individual",
                    key=f"apply_{esc['id']}", use_container_width=True,
                ):
                    apply_scenario(esc["id"])
                    st.rerun()
                if col_b.button(
                    "▶ Calcular y visualizar aquí",
                    key=f"calc_{esc['id']}", type="primary",
                    use_container_width=True,
                ):
                    with st.spinner(
                        "Ejecutando bootstrap paramétrico N=2000 réplicas..."
                    ):
                        calculate_scenario_sexenal(esc["id"])

    # Resultados del escenario sexenal
    st.markdown("---")
    st.markdown("### 📈 Resultados del escenario sexenal")

    if st.session_state.sexenio_result and \
            "fig" in st.session_state.sexenio_result:
        res = st.session_state.sexenio_result
        sx = res["sx"]; esc = res["esc"]
        tipo_lbl = ("Contrafactual" if esc["tipo"] == "contrafactual"
                    else "Prospectivo")
        tipo_col = "#7c3aed" if esc["tipo"] == "contrafactual" else "#0891b2"

        st.markdown(
            f"<div style='padding:14px 16px;background:{sx['color_bg']};"
            f"border-left:5px solid {sx['color_hex']};border-radius:8px;"
            f"margin-bottom:10px;'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;flex-wrap:wrap;gap:8px;'>"
            f"<div><b style='color:{sx['color_hex']};font-size:14px;'>"
            f"{sx['nombre']} · {sx['periodo']}</b><br>"
            f"<b style='font-size:12.5px;color:#1f2937;'>{esc['titulo']}</b><br>"
            f"<span style='font-size:11px;color:#4b5563;'>{esc['subtitulo']}"
            f"</span></div>"
            f"<span style='background:{tipo_col};color:white;padding:4px 10px;"
            f"border-radius:12px;font-size:10.5px;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:.5px;'>{tipo_lbl}</span>"
            f"</div>"
            f"<div style='font-size:11.5px;color:#374151;margin-top:8px;"
            f"line-height:1.55;'>{esc['descripcion']}</div></div>",
            unsafe_allow_html=True,
        )
        st.pyplot(res["fig"], clear_figure=False)
        st.markdown(html_informe(res["meta"]), unsafe_allow_html=True)

        st.info(
            "💡 **Tip para el examen.** Puedes exportar el PDF del "
            "escenario sexenal desde el botón **Generar PDF** en la pestaña "
            "**🎯 Escenario individual**. El estado del último cálculo "
            "(aquí o allá) se comparte entre pestañas."
        )
    else:
        st.info(
            "Selecciona un escenario y presiona **▶ Calcular y visualizar aquí**, "
            "o usa **📋 Aplicar** para transferir los valores a la pestaña "
            "Escenario individual."
        )


# =============================================================================
# TAB 4 — METODOLOGÍA
# =============================================================================
with tab4:
    st.markdown("""
### Arquitectura econométrica

Este simulador operacionaliza los coeficientes causales estimados en la tesis
de maestría **"The Leader's Paradox in Clean Technology: Catch-Up Returns to
Public R&D and Green Crowding-Out Across Heterogeneous Economies"**.

La arquitectura es un modelo híbrido en tres capas complementarias:

**Capa 1 · Panel FE + Driscoll–Kraay (largo plazo, evidencia primaria).**
Efectos fijos por país con errores estándar robustos a heteroscedasticidad,
autocorrelación y dependencia seccional cruzada. Ancho de banda BW=4
seleccionado por la regla de Newey-West adaptada al horizonte T=17.

**Capa 2 · DML Secuencial (corto plazo, exploratorio).**
Doble aprendizaje automático con XGBoost como estimador *nuisance* de primera
etapa (200 estimadores, profundidad 3, learning rate 0.05), validación
cruzada K=5 con particiones aleatorias. La causalidad se recupera mediante
regresión sobre residuos limpios: ε_Y = θ · ε_D + η.

**Capa 3 · Mediación bootstrap (canal GIDE → Patentes → ShareLC).**
Bootstrap paramétrico Monte Carlo con N=10,000 réplicas siguiendo Preacher &
Hayes (2008). Efecto indirecto a·b = +0.033 (IC 90% [+0.004, +0.075]).

### Heterogeneidad Líder / Seguidor

Clasificación *data-driven* por mediana del panel (spec S3b v11):

| Motor | Líderes | Seguidores |
|-------|---------|------------|
| M1 · Patentes Renovables | China, Japón, EE.UU., Corea del Sur, Alemania, Francia | Brasil, Canadá, Chile, Dinamarca, México |
| M2 · Share Low-Carbon | Francia, Brasil, Canadá, Dinamarca, Chile, Alemania | China, Japón, EE.UU., Corea del Sur, México |

Solo Alemania y Francia lideran ambos motores simultáneamente — el resto
exhibe la Paradoja del Líder (Innovation Leader ≠ Decarbonization Leader).

### Referencias metodológicas (verificables)

- Abramovitz, M. (1986). Catching up, forging ahead, and falling behind. *The Journal of Economic History, 46*(2), 385–406. https://doi.org/10.1017/S0022050700046209
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal, 21*(1), C1–C68. https://doi.org/10.1111/ectj.12097
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics, 80*(4), 549–560. https://doi.org/10.1162/003465398557825
- Fagerberg, J., & Srholec, M. (2008). National innovation systems, capabilities and economic development. *Research Policy, 37*(9), 1417–1435. https://doi.org/10.1016/j.respol.2008.06.003
- Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling strategies for assessing and comparing indirect effects in multiple mediator models. *Behavior Research Methods, 40*(3), 879–891. https://doi.org/10.3758/BRM.40.3.879
- Unruh, G. C. (2000). Understanding carbon lock-in. *Energy Policy, 28*(12), 817–830. https://doi.org/10.1016/S0301-4215(00)00070-7

### Limitaciones explícitas

1. **Ventana temporal T=17** (2007–2024): la proyección a 10 años cae fuera
   de muestra; interpretar como escenario ilustrativo, no como pronóstico.
2. **Coeficientes θ constantes**: no hay recalibración de los efectos ante
   cambios de régimen. Los shocks estructurales ocurridos fuera de muestra
   (COVID, guerra de Ucrania, IRA) pueden alterar las elasticidades reales.
3. **Spillovers ecosistémicos correlacionales**: los efectos colaterales
   sobre investigadores, exportaciones de alta tecnología y artículos
   científicos son de naturaleza correlacional (β_W|D), no causales.
4. **Un único bootstrap paramétrico**: el IC 95% supone normalidad de los θ.
   La inferencia principal de la tesis se apoya en Driscoll–Kraay, no en el
   bootstrap del simulador — este último es una herramienta de comunicación.
""")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown(f"""
<div class="footer-sim">
  Simulador Causal vFINAL · Streamlit Community Cloud  ·
  Última compilación: {datetime.now().strftime('%d-%m-%Y')}<br>
  <em>Tesis de Maestría · Angel A. Ramírez Martínez · UASLP / SECIHTI</em>
</div>
""", unsafe_allow_html=True)
