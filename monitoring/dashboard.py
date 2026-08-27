"""Dashboard de monitoring de l'API de scoring.

Lecture des logs de production, calcul des indicateurs operationnels,
detection de derive et exploration interactive.

Lancement :
    streamlit run monitoring/dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

RACINE = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Monitoring - Scoring credit",
    page_icon="📊",
    layout="wide",
)

# --- Palette : une couleur = un sens, tenue dans tout le tableau de bord ---
BLEU, VERT, ORANGE, ROUGE, GRIS = "#4C78A8", "#54A24B", "#F58518", "#E45756", "#79706E"


# ======================================================================
#  Chargement des donnees (mis en cache)
# ======================================================================
@st.cache_data(ttl=300)
def charger_logs() -> pd.DataFrame:
    chemin = RACINE / "monitoring" / "production_logs.parquet"
    if not chemin.exists():
        return pd.DataFrame()
    df = pd.read_parquet(chemin)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")


@st.cache_data(ttl=300)
def charger_erreurs() -> pd.DataFrame:
    chemin = RACINE / "monitoring" / "production_logs_errors.parquet"
    if not chemin.exists():
        return pd.DataFrame()
    df = pd.read_parquet(chemin)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data
def charger_reference() -> pd.DataFrame:
    chemin = RACINE / "data" / "reference" / "reference.parquet"
    return pd.read_parquet(chemin) if chemin.exists() else pd.DataFrame()


@st.cache_data
def charger_metadata() -> dict:
    chemin = RACINE / "models" / "metadata.json"
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}


def calculer_psi(reference: pd.Series, courant: pd.Series, n_classes: int = 10) -> float:
    ref, cur = reference.dropna(), courant.dropna()
    if len(ref) < 20 or len(cur) < 20:
        return np.nan
    bornes = np.unique(np.quantile(ref, np.linspace(0, 1, n_classes + 1)))
    if len(bornes) < 3:
        return 0.0
    bornes[0], bornes[-1] = -np.inf, np.inf
    pr = np.clip(np.histogram(ref, bins=bornes)[0] / len(ref), 1e-6, None)
    pc = np.clip(np.histogram(cur, bins=bornes)[0] / len(cur), 1e-6, None)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


# ======================================================================
#  En-tete et barre laterale
# ======================================================================
st.title("Monitoring de l'API de scoring credit")
st.caption("Pret a depenser - Projet 8, Confirmez vos competences en MLOps")

logs = charger_logs()
erreurs = charger_erreurs()
reference = charger_reference()
metadata = charger_metadata()
SEUIL = float(metadata.get("threshold", 0.09))

with st.sidebar:
    st.header("Parametres")
    url_api = st.text_input("URL de l'API", "http://localhost:8000")

    if st.button("Tester la connexion"):
        try:
            reponse = requests.get(f"{url_api}/health", timeout=10)
            if reponse.status_code == 200:
                st.success("API en ligne")
                st.json(reponse.json())
            else:
                st.error(f"Code HTTP {reponse.status_code}")
        except Exception as exc:                       # noqa: BLE001
            st.error(f"Injoignable : {exc}")

    st.divider()
    if not logs.empty:
        debut, fin = logs["timestamp"].min(), logs["timestamp"].max()
        st.metric("Periode couverte", f"{(fin - debut).total_seconds() / 3600:.1f} h")
        st.caption(f"du {debut:%d/%m %H:%M} au {fin:%d/%m %H:%M}")
    st.divider()
    st.caption(f"Modele v{metadata.get('model_version', '?')} · seuil {SEUIL}")
    if st.button("Vider le cache"):
        st.cache_data.clear()
        st.rerun()

if logs.empty:
    st.warning("Aucun log de production trouve. Lance d'abord les deux commandes ci-dessous.")
    st.code("python scripts/simulate_traffic.py --n 300\n"
            "python scripts/fetch_logs.py --local", language="bash")
    st.stop()


# ======================================================================
#  Indicateurs cles
# ======================================================================
n_total = len(logs) + len(erreurs)
taux_erreur = len(erreurs) / n_total if n_total else 0.0
p50 = logs["latency_ms"].quantile(0.50)
p95 = logs["latency_ms"].quantile(0.95)
taux_refus = (logs["decision"] == "REFUSE").mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Predictions", f"{len(logs):,}".replace(",", " "))
c2.metric("Latence p50", f"{p50:.1f} ms")
c3.metric("Latence p95", f"{p95:.1f} ms",
          delta=f"{p95 - 100:.0f} ms vs SLO", delta_color="inverse")
c4.metric("Taux d'erreur", f"{100 * taux_erreur:.2f} %",
          delta="OK" if taux_erreur < 0.05 else "eleve",
          delta_color="normal" if taux_erreur < 0.05 else "inverse")
c5.metric("Taux de refus", f"{100 * taux_refus:.1f} %")

st.divider()

onglets = st.tabs([
    "Performance", "Predictions", "Data drift", "Erreurs", "Tester l'API",
])


# ----------------------------------------------------------------------
#  Onglet 1 - Performance
# ----------------------------------------------------------------------
with onglets[0]:
    st.subheader("Performance operationnelle")

    ga, gb = st.columns(2)
    with ga:
        fig = px.histogram(logs, x="latency_ms", nbins=60,
                           title="Distribution de la latence totale",
                           color_discrete_sequence=[BLEU])
        fig.add_vline(x=p95, line_dash="dash", line_color=ROUGE,
                      annotation_text=f"p95 = {p95:.1f} ms")
        st.plotly_chart(fig, use_container_width=True)

    with gb:
        fig = px.histogram(logs, x="inference_ms", nbins=60,
                           title="Temps d'inference pur du modele",
                           color_discrete_sequence=[VERT])
        st.plotly_chart(fig, use_container_width=True)

    # Serie temporelle des percentiles
    temporel = logs.set_index("timestamp").resample("1min").agg(
        p50=("latency_ms", lambda s: s.quantile(0.50)),
        p95=("latency_ms", lambda s: s.quantile(0.95)),
        n=("latency_ms", "size"),
    ).dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temporel.index, y=temporel["p50"],
                             name="p50", line=dict(color=BLEU)))
    fig.add_trace(go.Scatter(x=temporel.index, y=temporel["p95"],
                             name="p95", line=dict(color=ORANGE)))
    fig.add_hline(y=100, line_dash="dot", line_color=ROUGE,
                  annotation_text="SLO 100 ms")
    fig.update_layout(title="Evolution de la latence",
                      yaxis_title="millisecondes")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Decomposition de la latence")
    part_modele = logs["inference_ms"].sum() / logs["latency_ms"].sum()
    d1, d2 = st.columns(2)
    d1.metric("Part du modele dans la latence", f"{100 * part_modele:.1f} %")
    d2.metric("Part de l'infrastructure (parsing, validation, log)",
              f"{100 * (1 - part_modele):.1f} %")
    st.info(
        "Sur un modele d'arbres, l'inference represente une part minoritaire "
        "de la latence : l'essentiel est consomme par la serialisation JSON, "
        "la validation et la construction du vecteur de features. C'est ce "
        "constat qui a oriente le travail d'optimisation."
    )

    if "backend" in logs.columns and logs["backend"].nunique() > 1:
        st.subheader("Comparaison des moteurs d'inference")
        comparaison = logs.groupby("backend").agg(
            n=("inference_ms", "size"),
            inference_p50=("inference_ms", lambda s: round(s.quantile(0.50), 3)),
            inference_p95=("inference_ms", lambda s: round(s.quantile(0.95), 3)),
            latence_p95=("latency_ms", lambda s: round(s.quantile(0.95), 2)),
        )
        st.dataframe(comparaison, use_container_width=True)


# ----------------------------------------------------------------------
#  Onglet 2 - Predictions
# ----------------------------------------------------------------------
with onglets[1]:
    st.subheader("Distribution des scores predits")

    fig = px.histogram(logs, x="probability_default", nbins=60,
                       color="decision",
                       color_discrete_map={"ACCORDE": VERT, "REFUSE": ROUGE},
                       title="Scores de probabilite de defaut")
    fig.add_vline(x=SEUIL, line_dash="dash", line_color="black",
                  annotation_text=f"Seuil metier = {SEUIL}")
    st.plotly_chart(fig, use_container_width=True)

    ea, eb = st.columns(2)
    with ea:
        repartition = logs["decision"].value_counts().reset_index()
        repartition.columns = ["decision", "n"]
        fig = px.pie(repartition, names="decision", values="n",
                     color="decision",
                     color_discrete_map={"ACCORDE": VERT, "REFUSE": ROUGE},
                     title="Repartition des decisions", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)

    with eb:
        temporel = logs.set_index("timestamp").resample("2min").agg(
            taux_refus=("decision", lambda s: (s == "REFUSE").mean()),
            score_moyen=("probability_default", "mean"),
        ).dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=temporel.index,
                                 y=100 * temporel["taux_refus"],
                                 name="Taux de refus (%)",
                                 line=dict(color=ROUGE)))
        fig.update_layout(title="Evolution du taux de refus",
                          yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction drift")
    if not reference.empty:
        st.caption(
            "Comparaison de la distribution des scores de production avec "
            "celle qu'aurait produite le modele sur les donnees d'entrainement."
        )
        # On approxime avec un echantillon pour rester reactif
        st.metric("Score moyen en production",
                  f"{logs['probability_default'].mean():.4f}")


# ----------------------------------------------------------------------
#  Onglet 3 - Data drift
# ----------------------------------------------------------------------
with onglets[2]:
    st.subheader("Derive des donnees d'entree")

    if reference.empty:
        st.warning("Jeu de reference absent (data/reference/reference.parquet).")
    else:
        courant = pd.json_normalize(logs["features"])
        remplissage = courant.notna().mean()
        candidates = [c for c in remplissage[remplissage >= 0.5].index
                      if c in reference.columns]

        if not candidates:
            st.warning("Aucune colonne comparable entre reference et production.")
        else:
            with st.spinner("Calcul du PSI..."):
                psi = pd.DataFrame([
                    {
                        "variable": c,
                        "psi": round(calculer_psi(reference[c], courant[c]), 4),
                        "moy_reference": round(float(reference[c].mean()), 3),
                        "moy_production": round(float(courant[c].mean()), 3),
                    }
                    for c in candidates[:40]
                ]).dropna(subset=["psi"]).sort_values("psi", ascending=False)

            psi["statut"] = pd.cut(
                psi["psi"], bins=[-np.inf, 0.10, 0.25, np.inf],
                labels=["stable", "derive moderee", "DERIVE IMPORTANTE"])

            a, b, c = st.columns(3)
            a.metric("Variables stables", int((psi["psi"] < 0.10).sum()))
            b.metric("Derive moderee", int(psi["psi"].between(0.10, 0.25).sum()))
            c.metric("Derive importante", int((psi["psi"] > 0.25).sum()))

            fig = px.bar(
                psi.head(15).sort_values("psi"), x="psi", y="variable",
                orientation="h", color="statut",
                color_discrete_map={"stable": VERT,
                                    "derive moderee": ORANGE,
                                    "DERIVE IMPORTANTE": ROUGE},
                title="PSI par variable")
            fig.add_vline(x=0.10, line_dash="dash", line_color=ORANGE)
            fig.add_vline(x=0.25, line_dash="dash", line_color=ROUGE)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(psi, use_container_width=True, hide_index=True)

            st.subheader("Comparaison d'une variable")
            choix = st.selectbox("Variable", psi["variable"].tolist())
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=reference[choix].dropna(),
                                       name="Reference", opacity=0.6,
                                       histnorm="probability density",
                                       marker_color=BLEU))
            fig.add_trace(go.Histogram(x=courant[choix].dropna(),
                                       name="Production", opacity=0.6,
                                       histnorm="probability density",
                                       marker_color=ROUGE))
            fig.update_layout(barmode="overlay", title=f"Distribution de {choix}")
            st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
#  Onglet 4 - Erreurs
# ----------------------------------------------------------------------
with onglets[3]:
    st.subheader("Analyse des erreurs")
    if erreurs.empty:
        st.success("Aucune erreur enregistree sur la periode.")
    else:
        a, b = st.columns(2)
        with a:
            par_code = erreurs["status"].value_counts().reset_index()
            par_code.columns = ["code HTTP", "n"]
            fig = px.bar(par_code, x="code HTTP", y="n",
                         title="Erreurs par code HTTP",
                         color_discrete_sequence=[ORANGE])
            st.plotly_chart(fig, use_container_width=True)
        with b:
            par_type = erreurs["error_type"].value_counts().reset_index()
            par_type.columns = ["type", "n"]
            fig = px.pie(par_type, names="type", values="n", hole=0.45,
                         title="Erreurs par type")
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Les erreurs 422 sont **attendues et saines** : ce sont des "
            "requetes invalides correctement rejetees par la validation. "
            "Ce qui doit alerter, ce sont les erreurs **500** (defaut cote "
            "serveur) et une hausse brutale du volume total."
        )
        st.dataframe(
            erreurs[["timestamp", "request_id", "endpoint", "status",
                     "error_type", "message"]].tail(50),
            use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------
#  Onglet 5 - Tester l'API en direct
# ----------------------------------------------------------------------
with onglets[4]:
    st.subheader("Interroger l'API en direct")
    st.caption("Utile pour la demonstration : on voit la requete et la reponse.")

    mode = st.radio("Mode", ["Par identifiant client", "Par features"],
                    horizontal=True)

    if mode == "Par identifiant client":
        client_id = st.number_input("Identifiant client", min_value=0,
                                    value=0, step=1)
        corps = {"client_id": int(client_id)}
    else:
        col1, col2 = st.columns(2)
        with col1:
            revenu = st.number_input("Revenu annuel (AMT_INCOME_TOTAL)",
                                     0.0, 5_000_000.0, 202500.0, step=10000.0)
            credit = st.number_input("Montant du credit (AMT_CREDIT)",
                                     0.0, 5_000_000.0, 406597.0, step=10000.0)
            age = st.slider("Age", 18, 80, 35)
        with col2:
            ext2 = st.slider("Score externe 2 (EXT_SOURCE_2)", 0.0, 1.0, 0.26)
            ext3 = st.slider("Score externe 3 (EXT_SOURCE_3)", 0.0, 1.0, 0.14)
            enfants = st.number_input("Nombre d'enfants", 0, 15, 0)
        corps = {"features": {
            "AMT_INCOME_TOTAL": revenu,
            "AMT_CREDIT": credit,
            "DAYS_BIRTH": -age * 365.25,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "CNT_CHILDREN": float(enfants),
        }}

    st.code(json.dumps(corps, indent=2), language="json")

    if st.button("Envoyer la requete", type="primary"):
        try:
            reponse = requests.post(f"{url_api}/predict", json=corps, timeout=30)
            if reponse.status_code == 200:
                resultat = reponse.json()
                a, b, c = st.columns(3)
                a.metric("Probabilite de defaut",
                         f"{100 * resultat['probability_default']:.2f} %")
                b.metric("Decision", resultat["decision"])
                c.metric("Latence", f"{resultat['latency_ms']:.1f} ms")

                jauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=100 * resultat["probability_default"],
                    number={"suffix": " %"},
                    title={"text": "Risque de defaut"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": GRIS},
                        "steps": [
                            {"range": [0, 100 * SEUIL], "color": VERT},
                            {"range": [100 * SEUIL, 100], "color": ROUGE},
                        ],
                        "threshold": {"line": {"color": "black", "width": 4},
                                      "value": 100 * SEUIL},
                    },
                ))
                jauge.update_layout(height=280)
                st.plotly_chart(jauge, use_container_width=True)
                with st.expander("Reponse complete de l'API"):
                    st.json(resultat)
            else:
                st.error(f"Code HTTP {reponse.status_code}")
                st.json(reponse.json())
        except Exception as exc:                       # noqa: BLE001
            st.error(f"Erreur : {exc}")


st.divider()
st.caption(
    "Projet 8 - Confirmez vos competences en MLOps · Yohan Parent · "
    f"Modele v{metadata.get('model_version', '?')} · seuil metier {SEUIL}"
)
