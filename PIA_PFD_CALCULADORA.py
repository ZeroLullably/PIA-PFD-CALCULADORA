import io
import zipfile
from pathlib import Path
from datetime import date
from math import log, exp, sqrt, erf, pi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

# ============================================================
# CALCULADORA WEB BLACK-SCHOLES-MERTON
# Ejecutar localmente: streamlit run PIA_PFD_CALCULADORA.py
# ============================================================

st.set_page_config(page_title="Calculadora BSM", page_icon="📈", layout="wide")

TRADING_DAYS = 252
DAYS_YEAR = 365
RATE_COMPOUNDING = 2
OUTPUT_ROOT = Path.cwd() / "Resultados"


# -------------------- FUNCIONES BSM --------------------

def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def norm_pdf(x):
    return exp(-0.5 * x * x) / sqrt(2 * pi)


def d1_d2(S, K, T, r, sigma, q=0.0):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T y volatilidad deben ser positivos.")
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return d1, d1 - sigma * sqrt(T)


def bsm(S, K, T, r, sigma, kind, q=0.0):
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    if kind.lower() == "call":
        return S * exp(-q*T) * norm_cdf(d1) - K * exp(-r*T) * norm_cdf(d2)
    if kind.lower() == "put":
        return K * exp(-r*T) * norm_cdf(-d2) - S * exp(-q*T) * norm_cdf(-d1)
    raise ValueError("Tipo de opción inválido.")


def greeks(S, K, T, r, sigma, kind, q=0.0):
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    gamma = exp(-q*T) * norm_pdf(d1) / (S * sigma * sqrt(T))
    vega = S * exp(-q*T) * norm_pdf(d1) * sqrt(T) / 100

    if kind.lower() == "call":
        delta = exp(-q*T) * norm_cdf(d1)
        theta = (-S*exp(-q*T)*norm_pdf(d1)*sigma/(2*sqrt(T))
                 - r*K*exp(-r*T)*norm_cdf(d2)
                 + q*S*exp(-q*T)*norm_cdf(d1))
        rho = K*T*exp(-r*T)*norm_cdf(d2)/100
    else:
        delta = exp(-q*T) * (norm_cdf(d1)-1)
        theta = (-S*exp(-q*T)*norm_pdf(d1)*sigma/(2*sqrt(T))
                 + r*K*exp(-r*T)*norm_cdf(-d2)
                 - q*S*exp(-q*T)*norm_cdf(-d1))
        rho = -K*T*exp(-r*T)*norm_cdf(-d2)/100

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega_por_1pct": vega,
        "Theta_diaria": theta / DAYS_YEAR,
        "Rho_por_1pct": rho
    }


def option_price_bounds(S, K, T, r, q, kind):
    """Límites teóricos europeos con dividend yield continuo."""
    discounted_spot = S * exp(-q*T)
    discounted_strike = K * exp(-r*T)

    if kind.lower() == "call":
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot

    return max(discounted_strike - discounted_spot, 0.0), discounted_strike


def vega_unscaled(S, K, T, r, sigma, q=0.0):
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return S * exp(-q*T) * norm_pdf(d1) * sqrt(T)


def implied_vol_bisection(price, S, K, T, r, kind, q=0.0):
    low, high = 1e-6, 5.0

    try:
        lower_bound, upper_bound = option_price_bounds(S, K, T, r, q, kind)
        tolerance = max(0.01, upper_bound * 1e-6)

        if price < lower_bound - tolerance or price > upper_bound + tolerance:
            return np.nan

        price = min(max(price, lower_bound), upper_bound)

        f_low = bsm(S, K, T, r, low, kind, q) - price
        f_high = bsm(S, K, T, r, high, kind, q) - price

        if abs(f_low) < 1e-9:
            return low

        if abs(f_high) < 1e-9:
            return high

        if f_low * f_high > 0:
            return np.nan

        for _ in range(250):
            mid = (low + high) / 2
            f_mid = bsm(S, K, T, r, mid, kind, q) - price

            if abs(f_mid) < 1e-8:
                return mid

            if f_low * f_mid < 0:
                high, f_high = mid, f_mid
            else:
                low, f_low = mid, f_mid

        return (low + high) / 2

    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan


def implied_vol_newton(price, S, K, T, r, kind, initial_sigma, q=0.0):
    lower_bound, upper_bound = option_price_bounds(S, K, T, r, q, kind)
    tolerance = max(0.01, upper_bound * 1e-6)

    if not np.isfinite(price) or price <= 0:
        return np.nan, "Sin cotización válida"

    if price < lower_bound - tolerance or price > upper_bound + tolerance:
        return np.nan, "Precio fuera de límites BSM"

    sigma = min(max(float(initial_sigma), 0.01), 5.0)

    for _ in range(100):
        try:
            error = bsm(S, K, T, r, sigma, kind, q) - price

            if abs(error) < 1e-7:
                return sigma, "Newton-Raphson"

            vega = vega_unscaled(S, K, T, r, sigma, q)

            if vega < 1e-10:
                break

            candidate = sigma - error / vega

            if not np.isfinite(candidate) or candidate <= 0.000001 or candidate > 5:
                break

            sigma = candidate

        except (ValueError, OverflowError, ZeroDivisionError):
            break

    result = implied_vol_bisection(price, S, K, T, r, kind, q)

    if np.isfinite(result):
        return result, "Bisección (respaldo)"

    return np.nan, "No convergió"


# -------------------- DATOS DE YAHOO FINANCE --------------------

@st.cache_data(ttl=900, show_spinner=False)
def get_history(ticker, ref_date, years):
    ref = pd.Timestamp(ref_date)
    start = ref - pd.DateOffset(years=int(years))

    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(ref + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data.empty:
        raise ValueError(f"No se encontraron históricos para {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.sort_index()

    if "Close" not in data.columns:
        raise ValueError("Yahoo Finance no devolvió la columna Close.")

    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"

    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=[price_col, "Close"])

    if len(data) < 3:
        raise ValueError("No hay suficientes observaciones históricas.")

    returns = np.log(data[price_col] / data[price_col].shift(1)).dropna()
    sigma = float(returns.std(ddof=1) * sqrt(TRADING_DAYS))
    spot = float(data["Close"].iloc[-1])

    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(spot) or spot <= 0:
        raise ValueError("No fue posible calcular un precio o volatilidad válidos.")

    return data, returns, sigma, spot, data.index[-1].date(), start.date()


@st.cache_data(ttl=300, show_spinner=False)
def get_expirations(ticker):
    return tuple(yf.Ticker(ticker).options)


@st.cache_data(ttl=300, show_spinner=False)
def get_option_chain(ticker, expiration):
    chain = yf.Ticker(ticker).option_chain(expiration)

    calls = chain.calls.copy()
    puts = chain.puts.copy()

    cols = ["strike", "bid", "ask", "volume", "openInterest"]

    for col in cols:
        if col not in calls.columns:
            calls[col] = np.nan
        if col not in puts.columns:
            puts[col] = np.nan

    calls = calls[cols].rename(columns={
        "strike": "Strike",
        "bid": "Call_Bid",
        "ask": "Call_Ask",
        "volume": "Call_Volumen",
        "openInterest": "Call_OI"
    })

    puts = puts[cols].rename(columns={
        "strike": "Strike",
        "bid": "Put_Bid",
        "ask": "Put_Ask",
        "volume": "Put_Volumen",
        "openInterest": "Put_OI"
    })

    result = pd.merge(calls, puts, on="Strike", how="outer").sort_values("Strike")

    numeric_cols = [
        "Strike", "Call_Bid", "Call_Ask", "Call_Volumen", "Call_OI",
        "Put_Bid", "Put_Ask", "Put_Volumen", "Put_OI"
    ]

    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    for col in ["Call_Volumen", "Call_OI", "Put_Volumen", "Put_OI"]:
        result[col] = result[col].fillna(0)

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=["Strike"])
    result = result[result["Strike"] > 0]

    return result.reset_index(drop=True)


# -------------------- VALIDACIÓN Y RESULTADOS --------------------

def valid_midpoint(bid, ask, S, K, T, r, q, kind):
    """Devuelve midpoint solo si bid/ask son positivos, ordenados y compatibles con BSM."""

    if pd.isna(bid) or pd.isna(ask):
        return np.nan, "Sin cotización"

    bid, ask = float(bid), float(ask)

    if bid <= 0 or ask <= 0:
        return np.nan, "Bid/Ask cero o vacío"

    if ask < bid:
        return np.nan, "Mercado cruzado (Ask < Bid)"

    midpoint = (bid + ask) / 2
    lower, upper = option_price_bounds(S, K, T, r, q, kind)
    tolerance = max(0.02, upper * 0.002)

    if midpoint < lower - tolerance:
        return np.nan, "Precio inferior al límite teórico"

    if midpoint > upper + tolerance:
        return np.nan, "Precio superior al límite teórico"

    return midpoint, "Válida"


def build_results(options, S, T, r, sigma, q):
    rows = []

    for _, row in options.iterrows():
        K = float(row["Strike"])

        cm, call_status = valid_midpoint(
            row["Call_Bid"], row["Call_Ask"], S, K, T, r, q, "call"
        )

        pm, put_status = valid_midpoint(
            row["Put_Bid"], row["Put_Ask"], S, K, T, r, q, "put"
        )

        call_theory = bsm(S, K, T, r, sigma, "call", q)
        put_theory = bsm(S, K, T, r, sigma, "put", q)

        call_iv, call_method = (
            implied_vol_newton(cm, S, K, T, r, "call", sigma, q)
            if pd.notna(cm) else (np.nan, call_status)
        )

        put_iv, put_method = (
            implied_vol_newton(pm, S, K, T, r, "put", sigma, q)
            if pd.notna(pm) else (np.nan, put_status)
        )

        parity_market = cm - pm if pd.notna(cm) and pd.notna(pm) else np.nan
        parity_theory = S * exp(-q*T) - K * exp(-r*T)

        rows.append({
            "Strike": K,
            "Moneyness_K_S0": K / S,

            "Call_Bid": row["Call_Bid"],
            "Call_Ask": row["Call_Ask"],
            "Call_Volumen": row["Call_Volumen"],
            "Call_OI": row["Call_OI"],
            "Call_Mercado": cm,
            "Call_Estado": call_status,
            "Call_BSM": call_theory,
            "Call_IV": call_iv,
            "Call_Metodo_IV": call_method,
            "Call_Diferencia": cm - call_theory if pd.notna(cm) else np.nan,

            "Put_Bid": row["Put_Bid"],
            "Put_Ask": row["Put_Ask"],
            "Put_Volumen": row["Put_Volumen"],
            "Put_OI": row["Put_OI"],
            "Put_Mercado": pm,
            "Put_Estado": put_status,
            "Put_BSM": put_theory,
            "Put_IV": put_iv,
            "Put_Metodo_IV": put_method,
            "Put_Diferencia": pm - put_theory if pd.notna(pm) else np.nan,

            "Paridad_C_menos_P": parity_market,
            "Paridad_S_menos_Ke_rt": parity_theory,
            "Diferencia_Paridad": (
                parity_market - parity_theory
                if pd.notna(parity_market) else np.nan
            )
        })

    return pd.DataFrame(rows)


def build_greeks(options, S, T, r, sigma, q):
    rows = []

    for _, row in options.iterrows():
        K = float(row["Strike"])

        for kind in ["call", "put"]:
            rows.append({
                "Strike": K,
                "Tipo": kind.capitalize(),
                **greeks(S, K, T, r, sigma, kind, q)
            })

    return pd.DataFrame(rows)


def make_figures(results, greek_table, ticker):
    figs = {}

    # Comparación de precios
    fig, ax = plt.subplots(figsize=(10, 5))

    for col, label, marker in [
        ("Call_Mercado", "Call mercado (cotización válida)", "o"),
        ("Call_BSM", "Call BSM", "o"),
        ("Put_Mercado", "Put mercado (cotización válida)", "s"),
        ("Put_BSM", "Put BSM", "s")
    ]:
        plot_data = results.dropna(subset=[col])
        ax.plot(plot_data["Strike"], plot_data[col], marker=marker, label=label)

    ax.set(
        xlabel="Strike",
        ylabel="Precio de opción",
        title=f"{ticker}: precios de mercado y teóricos BSM"
    )
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()

    figs["01_comparacion_precios.png"] = fig

    # Volatility Skew
    fig, ax = plt.subplots(figsize=(10, 5))

    for col, label, marker in [
        ("Call_IV", "Call", "o"),
        ("Put_IV", "Put", "s")
    ]:
        valid = results.dropna(subset=[col]).sort_values("Moneyness_K_S0")
        ax.plot(
            valid["Moneyness_K_S0"],
            valid[col] * 100,
            marker=marker,
            label=label
        )

    ax.set(
        xlabel="Moneyness (K / S₀)",
        ylabel="Volatilidad implícita (%)",
        title=f"{ticker}: Volatility Skew"
    )
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()

    figs["02_volatility_skew.png"] = fig

    # Volatility Smile
    fig, ax = plt.subplots(figsize=(10, 5))

    for col, label, marker in [
        ("Call_IV", "Call", "o"),
        ("Put_IV", "Put", "s")
    ]:
        valid = results.dropna(subset=[col]).sort_values("Strike")
        ax.plot(
            valid["Strike"],
            valid[col] * 100,
            marker=marker,
            label=label
        )

    ax.set(
        xlabel="Strike",
        ylabel="Volatilidad implícita (%)",
        title=f"{ticker}: Volatility Smile"
    )
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()

    figs["03_volatility_smile.png"] = fig

    # Letras griegas
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))

    cols = [
        ("Delta", "Delta"),
        ("Gamma", "Gamma"),
        ("Vega_por_1pct", "Vega por 1%"),
        ("Theta_diaria", "Theta diaria"),
        ("Rho_por_1pct", "Rho por 1%")
    ]

    for ax, (col, title) in zip(axes, cols):
        labels = (
            greek_table["Strike"].map(lambda x: f"{x:g}")
            + " "
            + greek_table["Tipo"]
        )

        ax.bar(range(len(greek_table)), greek_table[col].astype(float))
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=90, labelsize=6)

        if len(labels) <= 30:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
        else:
            step = max(1, len(labels) // 20)
            ax.set_xticks(range(0, len(labels), step))
            ax.set_xticklabels(labels.iloc[::step])

        ax.axhline(0, linewidth=.8)
        ax.grid(axis="y", alpha=.25)

    fig.suptitle(f"{ticker}: letras griegas")
    fig.tight_layout()

    figs["04_letras_griegas.png"] = fig

    # Put-Call Parity
    fig, ax = plt.subplots(figsize=(10, 5))

    valid_parity = results.dropna(subset=["Paridad_C_menos_P"])

    ax.plot(
        valid_parity["Strike"],
        valid_parity["Paridad_C_menos_P"],
        marker="o",
        label="C - P (mercado válido)"
    )

    ax.plot(
        results["Strike"],
        results["Paridad_S_menos_Ke_rt"],
        marker="s",
        label="S₀ exp(-qT) - K exp(-rT)"
    )

    ax.set(
        xlabel="Strike",
        ylabel="Valor",
        title=f"{ticker}: Put-Call Parity"
    )
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()

    figs["05_put_call_parity.png"] = fig

    return figs


def save_results(
    ticker, ref_date, expiration, history,
    options, results, greek_table, figures
):
    folder = OUTPUT_ROOT / ticker / f"{ref_date}_{expiration}"
    folder.mkdir(parents=True, exist_ok=True)

    history.to_csv(folder / "historicos_utilizados.csv")
    options.to_csv(folder / "opciones_descargadas.csv", index=False)
    results.to_csv(folder / "resultados_opciones.csv", index=False)
    greek_table.to_csv(folder / "letras_griegas.csv", index=False)

    for name, fig in figures.items():
        fig.savefig(folder / name, dpi=300, bbox_inches="tight")

    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in folder.iterdir():
            if file.is_file():
                zf.write(file, arcname=file.name)

    buffer.seek(0)

    return folder, buffer.getvalue()


# -------------------- INTERFAZ --------------------

st.title("Calculadora Black-Scholes-Merton")
st.caption("Valuación de opciones, volatilidad implícita y medidas de riesgo.")

with st.sidebar:
    st.header("Datos generales")

    with st.form("asset_form"):
        ticker = st.text_input("Ticker bursátil", value="TSLA").strip().upper()

        reference_date = st.date_input(
            "Fecha de referencia",
            value=date.today()
        )

        years = st.number_input(
            "Años para volatilidad histórica",
            min_value=1,
            max_value=100,
            value=3,
            step=1,
            help="Puedes introducir cualquier número entero positivo: 1, 2, 3, 4, 5, etc."
        )

        rate_pct = st.number_input(
            "Tasa libre de riesgo anual (%)",
            min_value=-99.0,
            max_value=100.0,
            value=4.05,
            step=0.05
        )

        dividend_pct = st.number_input(
            "Dividend yield anual (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.05
        )

        moneyness_range = st.slider(
            "Rango de strikes para mostrar (K / S₀)",
            min_value=0.10,
            max_value=3.00,
            value=(0.80, 1.20),
            step=0.05,
            help="Limita tablas y gráficas a strikes cercanos al precio del activo. "
                 "No elimina los contratos de los archivos originales."
        )

        consult = st.form_submit_button(
            "Consultar activo",
            type="primary",
            use_container_width=True
        )

    if st.button("Limpiar análisis", use_container_width=True):
        for key in ["asset", "expirations", "analysis", "folder", "zip"]:
            st.session_state.pop(key, None)

        st.rerun()


if consult:
    if not ticker:
        st.error("Introduce un ticker.")

    elif reference_date > date.today():
        st.error("La fecha de referencia no puede ser futura.")

    elif int(years) != years:
        st.error("El periodo debe ser un número entero de años.")

    else:
        try:
            with st.status("Consultando Yahoo Finance...", expanded=True) as status:
                st.write(f"Descargando históricos de {ticker}...")

                hist, returns, sigma, spot, actual_ref, start_date = get_history(
                    ticker,
                    reference_date,
                    int(years)
                )

                st.write("Consultando vencimientos disponibles...")

                all_expirations = get_expirations(ticker)

                valid_expirations = [
                    x for x in all_expirations
                    if pd.Timestamp(x).date() > actual_ref
                ]

                if not valid_expirations:
                    raise ValueError(
                        "No hay vencimientos posteriores a la fecha de referencia."
                    )

                st.session_state["asset"] = {
                    "ticker": ticker,
                    "history": hist,
                    "returns": returns,
                    "sigma": sigma,
                    "spot": spot,
                    "actual_ref": actual_ref,
                    "requested_ref": reference_date,
                    "start_date": start_date,
                    "years": int(years),
                    "rate": rate_pct / 100,
                    "q": dividend_pct / 100,
                    "moneyness_range": tuple(moneyness_range)
                }

                st.session_state["expirations"] = valid_expirations

                for key in ["analysis", "folder", "zip"]:
                    st.session_state.pop(key, None)

                status.update(
                    label="Datos del activo descargados.",
                    state="complete"
                )

        except Exception as e:
            st.error(f"No se pudo consultar Yahoo Finance: {e}")


if "asset" in st.session_state:
    asset = st.session_state["asset"]

    st.subheader(f"Información del activo: {asset['ticker']}")

    if asset["actual_ref"] != asset["requested_ref"]:
        st.warning(
            f"No hubo cotización en {asset['requested_ref']}. "
            f"Se utiliza el último cierre disponible: {asset['actual_ref']}."
        )

    if asset["actual_ref"] < date.today():
        st.warning(
            "Yahoo Finance proporciona la cadena de opciones disponible actualmente, "
            "no necesariamente cotizaciones históricas de la fecha de referencia. "
            "Para un análisis histórico, las cotizaciones de opciones deben obtenerse "
            "de una fuente que conserve datos históricos."
        )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Precio S₀", f"${asset['spot']:,.2f}")
    c2.metric("Volatilidad histórica", f"{asset['sigma']:.2%}")
    c3.metric("Tasa nominal", f"{asset['rate']:.2%}")
    c4.metric("Dividend yield", f"{asset['q']:.2%}")

    st.caption(
        f"Históricos: {asset['start_date']} a {asset['actual_ref']} · "
        f"Periodo: {asset['years']} año(s)"
    )

    st.subheader("Seleccionar vencimiento")

    with st.form("expiry_form"):
        expiry = st.selectbox(
            "Vencimientos disponibles en Yahoo Finance",
            st.session_state["expirations"]
        )

        run_analysis = st.form_submit_button(
            "Descargar opciones y calcular análisis",
            type="primary"
        )

    if run_analysis:
        try:
            with st.status("Descargando opciones y calculando...", expanded=True) as status:
                st.write("Descargando cadena de opciones...")

                options = get_option_chain(asset["ticker"], expiry)

                days = (
                    pd.Timestamp(expiry).date() - asset["actual_ref"]
                ).days

                if days <= 0:
                    raise ValueError(
                        "El vencimiento debe ser posterior a la fecha de referencia."
                    )

                T = days / DAYS_YEAR

                r = RATE_COMPOUNDING * log(
                    1 + asset["rate"] / RATE_COMPOUNDING
                )

                st.write("Validando cotizaciones y calculando precios BSM...")

                results = build_results(
                    options,
                    asset["spot"],
                    T,
                    r,
                    asset["sigma"],
                    asset["q"]
                )

                if results.empty:
                    raise ValueError("Yahoo Finance no devolvió contratos analizables.")

                st.write("Calculando griegas y generando gráficas...")

                low_m, high_m = asset["moneyness_range"]

                visible_results = results[
                    results["Moneyness_K_S0"].between(low_m, high_m)
                ].copy()

                visible_options = options[
                    (options["Strike"] / asset["spot"]).between(low_m, high_m)
                ].copy()

                greek_table = build_greeks(
                    visible_options,
                    asset["spot"],
                    T,
                    r,
                    asset["sigma"],
                    asset["q"]
                )

                figures = make_figures(
                    visible_results,
                    greek_table,
                    asset["ticker"]
                )

                st.write("Guardando tablas, históricos y gráficas...")

                folder, zip_data = save_results(
                    asset["ticker"],
                    str(asset["actual_ref"]),
                    expiry,
                    asset["history"],
                    options,
                    results,
                    greek_table,
                    figures
                )

                # Guardar también el subconjunto cercano al precio spot.
                visible_results.to_csv(
                    folder / "resultados_rango_cercano.csv",
                    index=False
                )

                visible_options.to_csv(
                    folder / "opciones_rango_cercano.csv",
                    index=False
                )

                buffer = io.BytesIO()

                with zipfile.ZipFile(
                    buffer,
                    "w",
                    zipfile.ZIP_DEFLATED
                ) as zf:
                    for file in folder.iterdir():
                        if file.is_file():
                            zf.write(file, arcname=file.name)

                buffer.seek(0)
                zip_data = buffer.getvalue()

                st.session_state["analysis"] = {
                    "options": options,
                    "results": results,
                    "visible_options": visible_options,
                    "visible_results": visible_results,
                    "greeks": greek_table,
                    "figures": figures,
                    "expiry": expiry,
                    "T": T,
                    "days": days,
                    "r": r
                }

                st.session_state["folder"] = str(folder)
                st.session_state["zip"] = zip_data

                status.update(
                    label="Análisis terminado. Archivos guardados.",
                    state="complete"
                )

        except Exception as e:
            st.error(f"No se pudo completar el análisis: {e}")


if "analysis" in st.session_state:
    asset = st.session_state["asset"]
    analysis = st.session_state["analysis"]

    all_results = analysis["results"]
    results = analysis["visible_results"]
    greek_table = analysis["greeks"]

    low_m, high_m = asset["moneyness_range"]

    st.divider()
    st.header("Resultados del análisis")

    m1, m2, m3 = st.columns(3)

    m1.metric("Vencimiento", analysis["expiry"])
    m2.metric("Días al vencimiento", analysis["days"])
    m3.metric("Tiempo T", f"{analysis['T']:.6f}")

    valid_calls = results["Call_Mercado"].notna().sum()
    valid_puts = results["Put_Mercado"].notna().sum()
    paired_quotes = results["Paridad_C_menos_P"].notna().sum()

    st.info(
        f"Rango mostrado: {low_m:.2f} ≤ K/S₀ ≤ {high_m:.2f}. "
        f"En este rango hay {len(results)} strikes, {valid_calls} Calls válidas, "
        f"{valid_puts} Puts válidas y {paired_quotes} pares Call-Put utilizables. "
        "Las cotizaciones rechazadas no se usan para volatilidad implícita ni paridad."
    )

    with st.expander("Consultar cadena completa descargada de Yahoo Finance"):
        st.caption(
            "Se conserva la cadena completa para consulta y exportación. "
            "Los cálculos y gráficas principales se concentran en el rango seleccionado."
        )

        st.dataframe(
            all_results[[
                "Strike",
                "Call_Bid",
                "Call_Ask",
                "Call_Mercado",
                "Call_Estado",
                "Put_Bid",
                "Put_Ask",
                "Put_Mercado",
                "Put_Estado"
            ]].style.format(precision=4, na_rep="—"),
            use_container_width=True,
            height=350,
            hide_index=True
        )

    if results.empty:
        st.warning(
            "No hay strikes dentro del rango seleccionado. "
            "Amplía el rango K/S₀ desde el panel lateral y vuelve a consultar el activo."
        )

    tabs = st.tabs([
        "Precios BSM",
        "Volatilidad implícita",
        "Griegas",
        "Put-Call Parity",
        "Gráficas",
        "Calculadora individual",
        "Exportar"
    ])

    # ---------------- PRECIOS BSM ----------------

    with tabs[0]:
        cols = [
            "Strike",
            "Call_Bid",
            "Call_Ask",
            "Call_Mercado",
            "Call_Estado",
            "Call_BSM",
            "Call_Diferencia",
            "Put_Bid",
            "Put_Ask",
            "Put_Mercado",
            "Put_Estado",
            "Put_BSM",
            "Put_Diferencia"
        ]

        st.dataframe(
            results[cols].style.format(precision=4, na_rep="—"),
            use_container_width=True,
            height=520,
            hide_index=True
        )

        st.download_button(
            "Descargar tabla CSV",
            results[cols].to_csv(index=False).encode("utf-8"),
            f"{asset['ticker']}_precios.csv",
            "text/csv"
        )

    # ---------------- VOLATILIDAD IMPLÍCITA ----------------

    with tabs[1]:
        cols = [
            "Strike",
            "Moneyness_K_S0",
            "Call_IV",
            "Call_Metodo_IV",
            "Put_IV",
            "Put_Metodo_IV"
        ]

        st.dataframe(
            results[cols].style.format({
                "Strike": "{:,.2f}",
                "Moneyness_K_S0": "{:.4f}",
                "Call_IV": "{:.2%}",
                "Put_IV": "{:.2%}"
            }, na_rep="—"),
            use_container_width=True,
            height=520,
            hide_index=True
        )

        st.download_button(
            "Descargar tabla CSV",
            results[cols].to_csv(index=False).encode("utf-8"),
            f"{asset['ticker']}_volatilidades.csv",
            "text/csv"
        )

    # ---------------- GRIEGAS ----------------

    with tabs[2]:
        st.dataframe(
            greek_table.style.format({
                "Strike": "{:,.2f}",
                "Delta": "{:.6f}",
                "Gamma": "{:.8f}",
                "Vega_por_1pct": "{:.6f}",
                "Theta_diaria": "{:.6f}",
                "Rho_por_1pct": "{:.6f}"
            }, na_rep="—"),
            use_container_width=True,
            height=520,
            hide_index=True
        )

        st.caption(
            "Vega y Rho se expresan por 1 punto porcentual. "
            "Theta se presenta por día. Las griegas se estiman usando "
            "la volatilidad histórica seleccionada."
        )

        st.download_button(
            "Descargar tabla CSV",
            greek_table.to_csv(index=False).encode("utf-8"),
            f"{asset['ticker']}_griegas.csv",
            "text/csv"
        )

    # ---------------- PUT-CALL PARITY ----------------

    with tabs[3]:
        cols = [
            "Strike",
            "Paridad_C_menos_P",
            "Paridad_S_menos_Ke_rt",
            "Diferencia_Paridad"
        ]

        st.dataframe(
            results[cols].style.format(precision=4, na_rep="—"),
            use_container_width=True,
            height=520,
            hide_index=True
        )

        st.caption(
            "La paridad observada solo se calcula cuando existen cotizaciones "
            "válidas tanto para Call como para Put del mismo strike y vencimiento."
        )

        if paired_quotes == 0:
            st.warning(
                "No hay pares Call-Put con cotizaciones válidas en el rango actual. "
                "No es posible evaluar empíricamente la paridad con esta cadena."
            )

        st.download_button(
            "Descargar tabla CSV",
            results[cols].to_csv(index=False).encode("utf-8"),
            f"{asset['ticker']}_paridad.csv",
            "text/csv"
        )

    # ---------------- GRÁFICAS ----------------

    with tabs[4]:
        for name, fig in analysis["figures"].items():
            st.pyplot(fig, clear_figure=False)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")

            st.download_button(
                f"Descargar {name}",
                buf.getvalue(),
                f"{asset['ticker']}_{name}",
                "image/png",
                key=f"img_{name}"
            )

    # ---------------- CALCULADORA INDIVIDUAL ----------------

    with tabs[5]:
        st.caption(
            "Modifica los parámetros y presiona Enter o «Calcular opción» "
            "para actualizar los resultados. Los cálculos no se ejecutan "
            "mientras escribes."
        )

        with st.form("individual_calculator_form"):
            col1, col2 = st.columns(2)

            with col1:
                strike = st.number_input(
                    "Strike K",
                    min_value=0.01,
                    value=float(asset["spot"]),
                    step=1.0,
                    format="%.2f",
                    key="individual_strike"
                )

                kind = st.selectbox(
                    "Tipo de opción",
                    ["Call", "Put"],
                    key="individual_kind"
                )

            with col2:
                vol_pct = st.number_input(
                    "Volatilidad (%)",
                    min_value=0.01,
                    max_value=500.0,
                    value=float(min(asset["sigma"] * 100, 500)),
                    step=0.5,
                    format="%.2f",
                    key="individual_volatility"
                )

                st.write(f"Precio S₀: ${asset['spot']:.4f}")

            calculate_individual = st.form_submit_button(
                "Calcular opción",
                type="primary"
            )

        if calculate_individual:
            price = bsm(
                asset["spot"],
                strike,
                analysis["T"],
                analysis["r"],
                vol_pct / 100,
                kind.lower(),
                asset["q"]
            )

            greek_values = greeks(
                asset["spot"],
                strike,
                analysis["T"],
                analysis["r"],
                vol_pct / 100,
                kind.lower(),
                asset["q"]
            )

            st.success(f"Precio teórico {kind}: ${price:,.6f}")

            st.dataframe(
                pd.DataFrame([greek_values]).style.format(precision=6),
                use_container_width=True,
                hide_index=True
            )

    # ---------------- EXPORTAR ----------------

    with tabs[6]:
        st.success("Los resultados se guardaron y están listos para descargar.")

        st.code(st.session_state["folder"])

        st.download_button(
            "Descargar todos los resultados en ZIP",
            st.session_state["zip"],
            f"{asset['ticker']}_resultados_BSM.zip",
            "application/zip",
            type="primary"
        )


st.divider()

st.caption(
    "Los datos de mercado dependen de la disponibilidad y fecha de las "
    "cotizaciones proporcionadas por Yahoo Finance."
)
