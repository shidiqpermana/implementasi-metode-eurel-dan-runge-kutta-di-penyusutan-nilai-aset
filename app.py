import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


DATA_FILE = "combined_financial_data_idx.csv"


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    """Load the financial data CSV."""
    df = pd.read_csv(path)
    return df


def filter_symbol_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Filter data for a given symbol and keep only rows relevant to PPE & Accumulated Depreciation.

    The instructions mention 'Property, Plant And Equipment', but in this dataset
    PPE is better represented by 'Machinery Furniture Equipment'. We keep both
    names if present to be robust.
    """
    accounts_of_interest = [
        "Accumulated Depreciation",
        "Property, Plant And Equipment",
        "Machinery Furniture Equipment",
    ]
    mask = (df["symbol"] == symbol) & (df["account"].isin(accounts_of_interest))
    return df.loc[mask].copy()


def get_nbv_series(symbol_df: pd.DataFrame):
    """
    Compute Net Book Value (NBV) time series for a symbol.

    NBV = PPE + Accumulated Depreciation
    where Accumulated Depreciation is expected to be negative in the dataset.
    """
    year_cols = [col for col in symbol_df.columns if col.isdigit()]
    if not year_cols:
        raise ValueError("Tidak ditemukan kolom tahun (misalnya 2020, 2021, ...) pada data.")

    # Sort year columns numerically (as strings)
    year_cols = sorted(year_cols, key=int)

    # Try to find PPE-like row
    ppe_row = None
    for acc_name in ["Property, Plant And Equipment", "Machinery Furniture Equipment"]:
        rows = symbol_df[symbol_df["account"] == acc_name]
        if not rows.empty:
            ppe_row = rows.iloc[0]
            break

    if ppe_row is None:
        raise ValueError(
            "Tidak ditemukan akun PPE untuk simbol ini "
            "('Property, Plant And Equipment' atau 'Machinery Furniture Equipment')."
        )

    dep_rows = symbol_df[symbol_df["account"] == "Accumulated Depreciation"]
    if dep_rows.empty:
        raise ValueError("Tidak ditemukan akun 'Accumulated Depreciation' untuk simbol ini.")

    dep_row = dep_rows.iloc[0]

    ppe_values = ppe_row[year_cols].astype(float).values
    dep_values = dep_row[year_cols].astype(float).values

    nbv_values = ppe_values + dep_values
    years = np.array([int(y) for y in year_cols])
    return years, nbv_values


def estimate_k_from_data(years: np.ndarray, nbv_values: np.ndarray) -> float:
    """
    Estimate depreciation rate k from historical NBV data using:

    k = - (1 / NBV_avg) * (ΔNBV / Δt)
    """
    if len(nbv_values) < 2:
        raise ValueError("Data NBV historis tidak cukup untuk mengestimasi k.")

    t_start = years[0]
    t_end = years[-1]
    delta_t = t_end - t_start
    if delta_t <= 0:
        raise ValueError("Rentang waktu tidak valid untuk estimasi k.")

    nbv_start = nbv_values[0]
    nbv_end = nbv_values[-1]
    nbv_avg = np.mean(nbv_values)
    if nbv_avg == 0:
        raise ValueError("NBV rata-rata bernilai nol, tidak dapat mengestimasi k.")

    delta_nbv = nbv_end - nbv_start
    k_est = -(1.0 / nbv_avg) * (delta_nbv / delta_t)
    return float(k_est)


def euler_method(f, V0: float, t: np.ndarray) -> np.ndarray:
    """
    Euler method for solving dV/dt = f(t, V).

    Parameters
    ----------
    f : callable
        Function f(t, V) representing the ODE.
    V0 : float
        Initial value V(t0).
    t : np.ndarray
        Array of time points where the solution is approximated.
    """
    V = np.zeros_like(t, dtype=float)
    V[0] = V0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        V[i] = V[i - 1] + h * f(t[i - 1], V[i - 1])
    return V


def runge_kutta_4(f, V0: float, t: np.ndarray) -> np.ndarray:
    """
    Runge-Kutta order 4 (RK4) for solving dV/dt = f(t, V).

    Parameters
    ----------
    f : callable
        Function f(t, V) representing the ODE.
    V0 : float
        Initial value V(t0).
    t : np.ndarray
        Array of time points where the solution is approximated.
    """
    V = np.zeros_like(t, dtype=float)
    V[0] = V0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        ti = t[i - 1]
        Vi = V[i - 1]

        k1 = f(ti, Vi)
        k2 = f(ti + 0.5 * h, Vi + 0.5 * h * k1)
        k3 = f(ti + 0.5 * h, Vi + 0.5 * h * k2)
        k4 = f(ti + h, Vi + h * k3)

        V[i] = Vi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return V


def analytical_solution(V0: float, k: float, t: np.ndarray) -> np.ndarray:
    """Analytical solution for dV/dt = -k V -> V(t) = V0 * exp(-k t)."""
    return V0 * np.exp(-k * t)


def main():
    st.title("Model Penyusutan Aset dengan Metode Euler dan Runge-Kutta Orde 4 (RK4)")
    st.markdown(
        """
        Aplikasi ini memodelkan **penyusutan aset tetap** menggunakan persamaan diferensial biasa (PDB)
        dengan model saldo menurun berikut:
        """
    )
    st.latex(r"\frac{dV}{dt} = -k \cdot V")
    st.markdown(
        """
        di mana:

        - \(V\) adalah **Net Book Value (NBV)** aset tetap  
        - \(t\) adalah waktu (tahun)  
        - \(k\) adalah tingkat penyusutan tahunan
        """
    )

    # Sidebar inputs
    st.sidebar.header("Pengaturan Model")
    symbol = st.sidebar.text_input("Simbol Emiten", value="AALI")
    method_option = st.sidebar.selectbox(
        "Pilih Metode Numerik",
        ("Euler", "Runge-Kutta Orde 4 (RK4)"),
    )
    k_input = st.sidebar.number_input(
        "Tingkat Penyusutan k (per tahun)",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
        format="%.2f",
    )
    h = 0.5  # step size (years)

    # Load and process data
    try:
        df = load_data(DATA_FILE)
    except Exception as e:
        st.error(f"Gagal membaca file data: {e}")
        return

    if "symbol" not in df.columns or "account" not in df.columns:
        st.error("File CSV tidak memiliki kolom 'symbol' dan/atau 'account' yang diperlukan.")
        return

    symbol_df = filter_symbol_data(df, symbol)
    if symbol_df.empty:
        st.error(f"Tidak ada data yang sesuai untuk simbol '{symbol}'.")
        return

    try:
        years, nbv_values = get_nbv_series(symbol_df)
    except Exception as e:
        st.error(str(e))
        return

    # Define time axis (t = 0 at first year)
    base_year = years[0]
    t_hist = years - base_year
    t_end = t_hist[-1]
    t_sim = np.arange(0.0, t_end + h, h)

    V0 = float(nbv_values[0])

    # Estimate k from data (for information)
    k_est_text = ""
    try:
        k_est = estimate_k_from_data(years, nbv_values)
        k_est_text = f"Estimasi k dari data historis ≈ {k_est:.4f} per tahun"
    except Exception as e:
        k_est_text = f"Gagal mengestimasi k dari data historis: {e}. Menggunakan nilai input."

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Informasi Estimasi k (dari data):**")
    st.sidebar.markdown(k_est_text)

    # Use user-provided k for simulation
    k = float(k_input)

    # Define ODE function for given k
    def ode_func(t, V):
        return -k * V

    # Numerical methods
    V_euler = euler_method(ode_func, V0, t_sim)
    V_rk4 = runge_kutta_4(ode_func, V0, t_sim)
    V_analytic = analytical_solution(V0, k, t_sim)

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 5))

    # Historical points
    ax.plot(
        t_hist,
        nbv_values,
        "o-",
        label="NBV Historis",
        color="black",
        linewidth=2,
        markersize=6,
    )

    # Numerical + analytical curves (hanya metode yang dipilih + solusi analitik)
    if method_option == "Euler":
        ax.plot(t_sim, V_euler, "--", label="Euler", color="tab:blue")
    else:
        ax.plot(t_sim, V_rk4, "-.", label="RK4", color="tab:orange")

    ax.plot(t_sim, V_analytic, ":", label="Solusi Analitik", color="tab:green")

    ax.set_xlabel(f"Waktu sejak {base_year} (tahun)")
    ax.set_ylabel("Net Book Value (NBV)")
    ax.set_title(f"Simulasi Penyusutan Aset Tetap - {symbol}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)

    # Comparison table at final time
    historical_final = float(nbv_values[-1])
    analytic_final = float(V_analytic[-1])

    if method_option == "Euler":
        method_name = "Euler"
        method_final = float(V_euler[-1])
    else:
        method_name = "RK4"
        method_final = float(V_rk4[-1])

    comparison_df = pd.DataFrame(
        {
            "Metode": ["Historis (NBV Data)", method_name, "Solusi Analitik"],
            "V Akhir Periode": [
                historical_final,
                method_final,
                analytic_final,
            ],
        }
    )

    st.markdown("### Perbandingan Nilai Akhir Periode")
    st.table(comparison_df.style.format({"V Akhir Periode": "{:,.2f}"}))

    st.markdown(
        """
        **Catatan:**

        - NBV historis diperoleh dari data `Accumulated Depreciation` dan `Machinery Furniture Equipment`
          (atau `Property, Plant And Equipment` jika tersedia).
        - Metode Euler dan RK4 menggunakan ukuran langkah \\(h = 0{,}5\\) tahun.
        - Solusi analitik digunakan sebagai acuan verifikasi dari hasil metode numerik.
        """
    )


if __name__ == "__main__":
    main()


