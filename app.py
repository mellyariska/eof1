# streamlit_eof_app.py
# ------------------------------------------------------------
# Interaktif EOF (Empirical Orthogonal Function) & Klaster Iklim
# - Upload NetCDF (rr: time x latitude x longitude)
# - Pilih subset waktu/area
# - Pilih opsi anomali (rata-rata atau klimatologi bulanan)
# - Hitung EOF (SVD) dengan pembobotan luas (cos(lat))
# - Visualisasi peta mode EOF & PC
# - Klasterisasi wilayah berdasarkan PC untuk memetakan zona iklim
# ------------------------------------------------------------

import io
import numpy as np
import xarray as xr
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
import scipy.linalg as sla
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EOF & Climate Regions", layout="wide")
st.title("🌧️ Empirical Orthogonal Function (EOF) • Peta Iklim Interaktif")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def _open_dataset(file_bytes: bytes | None, variable_hint: str = "rr"):
    """Open NetCDF from upload or fallback local path. Return xarray.Dataset and target variable name."""
    if file_bytes is not None:
        ds = xr.open_dataset(io.BytesIO(file_bytes))
    else:
        # Fallback ke path lokal (untuk demo/dev) — sesuaikan bila perlu
        ds = xr.open_dataset("data_kompres.nc")
    # Pilih variabel target
    var = variable_hint if variable_hint in ds.data_vars else list(ds.data_vars)[0]
    return ds, var


def _subset(ds: xr.Dataset, var: str, lat_range, lon_range, time_range):
    da = ds[var]
    # Pastikan koordinat bernama 'latitude'/'longitude' — fallback ke nama umum
    lat_name = None
    for cand in ["latitude", "lat", "Latitude", "y"]:
        if cand in da.coords:
            lat_name = cand; break
    lon_name = None
    for cand in ["longitude", "lon", "Longitude", "x"]:
        if cand in da.coords:
            lon_name = cand; break
    time_name = None
    for cand in ["time", "Time", "date"]:
        if cand in da.coords:
            time_name = cand; break

    if lat_name is None or lon_name is None or time_name is None:
        raise ValueError("Koordinat latitude/longitude/time tidak ditemukan.")

    da = da.sortby(lat_name).sortby(lon_name)

    # Subset spatial
    if lat_range is not None:
        lat_min, lat_max = lat_range
        da = da.sel({lat_name: slice(lat_min, lat_max)})
    if lon_range is not None:
        lon_min, lon_max = lon_range
        da = da.sel({lon_name: slice(lon_min, lon_max)})

    # Subset temporal
    if time_range is not None:
        t0, t1 = time_range
        da = da.sel({time_name: slice(np.datetime64(t0), np.datetime64(t1))})

    return da, lat_name, lon_name, time_name


def _make_anomaly(da: xr.DataArray, time_name: str, method: str):
    """Anomali: 'mean' (kurangi rata-rata keseluruhan) atau 'clim' (kurangi klimatologi bulanan)."""
    if method == "mean":
        return da - da.mean(time_name)
    elif method == "clim":
        if not np.issubdtype(da[time_name].dtype, np.datetime64):
            return da - da.mean(time_name)
        clim = da.groupby(f"{time_name}.month").mean()
        anom = (da.groupby(f"{time_name}.month") - clim)
        return anom
    else:
        return da


def _area_weight(lat_vals: np.ndarray):
    """Bobot luas berbasis cos(lat). Return sqrt(cos) untuk diterapkan ke data sebelum SVD."""
    w = np.cos(np.deg2rad(lat_vals))
    w[w < 0] = 0
    return np.sqrt(w)


def _compute_eof(da: xr.DataArray, lat_name: str, lon_name: str, time_name: str, n_modes: int):
    da = da.transpose(time_name, lat_name, lon_name)
    T, Ny, Nx = da.shape

    max_space = 30000
    if Ny * Nx > max_space:
        cf_lat = 2 if Ny >= 4 else 1
        cf_lon = 2 if Nx >= 4 else 1
        if cf_lat > 1 or cf_lon > 1:
            st.warning(f"Grid besar (Ny*Nx={Ny*Nx}). Melakukan coarsen sementara.")
            da = da.coarsen({lat_name: cf_lat, lon_name: cf_lon}, boundary="trim").mean()
            T, Ny, Nx = da.shape

    X = da.values.reshape(T, Ny * Nx)
    X = X - X.mean(axis=0, keepdims=True)

    lat_vals = da[lat_name].values
    sw = _area_weight(lat_vals)
    W = np.repeat(sw[:, None], Nx, axis=1).reshape(1, Ny * Nx)
    Xw = X * W

    Xw = np.nan_to_num(Xw, nan=0.0, posinf=0.0, neginf=0.0)
    col_std = Xw.std(axis=0)
    col_std[col_std < 1e-12] = 1.0
    Xw = (Xw - Xw.mean(axis=0, keepdims=True)) / col_std

    U = s = Vt = None
    try:
        U, s, Vt = sla.svd(Xw, full_matrices=False, lapack_driver="gesvd")
    except:
        try:
            U, s, Vt = sla.svd(Xw, full_matrices=False, lapack_driver="gesdd")
        except:
            k = min(max(n_modes, 20), min(Xw.shape) - 1)
            U, s, Vt = randomized_svd(Xw, n_components=k, n_iter=5, random_state=42)

    lam = s**2
    varfrac = lam / lam.sum()

    m = int(min(n_modes, Vt.shape[0]))
    Vt_m = Vt[:m, :]
    U_m = U[:, :m]
    s_m = s[:m]
    pcs = U_m * s_m

    patterns = []
    for k in range(m):
        vk = Vt_m[k, :].reshape(Ny, Nx)
        sw_rep = np.repeat(sw[:, None], Nx, axis=1)
        sw_rep[sw_rep == 0] = 1.0
        vk_unw = vk / sw_rep
        patterns.append(vk_unw)
    patterns = np.stack(patterns, axis=0)

    coords = {
        "mode": np.arange(1, m + 1),
        lat_name: da[lat_name].values,
        lon_name: da[lon_name].values,
        time_name: da[time_name].values,
    }

    return {
        "patterns": xr.DataArray(patterns, dims=("mode", lat_name, lon_name),
                                 coords={"mode": coords["mode"], lat_name: coords[lat_name], lon_name: coords[lon_name]}),
        "pcs": xr.DataArray(pcs, dims=(time_name, "mode"),
                            coords={time_name: coords[time_name], "mode": coords["mode"]}),
        "varfrac": xr.DataArray(varfrac[:m], dims=("mode",), coords={"mode": coords["mode"]}),
    }


def _plot_eof_map(pattern: xr.DataArray, title: str):
    df = pattern.to_pandas()
    fig = px.imshow(df, origin="lower", aspect="auto", labels=dict(color="Amplitude"), title=title)
    fig.update_xaxes(title_text=pattern.dims[1])
    fig.update_yaxes(title_text=pattern.dims[0])
    return fig


def _plot_pc_series(pcs: xr.DataArray, mode: int, title: str):
    ser = pcs.sel(mode=mode).to_pandas()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ser.index, y=ser.values, mode="lines", name=f"PC{mode}"))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="PC")
    return fig


def _cluster_regions(pcs: xr.DataArray, modes_for_cluster: int, n_clusters: int, lat_name: str, lon_name: str, template_da: xr.DataArray):
    patterns = template_da
    pat = patterns.sel(mode=slice(1, modes_for_cluster)).values
    m, Ny, Nx = pat.shape
    Xfeat = pat.reshape(m, Ny * Nx).T
    Xfeat = (Xfeat - Xfeat.mean(axis=0)) / (Xfeat.std(axis=0) + 1e-9)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(Xfeat)
    label_map = labels.reshape(Ny, Nx)

    lab_da = xr.DataArray(label_map.astype(int),
                          dims=(patterns.dims[1], patterns.dims[2]),
                          coords={patterns.dims[1]: patterns.coords[patterns.dims[1]],
                                  patterns.dims[2]: patterns.coords[patterns.dims[2]]},
                          name="cluster")
    return lab_da, km

# -----------------------------
# Sidebar • Inputs
# -----------------------------
st.sidebar.header("⚙️ Pengaturan")
file = st.sidebar.file_uploader("Upload NetCDF (.nc)", type=["nc", "nc4", "cdf"])
var_hint = st.sidebar.text_input("Nama variabel (opsional)", value="rr")

with st.sidebar.expander("Subset Spasial", expanded=False):
    lat_min = st.number_input("Latitude min", value=-15.0, step=0.25)
    lat_max = st.number_input("Latitude max", value=10.0, step=0.25)
    lon_min = st.number_input("Longitude min", value=80.0, step=0.25)
    lon_max = st.number_input("Longitude max", value=160.0, step=0.25)

with st.sidebar.expander("Subset Temporal", expanded=False):
    t0 = st.date_input("Tanggal awal", value=pd.Timestamp("1981-01-01").date())
    t1 = st.date_input("Tanggal akhir", value=pd.Timestamp("2017-12-31").date())

anom_method = st.sidebar.selectbox("Metode anomali", ["none", "mean", "clim"], index=2)
n_modes = st.sidebar.slider("Jumlah mode EOF", 1, 10, 4)

with st.sidebar.expander("Klaster Iklim", expanded=False):
    use_cluster = st.checkbox("Aktifkan klasterisasi", value=True)
    modes_for_cluster = st.slider("Mode untuk fitur klaster", 1, 10, 3)
    n_clusters = st.slider("Jumlah klaster", 2, 12, 6)

# -----------------------------
# Main • Compute
# -----------------------------
try:
    ds, var = _open_dataset(file.read() if file is not None else None, var_hint)
    da, lat_name, lon_name, time_name = _subset(
        ds, var, (lat_min, lat_max), (lon_min, lon_max), (t0, t1)
    )
    st.caption(f"Dataset: **{var}** | Dimensi: {list(da.dims)}")

    da_anom = _make_anomaly(da, time_name=time_name, method=anom_method)
    res = _compute_eof(da_anom, lat_name, lon_name, time_name, n_modes)
    patterns, pcs, varfrac = res["patterns"], res["pcs"], res["varfrac"]

    col_a, col_b = st.columns([1, 1])
    with col_a:
        df_var = pd.DataFrame({"Mode": patterns["mode"].values,
                               "Explained Variance (%)": (varfrac.values * 100.0)})
        fig_var = px.bar(df_var, x="Mode", y="Explained Variance (%)")
        st.subheader("📈 Variansi yang Dijelaskan")
        st.plotly_chart(fig_var, use_container_width=True)

    with col_b:
        st.subheader("🧭 Info Dataset")
        st.write(pd.DataFrame({
            "Latitude": [float(da[lat_name].values.min()), float(da[lat_name].values.max())],
            "Longitude": [float(da[lon_name].values.min()), float(da[lon_name].values.max())],
            "Time Start": [pd.to_datetime(da[time_name].values[0])],
            "Time End": [pd.to_datetime(da[time_name].values[-1])],
        }, index=["Min/Start", "Max/End"]))

    st.markdown("---")
    st.subheader("🗺️ Peta Mode EOF & Deret Waktu PC")
    tabs = st.tabs([f"Mode {int(m)}" for m in patterns["mode"].values])
    for i, m in enumerate(patterns["mode"].values):
        with tabs[i]:
            col1, col2 = st.columns([1.2, 1])
            fig_map = _plot_eof_map(patterns.sel(mode=int(m)), f"EOF Mode {int(m)}")
            col1.plotly_chart(fig_map, use_container_width=True)
            fig_pc = _plot_pc_series(pcs, int(m), f"PC{int(m)} • Var {varfrac.sel(mode=int(m)).item()*100:.2f}%")
            col2.plotly_chart(fig_pc, use_container_width=True)

    if use_cluster:
        st.markdown("---")
        st.subheader("🗂️ Pemetaan Zona Iklim")
        modes_for_cluster = int(min(modes_for_cluster, int(patterns.sizes["mode"])))
        lab_da, km = _cluster_regions(pcs, modes_for_cluster, n_clusters, lat_name, lon_name, patterns)

        df_lab = lab_da.to_pandas()
        fig_clu = px.imshow(df_lab, origin="lower", aspect="auto",
                            labels=dict(color="Cluster"),
                            title=f"Peta Klaster (K={n_clusters})")
        st.plotly_chart(fig_clu, use_container_width=True)

        st.markdown("**Centroid fitur EOF**")
        cent = pd.DataFrame(km.cluster_centers_, columns=[f"EOF{k}" for k in range(1, modes_for_cluster+1)])
        st.dataframe(cent.style.format("{:.2f}"))

        colx, coly = st.columns(2)
        with colx:
            ds_out = lab_da.to_dataset(name="cluster")
            fnc = ds_out.to_netcdf()
            if isinstance(fnc, (memoryview, bytearray)):
                fnc = bytes(fnc)
            elif isinstance(fnc, str):
                with open(fnc, "rb") as f:
                    fnc = f.read()
            st.download_button(
                "💾 Download cluster.nc",
                data=fnc,
                file_name="cluster_map.nc",
                mime="application/x-netcdf"
            )
        with coly:
            df_out = df_lab.reset_index().melt(
                id_vars=df_lab.index.names,
                var_name=lab_da.dims[1],
                value_name="cluster"
            )
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Download cluster.csv",
                data=csv,
                file_name="cluster_map.csv",
                mime="text/csv"
            )

    st.markdown("---")
    with st.expander("📚 Catatan Metodologi"):
        st.markdown(
            """
            **Ringkas Metode**
            - Data dibobotkan dengan √cos(lat).
            - EOF dihitung via SVD matriks [waktu x ruang].
            - Klasterisasi grid berdasarkan loading EOF (K-Means).
            """
        )

except Exception as e:
    st.error(f"Terjadi kesalahan: {type(e).__name__}: {e}")
    st.exception(e)
