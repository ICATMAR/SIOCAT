#!/usr/bin/env python3
"""
plot_boia.py
============
Llegeix el NetCDF de la boia Somorrostro i genera sèries temporals
de totes les variables en un PDF multi-pàgina.

Ús:
    python3 plot_boia.py --input boia_somorrostro.nc --output plots_boia.pdf
    python3 plot_boia.py --input boia_somorrostro.nc --output plots_boia.pdf --only-valid
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

# ── Estil ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "lines.linewidth":   1.0,
    "lines.markersize":  3,
})

BLUE = "#1F4E79"
GREY = "#7F8C8D"

SENSOR_COLORS = {
    "meteo":   "#2980B9",
    "aux":     "#7FB3D3",
    "ctd":     "#16A085",
    "sami":    "#8E44AD",
    "doppler": "#E67E22",
    "system":  "#7F8C8D",
}

SECTION_TITLES = {
    "meteo":   "Meteorologia",
    "aux":     "Variables auxiliars meteorologia",
    "ctd":     "CTD  SBE37",
    "sami":    "SAMI — pH",
    "doppler": "Doppler ADCP",
    "system":  "System — Bateria i potència",
}

FILL_VALUE = 9.969209968386869e+36

# ── Definició de les variables ─────────────────────────────────────────────────
# (grup_nc, nom_variable): (títol, secció, scatter, tipus)
#   tipus: "1d" → sèrie temporal normal
#          "heatmap" → matriu time×depth (HCSP, HCDT)
GROUPS = [
    # METEO
    ("meteo", "WDIR",            "Direcció del vent (WDIR)",                 "meteo",   True,  "1d"),
    ("meteo", "WSPD",            "Velocitat del vent (WSPD)",                "meteo",   False, "1d"),
    ("meteo", "ATMS",            "Pressió atmosfèrica (ATMS)",               "meteo",   False, "1d"),
    ("meteo", "RELH",            "Humitat relativa (RELH)",                  "meteo",   False, "1d"),
    ("meteo", "DRYT",            "Temperatura de l'aire (DRYT)",             "meteo",   False, "1d"),
    ("meteo", "DEWT",            "Punt de rosada (DEWT)",                    "meteo",   False, "1d"),
    ("meteo", "WETT",            "Temperatura bulb humit (WETT)",            "meteo",   False, "1d"),
    ("meteo", "LATITUDE",        "Latitud GPS (LATITUDE)",                   "meteo",   False, "1d"),
    ("meteo", "LONGITUDE",       "Longitud GPS (LONGITUDE)",                 "meteo",   False, "1d"),
    ("meteo", "HEIGHT",          "Alçada sobre el mar GPS (HEIGHT)",         "meteo",   False, "1d"),
    # AUX METEO
    ("meteo", "AUX_AIR_DENSITY", "Densitat de l'aire (AUX_AIR_DENSITY)",    "aux",     False, "1d"),
    ("meteo", "AUX_REL_WINDDIR", "Dir. vent relativa (AUX_REL_WINDDIR)",    "aux",     True,  "1d"),
    ("meteo", "AUX_CORR_WINDDIR","Dir. vent corregida (AUX_CORR_WINDDIR)",  "aux",     True,  "1d"),
    ("meteo", "AUX_REL_WS",      "Vel. vent relativa (AUX_REL_WS)",         "aux",     False, "1d"),
    # CTD
    ("oceanografia", "TEMP",     "Temperatura CTD (TEMP)",                  "ctd",     False, "1d"),
    ("oceanografia", "PSAL",     "Salinitat CTD (PSAL)",                    "ctd",     False, "1d"),
    ("oceanografia", "CNDC",     "Conductivitat CTD (CNDC)",                "ctd",     False, "1d"),
    ("oceanografia", "PRES",     "Pressió CTD (PRES)",                      "ctd",     False, "1d"),
    ("oceanografia", "DOX1",     "Oxigen dissolt CTD (DOX1)",               "ctd",     True,  "1d"),
    # SAMI
    ("oceanografia", "PHPH",       "pH SAMI (PHPH)",                        "sami",    True,  "1d"),
    ("oceanografia", "SAMI_TEMP",  "Temperatura SAMI (SAMI_TEMP)",          "sami",    False, "1d"),
    ("oceanografia", "SAMI_FLAG",  "Flag qualitat SAMI (SAMI_FLAG)",        "sami",    True,  "1d"),
    ("oceanografia", "SAMI_NPTS",  "Punts regressió SAMI (SAMI_NPTS)",      "sami",    True,  "1d"),
    ("oceanografia", "SAMI_NBLANK","Blanks SAMI (SAMI_NBLANK)",             "sami",    True,  "1d"),
    # DOPPLER escalars
    ("oceanografia", "SENSOR_HEAD",        "Heading Doppler",               "doppler", False, "1d"),
    ("oceanografia", "SENSOR_PITCH",       "Pitch Doppler",                 "doppler", False, "1d"),
    ("oceanografia", "SENSOR_ROLL",        "Roll Doppler",                  "doppler", False, "1d"),
    ("oceanografia", "SVEL",               "Velocitat so (SVEL)",           "doppler", False, "1d"),
    ("oceanografia", "SENSOR_VOLTAGE_ADCP","Tensió ADCP",                   "doppler", False, "1d"),
    ("oceanografia", "PRES_DOPP",          "Pressió Doppler",               "doppler", False, "1d"),
    ("oceanografia", "TEMP_DOPP",          "Temperatura Doppler",           "doppler", False, "1d"),
    # DOPPLER matrius (heatmap) — sempre inclosos independentment de --only-valid
    ("oceanografia", "HCSP", "Velocitat corrent HCSP (time × profunditat)", "doppler", False, "heatmap"),
    ("oceanografia", "HCDT", "Direcció corrent HCDT (time × profunditat)",  "doppler", False, "heatmap"),
    # SYSTEM
    ("oceanografia", "SENSOR_VOLTAGE_DATT","Tensió datalogger",             "system",  False, "1d"),
    ("oceanografia", "SENSOR_VOLTAGE",     "Tensió bateria",                "system",  False, "1d"),
    ("oceanografia", "AUX_SOLAR_VOLT",     "Tensió solar",                  "system",  False, "1d"),
    ("oceanografia", "AUX_SYS_CURRENT",    "Corrent sistema",               "system",  False, "1d"),
    ("oceanografia", "AUX_PANELTEMP",      "Temp. panell datalogger",       "system",  False, "1d"),
    ("oceanografia", "AUX_CTD_PWROFF",     "CTD power off",                 "system",  True,  "1d"),
    ("oceanografia", "AUX_ADP_PWROFF",     "ADP power off",                 "system",  True,  "1d"),
    ("oceanografia", "AUX_SAMI_PWROFF",    "SAMI power off",                "system",  True,  "1d"),
    ("oceanografia", "AUX_METEO_PWROFF",   "Meteo power off",               "system",  True,  "1d"),
]

# ── Lectura de dades ──────────────────────────────────────────────────────────
def read_times(ds, grp_name):
    return pd.to_datetime(ds.groups[grp_name].variables["time"][:],
                          unit="s", utc=True)


def read_1d(ds, grp_name, vname):
    var  = ds.groups[grp_name].variables[vname]
    raw  = var[:]
    # Usar .filled(nan) per respectar la màscara del NetCDF correctament
    data = np.ma.filled(raw.astype(float), np.nan)
    data[np.abs(data) > 1e35] = np.nan
    return data, getattr(var, "units", "")


def read_2d(ds, grp_name, vname):
    var  = ds.groups[grp_name].variables[vname]
    raw  = var[:]
    data = np.ma.filled(raw.astype(float), np.nan)
    data[np.abs(data) > 1e35] = np.nan
    return data, getattr(var, "units", "")


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_1d(ax, times, data, title, units, color, scatter=False):
    mask    = ~np.isnan(data)
    n_valid = int(mask.sum())
    n_total = len(data)

    if n_valid == 0:
        ax.text(0.5, 0.5, "Sense dades vàlides",
                ha="center", va="center", transform=ax.transAxes,
                color=GREY, style="italic", fontsize=9)
        ax.set_title(title, color=GREY, fontsize=9)
        ax.set_yticks([])
        return

    t_valid = times[mask]
    d_valid = data[mask]

    if scatter or n_valid < 50:
        ax.scatter(t_valid, d_valid, s=8, color=color, alpha=0.7,
                   linewidths=0, zorder=3)
    else:
        ax.plot(times, data, color=color, alpha=0.85, zorder=3)
        ax.fill_between(times, data, alpha=0.08, color=color)

    stats = (f"n={n_valid}/{n_total}  |  "
             f"min={np.nanmin(data):.3g}  "
             f"max={np.nanmax(data):.3g}  "
             f"mean={np.nanmean(data):.3g}")
    ax.set_title(f"{title}\n{stats}", fontsize=9, pad=4)
    if units:
        ax.set_ylabel(units, fontsize=7, color="#555555")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)


def plot_heatmap(pdf, times, depth, data, title, units, color):
    """Pàgina completa amb heatmap time×depth."""
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Doppler ADCP  —  {title}",
                 fontsize=13, fontweight="bold", color=color)

    n_valid = int(np.sum(~np.isnan(data)))

    if n_valid == 0:
        ax.set_facecolor("#F8F9FA")
        ax.text(0.5, 0.5,
                "Sense dades vàlides\n(ADCP no operatiu en el període actual)",
                ha="center", va="center", transform=ax.transAxes,
                color=GREY, style="italic", fontsize=13)
        ax.set_xlabel("Data")
        ax.set_ylabel("Profunditat (m)")
        # Posem eixos de temps i profunditat tot i que no hi hagi dades
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        t_num = mdates.date2num(times.to_pydatetime())
        ax.set_xlim(t_num[0], t_num[-1])
        ax.set_ylim(depth[-1], depth[0])
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    else:
        cmap = "RdBu_r" if "HCDT" in title else "viridis"
        t_num = mdates.date2num(times.to_pydatetime())
        im = ax.pcolormesh(t_num, depth, data.T, cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax, label=units)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_ylabel("Profunditat (m)")
        ax.invert_yaxis()

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Pàgines especials ─────────────────────────────────────────────────────────
def add_cover(pdf, nc_path, ds):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(BLUE)
    fig.text(0.5, 0.78, "Boia Somorrostro — ICATMAR",
             ha="center", color="white", fontsize=28, fontweight="bold")
    fig.text(0.5, 0.68, "Sèries temporals de totes les variables del NetCDF",
             ha="center", color="#BDD7EE", fontsize=16)
    times = read_times(ds, "meteo")
    lines = [
        f"Fitxer:      {os.path.basename(nc_path)}",
        f"Període:     {times[0].strftime('%d %b %Y')} → {times[-1].strftime('%d %b %Y')}",
        f"Timesteps:   {len(times)}  (15 min)",
        f"Grups:       /meteo  +  /oceanografia",
        f"Variables:   {len(GROUPS)} sèries / heatmaps",
    ]
    for i, line in enumerate(lines):
        fig.text(0.5, 0.50 - i * 0.065, line,
                 ha="center", color="white", fontsize=12,
                 fontfamily="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_section_divider(pdf, title, color):
    fig = plt.figure(figsize=(16, 2))
    fig.patch.set_facecolor(color)
    fig.text(0.5, 0.5, title, ha="center", va="center",
             color="white", fontsize=22, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Sèries temporals de totes les variables del NetCDF de la boia Somorrostro."
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Fitxer NetCDF d'entrada")
    parser.add_argument("--output", "-o", required=True,
                        help="Fitxer PDF de sortida")
    parser.add_argument("--only-valid", action="store_true",
                        help="Ometre variables 1D sense cap dada vàlida "
                             "(els heatmaps del Doppler s'inclouen sempre)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: fitxer no trobat: {args.input}")
        sys.exit(1)

    ds = nc.Dataset(args.input)
    for grp in ("meteo", "oceanografia"):
        if grp not in ds.groups:
            print(f"Error: grup /{grp} no trobat al NetCDF.")
            sys.exit(1)

    print(f"Llegint:       {args.input}")
    print(f"Generant PDF:  {args.output}")

    NCOLS    = 3
    NROWS    = 3
    per_page = NCOLS * NROWS

    # Agrupar variables per secció mantenint l'ordre
    sections = defaultdict(list)
    for entry in GROUPS:
        grp_nc, vname, title, section, scatter, tipus = entry
        sections[section].append((grp_nc, vname, title, scatter, tipus))

    section_order = ["meteo", "aux", "ctd", "sami", "doppler", "system"]

    with PdfPages(args.output) as pdf:
        add_cover(pdf, args.input, ds)

        for section_key in section_order:
            if section_key not in sections:
                continue

            all_items = sections[section_key]
            color     = SENSOR_COLORS[section_key]
            sec_title = SECTION_TITLES[section_key]

            # Separar 1d i heatmaps
            items_1d  = [(g, v, t, s) for g, v, t, s, tp in all_items if tp == "1d"]
            items_hm  = [(g, v, t, s) for g, v, t, s, tp in all_items if tp == "heatmap"]

            # Filtrar 1D si --only-valid (els heatmaps sempre s'inclouen)
            if args.only_valid:
                filtered = []
                for grp_nc, vname, title, scatter in items_1d:
                    if vname not in ds.groups[grp_nc].variables:
                        continue
                    data, _ = read_1d(ds, grp_nc, vname)
                    if np.any(~np.isnan(data)):
                        filtered.append((grp_nc, vname, title, scatter))
                items_1d = filtered

            # Si no hi ha res a mostrar en aquesta secció, saltar
            if not items_1d and not items_hm:
                continue

            add_section_divider(pdf, sec_title, color)

            # ── Pàgines de sèries 1D ──────────────────────────────────────
            n_items = len(items_1d)
            for page_start in range(0, max(n_items, 1), per_page):
                page_items = items_1d[page_start:page_start + per_page]
                if not page_items:
                    break
                n_plots = len(page_items)
                nrows   = (n_plots + NCOLS - 1) // NCOLS

                fig, axes = plt.subplots(nrows, NCOLS,
                                         figsize=(16, nrows * 3.2 + 0.5),
                                         squeeze=False)
                page_num = page_start // per_page + 1
                fig.suptitle(f"{sec_title}  —  pàg. {page_num}",
                             fontsize=11, fontweight="bold",
                             color=color, y=1.01)

                for idx, (grp_nc, vname, title, scatter) in enumerate(page_items):
                    ax = axes[idx // NCOLS][idx % NCOLS]
                    if vname not in ds.groups[grp_nc].variables:
                        ax.set_visible(False)
                        continue
                    times      = read_times(ds, grp_nc)
                    data, units = read_1d(ds, grp_nc, vname)
                    plot_1d(ax, times, data, title, units, color, scatter)

                for idx in range(n_plots, nrows * NCOLS):
                    axes[idx // NCOLS][idx % NCOLS].set_visible(False)

                fig.tight_layout(h_pad=3.0, w_pad=2.0)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print(f"  [{section_key}] pàgina {page_num} OK")

            # ── Heatmaps (sempre, independentment de --only-valid) ────────
            for grp_nc, vname, title, scatter in items_hm:
                if vname not in ds.groups[grp_nc].variables:
                    print(f"  [{section_key}] {vname} no trobat al NetCDF, saltat.")
                    continue
                times        = read_times(ds, grp_nc)
                depth        = ds.groups[grp_nc].variables["depth"][:]
                data, units  = read_2d(ds, grp_nc, vname)
                plot_heatmap(pdf, times, depth, data, title, units, color)
                n_valid = int(np.sum(~np.isnan(data)))
                print(f"  [{section_key}] {vname} heatmap OK  (n_valid={n_valid})")

        # Metadades del PDF
        d = pdf.infodict()
        d["Title"]   = "Boia Somorrostro — Sèries temporals"
        d["Author"]  = "ICATMAR"
        d["Subject"] = "NetCDF time series plots"

    ds.close()
    print(f"\nPDF generat: {args.output}")


if __name__ == "__main__":
    main()