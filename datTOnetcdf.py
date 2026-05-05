#!/usr/bin/env python3
"""
boia_to_netcdf.py
=================
Converteix els fitxers .dat (format TOA5 - Campbell Scientific) de la boia
oceanogràfica Somorrostro a un fitxer NetCDF seguint les convencions CF/ICATMAR.

Estructura del NetCDF de sortida (NETCDF4 amb grups):
  /meteo        → dades meteorològiques (15 min)
  /oceanografia → dades oceanogràfiques + Doppler + SAMI + System (15 min)
                  Inclou variables del Status interpolades al nearest 15 min,
                  i STATUS_TIME amb el timestamp original real del Status.

S'executa via cron per anar augmentant el dataset incrementalment:
    python3 boia_to_netcdf.py --input /ruta/ftpdir --output /ruta/sortida/boia.nc

Autor: ICATMAR
"""

import argparse
import os
import sys
import glob
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import netCDF4 as nc

# ---------------------------------------------------------------------------
# Configuració de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants de la boia
# ---------------------------------------------------------------------------

# Data d'inici del dataset (format "YYYY-MM-DD").
# Tots els fitxers .dat es filtraran a partir d'aquesta data.
# Posar None per incloure totes les dades disponibles.
START_DATE = "2026-02-05"

STATION_NAME  = "BoiaSomorrostro"
INSTITUTION   = "ICATMAR"
NOMINAL_LAT   = 41.3850   # graus N (referència)
NOMINAL_LON   = 2.1960    # graus E (referència)
FILL_VALUE    = 9.969209968386869e+36   # valor estàndard NetCDF per dades no vàlides

# Profunditat nominal del CTD i SAMI (metres)
DEPTH_CTD     = 1.0   # m (ajustar si cal)
DEPTH_SAMI    = 1.0   # m

# Primers nivells de cel·la del Doppler i gruix de cel·la (metres)
DOPPLER_FIRST_CELL_DEPTH = 2.0   # m (ajustar)
DOPPLER_CELL_SIZE        = 1.0   # m (ajustar)
DOPPLER_N_CELLS          = 40

# ---------------------------------------------------------------------------
# Noms esperats dels fitxers (glob patterns)
# ---------------------------------------------------------------------------
FILE_PATTERNS = {
    "meteo":   "*_Meteo.dat",
    "sbe37":   "*_SBE37_SMPO.dat",
    "sami":    "*_Sami.dat",
    "doppler": "*_Doppler.dat",
    "system":  "*_System.dat",
    "status":  "*_Status.dat",
}

# ---------------------------------------------------------------------------
# Lectura de fitxers TOA5
# ---------------------------------------------------------------------------
def read_toa5(filepath: str) -> pd.DataFrame:
    """
    Llegeix un fitxer TOA5 de Campbell Scientific.
    Estructura: 4 línies de capçalera
      Línia 0: metadades de l'estació
      Línia 1: noms de columnes
      Línia 2: unitats
      Línia 3: tipus de procés
    Retorna un DataFrame amb TIMESTAMP com a DatetimeIndex.
    """
    df = pd.read_csv(
        filepath,
        skiprows=1,
        header=0,
        encoding="utf-8-sig",
        low_memory=False,
    )
    df = df.iloc[2:].reset_index(drop=True)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df = df.dropna(subset=["TIMESTAMP"])
    df = df.set_index("TIMESTAMP").sort_index()
    if "RECORD" in df.columns:
        df = df.drop(columns=["RECORD"])
    df = df.apply(lambda col: pd.to_numeric(col, errors="coerce")
                  if col.dtype == object else col)
    # Filtre per data d'inici
    if START_DATE is not None:
        start = pd.Timestamp(START_DATE)
        df = df[df.index >= start]

    if len(df) == 0:
        log.warning(f"  {os.path.basename(filepath)} — cap dada després de START_DATE={START_DATE}")
        return df

    log.info(f"  Llegit: {os.path.basename(filepath)} — {len(df)} files, "
             f"{df.index[0]} → {df.index[-1]}")
    return df


def find_file(input_dir: str, pattern: str) -> str | None:
    matches = glob.glob(os.path.join(input_dir, pattern))
    if not matches:
        return None
    if len(matches) > 1:
        log.warning(f"Diversos fitxers trobats per '{pattern}': S'utilitza: {matches[0]}")
    return matches[0]


# ---------------------------------------------------------------------------
# Parsejat de les cel·les del Doppler
# ---------------------------------------------------------------------------
def parse_doppler_cells(df_dopp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extreu les velocitats (mm/s → m/s) i les direccions (°) de cada cel·la.
    Format de cada cel·la: "noCell velocitat_mm/s direccio_deg"
    Retorna speed_df i dir_df, ambdós indexats per TIMESTAMP.
    """
    cell_cols = [f"DoppCell({i})" for i in range(1, DOPPLER_N_CELLS + 1)]
    speeds, dirs = {}, {}

    for col in cell_cols:
        cell_num = col.replace("DoppCell(", "").replace(")", "")
        spd_list, dir_list = [], []
        for val in df_dopp[col]:
            val_str = str(val).strip()
            if val_str in ("", "nan", "NAN", "None"):
                spd_list.append(np.nan)
                dir_list.append(np.nan)
            else:
                parts = val_str.split()
                try:
                    spd_list.append(float(parts[1]) / 1000.0)  # mm/s → m/s
                    dir_list.append(float(parts[2]))
                except (IndexError, ValueError):
                    spd_list.append(np.nan)
                    dir_list.append(np.nan)
        speeds[f"cell_{cell_num}"] = spd_list
        dirs[f"cell_{cell_num}"]   = dir_list

    speed_df = pd.DataFrame(speeds, index=df_dopp.index)
    dir_df   = pd.DataFrame(dirs,   index=df_dopp.index)
    return speed_df, dir_df


# ---------------------------------------------------------------------------
# Helpers NetCDF
# ---------------------------------------------------------------------------
def make_time_var(grp, time_vals: np.ndarray):
    """Crea o actualitza la variable time en un grup."""
    if "time" not in grp.variables:
        tvar = grp.createVariable("time", "f8", ("time",), zlib=True)
    else:
        tvar = grp.variables["time"]
    tvar[:] = time_vals
    tvar.long_name     = "time"
    tvar.standard_name = "time"
    tvar.units         = "seconds since 1970-01-01 00:00:00 UTC"
    tvar.calendar      = "gregorian"
    tvar.axis          = "T"
    return tvar


def write_float_var(grp, nc_name, arr, dims, long_name, units,
                    standard_name="", icatmar_name="", extra_attrs=None):
    """Crea o actualitza una variable float32 en un grup."""
    arr_f = arr.astype(np.float32) if arr is not None else None
    if arr_f is None:
        return
    if nc_name not in grp.variables:
        v = grp.createVariable(nc_name, "f4", dims,
                               fill_value=FILL_VALUE, zlib=True)
    else:
        v = grp.variables[nc_name]
    v[:] = np.where(np.isnan(arr_f), FILL_VALUE, arr_f)
    v.long_name = long_name
    v.units     = units
    if standard_name:
        v.standard_name = standard_name
    if icatmar_name:
        v.icatmar_name = icatmar_name
    if extra_attrs:
        for k, val in extra_attrs.items():
            setattr(v, k, val)
    return v


def write_string_var(grp, nc_name, arr_str, long_name, comment=""):
    """Crea o actualitza una variable string en un grup."""
    if nc_name not in grp.variables:
        v = grp.createVariable(nc_name, str, ("time",))
    else:
        v = grp.variables[nc_name]
    for idx, s in enumerate(arr_str):
        v[idx] = s
    v.long_name = long_name
    if comment:
        v.comment = comment
    return v


def col(df, name):
    """Retorna la columna com array float32, o None si no existeix."""
    if df is None or name not in df.columns:
        return None
    return df[name].values.astype(np.float32)


# ---------------------------------------------------------------------------
# Atributs globals ACDD-1.3 + CF-1.8 (perfil Copernicus/CMEMS)
# ---------------------------------------------------------------------------
def set_global_attrs(ds, all_times):
    """
    Escriu els atributs globals al dataset arrel.
    Camps marcats [OMPLIR] s'han de completar abans de publicar a CMEMS.
    Camps marcats [AUTO] es generen automàticament.
    """
    # Referència: https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery
    # Vocabulari GCMD: https://gcmd.earthdata.nasa.gov/KeywordViewer/

    # --- Identificació ---
    ds.Conventions        = "CF-1.8, ACDD-1.3"
    ds.title              = "Boia oceanogràfica Somorrostro - Dades en temps real"  # [OMPLIR]
    ds.summary            = (                                                         # [OMPLIR]
        "Dades oceanogràfiques i meteorològiques en temps real de la boia "
        "Somorrostro, situada a la costa de Barcelona. Inclou mesures de "
        "corrent (ADCP), CTD, pH (SAMI) i meteorologia."
    )
    ds.id                 = "ICATMAR-BOIA-SOMORROSTRO-NRT"   # [OMPLIR] o DOI
    ds.naming_authority   = "cat.icatmar"                    # [OMPLIR]
    ds.product_version    = "1.0"                            # [OMPLIR]

    # --- Keywords GCMD (obligatori CMEMS) ---
    ds.keywords = (                                          # [OMPLIR/REVISAR]
        "Earth Science > Oceans > Ocean Circulation > Ocean Currents, "
        "Earth Science > Oceans > Salinity/Density > Salinity, "
        "Earth Science > Oceans > Ocean Temperature > Water Temperature, "
        "Earth Science > Oceans > Ocean Chemistry > Oxygen, "
        "Earth Science > Oceans > Ocean Chemistry > pH, "
        "Earth Science > Atmosphere > Atmospheric Winds > Surface Winds, "
        "Earth Science > Atmosphere > Atmospheric Pressure > Sea Level Pressure, "
        "Earth Science > Atmosphere > Atmospheric Water Vapor > Humidity"
    )
    ds.keywords_vocabulary = "GCMD Science Keywords"

    # --- Cobertura espacial ---
    ds.geospatial_lat_min           = NOMINAL_LAT
    ds.geospatial_lat_max           = NOMINAL_LAT
    ds.geospatial_lon_min           = NOMINAL_LON
    ds.geospatial_lon_max           = NOMINAL_LON
    ds.geospatial_lat_units         = "degrees_north"
    ds.geospatial_lon_units         = "degrees_east"
    ds.geospatial_vertical_min      = 0.0    # [OMPLIR]
    ds.geospatial_vertical_max      = 40.0   # [OMPLIR] profunditat cel·la 40 Doppler
    ds.geospatial_vertical_units    = "m"
    ds.geospatial_vertical_positive = "down"

    # --- Cobertura temporal [AUTO] ---
    ds.time_coverage_start      = all_times[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    ds.time_coverage_end        = all_times[-1].strftime("%Y-%m-%dT%H:%M:%SZ")
    ds.time_coverage_resolution = "PT15M"
    ds.time_coverage_duration   = str(all_times[-1] - all_times[0])

    # --- Plataforma ---
    ds.platform              = "moored surface buoy"
    ds.platform_vocabulary   = "GCMD Platforms"
    ds.platform_name         = STATION_NAME                  # [OMPLIR]
    ds.platform_code         = "XXXXXXXXX"                   # [OMPLIR] codi WMO o EuroGOOS
    ds.wmo_platform_code     = "XXXXXXXXX"                   # [OMPLIR]
    ds.instrument            = (                             # [OMPLIR/REVISAR]
        "Campbell Scientific CR1000X datalogger; "
        "Nortek ADCP (perfilador de corrents); "
        "SBE37 CTD+O2 (SeaBird Scientific); "
        "SAMI pH (Sunburst Sensors); "
        "Estació meteorològica"
    )
    ds.instrument_vocabulary = "NERC SeaVoX Device Catalogue"

    # --- Origen ---
    ds.source                   = "moored surface buoy"
    ds.featureType              = "timeSeries"
    ds.cdm_data_type            = "TimeSeries"
    ds.processing_level         = "1"                        # [OMPLIR]
    ds.standard_name_vocabulary = "CF Standard Name Table v85"

    # --- Institució i responsables ---
    ds.institution           = "Institut Català de Recerca per a la Governança del Mar (ICATMAR)"  # [OMPLIR]
    ds.institution_edmo_code = "XXXXX"                       # [OMPLIR] codi EDMO SeaDataNet
    ds.creator_name          = "ICATMAR"                     # [OMPLIR]
    ds.creator_email         = "info@icatmar.cat"            # [OMPLIR]
    ds.creator_url           = "https://www.icatmar.cat"     # [OMPLIR]
    ds.creator_type          = "institution"
    ds.publisher_name        = "ICATMAR"                     # [OMPLIR]
    ds.publisher_email       = "info@icatmar.cat"            # [OMPLIR]
    ds.publisher_url         = "https://www.icatmar.cat"     # [OMPLIR]
    ds.publisher_type        = "institution"
    ds.project               = "ICATMAR Xarxa de Boies"     # [OMPLIR]
    ds.program               = ""                            # [OMPLIR]

    # --- Llicència ---
    ds.license           = "CC-BY-4.0"                      # [OMPLIR]
    ds.access_constraint = "otherRestrictions"
    ds.acknowledgment    = (                                 # [OMPLIR]
        "Dades generades per ICATMAR. Si s'utilitzen, citar: "
        "ICATMAR (any), Boia Somorrostro, https://www.icatmar.cat"
    )

    # --- Qualitat ---
    ds.quality_control = "No QC aplicat (NRT)"              # [OMPLIR]
    ds.data_mode       = "R"   # R=Real-time, P=Provisional, D=Delayed

    # --- Traçabilitat [AUTO] ---
    ds.history      = (
        f"Creat/actualitzat: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"mitjançant boia_to_netcdf.py v2.0 (ICATMAR)"
    )
    ds.source_files = ", ".join(FILE_PATTERNS.values())
    ds.comment      = (
        "NetCDF estructurat en dos grups: /meteo (meteorologia, 15 min) i "
        "/oceanografia (oceanografia + Doppler + SAMI + System, 15 min). "
        "Variables sense equivalent CF/ICATMAR guardades amb prefix AUX_. "
        "SamiData/SamiMessage pendents de descodificació de pH (PHPH)."
    )
    ds.references    = ""                                    # [OMPLIR]
    ds.station_name  = STATION_NAME
    ds.nominal_latitude  = NOMINAL_LAT
    ds.nominal_longitude = NOMINAL_LON
    ds.mooring_depth     = 0.0                              # [OMPLIR] profunditat de fondejament (m)


# ---------------------------------------------------------------------------
# Funció principal
# ---------------------------------------------------------------------------
def build_netcdf(input_dir: str, output_file: str) -> None:
    log.info(f"Directori d'entrada : {input_dir}")
    log.info(f"Fitxer de sortida   : {output_file}")

    # ------------------------------------------------------------------
    # 1. Llegir fitxers
    # ------------------------------------------------------------------
    dfs = {}
    for key, pattern in FILE_PATTERNS.items():
        path = find_file(input_dir, pattern)
        if path is None:
            log.warning(f"No s'ha trobat '{pattern}' — s'omiteix.")
            continue
        try:
            dfs[key] = read_toa5(path)
        except Exception as e:
            log.error(f"Error llegint {path}: {e}")

    if not dfs:
        log.error("No s'ha pogut llegir cap fitxer. Aturant.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Eix temporal unificat de 15 min (sense el Status)
    #    El Status té timestamps irregulars i es tracta per separat.
    # ------------------------------------------------------------------
    scientific_keys = [k for k in dfs if k != "status"]
    all_times = pd.DatetimeIndex([])
    for k in scientific_keys:
        all_times = all_times.union(dfs[k].index)
    all_times = all_times.sort_values()
    n_time = len(all_times)

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    time_vals = np.array(
        [(t.to_pydatetime().replace(tzinfo=timezone.utc) - epoch).total_seconds()
         for t in all_times],
        dtype=np.float64
    )

    log.info(f"Eix temporal: {n_time} timesteps, "
             f"{all_times[0]} → {all_times[-1]}")

    # ------------------------------------------------------------------
    # 3. Eix de profunditat del Doppler
    # ------------------------------------------------------------------
    depth_dopp = np.array(
        [DOPPLER_FIRST_CELL_DEPTH + i * DOPPLER_CELL_SIZE
         for i in range(DOPPLER_N_CELLS)],
        dtype=np.float32
    )

    # ------------------------------------------------------------------
    # 4. Reindex de cada DataFrame a l'eix temporal comú
    # ------------------------------------------------------------------
    def rx(key):
        return dfs[key].reindex(all_times) if key in dfs else None

    df_meteo  = rx("meteo")
    df_sbe37  = rx("sbe37")
    df_sami   = rx("sami")
    df_dopp   = rx("doppler")
    df_sys    = rx("system")

    # ------------------------------------------------------------------
    # 5. Status: interpolar al nearest timestamp de l'eix comú
    # ------------------------------------------------------------------
    df_status_raw = dfs.get("status", None)
    df_status     = None
    status_orig_times = None   # timestamps originals del Status

    if df_status_raw is not None:
        # Per cada timestamp del Status, trobar el nearest a l'eix comú
        status_idx = df_status_raw.index
        nearest_idx = all_times[
            np.abs(
                np.subtract.outer(all_times.astype(np.int64),
                                  status_idx.astype(np.int64))
            ).argmin(axis=0)
        ]
        # Guardem el timestamp original abans de reindexar
        status_orig_times_series = pd.Series(
            status_idx.astype(np.int64) / 1e6,  # datetime64[us] → epoch seconds
            index=nearest_idx
        )
        # Si hi ha duplicats (dos Status map al mateix nearest), fem mitjana
        # per als numèrics i keepfirst per als strings
        df_status_reindexed = df_status_raw.copy()
        df_status_reindexed.index = nearest_idx
        df_status_reindexed = df_status_reindexed[
            ~df_status_reindexed.index.duplicated(keep='first')
        ]
        df_status = df_status_reindexed.reindex(all_times)

        # Idem per als temps originals
        status_orig_times_series = status_orig_times_series[
            ~status_orig_times_series.index.duplicated(keep='first')
        ]
        status_orig_times = status_orig_times_series.reindex(all_times).values.astype(np.float64)

        log.info(f"Status: {len(status_idx)} registres interpolats al nearest 15 min.")

    # ------------------------------------------------------------------
    # 6. Parsejat de les cel·les del Doppler
    # ------------------------------------------------------------------
    hcsp_mat = np.full((n_time, DOPPLER_N_CELLS), np.nan, dtype=np.float32)
    hcdt_mat = np.full((n_time, DOPPLER_N_CELLS), np.nan, dtype=np.float32)

    if df_dopp is not None:
        speed_df, dir_df = parse_doppler_cells(df_dopp)
        for i, c in enumerate([f"cell_{j}" for j in range(1, DOPPLER_N_CELLS + 1)]):
            hcsp_mat[:, i] = speed_df[c].values.astype(float)
            hcdt_mat[:, i] = dir_df[c].values.astype(float)

    # ------------------------------------------------------------------
    # 7. Crear el NetCDF
    # ------------------------------------------------------------------
    mode = "r+" if os.path.exists(output_file) else "w"
    log.info(f"Obrint NetCDF en mode '{mode}'...")

    with nc.Dataset(output_file, mode, format="NETCDF4") as ds:

        # ---- Atributs globals ----------------------------------------
        set_global_attrs(ds, all_times)

        # ==============================================================
        # GRUP /meteo
        # ==============================================================
        grp_m = ds.createGroup("meteo") if "meteo" not in ds.groups else ds.groups["meteo"]
        grp_m.description = "Dades meteorològiques de la boia (15 min)"
        grp_m.featureType = "timeSeries"

        if "time" not in grp_m.dimensions:
            grp_m.createDimension("time", None)
        make_time_var(grp_m, time_vals)

        if df_meteo is not None:
            def mv(c): return col(df_meteo, c)

            for nc_name, c, lname, iname, units, sname in [
                ("LATITUDE",  "Latitude",     "Latitude of each location",
                 "LATITUDE",  "degrees_north", "latitude"),
                ("LONGITUDE", "Longitude",    "Longitude of each location",
                 "LONGITUDE", "degrees_east",  "longitude"),
                ("WDIR",  "WindDir_True", "Wind from direction relative true north",
                 "WDIR",  "deg",          "wind_from_direction"),
                ("WSPD",  "Corr_WindS",   "Horizontal wind speed",
                 "WSPD",  "m s-1",        "wind_speed"),
                ("ATMS",  "BP",           "Atmospheric pressure at sea level",
                 "ATMS",  "hPa",          "air_pressure_at_sea_level"),
                ("RELH",  "RH",           "Relative humidity",
                 "RELH",  "%",            "relative_humidity"),
                ("DRYT",  "AirTemp",      "Air temperature in dry bulb",
                 "DRYT",  "degrees_Celsius", "air_temperature"),
                ("DEWT",  "DP",           "Dew point temperature",
                 "DEWT",  "degrees_Celsius", "dew_point_temperature"),
                ("WETT",  "WBT",          "Air temperature in wet bulb",
                 "WETT",  "degrees_Celsius", "wet_bulb_temperature"),
            ]:
                a = mv(c)
                if a is None:
                    log.warning(f"METEO: columna '{c}' no trobada.")
                    continue
                write_float_var(grp_m, nc_name, a, ("time",),
                                lname, units, sname, iname)

            # Variables auxiliars sense nom CF/ICATMAR
            log.warning("METEO: 'AD', 'HASL', 'Rel_WindDir', 'Corr_WindDir', 'Rel_WS' "
                        "sense equivalent CF/ICATMAR — guardades com AUX_.")
            for nc_name, c, lname, units in [
                ("AUX_AIR_DENSITY",  "AD",          "Air density",                       "kg m-3"),
                ("AUX_HASL",         "HASL",         "Height above sea level (GPS)",       "m"),
                ("AUX_REL_WINDDIR",  "Rel_WindDir",  "Relative wind direction",            "deg"),
                ("AUX_CORR_WINDDIR", "Corr_WindDir", "Corrected (magnetic) wind direction","deg"),
                ("AUX_REL_WS",       "Rel_WS",       "Relative wind speed",               "m s-1"),
            ]:
                a = mv(c)
                if a is None:
                    continue
                write_float_var(grp_m, nc_name, a, ("time",), lname, units,
                                extra_attrs={"comment": "Variable auxiliar sense nom CF/ICATMAR."})

        # ==============================================================
        # GRUP /oceanografia
        # ==============================================================
        grp_o = ds.createGroup("oceanografia") if "oceanografia" not in ds.groups \
                else ds.groups["oceanografia"]
        grp_o.description = ("Dades oceanogràfiques, Doppler, SAMI i System (15 min). "
                             "Variables Status interpolades al nearest 15 min.")
        grp_o.featureType = "timeSeries"

        if "time" not in grp_o.dimensions:
            grp_o.createDimension("time", None)
        if "depth" not in grp_o.dimensions:
            grp_o.createDimension("depth", DOPPLER_N_CELLS)
        make_time_var(grp_o, time_vals)

        # ---- Eix de profunditat --------------------------------------
        if "depth" not in grp_o.variables:
            dvar = grp_o.createVariable("depth", "f4", ("depth",), zlib=True)
        else:
            dvar = grp_o.variables["depth"]
        dvar[:] = depth_dopp
        dvar.long_name     = "depth"
        dvar.standard_name = "depth"
        dvar.units         = "m"
        dvar.positive      = "down"
        dvar.axis          = "Z"
        dvar.comment       = (f"Primera cel·la a {DOPPLER_FIRST_CELL_DEPTH} m, "
                              f"gruix de cel·la {DOPPLER_CELL_SIZE} m")

        # ---- SBE37 ---------------------------------------------------
        if df_sbe37 is not None:
            def sv(c): return col(df_sbe37, c)

            for nc_name, c, lname, iname, units, sname in [
                ("TEMP", "SBE37Temp", "Sea temperature",          "TEMP",
                 "degrees_Celsius", "sea_water_temperature"),
                ("CNDC", "SBE37Cond", "Electrical conductivity",  "CNDC",
                 "S m-1",           "sea_water_electrical_conductivity"),
                ("PRES", "SBE37Pres", "Sea pressure",             "PRES",
                 "dbar",            "sea_water_pressure"),
                ("DOX1", "SBE37OXY",  "Dissolved oxygen",         "DOX1",
                 "ml l-1",          "volume_fraction_of_oxygen_in_sea_water"),
                ("PSAL", "SBE37Sal",  "Practical salinity",       "PSAL",
                 "1",               "sea_water_practical_salinity"),
            ]:
                a = sv(c)
                if a is None:
                    log.warning(f"SBE37: columna '{c}' no trobada.")
                    continue
                write_float_var(grp_o, nc_name, a, ("time",), lname, units, sname, iname,
                                extra_attrs={"sensor": "SBE37",
                                             "depth": f"{DEPTH_CTD} m (nominal)"})

            if "SBE37Sn" in df_sbe37.columns:
                sn_vals = df_sbe37["SBE37Sn"].dropna().unique()
                if len(sn_vals) > 0:
                    grp_o.sbe37_serial_number = str(sn_vals[0])

        # ---- SAMI ----------------------------------------------------
        if df_sami is not None:
            log.warning("SAMI: SamiData/SamiMessage guardats en brut (pendent descodificació pH).")

            if "SamiNBlank" in df_sami.columns:
                a = col(df_sami, "SamiNBlank")
                write_float_var(grp_o, "AUX_SAMI_NBLANK", a, ("time",),
                                "SAMI number of blank measurements", "1",
                                extra_attrs={
                                    "comment": "Auxiliar. pH (PHPH) pendent descodificació.",
                                    "icatmar_name": "PHPH (pendent)",
                                    "depth": f"{DEPTH_SAMI} m (nominal)"
                                })

            for nc_name, c, lname in [
                ("RAW_SAMI_DATA",    "SamiData",    "Missatge SAMI en brut. Pendent descodificació pH."),
                ("RAW_SAMI_MESSAGE", "SamiMessage", "Missatge de diagnòstic SAMI."),
            ]:
                if c not in df_sami.columns:
                    continue
                arr_str = df_sami[c].fillna("").astype(str).values
                write_string_var(grp_o, nc_name, arr_str, lname,
                                 comment=lname)

        # ---- Doppler escalar -----------------------------------------
        if df_dopp is not None:
            def dv(c): return col(df_dopp, c)

            for nc_name, c, lname, iname, units, sname in [
                ("SENSOR_VOLTAGE", "DoppVolts",      "Input voltage from battery or power supply",
                 "SENSOR_VOLTAGE", "V",    ""),
                ("SVEL",           "DoppSoundSpeed", "Sound velocity in sea water",
                 "SVEL",           "m s-1","speed_of_sound_in_sea_water"),
                ("SENSOR_HEAD",    "DoppHeading",    "Sensor Heading",
                 "SENSOR_HEAD",    "deg",  ""),
                ("SENSOR_PITCH",   "DoppPitch",      "Sensor Pitch",
                 "SENSOR_PITCH",   "deg",  "platform_pitch"),
                ("SENSOR_ROLL",    "DoppRoll",       "Sensor Roll",
                 "SENSOR_ROLL",    "deg",  "platform_roll"),
                ("PRES_DOPP",      "DoppPress",      "Sea pressure (Doppler)",
                 "PRES",           "dbar", "sea_water_pressure"),
                ("TEMP_DOPP",      "DoppTemp",       "Sea temperature (Doppler)",
                 "TEMP",           "degrees_Celsius","sea_water_temperature"),
            ]:
                a = dv(c)
                if a is None:
                    log.warning(f"DOPPLER: columna '{c}' no trobada.")
                    continue
                write_float_var(grp_o, nc_name, a, ("time",), lname, units, sname, iname,
                                extra_attrs={"sensor": "Doppler"})

        # ---- Doppler perfils (time × depth) --------------------------
        for nc_name, mat, lname, iname, units, sname in [
            ("HCSP", np.where(np.isnan(hcsp_mat), FILL_VALUE, hcsp_mat),
             "Horizontal current speed",              "HCSP", "m s-1",
             "sea_water_speed"),
            ("HCDT", np.where(np.isnan(hcdt_mat), FILL_VALUE, hcdt_mat),
             "Current to direction relative true north", "HCDT", "deg",
             "direction_of_sea_water_velocity"),
        ]:
            if nc_name not in grp_o.variables:
                v = grp_o.createVariable(nc_name, "f4", ("time", "depth"),
                                         fill_value=FILL_VALUE, zlib=True)
            else:
                v = grp_o.variables[nc_name]
            v[:] = mat
            v.long_name     = lname
            v.standard_name = sname
            v.units         = units
            v.icatmar_name  = iname
            v.sensor        = "Doppler"
            v.coordinates   = "time depth"

        # ---- System --------------------------------------------------
        if df_sys is not None:
            def syv(c): return col(df_sys, c)

            log.warning("SYSTEM: CTD/ADP/SAMI/Meteo_PwrOff i PTemp sense equivalent "
                        "CF/ICATMAR — guardats com AUX_.")

            existing_sv = set(grp_o.variables.keys())
            for nc_name, c, lname, units, iname in [
                ("SENSOR_VOLTAGE_BATT", "Datalogg_Batt", "Datalogger battery voltage",
                 "V", "SENSOR_VOLTAGE"),
                ("AUX_BATT_VOLT",       "Batt_Volt",     "Battery voltage",          "V", ""),
                ("AUX_SOLAR_VOLT",      "Solar_Voltage", "Solar panel voltage",       "V", ""),
                ("AUX_SYS_CURRENT",     "System_Current","System current",            "A", ""),
                ("AUX_PTEMP",           "PTemp",         "Panel temperature",
                 "degrees_Celsius", ""),
                ("AUX_CTD_PWROFF",      "CTD_PwrOff",    "CTD power off flag",       "1", ""),
                ("AUX_ADP_PWROFF",      "ADP_PwrOff",    "ADP power off flag",       "1", ""),
                ("AUX_SAMI_PWROFF",     "SAMI_PwrOff",   "SAMI power off flag",      "1", ""),
                ("AUX_METEO_PWROFF",    "Meteo_PwrOff",  "Meteo power off flag",     "1", ""),
            ]:
                a = syv(c)
                if a is None:
                    log.warning(f"SYSTEM: columna '{c}' no trobada.")
                    continue
                write_float_var(grp_o, nc_name, a, ("time",), lname, units,
                                icatmar_name=iname)

        # ---- Status (interpolat al nearest 15 min) -------------------
        if df_status is not None:

            # Timestamp original del Status com a variable float64
            if "STATUS_TIME" not in grp_o.variables:
                stv = grp_o.createVariable("STATUS_TIME", "f8", ("time",),
                                           fill_value=-9999.0, zlib=True)
            else:
                stv = grp_o.variables["STATUS_TIME"]
            stv[:] = np.where(np.isnan(status_orig_times), -9999.0, status_orig_times)
            stv.long_name  = "Original timestamp of the Status record"
            stv.units      = "seconds since 1970-01-01 00:00:00 UTC"
            stv.calendar   = "gregorian"
            stv.comment    = ("Timestamp real del registre de Status del datalogger, "
                              "abans de la interpolació al nearest 15 min. "
                              "Usar per verificar la proximitat de la interpolació.")

            log.warning("STATUS: camps de diagnòstic sense equivalent CF/ICATMAR "
                        "— guardats com AUX_STATUS_.")

            for nc_name, c, lname, units in [
                ("AUX_STATUS_PANELTEMP",   "PanelTemp",      "Panel temperature",          "degrees_Celsius"),
                ("AUX_STATUS_BATTERY",     "Battery",        "Datalogger battery voltage",  "V"),
                ("AUX_STATUS_LITHIUM",     "LithiumBattery", "Lithium battery voltage",     "V"),
                ("AUX_STATUS_WATCHDOG",    "WatchdogErrors", "Watchdog error count",        "1"),
                ("AUX_STATUS_SKIPPEDSCAN", "SkippedScan",    "Skipped scan count",          "1"),
                ("AUX_STATUS_MEMORYFREE",  "MemoryFree",     "Free memory",                 "bytes"),
            ]:
                a = col(df_status, c)
                if a is None:
                    continue
                write_float_var(grp_o, nc_name, a, ("time",), lname, units,
                                extra_attrs={"comment":
                                    "Diagnòstic del datalogger. Interpolat al nearest 15 min. "
                                    "Vegeu STATUS_TIME per al timestamp original."})

            if "CardStatus" in df_status.columns:
                arr_str = df_status["CardStatus"].fillna("").astype(str).values
                write_string_var(grp_o, "AUX_STATUS_CARD", arr_str,
                                 "Memory card status",
                                 comment=("Diagnòstic del datalogger. Interpolat al nearest 15 min. "
                                          "Vegeu STATUS_TIME per al timestamp original."))

    log.info(f"NetCDF generat correctament: {output_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Converteix fitxers .dat TOA5 de la boia Somorrostro a NetCDF (CF/ICATMAR)."
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Directori d'entrada amb els fitxers .dat")
    parser.add_argument("--output", "-o", required=True,
                        help="Ruta del fitxer NetCDF de sortida")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        log.error(f"El directori d'entrada no existeix: {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    build_netcdf(args.input, args.output)


if __name__ == "__main__":
    main()