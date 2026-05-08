# datTOnetcdf.py

Converteix els fitxers `.dat` de la boia oceanogràfica Somorrostro (CR1000X) a NetCDF4 (CF-1.8, ACDD-1.3).

## Instal·lació

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ús

```bash
python3 datTOnetcdf.py --input /boies/ --output /boies/ICATMAR_BOY_FIXD_L0_BSOMO_1Hxxxxxxx_YYYYMMDDHHMM_YYYYMMDDHHMM_vYYYYMMDD.nc

```

## Cron

```bash
0 * * * * /venv/bin/python3 /datTOnetcdf.py --input /data/ftp/ --output /data/ICATMAR_BOY_FIXD_L0_BSOMO_1Hxxxxxxx_YYYYMMDDHHMM_YYYYMMDDHHMM_vYYYYMMDD.nc
```

## Paràmetres

Els paràmetres configurables (data d'inici, coordenades, profunditats, salinitat de fallback...) es troben al principi del fitxer `datTOnetcdf.py`.

## Documentació

Vegeu https://www.overleaf.com/read/mqdzpktpcvpt#cc50a5

## PLOT DE DATA
python3 pdf.py --input ICATMAR_BOY_FIXD_L0_BSOMO_1Hxxxxxxx_YYYYMMDDHHMM_YYYYMMDDHHMM_v202605080605.nc --output plots_boia.pdf --only-valid 

