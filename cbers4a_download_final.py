# %% [markdown]
# Author: Victor A. C. Rosário
# Credits: Gabriel Russo (https://github.com/gabriel-russo/cbers4asat)
# AmbGEO (Computer Vision course)
# INPE: https://data.inpe.br/bdc/

# %% [markdown]
# Installing required packages

# %%
#pip install psycopg2-binary
# pip install "cbers4asat[tools]"
# pip install datacube
#pip install cbers4asat

# %% [markdown]
# Importing for the packages

# %%
import geopandas as gpd
from datacube import Datacube
import rasterio
from rasterio.transform import from_bounds
import numpy as np
import os

from cbers4asat import Cbers4aAPI
from cbers4asat import Collections as coll
#from cbers4a import Collections
from datetime import date
from tqdm import tqdm

# %% [markdown]
# Setting the bbox area to be evaluated

# %%

# Shapefile used as bbox for searching CBERS-4A images
sp = gpd.read_file(r"H:/download/shapes/BR_Municipios_2022/Municipios_SP_python.shp")
sc = sp[sp["NM_MUN"] == "São Carlos"]
os.makedirs("R:/download/cbers4a/", exist_ok=True)
output_dir = "R:/download/cbers4a/"
gdf = sc.copy().to_crs(epsg=4326)

# Extracting bounding box (xmin, ymin, xmax, ymax)
bounds = gdf.total_bounds
bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]  # [minx, miny, maxx, maxy]

### Important: Use your own email registered in the CBERS-4A portal
### https://www.dgi.inpe.br/catalogo/explore (At INPE website, you can create an account for free)
api = Cbers4aAPI('xxxxxxxxxxxxxx')

# Searching for CBERS-4A products over the bounding box and date range
produtos = api.query(
    location=bbox,
    initial_date=date(2025, 11, 1),
    end_date=date(2025, 12, 31),
    cloud=100,
    limit=10,
    collections=[coll.CBERS4A_WPM_L4_DN]
)

# Showing the found products
if produtos and 'features' in produtos and produtos['features']:
    lista_produtos = produtos['features']
    print(f"Produtos encontrados: {len(lista_produtos)}")
    for i, produto in enumerate(lista_produtos):
        print(f"Produto {i+1}: {produto['id']}")
else:
     print("No products found in this area or time range.")

# %% [markdown]
# After that the product were found, use the following block of code to downloading it.
# U r gonna have the 100% bar completed when all the download scenes complete

# %%
def baixar_produtos(produtos):
    # Garante que produtos seja uma lista
    if not produtos:
        print("No produtos found.")
        return

    
    if not isinstance(produtos, list):
        produtos = [produtos]

    #print(f"Productuds found: {len(produtos)}")
    try:
        for produto in tqdm(produtos, desc="Downloading products"):
                produto,
                bands=['red', 'green', 'blue', 'nir', 'pan'],
                outdir=output_dir,
                with_folder=True
            )
        print("Download done!")
    except Exception as e:
        print(f"Download error: {e}")

# Chame a função após buscar os produtos
baixar_produtos(produtos)


# %% [markdown]
#  Stacking manually each scene

# %%
blue = r"R:\download\cbers4a\CBERS4A_WPM20514020251224ETC2\CBERS_4A_WPM_20251224_205_140_L4_BAND1.tif"
green = r"R:\download\cbers4a\CBERS4A_WPM20514020251224ETC2\CBERS_4A_WPM_20251224_205_140_L4_BAND2.tif"
red = r"R:\download\cbers4a\CBERS4A_WPM20514020251224ETC2\CBERS_4A_WPM_20251224_205_140_L4_BAND3.tif"
nir = r"R:\download\cbers4a\CBERS4A_WPM20514020251224ETC2\CBERS_4A_WPM_20251224_205_140_L4_BAND4.tif"
pan = r"R:\download\cbers4a\CBERS4A_WPM20514020251224ETC2\CBERS_4A_WPM_20251224_205_140_L4_BAND0.tif"

with rasterio.open(blue) as src:
    b = src.read(1).astype(np.float32)
    profile = src.profile.copy()   # <-- pegar profile do src, não do array

with rasterio.open(green) as src:
    g = src.read(1).astype(np.float32)

with rasterio.open(red) as src:
    r = src.read(1).astype(np.float32)

with rasterio.open(nir) as src:
    n = src.read(1).astype(np.float32)

# stack para (H, W, C)
rgbn = np.dstack((r, g, b, n)).astype(np.float32)

# atualizar profile para multibanda
profile.update({
    'count': rgbn.shape[2],       # número de bandas
    'dtype': 'float32'
    # se necessário, ajustar 'driver', 'compress', etc.
})

save_path = r"R:\download\cbers4a\CBERS4A_WPM20514020251224ETC2\RGBNIR_CBERS4A_20251224_python.tif"
with rasterio.open(save_path, 'w', **profile) as dst:
    # rasterio espera (bands, rows, cols)
    dst.write(np.transpose(rgbn, (2, 0, 1)))


# %% [markdown]
# Stacking using for looping in the created folders for each scene!

# %%
import os
import re
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

ROOT_DIR = output_dir   
OVERWRITE = False                  
COMPRESS = "LZW"                   

RE_META = re.compile(r".*_(\d{8})_(\d{3})_(\d{3})_L4_BAND([0-4])\.tif$", re.IGNORECASE)

def _find_band_files(scene_dir: str) -> dict:
    """
    Retorna dict {1: path_band1, 2: path_band2, 3: path_band3, 4: path_band4}
    (ignora BAND0).
    """
    band_files = {}
    for f in glob.glob(os.path.join(scene_dir, "*.tif")):
        m = RE_META.match(os.path.basename(f))
        if not m:
            continue
        band = int(m.group(4))
        if band in (1, 2, 3, 4):
            band_files[band] = f
    return band_files

def _extract_scene_meta(any_band_path: str):
    """
    Extrai (date_str, orbit, row) do nome do arquivo.
    """
    m = RE_META.match(os.path.basename(any_band_path))
    if not m:
        return None, None, None
    date_str = m.group(1)      # YYYYMMDD
    orbit = m.group(2)         # 205
    row = m.group(3)           # 140
    return date_str, orbit, row

def _read_as_ref_grid(src_path: str, ref_profile: dict, ref_transform, ref_crs, ref_shape):
    """
    Lê um raster e, se necessário, reprojeta/reamostra para bater com a grade de referência.
    Retorna array float32 (rows, cols).
    """
    with rasterio.open(src_path) as src:
        arr = src.read(1).astype(np.float32)

        
        if (src.crs == ref_crs) and (src.transform == ref_transform) and (arr.shape == ref_shape):
            return arr

       
        dest = np.empty(ref_shape, dtype=np.float32)

        reproject(
            source=arr,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest  # pode trocar para bilinear/cubic se preferir
        )
        return dest

def build_rgbnir_stack_for_scene(scene_dir: str):
    band_files = _find_band_files(scene_dir)
    missing = [b for b in (1, 2, 3, 4) if b not in band_files]

    if missing:
        print(f"[SKIP] {os.path.basename(scene_dir)}: faltando bandas {missing}")
        return

    
    ref_path = band_files[1]
    date_str, orbit, row = _extract_scene_meta(ref_path)
    if not (date_str and orbit and row):
        print(f"[SKIP] {os.path.basename(scene_dir)}: padrão de nome não reconhecido")
        return

    orbit_row = f"{orbit}{row}"  # ex: "205140" (ou se preferir com underscore: f"{orbit}_{row}")

    out_name = f"RGBNIR_CBERS4A_{date_str}_{orbit_row}.tif"
    out_path = os.path.join(scene_dir, out_name)

    if (not OVERWRITE) and os.path.exists(out_path):
        print(f"[OK] Já existe: {out_path}")
        return

    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)

    b = _read_as_ref_grid(band_files[1], ref_profile, ref_transform, ref_crs, ref_shape)
    g = _read_as_ref_grid(band_files[2], ref_profile, ref_transform, ref_crs, ref_shape)
    r = _read_as_ref_grid(band_files[3], ref_profile, ref_transform, ref_crs, ref_shape)
    n = _read_as_ref_grid(band_files[4], ref_profile, ref_transform, ref_crs, ref_shape)

    
    stack = np.stack([r, g, b, n], axis=0).astype(np.float32)  # (bands, rows, cols)

    
    ref_profile.update(
        count=4,
        dtype="float32",
        compress=COMPRESS
    )

    with rasterio.open(out_path, "w", **ref_profile) as dst:
        dst.write(stack)

    print(f"[DONE] Stack salvo: {out_path}")

def process_all_scenes(root_dir: str):
    scene_dirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    if not scene_dirs:
        print(f"Nenhuma pasta/cena encontrada em: {root_dir}")
        return

    for scene_dir in tqdm(scene_dirs, desc="Processando cenas"):
        build_rgbnir_stack_for_scene(scene_dir)

if __name__ == "__main__":
    process_all_scenes(ROOT_DIR)


# %% [markdown]
# Fusioning images using for looping
# #### Did not finish it yet duw to lack of memory

# %%
import os
import re
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

ROOT_DIR = output_dir
OVERWRITE = False
COMPRESS = "LZW"

RE_META = re.compile(r".*_(\d{8})_(\d{3})_(\d{3})_L4_BAND([0-4])\.tif$", re.IGNORECASE)

def find_band_files(scene_dir: str):
    """Retorna dict {0: pan, 1: blue, 2: green, 3: red, 4: nir} quando existirem."""
    band_files = {}
    for f in glob.glob(os.path.join(scene_dir, "*.tif")):
        m = RE_META.match(os.path.basename(f))
        if not m:
            continue
        band = int(m.group(4))
        if band in (0, 1, 2, 3, 4):
            band_files[band] = f
    return band_files

def extract_meta(any_path: str):
    """Extrai (date_str, orbit, row) do nome do arquivo."""
    m = RE_META.match(os.path.basename(any_path))
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)

def read_to_ref_grid(src_path: str, ref_crs, ref_transform, ref_shape, resampling=Resampling.bilinear):
    """
    Lê um raster e devolve array float32 reamostrado/reprojetado para a grade de referência.
    """
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)
        
        if (src.crs == ref_crs) and (src.transform == ref_transform) and (src_arr.shape == ref_shape):
            return src_arr

        dest = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling
        )
        return dest

def match_pan_to_intensity(pan, intensity, eps=1e-6):
    """
    Ajusta PAN para ter média e desvio-padrão similares ao intensity (MS),
    reduzindo mudanças de brilho/contraste no pansharpen.
    """
    pan_mean, pan_std = np.nanmean(pan), np.nanstd(pan)
    int_mean, int_std = np.nanmean(intensity), np.nanstd(intensity)

    pan_std = max(pan_std, eps)
    int_std = max(int_std, eps)

    pan_matched = (pan - pan_mean) * (int_std / pan_std) + int_mean
    return pan_matched

def pansharpen_ratio(ms_up, pan, weights=(0.30, 0.30, 0.30, 0.10), eps=1e-6):
    """
    ms_up: array (4, H, W) com [R,G,B,NIR] já na grade do PAN
    pan:   array (H, W) PAN na sua própria grade
    weights: pesos para compor a intensidade do MS (soma ~1).
             (R,G,B,NIR) default dá mais peso ao RGB e um pouco ao NIR.
    """
    w = np.array(weights, dtype=np.float32).reshape(4, 1, 1)

    
    intensity = np.sum(ms_up * w, axis=0)

    
    pan_adj = match_pan_to_intensity(pan, intensity, eps=eps)

    
    ratio = pan_adj / (intensity + eps)

    
    out = ms_up * ratio[np.newaxis, :, :]
    return out

def write_multiband(out_path, arr, ref_profile):
    """
    Grava GeoTIFF multibanda (bands, rows, cols).
    """
    profile = ref_profile.copy()
    profile.update(
        count=arr.shape[0],
        dtype="float32",
        compress=COMPRESS
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32))

def process_scene(scene_dir: str):
    bands = find_band_files(scene_dir)
    required = [0, 1, 2, 3, 4]
    missing = [b for b in required if b not in bands]
    if missing:
        print(f"[SKIP] {os.path.basename(scene_dir)}: faltando bandas {missing}")
        return

    date_str, orbit, row = extract_meta(bands[0])
    if not (date_str and orbit and row):
        print(f"[SKIP] {os.path.basename(scene_dir)}: metadados não reconhecidos no nome.")
        return

    orbit_row = f"{orbit}{row}"  # ou f"{orbit}_{row}"
    out_name = f"PANSHARP_RGBNIR_{date_str}_{orbit_row}.tif"
    out_path = os.path.join(scene_dir, out_name)

    if (not OVERWRITE) and os.path.exists(out_path):
        print(f"[OK] Já existe: {out_path}")
        return

   
    with rasterio.open(bands[0]) as pan_src:
        pan = pan_src.read(1).astype(np.float32)
        pan_profile = pan_src.profile.copy()
        ref_crs = pan_src.crs
        ref_transform = pan_src.transform
        ref_shape = (pan_src.height, pan_src.width)

    
    b = read_to_ref_grid(bands[1], ref_crs, ref_transform, ref_shape, resampling=Resampling.bilinear)
    g = read_to_ref_grid(bands[2], ref_crs, ref_transform, ref_shape, resampling=Resampling.bilinear)
    r = read_to_ref_grid(bands[3], ref_crs, ref_transform, ref_shape, resampling=Resampling.bilinear)
    n = read_to_ref_grid(bands[4], ref_crs, ref_transform, ref_shape, resampling=Resampling.bilinear)

    ms_up = np.stack([r, g, b, n], axis=0).astype(np.float32)

    # Pansharpen
    ps = pansharpen_ratio(ms_up, pan)

   
    lo = np.nanpercentile(ms_up, 0.5)
    hi = np.nanpercentile(ms_up, 99.5)
    ps = np.clip(ps, lo, hi)

    
    write_multiband(out_path, ps, pan_profile)
    print(f"[DONE] {out_path}")

def process_all_scenes(root_dir: str):
    scene_dirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    if not scene_dirs:
        print(f"Nenhuma cena em {root_dir}")
        return

    for sd in scene_dirs:
        process_scene(sd)

if __name__ == "__main__":
    process_all_scenes(ROOT_DIR)


# %% [markdown]
# Teste de layout e fusionamento


