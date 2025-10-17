import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2019-v1.7/"

LAYERS = ['treecover2000', 'gain', 'lossyear', 'datamask']

# Generate latitude strings: 80N to 00N, then 10S to 60S
lats = []
for lat in range(80, -1, -10):
    lats.append(f"{lat:02d}N")
for lat in range(10, 70, 10):
    lats.append(f"{lat:02d}S")

# Generate longitude strings: 180W to 010W, 000E to 170E
lons = []
for lon in range(180, 0, -10):
    lons.append(f"{lon:03d}W")
for lon in range(0, 180, 10):
    lons.append(f"{lon:03d}E")

# Function to generate URL for a specific tile
def get_tile_url(layer, lat_str, lon_str):
    filename = f"Hansen_GFC-2019-v1.7_{layer}_{lat_str}_{lon_str}.tif"
    return BASE_URL + filename

# Function to download a single file
def download_file(url, output_dir):
    local_filename = os.path.join(output_dir, os.path.basename(url))
    if os.path.exists(local_filename):
        print(f"Skipping {local_filename} (already exists)")
        return True
    try:
        with requests.get(url, stream=True) as r:
            if r.status_code != 200:
                print(f"Failed to download {url} (status: {r.status_code})")
                return False
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {local_filename}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

# Main function to download a layer
def download_layer(layer, output_root='global_forest_data', max_workers=5):
    output_dir = os.path.join(output_root, layer)
    os.makedirs(output_dir, exist_ok=True)

    urls = [get_tile_url(layer, lat, lon) for lat in lats for lon in lons]

    print(f"Downloading {len(urls)} tiles for layer '{layer}'...")

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_file, url, output_dir): url for url in urls}
        for future in tqdm(as_completed(future_to_url), total=len(urls)):
            if future.result():
                success_count += 1

    print(f"Completed {layer}: {success_count}/{len(urls)} tiles downloaded successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Hansen Global Forest Change dataset v1.7 (2000-2019)")
    parser.add_argument('--output',
                        help="Root output directory (default: global_forest_data)")
    parser.add_argument('--workers', type=int, default=5,
                        help="Number of parallel downloads (default: 5)")

    args = parser.parse_args()

    for layer in LAYERS:
        download_layer(layer, args.output, args.workers)
