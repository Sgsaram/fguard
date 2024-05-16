import datetime
import os

import dotenv

import fguard.core.communication
import fguard.core.utils
import numpy as np
import PIL.Image


dotenv.load_dotenv(dotenv.find_dotenv())

BASE_DIR = dir_path = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, ".fguard-config.json")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
DATA_FOLDER = os.path.join(BASE_DIR, "assets", "dataset")
SH_CLIENT_ID = os.environ.get("SH_CLIENT_ID", "")
SH_CLIENT_SECRET = os.environ.get("SH_CLIENT_SECRET", "")
EOLEARN_CACHE_FOLDER = os.path.join(BASE_DIR, ".eolearn_cache")

def main():
#     fguard.utils.remove_folder_content(OUTPUT_FOLDER)
#     detector = fguard.vision.UNetDetector(
#         os.path.join(BASE_DIR, "MODEL.pth"),
#     )
#     d_image = PIL.Image.open(os.path.join(BASE_DIR, "test_image.png"))
#     mask = detector.detect(np.array(d_image))
#     s_mask = PIL.Image.fromarray(fguard.utils.cluster_array_to_image(mask, [(0, 0, 0), (255, 255, 255)]), "RGB")
#     s_mask.save(os.path.join(OUTPUT_FOLDER, "unet_output.png"))

#     return
    comclient = fguard.core.communication.CommunicationClient(
        sh_client_id=SH_CLIENT_ID,
        sh_client_secret=SH_CLIENT_SECRET,
        cache_folder=EOLEARN_CACHE_FOLDER,
    )
    time_interval = (
        datetime.datetime.strptime("2024.01.01", "%Y.%m.%d"),
        datetime.datetime.strptime("2024.02.28", "%Y.%m.%d") + datetime.timedelta(days=1),
    )
    print("Started downloading data...")
    print("(Please wait. This may take no more than 2 minutes)")
#     coords = (
#         41.17414, # Lng 1
#         55.55250, # Lat 1
#         41.18757, # Lng 2
#         55.54184, # Lat 2
#     )
    coords = (
        42.65614,
        59.52529,
        42.73579,
        59.58242,
    )
    data = comclient.get_winter_data(
        coords=coords,
        time_interval=time_interval,
        size=(512, 512),
        data_collection="sentinel2_l1c",
#         time_difference=datetime.timedelta(days=5),
    )
    _res = data[0][0].shape[:2] if len(data) > 0 else (0, 0)
    print(f"Downloaded data successfully -> got {len(data)} images with {_res} shape")
#     print(
#         round(
#             geopy.distance.geodesic(
#                 (coords[1], coords[0]), # (Lat, Lng)
#                 (coords[3], coords[0]),
#             ).meters / 150,
#         ),
#     )
#     print(
#         round(
#             geopy.distance.geodesic(
#                 (coords[1], coords[0]),
#                 (coords[1], coords[2]),
#             ).meters / 150,
#         ),
#     )
    cur_ind = len(os.listdir(OUTPUT_FOLDER))
    for tc, mask, tmp in data:
        print(tmp)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if not mask[x, y]:
                    tc[x, y] = np.array([0, 0, 0])
        image_to_print = PIL.Image.fromarray(tc)
        image_to_print.save(
            os.path.join(
                OUTPUT_FOLDER,
                "img_" + fguard.core.utils.get_right_aligned_number(cur_ind, 5) + ".png"
            ),
            "png",
        )
        cur_ind += 1

if __name__ == "__main__":
    main()
