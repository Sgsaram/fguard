import datetime
import json
import os
import PIL.Image
import tomllib

import click
import dotenv
import numpy as np
import platformdirs
import tqdm

import fguard.communication
import fguard.handler
import fguard.utils
import fguard.vision

dotenv.load_dotenv(dotenv.find_dotenv())

BASE_DIR = platformdirs.user_data_dir("fguard", False)
CONFIG_FILE = os.path.join(BASE_DIR, ".fguard-config.json")
# OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
# SH_CLIENT_ID = os.environ.get("SH_CLIENT_ID", "")
# SH_CLIENT_SECRET = os.environ.get("SH_CLIENT_SECRET", "")
EOLEARN_CACHE_FOLDER = os.path.join(BASE_DIR, ".fguard_cache")


@click.group()
def cli():
    pass


@click.command()
@click.option("-i", "--id", type=str)
@click.option("-t", "--token", type=str)
@click.option("-f", "--file", type=click.Path(exists=True, file_okay=True))
def config(id, token, file):
    client_id = None
    client_token = None
    if file is not None:
        with open(file, "rb") as f:
            data = tomllib.load(f)
            if "config" in data:
                if "id" in data["config"]:
                    client_id = data["config"]["id"]
                if "token" in data["config"]:
                    client_token = data["config"]["token"]
    if id is not None:
        client_id = id
    if token is not None:
        client_token = token
    if client_id is None:
        print("Id wasn't specified.")
        exit(-1)
    if client_token is None:
        print("Token wasn't specified.")
        exit(-1)
    obj = dict()
    obj["id"] = client_id
    obj["token"] = client_token
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    with open(CONFIG_FILE, "w") as f:
        json.dump(obj, f)


@click.command()
@click.option("-c", "--coordinates", nargs=4, type=float)
@click.option("-t", "--time", nargs=2, type=str)
@click.option("-f", "--file", type=click.Path(exists=True, file_okay=True))
@click.option("-d", "--detector", default="net", type=click.Choice(["net", "cluster"]))
@click.option("-s", "--separate", is_flag=True)
# @click.option("-l", "--landsat", type=str)
@click.argument("output-folder", type=click.Path(exists=True, dir_okay=True), required=True)
def request(coordinates, time, file, detector, separate, output_folder):
    id = None
    token = None
    with open(CONFIG_FILE, "r") as f:
        data = json.load(f)
        if "id" in data:
            id = data["id"]
        if "token" in data:
            token = data["token"]
    if id is None or token is None:
        print("You didn't specify credentials. Use `fguard config`.")
        exit(-1)
    timestamp = None
    coords = None
    m_detector = None
    if file is not None:
        with open(file, "rb") as f:
            data = tomllib.load(f)
            if "request" in data:
                if "time" in data["request"]:
                    timestamp = data["request"]["time"]
                if "coordinates" in data["request"]:
                    coords = data["request"]["coordinates"]
                if "detector" in data["request"]:
                    m_detector = data["request"]["detector"]
    if coordinates is not None:
        coords = coordinates
    if time is not None:
        timestamp = time
    if detector is not None:
        m_detector = detector
    if coords is None:
        print("Coordinates weren't specified.")
        exit(-1)
    if timestamp is None:
        print("Time wasn't specified.")
        exit(-1)




    fguard.utils.remove_folder_content(output_folder)
    print(f"Emptied {output_folder}.")



    comclient = fguard.communication.CommunicationClient(
        sh_client_id=id,
        sh_client_secret=token,
        cache_folder=EOLEARN_CACHE_FOLDER,
    )
    time_interval = (
        datetime.datetime.strptime(timestamp[0], "%Y.%m.%d"),
        datetime.datetime.strptime(timestamp[1], "%Y.%m.%d") + datetime.timedelta(days=1),
    )
    print("Started downloading data.")
    data = comclient.get_winter_data(
        coords=coords,
        time_interval=time_interval,
        size=(512, 512),
        data_collection="sentinel2_l1c",
    )
    _res = data[0][0].shape[:2] if len(data) > 0 else (0, 0)
    print(f"Downloaded {len(data)} images with {_res} shape.")


    
    print("Started detection.")
    ready_data = []
    for tc, mask, _ in data:
        ready_data.append((tc, mask))
    detector_obj = None
    if m_detector == "net":
        detector_obj = fguard.vision.UNetDetector("MODEL.pth")
    else:
        detector_obj = fguard.vision.KMeansDetector(c_blur=9)
    result_array = detector_obj.pipe_detect_deforestation(ready_data)



    print("Started processing.")
#     df_handler = fguard.handler.DeforestationHandler(result_array)
#     result_clusters, unique_comp_cnt, events = df_handler()
    result_clusters = result_array


    print("Saving.")
    acg_list = fguard.utils.AutoColorGenerator(bottom_limit=150)
    array_images = fguard.utils.cluster_arrays_to_images(result_clusters, acg_list)
    max_number_length = len(str(len(array_images)))
    for i in tqdm.tqdm(range(len(array_images)), desc="Saving images", unit="pair"):
        rgba_mask = PIL.Image.fromarray(array_images[0]).convert("RGBA")
        pix = rgba_mask.load()
        for x in range(rgba_mask.width):
            for y in range(rgba_mask.height):
                if pix[x, y][:3] == (0, 0, 0):
                    pix[x, y] = (0, 0, 0, 0)
        rgba_real = PIL.Image.fromarray(data[i][0]).convert("RGBA")
        final_image = PIL.Image.blend(rgba_real, rgba_mask, 1)
        final_image.save(
            os.path.join(
                output_folder,
                "img-" + fguard.utils.get_right_aligned_number(
                    i,
                    max_number_length,
                ) + "-" + data[i][2].strftime('%Y-%m-%d') + ".png",
            ),
            "png",
        )
#         cur_cluster = PIL.Image.fromarray(array_images[i])
#         cur_cluster.save(
#             os.path.join(
#                 output_folder,
#                 "cluster_" + utils.get_right_aligned_number(i, max_number_length) + ".png"
#             ),
#             "png",
#         )
#         cur_tc = PIL.Image.fromarray(data[i][0])
#         cur_tc.save(
#             os.path.join(
#                 output_folder,
#                 "normal_img_" + utils.get_right_aligned_number(i, max_number_length) + ".png"
#             ),
#             "png",
#         )
    return
    dict_events = list()
    for event in events:
        dict_events.append(event.get_dict())
    info_obj = dict()
    info_obj["colors"] = list()
    info_obj["timestamps"] = list()
    for i in range(len(data)):
        info_obj["timestamps"].append(str(data[i][2]))
    for i in range(unique_comp_cnt):
        info_obj["colors"].append(acg_list[i])
    info_obj["events"] = dict_events
    with open(os.path.join(output_folder, "info.json"), "w") as f:
        json.dump(info_obj, f, indent=4)
    print(f"Saved results in '{output_folder}' directory")


def main():
    cli.add_command(config)
    cli.add_command(request)
    cli()

if __name__ == "__main__":
    main()



#     if not os.path.exists(OUTPUT_FOLDER):
#         os.makedirs(OUTPUT_FOLDER)
#     utils.remove_folder_content(OUTPUT_FOLDER)
#     comclient = communication.CommunicationClient(
#         sh_client_id=SH_CLIENT_ID,
#         sh_client_secret=SH_CLIENT_SECRET,
#         cache_folder=EOLEARN_CACHE_FOLDER,
#     )
#     coords = (
#         42.65614,
#         59.58242,
#         42.73579,
#         59.52529,
#     )
#     time_interval = (datetime.datetime(2017, 12, 1), datetime.datetime(2023, 2, 28))
# 
# 
# 
#     print("Started downloading data...")
#     print("(Please wait. This may take no more than 2 minutes)")
#     data = comclient.get_winter_data(
#         coords=coords,
#         time_interval=time_interval,
#         resolution=60,
#     )
#     _res = data[0][0].shape[:2] if len(data) > 0 else (0, 0)
#     print(f"Downloaded data successfully -> got {len(data)} images with {_res} shape")
# 
# 
#     
#     print("Started detection...")
#     ready_data = []
#     for tc, mask, _ in data:
#         ready_data.append((tc, mask))
#     detector = vision.Detector()
#     result_array = detector.pipe_detect_deforestation(ready_data)
#     assert len(result_array) > 0, "vision is trash"
#     print("Detected successfully")
# 
# 
# 
#     print("Started processing...")
#     df_handler = handler.DeforestationHandler(result_array)
#     result_clusters, unique_comp_cnt, events = df_handler()
#     assert len(result_clusters) > 0, "handler is trash"
#     print("Processed successfully")
# 
# 
#     print("Saving...")
#     acg_list = utils.AutoColorGenerator()
#     array_images = utils.cluster_arrays_to_images(result_clusters, acg_list)
#     max_number_length = len(str(len(array_images)))
#     for i in range(len(array_images)):
#         cur_cluster = PIL.Image.fromarray(array_images[i])
#         cur_cluster.save(
#             os.path.join(
#                 OUTPUT_FOLDER,
#                 "cluster_" + utils.get_right_aligned_number(i, max_number_length) + ".png"
#             ),
#             "png",
#         )
#         cur_tc = PIL.Image.fromarray(data[i][0])
#         cur_tc.save(
#             os.path.join(
#                 OUTPUT_FOLDER,
#                 "normal_img_" + utils.get_right_aligned_number(i, max_number_length) + ".png"
#             ),
#             "png",
#         )
# 
#     dict_events = list()
#     for event in events:
#         dict_events.append(event.get_dict())
#     info_obj = dict()
#     info_obj["colors"] = list()
#     info_obj["timestamps"] = list()
#     for i in range(len(data)):
#         info_obj["timestamps"].append(str(data[i][2]))
#     for i in range(unique_comp_cnt):
#         info_obj["colors"].append(acg_list[i])
#     info_obj["events"] = dict_events
#     with open(os.path.join(OUTPUT_FOLDER, "info.json"), "w") as f:
#         json.dump(info_obj, f, indent=4)
#     print(f"Saved result in '{OUTPUT_FOLDER}' directory")
