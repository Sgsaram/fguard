import datetime
import json
import os
import PIL.Image
import shutil
import tomllib

import click
import dotenv
import numpy as np
import oauthlib.oauth2.rfc6749.errors
import platformdirs
import sentinelhub
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
@click.option("-s", "--size", nargs=2, type=int)
@click.option("-i", "--isolate", is_flag=True)
# @click.option("-l", "--landsat", type=str)
@click.argument("output-folder", type=click.Path(exists=True, dir_okay=True), required=True)
def request(coordinates, time, file, detector, size, isolate, output_folder):
    id = None
    token = None
    m_size = (256, 256)
    if not os.path.exists(CONFIG_FILE):
        print("You didn't specify credentials. Use `fguard config`.")
        exit(-1)
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
                if "size" in data["request"]:
                    m_size = data["request"]["size"]
    if coordinates is not None:
        coords = coordinates
    if time is not None:
        timestamp = time
    if detector is not None:
        m_detector = detector
    if size is not None:
        m_size = size
    if coords is None:
        print("Coordinates weren't specified.")
        exit(-1)
    if timestamp is None:
        print("Time wasn't specified.")
        exit(-1)
    coords = (coords[1], coords[0], coords[3], coords[2])




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
    try:
        data = comclient.get_winter_data(
            coords=coords,
            time_interval=time_interval,
            size=m_size,
            data_collection="sentinel2_l1c",
        )
    except sentinelhub.exceptions.OutOfRequestsException:
        print("No processing units on account.")
        exit(-1)
    except oauthlib.oauth2.rfc6749.errors.InvalidClientError:
        print("Account with that id wasn't found.")
        exit(-1)
    except oauthlib.oauth2.rfc6749.errors.UnauthorizedClientError:
        print("Account with that token wasn't found.")
        exit(-1)
    except:
        print("Something went wrong while downloading data")
        exit(-1)
    _res = data[0][0].shape[:2] if len(data) > 0 else (0, 0)
    print(f"Downloaded {len(data)} images with {_res} shape and photo shooting timestamps:")
    for ind, da in enumerate(data):
        print(str(ind + 1) + " --> " + str(da[2]))
    
    print("Started detection", end=" ")
    ready_data = []
    for tc, mask, _ in data:
        ready_data.append((tc, mask))
    detector_obj = None
    if m_detector == "net":
        print("using neural networks.")
        detector_obj = fguard.vision.UNetDetector("MODEL.pth")
    else:
        print("using unsupervised clustering.")
        detector_obj = fguard.vision.KMeansDetector(c_blur=9)
    result_array = detector_obj.pipe_detect_deforestation(ready_data)



    print("Started processing.")
    df_handler = fguard.handler.DeforestationHandler(result_array)
    result_clusters, unique_comp_cnt, events = df_handler()


    print("Started exporting.")
    acg_list = fguard.utils.AutoColorGenerator(bottom_limit=100, upper_limit=200)
    array_images = fguard.utils.cluster_arrays_to_images(result_clusters, acg_list)
    max_number_length = len(str(len(array_images)))
    for i in tqdm.tqdm(range(len(array_images)), desc="Saving images", unit="pair"):
        rgba_mask = PIL.Image.fromarray(array_images[i]).convert("RGBA")
        pix = rgba_mask.load()
        for x in range(rgba_mask.width):
            for y in range(rgba_mask.height):
                if pix[x, y][:3] == (0, 0, 0):
                    pix[x, y] = (0, 0, 0, 0)
                else:
                    cur = list(pix[x, y])
                    cur[3] = round(1 * 255)
                    pix[x, y] = tuple(cur)
        rgba_real = PIL.Image.fromarray(data[i][0]).convert("RGBA")
        if isolate:
            rgba_real.save(
                os.path.join(
                    output_folder,
                    "real-" + data[i][2].strftime('%Y-%m-%d') + ".png",
                ),
                "png",
            )
            rgba_mask.save(
                os.path.join(
                    output_folder,
                    "mask-" + data[i][2].strftime('%Y-%m-%d') + ".png",
                ),
                "png",
            )
        else:
            final_image = PIL.Image.alpha_composite(rgba_real, rgba_mask)
            final_image.save(
                os.path.join(
                    output_folder,
                    "img-" + data[i][2].strftime('%Y-%m-%d') + ".png",
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
    print(f"Saved results in '{output_folder}' directory.")
    print("For more information about received images check `info.json`.")


@click.command()
@click.option("--cache", is_flag=True)
@click.option("--config", is_flag=True)
def remove(cache, config):
    if cache and os.path.exists(EOLEARN_CACHE_FOLDER):
        shutil.rmtree(EOLEARN_CACHE_FOLDER)
    if config and os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)


def main():
    try:
        cli.add_command(config)
        cli.add_command(request)
        cli.add_command(remove)
        cli()
    except Exception:
        print("Something went wrong")

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
