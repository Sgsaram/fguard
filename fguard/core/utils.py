import os
import random

import numpy as np
import tqdm


def remove_folder_content(
    path: str,
) -> None:
    """
    EXAMPLE:
    remove_folder_content(OUTPUT_IMAGES_DIR)
    """
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

def get_right_aligned_number(
    n: int,
    length: int,
) -> str:
    res_str = str(n)[::-1]
    assert(len(res_str) <= length)
    res_str += ('0' * (length - len(res_str)))
    return res_str[::-1]

class AutoColorGenerator:
    def __init__(
        self,
        zero_cluster_color: tuple[int, int, int] = (0, 0, 0),
        bottom_limit: int = 20,
        upper_limit: int = 235,
    ) -> None:
        self.size = 1
        self.zero_cluster_color = zero_cluster_color
        self.colors = [self.zero_cluster_color]
        self.bottom_limit = bottom_limit
        self.upper_limit = upper_limit

    def __add_elements(
        self,
        cnt,
    ) -> None:
        for _ in range(cnt):
            cur_color = (
                random.randint(self.bottom_limit, self.upper_limit),
                random.randint(self.bottom_limit, self.upper_limit),
                random.randint(self.bottom_limit, self.upper_limit),
            )
            self.colors.append(cur_color)
        self.size += cnt

    def __getitem__(
        self,
        key: int,
    ) -> tuple[int, int, int]:
        if key >= self.size:
            self.__add_elements(key + 1 - self.size)
        return self.colors[key]

acg_list = AutoColorGenerator()

def cluster_array_to_image(
    image: np.ndarray,
    colors: list[tuple] | AutoColorGenerator,
) -> np.ndarray:
    cur_new_image = np.zeros(
        image.shape[:2] + (3,),
        dtype=np.uint8,
    )
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for k in range(3):
                if isinstance(image[x, y], list) or isinstance(image[x, y], np.ndarray):
                    cur_val = image[x, y][0]
                else:
                    cur_val = image[x, y]
                cur_new_image[x, y, k] = colors[cur_val][k]
    return cur_new_image

def cluster_arrays_to_images(
    images: np.ndarray,
    colors: list[tuple] | AutoColorGenerator,
) -> np.ndarray:
    res = []
    for i in tqdm.tqdm(range(images.shape[0]), desc="Preparing images", unit="mask"):
        res.append(cluster_array_to_image(images[i], colors))
    return np.array(res)


# [
#     (0, 0, 0),
#     (249, 65, 68),
#     (144, 190, 109),
#     (249, 199, 79),
# ]
