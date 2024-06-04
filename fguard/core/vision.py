import typing

import cv2 as cv
import numpy as np
import PIL.Image
import onnxruntime as ort
import tqdm



class Detector:
    def __init__(self) -> None:
        pass


class KMeansDetector(Detector):
    def __init__(
        self,
        c_delta = 20,
        c_blur = 9,
    ) -> None:
        self.c_delta = c_delta
        self.c_blur = c_blur

    @staticmethod
    def _get_gray_scale(a: np.ndarray):
        """
        Variable "a" format is rgb
        """
        return np.round(
            0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2],
        )

    def detect(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        k = 2
        blured_image = cv.medianBlur(image, self.c_blur)
        line_element = np.zeros(image.shape[:2], dtype=np.uint32)
        res_image = np.zeros(image.shape[:2], dtype=bool)
        image_line = []
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if mask[x, y][0]:
                    line_element[x, y] = len(image_line)
                    image_line.append(np.float32(blured_image[x, y]))
        image_line = np.array(image_line)
        if len(image_line) == 0:
            return res_image
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, centers = cv.kmeans(
            image_line,
            k,
            typing.cast(np.ndarray, None),
            criteria,
            20,
            cv.KMEANS_RANDOM_CENTERS,
        )
        gray_centers = np.zeros(k, dtype=np.float32)
        for i in range(len(centers)):
            gray_centers[i] = KMeansDetector._get_gray_scale(centers[i].astype(np.uint8))
        truth_center = max(gray_centers)
        if abs(gray_centers[0] - gray_centers[1]) < self.c_delta:
            return res_image
        line_res = gray_centers[labels.flatten()] #pyright: ignore
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if mask[x, y][0] and line_res[line_element[x, y]] == truth_center:
                    res_image[x, y] = True
        return res_image
    
    def pipe_detect_deforestation(
        self,
        data: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        final_masks = []
        for tc_image, mask in tqdm.tqdm(data, desc="Detecting deforestation", unit="image"):
            cur_mask = self.detect(tc_image, mask)
            if len(final_masks) > 0:
                cur_mask |= final_masks[-1]
            final_masks.append(cur_mask)
        return np.array(final_masks)


class UNetDetector(Detector):
    def __init__(
        self,
        model_path: str,
    ) -> None:
        self.ort_session = ort.InferenceSession(model_path)

    def detect(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        img = np.copy(image)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if not mask[x, y][0]:
                    img[x, y] = (0, 0, 0)

        old_size = (image.shape[1], image.shape[0])
        new_size = (256, 256)
        img = PIL.Image.fromarray(img)
        img = img.resize(new_size, PIL.Image.Resampling.BICUBIC)
        img = np.array(img, dtype=np.float32)
        
        img = img.transpose((2, 0, 1))
        if (img > 1).any():
            img = img / 255.0
        img = np.expand_dims(img, 0)

        output = self.ort_session.run(None, {"input": img})[0]

#       mid_img = PIL.Image.fromarray(output)
#       mid_img = mid_img.resize(old_size, PIL.Image.Resampling.BILINEAR)
#       mid_arr = np.array(mid_img)
        res_mask = np.argmax(output, axis=1)

        res = np.squeeze(res_mask[0])
        res = (res * 255).astype(np.uint8)
        res = PIL.Image.fromarray(res)
        res = res.resize(old_size, PIL.Image.Resampling.BICUBIC)
        res = np.array(res)

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if not mask[x, y][0]:
                    res[x][y] = 0

        return res > 0

    def pipe_detect_deforestation(
        self,
        data: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        final_masks = []
        for tc_image, mask in tqdm.tqdm(data, desc="Detecting deforestation", unit="image"):
            cur_mask = self.detect(tc_image, mask)
            if len(final_masks) > 0:
                cur_mask |= final_masks[-1]
            final_masks.append(cur_mask)
        return np.array(final_masks)

