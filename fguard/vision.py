import typing

import cv2 as cv
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import tqdm

from fguard.unet import UNet


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
        bilinear: bool = False,
        out_threshold: float = 0.5,
    ) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, map_location=self.device)
        state_dict.pop("mask_values")
        self.net = UNet(n_channels=3, n_classes=2, bilinear=bilinear)
        self.net.to(device=self.device)
        self.net.load_state_dict(state_dict)
        self.out_threshold = out_threshold

    def detect(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        self.net.eval()
        img = np.copy(image)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if not mask[x, y][0]:
                    img[x, y] = (0, 0, 0)

        old_size = (image.shape[1], image.shape[0])
        new_size = (256, 256)
        img = PIL.Image.fromarray(img)
        img = img.resize(new_size, PIL.Image.Resampling.BICUBIC)
        img = np.array(img)
        
        img = img.transpose((2, 0, 1))
        if (img > 1).any():
            img = img / 255.0
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(img).cpu()
            output = F.interpolate(output, (image.shape[1], image.shape[0]), mode='bilinear')
            if self.net.n_classes > 1:
                mask = output.argmax(dim=1)
            else:
                mask = torch.sigmoid(output) > self.out_threshold


        res = mask[0].long().squeeze().numpy()
        res = (res * 255).astype(np.uint8)
        res = PIL.Image.fromarray(res)
        res = res.resize(old_size, PIL.Image.Resampling.NEAREST)
        res = np.array(res)
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
