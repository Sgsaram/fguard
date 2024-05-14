import datetime
import typing

import eolearn.core
import eolearn.io
import numpy as np
import sentinelhub as sh
import tqdm


class CommunicationClient:
    class __AddValidDataMaskTask(eolearn.core.eotask.EOTask):
        def execute(self, eopatch: eolearn.core.eodata.EOPatch): # pyright: ignore
            eopatch.mask["validMask"] = eopatch.mask["dataMask"].astype(
                bool
            ) & ~eopatch.mask["CLM"].astype(bool)
            return eopatch

    __STR_TO_DATA_COLLECTION = {
        "sentinel2_l1c": sh.data_collections.DataCollection.SENTINEL2_L1C,
        "sentinel2_l2a": sh.data_collections.DataCollection.SENTINEL2_L2A,
    }

    def __init__(
        self,
        sh_client_id: str,
        sh_client_secret: str,
        cache_folder: str | None = None,
    ) -> None:
        self.config = sh.config.SHConfig(
            sh_client_id=sh_client_id,
            sh_client_secret=sh_client_secret,
            use_defaults=True,
        )
        self.cache_folder = cache_folder

    @staticmethod
    def _get_last_month_day(
        date: datetime.datetime,
    ) -> datetime.datetime:
        next_month = date.replace(day=28) + datetime.timedelta(days=4)
        return next_month - datetime.timedelta(days=next_month.day)

    @staticmethod
    def _get_winter_intervals(
        time_interval: tuple[datetime.datetime, datetime.datetime]
    ) -> list[tuple[datetime.datetime, datetime.datetime]]:
        """
        On [from, to) <- half interval
        """
        from_date, to_date = time_interval
        res = list()
        for cur_year in range(from_date.year - 1, to_date.year):
            winter_first_day = max(from_date, datetime.datetime(cur_year, 12, 1))
            sec_value = CommunicationClient._get_last_month_day(datetime.datetime(cur_year + 1, 2, 1))
            winter_last_day = min(
                to_date,
                sec_value + datetime.timedelta(days=1),
            )
            if from_date <= winter_first_day and winter_last_day <= to_date and winter_first_day <= winter_last_day:
                res.append((winter_first_day, winter_last_day))
        return res

    @staticmethod
    def _split_time_intervals(
        intervals: list[tuple[datetime.datetime, datetime.datetime]],
        parts: int = 5,
    ) -> list[tuple[datetime.datetime, datetime.datetime]]:
        final_intervals = list()
        for inter in intervals:
            all_days = (inter[1] - inter[0]).days
            parts = min(parts, all_days)
            inter_days = datetime.timedelta(days=all_days // parts)
            cur_day = inter[0]
            for _ in range(parts - 1):
                final_intervals.append((cur_day, cur_day + inter_days))
                cur_day += inter_days
            final_intervals.append((cur_day, inter[1]))
        return final_intervals


    def get_data_otp(
        self,
        coords: tuple[float, float, float, float],
        time_interval: tuple[datetime.datetime, datetime.datetime],
        data_collection: str = "sentinel2_l2a",
        resolution: float | None = None,
        size: tuple[int, int] | None = None,
        time_difference: datetime.timedelta = datetime.timedelta(hours=12),
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns array of tuples containing true color image,
        merge of two valid masks (cloud coverage + data zones)
        and timestamps
        in one time period data collection
        """
        aoi_bbox = sh.geometry.BBox(
            bbox=coords,
            crs=sh.constants.CRS.WGS84,
        )
        input_task = eolearn.io.sentinelhub_process.SentinelHubInputTask(
            data_collection=CommunicationClient.__STR_TO_DATA_COLLECTION[
                data_collection
            ],
            bands=["B04", "B03", "B02"],
            bands_feature=(eolearn.core.constants.FeatureType.DATA, "sentinel_data"),
            additional_data=[
                (eolearn.core.constants.FeatureType.MASK, "dataMask"),
                (eolearn.core.constants.FeatureType.MASK, "CLM"),
            ],
            size=size,
            resolution=resolution,
            time_difference=time_difference,
            config=self.config,
            max_threads=5,
            mosaicking_order=sh.constants.MosaickingOrder.LEAST_RECENT,
            maxcc=1,
            cache_folder=self.cache_folder,
        )
        add_valid_data_task = CommunicationClient.__AddValidDataMaskTask()
        output_task = eolearn.core.eoworkflow_tasks.OutputTask("eopatch")
        input_node = eolearn.core.eonode.EONode(input_task)
        add_valid_data_node = eolearn.core.eonode.EONode(
            add_valid_data_task,
            inputs=[input_node],
        )
        output_node = eolearn.core.eonode.EONode(
            output_task,
            inputs=[add_valid_data_node],
        )
        workflow = eolearn.core.eoworkflow.EOWorkflow(
            [
                input_node,
                add_valid_data_node,
                output_node,
            ],
        )
        result = workflow.execute(
            {
                input_node: {
                    "bbox": aoi_bbox,
                    "time_interval": time_interval,
                },
            },
        )
        v_min = np.vectorize(min)
        eopatch: eolearn.core.eodata.EOPatch = typing.cast(
            eolearn.core.eodata.EOPatch,
            result.outputs["eopatch"]
        )
        if len(eopatch.data["sentinel_data"]) == 0:
            return list()
        return list(
            zip(
                v_min(eopatch.data["sentinel_data"] * 255, 255).astype(np.uint8),
                eopatch.mask["validMask"],
                np.array(eopatch.timestamps),
            )
        )

    def get_data_mtp(
        self,
        coords: tuple[float, float, float, float],
        time_intervals: list[tuple[datetime.datetime, datetime.datetime]],
        data_collection: str = "sentinel2_l2a",
        resolution: float | None = None,
        size: tuple[int, int] | None = None,
        time_difference: datetime.timedelta = datetime.timedelta(hours=12),
        verbose: bool = True,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns array of tuples containing true color image,
        merge of two valid masks (cloud coverage + data zones)
        and timestamps
        in many time periods data collection
        """
        res = []
        if verbose:
            time_intervals = CommunicationClient._split_time_intervals(time_intervals, parts=10)
            for a, b in time_intervals:
                print(a, b)
            for timep in tqdm.tqdm(time_intervals, desc="Downloading data", unit="batch"):
                res.extend(
                    self.get_data_otp(
                        coords,
                        timep,
                        data_collection,
                        resolution,
                        size,
                        time_difference,
                    )
                )
        else:
            for timep in time_intervals:
                res.extend(
                    self.get_data_otp(
                        coords,
                        timep,
                        data_collection,
                        resolution,
                        size,
                        time_difference,
                    )
                )
        return res
    
    def get_winter_data(
        self,
        coords: tuple[float, float, float, float],
        time_interval: tuple[datetime.datetime, datetime.datetime],
        data_collection: str = "sentinel2_l2a",
        resolution: float | None = None,
        size: tuple[int, int] | None = None,
        time_difference: datetime.timedelta = datetime.timedelta(hours=12),
        verbose: bool = True,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns array of tuples containing true color image,
        merge of two valid masks (cloud coverage + data zones)
        and timestamps
        in only winter time periods data collection
        """
        winter_intervals = CommunicationClient._get_winter_intervals(time_interval)
        return self.get_data_mtp(
            coords,
            winter_intervals,
            data_collection,
            resolution,
            size,
            time_difference,
            verbose,
        )

