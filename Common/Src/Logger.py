import logging
from Common.Src.DeviceWrapper import DeviceWrapper
import pandas as pd
from pathlib import Path

# https://jh-bk.tistory.com/40
class Logger():
    def __init__(self, device_wrapper : DeviceWrapper, *args, **kwargs):
        logging.basicConfig(level=logging.DEBUG)

        parsed_args = kwargs["args"]

        self.excel_log_dir : Path = Path(parsed_args.root_dir) / "Model" / parsed_args.save_dir_name
        self.excel_log_dir.mkdir(parents=True, exist_ok=True)
        self.excel_log_path = self.excel_log_dir / "log.xlsx"

        self.device_wrapper : DeviceWrapper = device_wrapper
        self.df_map : dict = {}

    def append_to_excel(self, sheet : str, col : str, value) -> None:
        if not self.device_wrapper.is_multi_gpu_mode(): return
        if not self.device_wrapper.is_main_device(): return

        if sheet not in self.df_map.keys():
            self.df_map[sheet] = pd.DataFrame()
        df : pd.DataFrame = self.df_map[sheet]

        if not self.excel_log_path.exists():
            with pd.ExcelWriter(self.excel_log_path, mode="w") as f:
                df.to_excel(f, sheet_name=sheet, index=False)

        if col not in df.columns:
            df.loc[0, col] = value
            with pd.ExcelWriter(self.excel_log_path, mode="a", if_sheet_exists="overlay") as f:
                df.to_excel(f, sheet_name=sheet, index=False)
        else:
            index : int = len(df[col].dropna())
            df.loc[index, col] = value
            with pd.ExcelWriter(self.excel_log_path, mode="a", if_sheet_exists="overlay") as f:
                df.to_excel(f, sheet_name=sheet, index=False)
        

    def debug(self, log : str):
        if self.device_wrapper.is_multi_gpu_mode():
            if self.device_wrapper.is_main_device():
                logging.debug(log)
        else:
            logging.debug(log)

    def info(self, log : str):
        if self.device_wrapper.is_multi_gpu_mode():
            if self.device_wrapper.is_main_device():
                logging.info(log)
        else:
            logging.info(log)

    def warning(self, log : str):
        if self.device_wrapper.is_multi_gpu_mode():
            if self.device_wrapper.is_main_device():
                logging.warning(log)
        else:
            logging.warning(log)

    def error(self, log : str):
        if self.device_wrapper.is_multi_gpu_mode():
            if self.device_wrapper.is_main_device():
                logging.error(log)
        else:
            logging.error(log)

    def critical(self, log : str):
        if self.device_wrapper.is_multi_gpu_mode():
            if self.device_wrapper.is_main_device():
                logging.critical(log)
        else:
            logging.critical(log)