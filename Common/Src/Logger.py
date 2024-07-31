import logging
from Common.Src.DeviceWrapper import DeviceWrapper

# https://jh-bk.tistory.com/40
class Logger():
    def __init__(self, device_wrapper : DeviceWrapper):
        self.device_wrapper = device_wrapper
        logging.basicConfig(level=logging.DEBUG)

    # TODO: (Train, Validation, Test), epoch, metric, 

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