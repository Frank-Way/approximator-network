""" Модуль с некоторыми утилитарными функциями """

import logging
import sys
from logging import Logger
from typing import Tuple, TextIO, Any, List

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor, dtype
from torch.types import Device

import constants


def linear_activation(x: Any) -> Any:
    """
    Линейная функция активации, которая возвращает переданный аргумент
    :param x: аргумент
    :return: результат (тот же аргумент)
    """
    return x


def extend_array(base: ndarray, factor: int, scale: float = 0.01) -> ndarray:
    """
    Расширение массива в заданное количество раз путём добавления шума с нормальным распределением (МО = 0)
    :param base: исходный массив
    :param factor: коэффииент расширения (1, 2, ...)
    :param scale: СКО нормального распределения
    :return: расширенный массив
    """
    rows, cols = base.shape
    result = np.zeros(shape=(rows * factor, cols))
    for i in range(rows):
        for j in range(factor):
            result[i * factor + j] = base[i] + np.random.normal(loc=0, scale=scale, size=cols)
    return result


def set_dtype_and_device(tensor: Tensor, data_type: dtype = torch.float32,
                         device: Device = torch.device("cpu")) -> Tensor:
    """
    Настройка тензора (присвоение типа и перенос на целевое устройство)
    :param tensor: исходный тензор
    :param data_type: тип данных тензора
    :param device: устройство для работы с тензорами
    :return: настроенный тензор
    """
    tensor = tensor.to(dtype=data_type)
    tensor = tensor.to(device)
    return tensor


def cartesian(arrays: Tuple[ndarray, ...]) -> ndarray:
    """
    Вычисление декартового произведения массивов
    :param arrays: одномерные массивы
    :return: декартово произведение arrays
    """
    n = 1
    for x in arrays:
        n *= x.size
    out = np.zeros((n, len(arrays)))

    for i in range(len(arrays)):
        m = int(n / arrays[i].size)
        out[:n, i] = np.repeat(arrays[i], m)
        n //= arrays[i].size

    n = arrays[-1].size
    for k in range(len(arrays) - 2, -1, -1):
        n *= arrays[k].size
        m = int(n / arrays[k].size)
        for j in range(1, arrays[k].size):
            out[j * m: (j + 1) * m, k + 1:] = out[0: m, k + 1:]
    return out


def ndarray1d_to_2d(a: ndarray, array_type: str = "col") -> ndarray:
    """
    Формирование двумерного массива из одномерного
    :param a: одномерный массив
    :param array_type: тип массива (строка "row" или столбец "col")
    :return: двумерный массив
    """
    if a.ndim == 1:
        if array_type == "col":
            return a.reshape(-1, 1)
        elif array_type == "row":
            return a.reshape(1, -1)
    else:
        return a


def tensor_to_str(a: Tensor) -> str:
    """
    Формирование строки только со значениями элементов тензора
    :param a: тензор
    :return: строка значений тензора
    """
    string = str(a)
    prefix = "tensor("
    if string.startswith(prefix):
        string = string.replace(prefix, "").replace(",\n" + " " * len(prefix), ",\n")
        string = string[:string.rfind("]") + 1]
    return string


def _get_file_handler(filename: str, level: str = "DEBUG") -> logging.FileHandler:
    """
    Получение обработчика для вывода в файл
    :param filename: имя файла
    :param level: уровень логирования
    :return: обработчик
    """
    handler = logging.FileHandler(filename=filename)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(constants.LOG_FORMAT))
    return handler


def _get_stream_handler(stream: TextIO = sys.stdout, level: str = "DEBUG") -> logging.StreamHandler:
    """
    Получение обработчика для вывода в поток
    :param stream: поток
    :param level: уровень логирования
    :return: обработчик
    """
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(constants.LOG_FORMAT))
    return handler


def get_logger(name: str = "unnamedLogger", streams: List[TextIO] = None, filenames: List[str] = None,
               level: str = "DEBUG") -> Logger:
    """
    Получение логера для вывода в указанные файлы и потоки
    :param name: имя логера
    :param streams: потоки для вывода логов
    :param filenames: имена файлов для вывода логов
    :param level: уровень логирования
    :return: логер
    """
    logger = logging.getLogger(name=name)
    logger.setLevel(level)
    if filenames is not None:
        for filename in filenames:
            logger.addHandler(_get_file_handler(filename=filename, level=level))
    if streams is not None:
        for stream in streams:
            logger.addHandler(_get_stream_handler(stream=stream, level=level))
    return logger
