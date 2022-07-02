""" Модуль с описанием различных классов """

from __future__ import annotations
import logging
from typing import List, Callable, NamedTuple, Union, Tuple, Generator, Any

import numpy as np
import sympy
import torch
from torch import Tensor, nn, dtype
from torch.nn.modules import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.types import Device

import constants
from exceptions import EmptyValueException, ValuesRelationException, WrongFormatException, RangeMismatchException, \
    FunctionExtendingException
from utils import get_logger, cartesian, set_dtype_and_device, ndarray1d_to_2d, extend_array


class LayerConfiguration(NamedTuple):
    """ Конфигурация слоя нейросети """
    # inputs: int  # количество входов
    neurons: int  # размер слоя (количество нейронов)
    module: Module  # тип слоя
    activation: Callable  # функция активации


class NetworkConfiguration(NamedTuple):
    """ Конфигурация нейросети """
    inputs: int  # количество входов
    layer_configurations: List[LayerConfiguration]  # конфигурации слоёв


class ApproximatorNetwork(nn.Module):
    """ Нейронная сеть - нейросетевой аппроксиматор """
    inputs: int  # количество входов
    modules: List[Module]  # слои нейросети
    activations: List[Callable]  # функции активации
    logger: logging.Logger = get_logger(name="ApproximatorNetwork",
                                        streams=constants.DEFAULT_LOG_STREAMS,
                                        filenames=constants.DEFAULT_LOG_FILENAMES,
                                        level=constants.DEFAULT_LOG_LEVEL)

    def __int__(self) -> None:
        """
        Конструктор
        """
        super(ApproximatorNetwork, self).__int__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Прямой проход нейросети
        :param x: исходные данные из обучающей выборки
        :return: выход нейросети
        """
        x = x.reshape(-1, self.inputs)
        for activation, module in zip(self.activations, self.modules):
            x = activation(module(x))
        return x

    def setup(self, configuration: NetworkConfiguration) -> None:
        """
        Настройка нейросети
        :param configuration: настройки
        """
        s = str(self).replace("\n", " ")
        self.logger.info(f"Начало настройки нейросети ({s})")

        self.inputs = configuration.inputs

        size_pairs = [(configuration.inputs, configuration.layer_configurations[0].neurons)]
        for i, layer_configuration in enumerate(configuration.layer_configurations[:-1]):
            size_pairs.append((layer_configuration.neurons, configuration.layer_configurations[i + 1].neurons))

        if not (len(size_pairs) == len(configuration.layer_configurations)):
            raise ValuesRelationException(
                f"Не получилось определить количество входов и выходов для каждого слоя нейросети")

        self.modules = []
        self.activations = []
        for i, (size_pair, layer_configuration) in enumerate(zip(size_pairs, configuration.layer_configurations)):
            self.modules.append(layer_configuration.module(*size_pair))
            self.activations.append(layer_configuration.activation)
            self.register_module(f"layer{i}", self.modules[-1])

        if not (len(self.modules) == len(self.activations) == len(configuration.layer_configurations)):
            raise ValuesRelationException(
                f"Не совпадают количество слоёв с указанным в конфигурации (получено len(modules)={len(self.modules)},"
                f" len(activations)={len(self.activations)}, layers_count={len(configuration.layer_configurations)})")

        s = str(self).replace("\n", " ")
        self.logger.info(f"Завершение настройки нейросети ({s})")


class FloatFrom0To1(tuple):
    """ Класс для представления числа с плавающей точкой в диапазоне [0; 1].
        Класс является обёрткой для кортежа, что делает его неизменяемым. """

    def __new__(cls, value: float):
        """
        Создание нового числа с проверкой принадлежности диапазону
        :param value: число
        """
        if value.__class__ == tuple:
            val = float(value[0])
        else:
            val = float(value)
        if 0 <= val <= 1:
            return tuple.__new__(cls, (val,))
        else:
            raise ValueError("Число должно находиться в пределах [0.0; 1.0]")

    @property
    def value(self):
        """ Получение значения """
        return tuple.__getitem__(self, 0)

    def __str__(self):
        """ Строковое представление """
        return "class: {c}, value: {v}".format(c=FloatFrom0To1.__name__, v=self.value)


class VariableLimits(NamedTuple):
    """ Границы изменения входной переменной для аппроксимируемой функции """
    left: float  # левая граница
    right: float  # правая граница


class GenerateFunctionArguments(NamedTuple):
    """ Аргументы метода ApproximatorDataset.generate_function """
    function: str  # строковое представление аппроксимируемой функции
    inputs: int  # количество входов аппроксимируемой функции
    size: int  # размер обучающей выборки (количество отсчётов по каждой входной переменной)
    limits: List[VariableLimits]  # границы изменения входных переменных
    extending: FloatFrom0To1 = FloatFrom0To1(0.0)  # доля расширения обучающей выборки
    testing: FloatFrom0To1 = FloatFrom0To1(0.3)  # доля выборки, которая резервируется для тестов
    validating: FloatFrom0To1 = FloatFrom0To1(0.1)  # доля выборки, которая резервируется для валидации
    data_type: dtype = torch.float32  # тип данных тензоров
    device: Device = torch.device("cpu")  # устройство для работы с тензорами


class MyDataset:
    """ Обучающая выборка """
    x: Tensor  # входные значения
    y: Tensor  # требуемые выходные значения

    def __init__(self, x: Tensor, y: Tensor) -> None:
        """
        Конструктор
        :param x: входные значения
        :param y: выходные значения
        """
        MyDataset._check_rows_count(x, y)

        self.x = x
        self.y = y

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Получение элемента выборки
        :param index: индекс
        :return: элемент выборки
        """
        return self.x[index], self.y[index]

    @staticmethod
    def _permute_data(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        (protected) Перемешивание выборки
        :param x: входные значения
        :param y: выходные значения
        :return: перемешанные входные и выходные значения
        """
        MyDataset._check_rows_count(x, y)

        indices = np.random.permutation(x.shape[0])
        return x[indices], y[indices]

    @staticmethod
    def _check_rows_count(x: Tensor, y: Tensor) -> None:
        """
        (protected) Проверка соответсвия размеров обучающей выборке по входам и выходам
        :param x: входные значения
        :param y: выходные значения
        """
        if not (x.shape[0] == y.shape[0]):
            raise ValuesRelationException(
                f"Количество строк обучающей выборки разное для входов и выходов (получено x={x.shape[0]} и"
                f" y={y.shape[0]}, соответственно)")

    @staticmethod
    def get_batch_generator(data: MyDataset, size: int = 32) -> Generator[Tuple[Tensor, Tensor], Any, Any]:
        """
        Получение генератора для разбиения обучающей выборки на пакеты фиксированного размера
        :param data: обучающая выборка
        :param size: размер пакета
        :return: генератор/итератор
        """
        x, y = data.x, data.y
        rows = x.shape[0]
        size = min(rows, size)
        for i in range(0, rows, size):
            x_permuted, y_permuted = MyDataset._permute_data(x, y)
            yield x_permuted[i: i + size], y_permuted[i: i + size]

    def to_str_table(self, precision: int = 4) -> str:
        """
        Получение строкового представления обучающей выборки в виде таблицы
        :param precision: точность (количество цифр после десятичной точки)
        :return: обучающая выборка в виде таблицы
        """
        x, x_rows, x_cols = self.x, self.x.shape[0], self.x.shape[1]
        y, y_rows, y_cols = self.y, self.y.shape[0], self.y.shape[1]
        MyDataset._check_rows_count(x, y)

        if not (len(x.shape) == 2):
            raise WrongFormatException(
                f"Размерность входов обучающей выборки должна равняться двум (получено x.shape={x.shape})")

        if not (precision >= 0):
            raise RangeMismatchException(
                f"Точноть представления чисел с плавающей точкой должна быть неотрицательной (получено "
                f"precision={precision})")

        x_strings, y_strings = [f"x{i + 1}" for i in range(x_cols)], [f"y{i + 1}" for i in range(y_cols)]

        width = min(15, precision + 10)

        header = ""
        for i in range(x_cols):
            header += f"{x_strings[i]: >{width}}"
        header += " |"
        for i in range(y_cols):
            header += f"{y_strings[i]: >{width}}"

        index = header.find("|")
        separator = "-" * len(header)
        separator = separator[:index] + "+" + separator[index + 1:]
        body = ""

        for i in range(x_rows):
            line = ""
            for j in range(x_cols):
                line += f"{x[i][j]:>{width}.{precision}f}"

            line += " |"
            for j in range(y_cols):
                line += f"{y[i][j]:>{width}.{precision}f}"
            body += f"{line}\n"
        return f"{header}\n{separator}\n{body}"


class ApproximatorDataset:
    """ Набор обучающих выборок для нейросетевого аппроксиматора """
    train_data: MyDataset  # данные для обучения
    test_data: MyDataset  # данные для тестов (оценка во время обучения)
    valid_data: MyDataset  # данные для валидации (оценка после обучения)
    logger: logging.Logger = get_logger(name="ApproximatorDataset",
                                        streams=constants.DEFAULT_LOG_STREAMS,
                                        filenames=constants.DEFAULT_LOG_FILENAMES,
                                        level=constants.DEFAULT_LOG_LEVEL)

    def __init__(self, train_data: MyDataset, test_data: MyDataset, valid_data: MyDataset) -> None:
        """
        Конструктор
        :param train_data: данные для обучения
        :param test_data: данные для тестов (оценка во время обучения)
        :param valid_data: данные для валидации (оценка после обучения)
        """
        self.train_data = train_data
        self.test_data = test_data
        self.valid_data = valid_data

    @staticmethod
    def generate_function(arguments: GenerateFunctionArguments) -> ApproximatorDataset:
        """
        Генерация обучающих выборок для воспроизведения аппроксимируемой математической функции
        :param arguments: аргументы
        :return: обучающие выборки
        """
        ApproximatorDataset.logger.info(f"Начало генерации обучающей выборки для параметров: {arguments}")

        if not (arguments.inputs > 0):
            raise RangeMismatchException(
                f"Количество аргументов функции должно быть больше нуля (получено inputs={arguments.inputs})")

        if not (len(arguments.function) > 0):
            raise EmptyValueException(f"Функция не заполнена (получено {arguments.function})")

        if not (arguments.size > arguments.inputs):
            raise ValuesRelationException(
                f"Размер выборки должен превышать количество входов функции (получено size={arguments.size} и "
                f"inputs={arguments.inputs}, соответственно)")

        if not (arguments.inputs == len(arguments.limits)):
            raise ValuesRelationException(
                f"Количество границ входных переменных должно соответствовать числу входов функции (получено "
                f"inputs={arguments.inputs} и len(limits)={len(arguments.limits)}, соответственно)")

        ApproximatorDataset.logger.debug("Начало получение функции с помощью SymPy")

        simplified_expression = sympy.simplify(expr=arguments.function.lower())
        ApproximatorDataset.logger.debug(f"Функция \"{arguments.function}\" преобразована в"
                                         f" \"{simplified_expression}\"")
        input_symbols = [sympy.Symbol(name=f"x{i + 1}") for i in range(arguments.inputs)]
        callable_function = sympy.lambdify(args=input_symbols, expr=simplified_expression)

        train_size = arguments.size
        test_size = int(arguments.size * arguments.testing.value)
        valid_size = int(arguments.size * arguments.validating.value)
        ApproximatorDataset.logger.debug(f"Размер выборки для обучения: {train_size}, тестов: {test_size}, "
                                         f"валидации: {valid_size}")

        x_train_list, x_train_extended_list, x_test_list, x_valid_list = [], [], [], []

        for i in range(arguments.inputs):
            left, right = arguments.limits[i]

            x_train_list.append(np.linspace(left, right, num=train_size))
            x_test_list.append(np.linspace(left, right, num=test_size))
            x_valid_list.append(np.linspace(left, right, num=valid_size))

            if arguments.extending.value:
                delta = np.abs(right - left)
                extended_left = left - arguments.extending.value * delta
                extended_right = right + arguments.extending.value * delta
                ApproximatorDataset.logger.debug(
                    f"Попытка расширения границ для {i + 1}-го входа с [{left}; {right}] до [{extended_left};"
                    f" {extended_right}]")
                x_train_extended_list.append(np.linspace(start=extended_left, stop=extended_right, num=train_size))

        x_train = cartesian(tuple(x_train_list))
        x_train_extended = cartesian(tuple(x_train_extended_list))
        x_test = cartesian(tuple(x_test_list))
        x_valid = cartesian(tuple(x_valid_list))

        y_train = callable_function(*[x_train[:, i] for i in range(arguments.inputs)])
        y_test = callable_function(*[x_test[:, i] for i in range(arguments.inputs)])
        y_valid = callable_function(*[x_valid[:, i] for i in range(arguments.inputs)])

        try:
            ApproximatorDataset.logger.debug("Попытка вычисления функции от переменных для расширенного диапазона")
            y_train_extended = callable_function(*[x_train_extended[:, i] for i in range(arguments.inputs)])
            x_train = x_train_extended
            y_train = y_train_extended
        except Union[ValueError, TypeError, AttributeError]:
            ranges = ", ".join(
                map(lambda t: f"x{t[0] + 1} = [{np.min(t[1])}; {np.max(t[1])}", enumerate(x_train_extended)))
            raise FunctionExtendingException(
                f"Не удалось расширить выборку для функции {simplified_expression} на интервалах: {ranges}")

        train_data = MyDataset(x=set_dtype_and_device(tensor=torch.tensor(ndarray1d_to_2d(x_train)),
                                                      data_type=arguments.data_type, device=arguments.device),
                               y=set_dtype_and_device(tensor=torch.tensor(ndarray1d_to_2d(y_train)),
                                                      data_type=arguments.data_type, device=arguments.device))

        test_data = MyDataset(x=set_dtype_and_device(tensor=torch.tensor(ndarray1d_to_2d(x_test)),
                                                     data_type=arguments.data_type, device=arguments.device),
                              y=set_dtype_and_device(tensor=torch.tensor(ndarray1d_to_2d(y_test)),
                                                     data_type=arguments.data_type, device=arguments.device))

        valid_data = MyDataset(x=set_dtype_and_device(tensor=torch.tensor(ndarray1d_to_2d(x_valid)),
                                                      data_type=arguments.data_type, device=arguments.device),
                               y=set_dtype_and_device(tensor=torch.tensor(ndarray1d_to_2d(y_valid)),
                                                      data_type=arguments.data_type, device=arguments.device))

        ApproximatorDataset.logger.info(f"Завершение генерации обучающей выборки для параметров: {arguments}")
        return ApproximatorDataset(train_data, test_data, valid_data)

    @staticmethod
    def generate_xor(train_factor: int = 16,
                     train_scale: float = 0.01,
                     test_factor: int = 4,
                     test_scale: float = 0.001,
                     data_type: dtype = torch.float32,
                     device: Device = torch.device("cpu")) -> ApproximatorDataset:
        """
        Генерация обучающих выборок для обучения нейросети на воспроизведение функции "ИСКЛЮЧАЮЩЕЕ ИЛИ"
        :param train_factor: во сколько раз расширить выборку для обучения
        :param train_scale: СКО для зашумления выборки для обучения
        :param test_factor: во сколько раз расширить выборку для тестов
        :param test_scale: СКО для зашумления выборки для тестов
        :param data_type: тип данных для тензоров
        :param device: устройство для работы с тензорами
        :return: обучающие выборки
        """
        x_array = np.array([[0.0, 0.0],
                            [0.0, 1.0],
                            [1.0, 0.0],
                            [1.0, 1.0]])

        y_array = np.array([[0.0],
                            [1.0],
                            [1.0],
                            [0.0]])

        valid_data = MyDataset(x=set_dtype_and_device(tensor=torch.tensor(x_array), data_type=data_type, device=device),
                               y=set_dtype_and_device(tensor=torch.tensor(y_array), data_type=data_type, device=device))

        test_data = MyDataset(x=set_dtype_and_device(tensor=torch.tensor(
            extend_array(base=x_array, factor=test_factor, scale=test_scale)), data_type=data_type, device=device),
                              y=set_dtype_and_device(tensor=torch.tensor(
            extend_array(base=y_array, factor=test_factor, scale=test_scale)), data_type=data_type, device=device))

        train_data = MyDataset(x=set_dtype_and_device(tensor=torch.tensor(
            extend_array(base=x_array, factor=train_factor, scale=train_scale)), data_type=data_type, device=device),
                               y=set_dtype_and_device(tensor=torch.tensor(
            extend_array(base=y_array, factor=train_factor, scale=train_scale)), data_type=data_type, device=device))

        return ApproximatorDataset(train_data, test_data, valid_data)

    @staticmethod
    def generate_random(inputs: int = 3,
                        outputs: int = 4,
                        size: int = 8,
                        train_factor: int = 16,
                        train_scale: float = 0.01,
                        test_factor: int = 4,
                        test_scale: float = 0.001,
                        data_type: dtype = torch.float32,
                        device: Device = torch.device("cpu")) -> ApproximatorDataset:
        """
        Генерация обучающих выборок для обучения нейросети на воспроизведение случайной логической функции
        :param inputs: количество входов
        :param outputs: количество выходов
        :param size: базовый размер выборки
        :param train_factor: во сколько раз расширить выборку для обучения
        :param train_scale: СКО для зашумления выборки для обучения
        :param test_factor: во сколько раз расширить выборку для тестов
        :param test_scale: СКО для зашумления выборки для тестов
        :param data_type: тип данных для тензоров
        :param device: устройство для работы с тензорами
        :return: обучающие выборки
        """
        x_array = np.random.binomial(n=1, p=0.5, size=(size, inputs)).astype(np.float32)
        y_array = np.random.binomial(n=1, p=0.5, size=(size, outputs)).astype(np.float32)

        valid_data = MyDataset(x=set_dtype_and_device(tensor=torch.tensor(x_array), data_type=data_type, device=device),
                               y=set_dtype_and_device(tensor=torch.tensor(y_array), data_type=data_type, device=device))

        test_data = MyDataset(x=set_dtype_and_device(tensor=torch.tensor(
            extend_array(x_array, test_factor, test_scale)), data_type=data_type, device=device),
                              y=set_dtype_and_device(tensor=torch.tensor(
            extend_array(y_array, test_factor, test_scale)), data_type=data_type, device=device))

        train_data = MyDataset(x=set_dtype_and_device(tensor=torch.tensor(
            extend_array(x_array, train_factor, train_scale)), data_type=data_type, device=device),
                               y=set_dtype_and_device(tensor=torch.tensor(
            extend_array(y_array, train_factor, train_scale)), data_type=data_type, device=device))

        return ApproximatorDataset(train_data, test_data, valid_data)


class TrainArguments:
    """ Аргументы метода Trainer.train """
    model: ApproximatorNetwork  # нейросеть для обучения
    loss_function: _Loss  # функция потерь
    data: ApproximatorDataset  # обучающая выборка
    optimizer: Optimizer  # оптимизатор
    epochs: int = 10000  # количество эпох обучения
    start_lr: float = 0.1  # начальная скорость обучения
    final_lr: float = 0.00001  # конечная скорость обучения
    momentum: FloatFrom0To1 = FloatFrom0To1(0.0)  # величина инерции
    batch_size: int = 32  # размер пакета
    queries: int = 10  # количество опросов во время обучения

    def __init__(self, model: ApproximatorNetwork, loss_function: _Loss, epochs: int = 10000, start_lr: float = 0.1,
                 final_lr: float = 0.00001, momentum: FloatFrom0To1 = FloatFrom0To1(0.0), batch_size: int = 32,
                 queries: int = 10) -> None:
        """
        Конструктор
        :param model: нейросеть для обучения
        :param loss_function: функция потерь
        :param epochs: количество эпох обучения
        :param start_lr: начальная скорость обучения
        :param final_lr: конечная скорость обучения
        :param momentum: величина инерции
        :param batch_size: размер пакета
        :param queries: количество опросов во время обучения
        """
        self.model = model
        self.loss_function = loss_function
        self.epochs = epochs
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.batch_size = batch_size
        self.queries = queries
        self.momentum = momentum

    def __repr__(self) -> str:
        """ Строковое представление """
        return str(self.__dict__).replace("\n", " ")


class TrainResults(NamedTuple):
    """ Промежуточные результаты обучения нейросети """
    test_query_results: List[float]  # результаты оценки на тестовых данных во время обучения


class Trainer:
    logger: logging.Logger = get_logger(name="Trainer",
                                        streams=constants.DEFAULT_LOG_STREAMS,
                                        filenames=constants.DEFAULT_LOG_FILENAMES,
                                        level=constants.DEFAULT_LOG_LEVEL)

    @staticmethod
    def train(arguments: TrainArguments) -> TrainResults:
        """
        Обучение нейросети
        :param arguments: аргументы
        :return: результаты
        """
        Trainer.logger.info(f"Начало обучения для параметров: {arguments}")

        # объект для снижения скорости обучения
        scheduler = ExponentialLR(optimizer=arguments.optimizer,
                                  gamma=np.power(arguments.final_lr / arguments.start_lr, 1.0 / (arguments.epochs - 1)),
                                  last_epoch=-1)

        # эпохи, на которых должна выполняться оценка результатов работы нейросети
        query_at = [e for e in range(1, arguments.epochs + 1) if e % (arguments.epochs // arguments.queries) == 0]
        Trainer.logger.debug(f"Опросы будут совершены в следующие эпохи: {query_at}")

        query_results = []
        for epoch in range(1, arguments.epochs + 1):  # запуск обучения
            training_loss = 0.0
            test_loss = 0.0
            arguments.model.train()  # перевод нейросети в режим обучения

            # итерация по пакетам обучающей выборки (данные для обучения)
            for batch in MyDataset.get_batch_generator(arguments.data.train_data, arguments.batch_size):

                arguments.optimizer.zero_grad()  # сброс градиентов
                inputs, targets = batch
                output = arguments.model(inputs)  # вычисление выходов (прямой проход)
                loss = arguments.loss_function(output, targets)  # оценка потерь
                loss.backward()  # вычисление градиентов (обратный проход)
                arguments.optimizer.step()  # корректировка параметров нейросети

                training_loss += loss.data.item() * inputs.size(0) / arguments.data.train_data.x.shape[0]
            scheduler.step()  # снижение скорости обучения

            if epoch in query_at:
                arguments.model.eval()  # перевод нейросети в режим оценки

                # итерация по пакетам обучающей выборки (данные для тестов)
                for batch in MyDataset.get_batch_generator(arguments.data.test_data, arguments.batch_size):
                    inputs, targets = batch
                    output = arguments.model(inputs)
                    loss = arguments.loss_function(output, targets)
                    test_loss += loss.data.item() * inputs.size(0) / arguments.data.test_data.x.shape[0]
                query_results.append(test_loss)

                Trainer.logger.info(f'Epoch: {epoch}, Training Loss: {training_loss:.6f},'
                                    f'Test Data Loss: {test_loss:.6f}')

        Trainer.logger.info(f"Завершение обучения для параметров: {arguments}")
        return TrainResults(test_query_results=query_results)


class RunConfiguration(NamedTuple):
    """ Конфигурация запуска программы """
    network_configuration: NetworkConfiguration  # параметры нейросети
    generate_function_arguments: GenerateFunctionArguments  # параметры обучающей выборки
    train_arguments: TrainArguments  # параметры обучения


class RunResults:
    """ Конечные результаты обучения нейросети """
    absolute_delta: float  # максимальная абсолютная ошибка
    relative_delta: float  # максимальная относительная ошибка
    mean_delta: float  # средняя абсолютная ошибка
    model: nn.Module  # нейросеть
    dataset: ApproximatorDataset  # обучающие выборки
    logger: logging.Logger = get_logger(name="RunResults",
                                        streams=constants.DEFAULT_LOG_STREAMS,
                                        filenames=constants.DEFAULT_LOG_FILENAMES,
                                        level=constants.DEFAULT_LOG_LEVEL)

    def __init__(self, absolute_delta: float, relative_delta: float, mean_delta: float, model: nn.Module = None,
                 dataset: ApproximatorDataset = None):
        """
        Конструктор
        :param absolute_delta: максимальная абсолютная ошибка
        :param relative_delta: максимальная относительная ошибка
        :param mean_delta: средняя абсолютная ошибка
        :param model: нейросеть
        :param dataset: обучающие выборки
        """
        self.absolute_delta = absolute_delta
        self.relative_delta = relative_delta
        self.mean_delta = mean_delta
        self.model = model
        self.dataset = dataset

    @staticmethod
    def build_blank() -> RunResults:
        """
        Получение пустых результатов
        :return: пустые результаты
        """
        return RunResults(absolute_delta=float("+inf"), relative_delta=float("+inf"), mean_delta=float("+inf"))

    @staticmethod
    def build_from_model(network: nn.Module, dataset: ApproximatorDataset) -> RunResults:
        """
        Получение результатов для обученной нейросети
        :param network: нейросеть
        :param dataset: обучающие выборки
        :return: результаты обучения
        """
        RunResults.logger.info("Начало получения результатов по окончании обучения")

        predictions = network(dataset.valid_data.x)  # вычисление выхода нейросети
        errors = predictions - dataset.valid_data.y  # получение ошибок
        absolute_errors = torch.abs(errors)  # получение абсолютных ошибок

        # диапазон значений в выходе нейросети
        range_length = torch.abs(torch.max(predictions) - torch.min(predictions)).item()
        # вычисление ошибок
        absolute_delta = torch.max(errors).item()
        relative_delta = absolute_delta / range_length * 100
        mean_delta = torch.mean(absolute_errors).item()

        RunResults.logger.info("Завершение получения результатов по окончании обучения")
        return RunResults(absolute_delta=absolute_delta, relative_delta=relative_delta, mean_delta=mean_delta,
                          model=network, dataset=dataset)

    def deltas_to_str(self, verbose: bool = False) -> str:
        """
        Получение строки с ошибками
        :param verbose: нужно ли словесное описание
        :return: строкове представление ошибок
        """
        if verbose:
            return f"максимальная абсолютная ошибка = {self.absolute_delta};" \
                   f"\tмаксимальная относительная ошибка = {self.relative_delta} %;" \
                   f"\tсредняя абсолютная ошибка = {self.mean_delta}"
        return f"absolute_delta = {self.absolute_delta};" \
               f"\trelative_delta = {self.relative_delta} %;" \
               f"\tmean_delta = {self.mean_delta}"

    def to_str_table(self, part: FloatFrom0To1 = FloatFrom0To1(1.0), precision: int = 4) -> str:
        """
        Получение строки с результатами обучения в виде таблицы
        :param part: какая часть таблицы должна быть выведена
        :param precision: точность (количество цифр после десятичной точки)
        :return: результаты обучения в виде таблицы
        """
        if not (self.model is not None and self.dataset is not None):
            raise EmptyValueException(
                f"Не задана нейросеть или набор данных (получено model={self.model} и"
                f" dataset={self.dataset}, соответственно)")

        x = self.dataset.valid_data.x
        y = self.model(x)
        t = self.dataset.valid_data.y
        e = torch.abs(y - t)

        if not (x.shape[0] == y.shape[0] == t.shape[0]):
            raise ValuesRelationException(
                f"Не соответствуют количество строк в обучающей выборке и результате (получено x={x.shape[0]},"
                f" y={y.shape[0]}, t={t.shape[0]})")

        if not (y.shape == t.shape == e.shape):
            raise ValuesRelationException(
                f"Не соответствуют размеры обучающей выборки и результата (получено y={y.shape}, t={t.shape})")

        if not (len(x.shape) == 2):
            raise WrongFormatException(
                f"Размерность входов обучающей выборки должна равняться двум (получено x.shape={x.shape})")

        if not (precision >= 0):
            raise RangeMismatchException(
                f"Точноть представления чисел с плавающей точкой должна быть неотрицательной (получено "
                f"precision={precision})")

        x_rows, x_cols = x.shape
        y_rows, y_cols = y.shape
        t_rows, t_cols = t.shape
        e_rows, e_cols = e.shape
        x_strings = [f"x{i + 1}" for i in range(x_cols)]
        y_strings = [f"y{i + 1}" for i in range(y_cols)]
        t_strings = [f"t{i + 1}" for i in range(t_cols)]
        e_strings = [f"e{i + 1}" for i in range(e_cols)]
        width = min(15, precision + 10)
        header = ""
        for i in range(x_cols):
            header += f"{x_strings[i]:>{width}}"
        header += " |"
        for i in range(y_cols):
            header += f"{y_strings[i]:>{width}}{t_strings[i]:>{width}}{e_strings[i]:>{width}}"
        index = header.find("|")
        separator = "-" * len(header)
        separator = separator[:index] + "+" + separator[index + 1:]
        step = int(1 / part.value) if part.value else 1
        body = ""
        for i in range(0, x_rows, step):
            line = ""
            for j in range(x_cols):
                line += f"{x[i][j]:>{width}.{precision}f}"
            line += " |"
            for j in range(y_cols):
                line += f"{y[i][j]:>{width}.{precision}f}{t[i][j]:>{width}.{precision}f}{e[i][j]:>{width}.{precision}f}"
            body += f"{line}\n"
        return f"{header}\n{separator}\n{body}"
