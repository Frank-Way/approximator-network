""" Основной модуль """
from copy import deepcopy

import torch.optim as optim
import torch.utils.data
from torch import nn

import constants
from models import ApproximatorDataset, ApproximatorNetwork, Trainer, RunConfiguration, RunResults, \
    GenerateFunctionArguments, VariableLimits, FloatFrom0To1, TrainArguments, NetworkConfiguration, LayerConfiguration
from utils import get_logger, tensor_to_str, linear_activation


def main() -> None:
    """
    Основной метод для запуска обучения, вызываемый при запускеА
    """
    logger = get_logger(name="main",
                        streams=constants.DEFAULT_LOG_STREAMS,
                        filenames=constants.DEFAULT_LOG_FILENAMES,
                        level=constants.DEFAULT_LOG_LEVEL)
    logger.info("Начало выполнения программы")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # выбор устройства
    logger.debug(f"Установлено устройство: {device}")
    data_type = torch.float64

    configs = []  # конфигурации для запуска обучения
    try:  # наполнение списка конфигураций
        configs.append(RunConfiguration(
            network_configuration=NetworkConfiguration(inputs=1, layer_configurations=[
                LayerConfiguration(neurons=6, module=nn.Linear, activation=torch.tanh),
                LayerConfiguration(neurons=1, module=nn.Linear, activation=linear_activation)]),
            generate_function_arguments=GenerateFunctionArguments(function="sin(pi * x1)", inputs=1, size=64 ** 2,
                                                                  data_type=data_type, device=device,
                                                                  limits=[VariableLimits(0, 1)],
                                                                  extending=FloatFrom0To1(0.1),
                                                                  testing=FloatFrom0To1(0.3),
                                                                  validating=FloatFrom0To1(0.2)),
            train_arguments=TrainArguments(model=ApproximatorNetwork(), epochs=10000, loss_function=torch.nn.MSELoss(),
                                           batch_size=64, start_lr=0.1, final_lr=0.0001, queries=50,
                                           momentum=FloatFrom0To1(0.0))))

        configs.append(RunConfiguration(
            network_configuration=NetworkConfiguration(inputs=1, layer_configurations=[
                LayerConfiguration(neurons=10, module=nn.Linear, activation=torch.tanh),
                LayerConfiguration(neurons=1, module=nn.Linear, activation=linear_activation)]),
            generate_function_arguments=GenerateFunctionArguments(function="sin(pi * x1)", inputs=1, size=64 ** 2,
                                                                  data_type=data_type, device=device,
                                                                  limits=[VariableLimits(0, 1)],
                                                                  extending=FloatFrom0To1(0.1),
                                                                  testing=FloatFrom0To1(0.3),
                                                                  validating=FloatFrom0To1(0.2)),
            train_arguments=TrainArguments(model=ApproximatorNetwork(), epochs=10000, loss_function=torch.nn.MSELoss(),
                                           batch_size=64, start_lr=0.1, final_lr=0.0001, queries=50,
                                           momentum=FloatFrom0To1(0.0))))

        configs.append(RunConfiguration(
            network_configuration=NetworkConfiguration(inputs=1, layer_configurations=[
                LayerConfiguration(neurons=14, module=nn.Linear, activation=torch.tanh),
                LayerConfiguration(neurons=1, module=nn.Linear, activation=linear_activation)]),
            generate_function_arguments=GenerateFunctionArguments(function="sin(pi * x1)", inputs=1, size=64 ** 2,
                                                                  data_type=data_type, device=device,
                                                                  limits=[VariableLimits(0, 1)],
                                                                  extending=FloatFrom0To1(0.1),
                                                                  testing=FloatFrom0To1(0.3),
                                                                  validating=FloatFrom0To1(0.2)),
            train_arguments=TrainArguments(model=ApproximatorNetwork(), epochs=10000, loss_function=torch.nn.MSELoss(),
                                           batch_size=64, start_lr=0.1, final_lr=0.0001, queries=50,
                                           momentum=FloatFrom0To1(0.0))))
        logger.debug(f"Успешно прочитано {len(configs)} конфигураций запуска")
    except Exception as e:
        logger.exception(
            f"Неожиданное исключение возникло при создании указанной конфигурации: {e}\n"
            f"Просьба исправить конфигурацию и запустить программу повторно")
        return

    best_results = RunResults.build_blank()
    best_config = None

    for i, config in enumerate(configs):
        logger.info(f"Начало обработки {i + 1}-й конфигурации запуска: {config}")

        # настройка нейросети
        net = config.train_arguments.model

        net.setup(configuration=config.network_configuration)
        net.to(device)
        net.to(dtype=data_type)

        try:  # формирование обучающей выборки
            # data = ApproximatorDataset.generate_random(data_type=dtype, device=device)
            # data = ApproximatorDataset.generate_xor(data_type=dtype, device=device)
            data = ApproximatorDataset.generate_function(arguments=config.generate_function_arguments)
        except Exception as e:
            logger.exception(
                f"Неожиданное исключение возникло при генерации данных на {i + 1}-й интерации: {e}\n"
                f"Итерация будет пропущена")
            continue

        if constants.IS_DATASET_TABLE_REQUIRED:
            logger.info(f"Обучающая выборка в виде таблицы:\n{data.train_data.to_str_table()}")

        # настройка оптимизатора
        optimizer = optim.SGD(net.parameters(), lr=config.train_arguments.start_lr,
                              momentum=config.train_arguments.momentum.value)

        config.train_arguments.optimizer = optimizer
        config.train_arguments.data = data

        # запуск обучения
        Trainer.train(arguments=config.train_arguments)

        # получение результатов обучения
        results = RunResults.build_from_model(network=net, dataset=data)

        if results.absolute_delta < best_results.absolute_delta:  # сохранение наилучших результатов
            best_results = results
            best_config = config
        logger.info(f"{i + 1}) {results.deltas_to_str(verbose=False)}")

    logger.info(f"Наилучшая точность для всех конфигураций: {best_results.deltas_to_str(verbose=True)}")

    if constants.IS_RESULTS_TABLE_REQUIRED:
        logger.info(f"Результат в виде таблицы:\n{best_results.to_str_table(FloatFrom0To1(constants.TABLE_PART))}")

    if constants.IS_MODEL_EXPORT_REQUIRED:
        msg = "Нейронная сеть, обученная для воспроизведения зависимости:\n"

        if best_config.generate_function_arguments.inputs == 1:
            msg += "    F(x1)"
        elif best_config.generate_function_arguments.inputs == 2:
            msg += "    F(x1, x2)"
        else:
            msg += f"    F(x1, ..., x{best_config.generate_function_arguments.inputs})"
        msg += f" = {best_config.generate_function_arguments.function}\n"

        msg += "в пределах изменения входных переменных:\n"
        for i in range(best_config.generate_function_arguments.inputs):
            msg += f"    x{i + 1}: [{best_config.generate_function_arguments.limits[i].left}; " \
                   f"{best_config.generate_function_arguments.limits[i].right}]\n"

        msg += "\n".join(("с точностью: ", *map(lambda s: f"    {s.strip()}",
                                                best_results.deltas_to_str(verbose=True).split(";")), ""))

        msg += "Конфигурация слоёв\n"
        for i, layer_configuration in enumerate(best_config.network_configuration.layer_configurations):
            msg += f"{i + 1})\n" \
                   f"    нейронов: {layer_configuration.neurons}\n" \
                   f"    активация: {layer_configuration.activation}\n"

        msg += "Параметры модели\n"
        for i, module in enumerate(best_config.train_arguments.model.modules):
            msg += f"{i + 1})\n"\
                   f"    веса:\n{tensor_to_str(module.weight.data)}\n" \
                   f"    смещения:\n{tensor_to_str(module.bias.data)}\n"
        logger.info(msg)

    logger.info("Завершение выполнения программы")


if __name__ == "__main__":  # точка входа в программу
    torch.set_printoptions(threshold=constants.TENSOR_PRINT_THRESHOLD)
    main()