""" Модуль с некоторыми общими параметрами """

import sys

# формат, в котором формируются записи в логере
LOG_FORMAT = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
# LOG_FORMAT = f"%(message)s"  # формат без префиксов

# формат представления времени
DATETIME_FORMAT = "%d/%m/%Y %H:%M:%S"

# максимальное число печатаемых при выводе элементов тензоров
TENSOR_PRINT_THRESHOLD = 10000

# имена файлов для хранения логов (если None, то логи не выводятся в какие-либо файлы)
DEFAULT_LOG_FILENAMES = ["log.txt"]
# DEFAULT_LOG_FILENAMES = None

# потоки для вывода логов (если None, то логи не выводятся в какие-либо потоки)
DEFAULT_LOG_STREAMS = [sys.stdout]
# DEFAULT_LOG_STREAMS = None

# уровень логирования
DEFAULT_LOG_LEVEL = "DEBUG"

# нужен ли вывод обучеющей выборки в виде таблицы
# IS_DATASET_TABLE_REQUIRED = True
IS_DATASET_TABLE_REQUIRED = False

# нужен ли вывод результатов обучения в виде таблицы
# IS_RESULTS_TABLE_REQUIRED = True
IS_RESULTS_TABLE_REQUIRED = False

# какая часть таблицы должна быть выведена
# TABLE_PART = 0.2
TABLE_PART = 1.0

# нужен ли вывод параметров модели после обучения
IS_MODEL_EXPORT_REQUIRED = True
# IS_MODEL_EXPORT_REQUIRED = False
