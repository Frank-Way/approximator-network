""" Модуль с исключениями """

class FunctionExtendingException(ValueError):
    pass


class RangeMismatchException(ValueError):
    pass


class EmptyValueException(ValueError):
    pass


class ValuesRelationException(ValueError):
    pass


class WrongFormatException(ValueError):
    pass
