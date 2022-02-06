from enum import Enum


class Labels(Enum):
    """
    Fake is set to be the positive class.
    Useful so that we don't have to remember the integer labels elsewhere in the code.
    """
    FAKE = 1
    REAL = 0
