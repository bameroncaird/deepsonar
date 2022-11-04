from enum import Enum


class Labels(Enum):
    """
    Useful so that we don't have to remember the integer labels elsewhere in the code.
    Set real to the positive class (1).
    My intuition for this is that a positive prediction means that the voice passes the detection system and moves on to authentication.
    E.g., a false positive should be a fake voice that is accepted by the detector as real.
    """
    FAKE = 0.0
    REAL = 1.0
