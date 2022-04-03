import logging


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    yellow = "\x1b[93;1m"
    red = "\x1b[31;1m"
    blue = "\x1b[36;1m"
    green = "\x1b[32;1m"
    reset = "\x1b[0m"
    orange = "\x1b[33;1m"
    date = "%(asctime)s ["
    level_name = "%(levelname)s"
    prefix = "]\t"
    other = "%(name)s\t%(message)s"
    datefmt = "%H:%M:%S"

    FORMATS = {
        logging.DEBUG: date + green + level_name + reset + prefix + other,
        logging.INFO: date + blue + level_name + reset + prefix + "\t" + other,
        logging.WARNING: date + yellow + level_name + reset + prefix + other,
        logging.ERROR: date + red + level_name + reset + prefix + other,
        logging.CRITICAL: date + orange + level_name + reset + prefix + other,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%H:%M:%S")
        return formatter.format(record)


def init_logger(name, log_level):

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger
