import logging
import os
import pandas as pd

"""
This file provides some functions which can be used for logging some metrics while training
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def log(content, *args):
    for arg in args:
        content += str(arg)
    logger.info(content)


def log_in_csv(name, epoch, stats):
    if not os.path.isfile(f'{name}.csv'):
        headers = ["Epoch", "mAP@0.5", "mAP@0.5_0.95"]
        df = pd.DataFrame(columns=headers)
        df.to_csv(f'{name}.csv', index=False)

    data = {"Epoch": [epoch], "mAP@0.5": [stats[1]], "mAP@0.5_0.95": [stats[0]]}
    df = pd.DataFrame(data)
    df.to_csv(f'{name}.csv', mode="a", header=False, index=False)
    print(f"Results logged to {name}.csv")
