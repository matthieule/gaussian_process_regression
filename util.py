"""Utility functions"""
import sys
import logging


def get_logger():
    """Get logger, stolen from the OpenAI logger:

    https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
    """
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger
