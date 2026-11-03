"""
config_loader.py
----------------
Helper function to load settings from config.yaml into a Python dictionary.
Using a central config file allows you to change settings (e.g. chunk size, models)
without having to edit the Python code.
"""

import os

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads and parses the YAML configuration file.

    Parameters:
        config_path (str): Relative or absolute path to config.yaml.

    Returns:
        dict: Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. Please ensure 'config.yaml' exists."
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config

