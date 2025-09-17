from os.path import abspath, join, dirname

DATASET_PATH = abspath(join(dirname(__file__), "..", "data/processed/dataset.csv"))
RAW_DATA_PATH = abspath(join(dirname(__file__), "..", "..", "data/raw"))
PROCESSED_DATA_PATH = abspath(join(dirname(__file__), "..", "..", "data/processed"))