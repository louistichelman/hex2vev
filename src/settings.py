from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR.joinpath("src/models")
NOTEBOOKS_DIR = PROJECT_DIR.joinpath("notebooks")
REPORTS_DIR = PROJECT_DIR.joinpath("reports")
FIGURES_DIR = REPORTS_DIR.joinpath("figures")

DATA_DIR = PROJECT_DIR.joinpath("src/data")
DATA_INTERIM_DIR = DATA_DIR.joinpath("osm/interim")
DATA_RAW_DIR = DATA_DIR.joinpath("osm/raw")
DATA_PROCESSED_DIR = DATA_DIR.joinpath("osm/processed")
DATA_EMBEDDED_DIR = DATA_DIR.joinpath("osm/embedded")
DATA_BRW = DATA_DIR.joinpath("brw")
KEPLER_CONFIG_DIR = PROJECT_DIR.joinpath("config")
FILTERS_DIR = PROJECT_DIR.joinpath("filters")
