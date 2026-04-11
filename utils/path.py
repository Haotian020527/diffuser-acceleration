from pathlib import Path

ASSETS_VERSION = "v0.0.6"

class RootPath:
    ENV_PATH = Path("/home/lht/01diffuser-generation/env_model")
    AGENT = ENV_PATH / "agent"
    SCENE = ENV_PATH / "physcene"