from enum import Enum

# model selection options
class ModelType(Enum):
    MISTRAL = "mistral"
    MELTEMI_Q8 = "meltemi_q8"
    MISTRAL_COMPRESSED = "mistral_compressed"
    MELTEMI_COMPRESSED_Q4 = "meltemi_compressed_q4"
    MELTEMI_COMPRESSED_Q2 = "meltemi_compressed_q2"

# set this to choose which model to use
SELECTED_MODEL = ModelType.MELTEMI_Q8

# model configurations
MODELS = {
    ModelType.MISTRAL.value: {
        "name": "mistral",
        "modelfile": "Modelfile",
        "description": "Mistral-7B-Instruct-v0.3 (4.1GB)"
    },
    ModelType.MELTEMI_Q8.value: {
        "name": "meltemi_q8",
        "modelfile": "Modelfile.meltemi_q8",
        "description": "Greek-specialized Meltemi Q8 (8.0GB)"
    },
    ModelType.MISTRAL_COMPRESSED.value: {
        "name": "mistral_compressed",
        "modelfile": "Modelfile.compressed",
        "description": "Compressed Mistral Q4_K (4.07GB)"
    },
    ModelType.MELTEMI_COMPRESSED_Q4.value: {
        "name": "meltemi_compressed_q4",
        "modelfile": "Modelfile.meltemi_compressed_q4",
        "description": "Compressed Meltemi Q4_XS (4.07GB)"
    },
    ModelType.MELTEMI_COMPRESSED_Q2.value: {
        "name": "meltemi_compressed_q2",
        "modelfile": "Modelfile.meltemi_compressed_q2",
        "description": "Ultra Compressed Meltemi IQ2_M (2.64GB)"
    }
}

# default model from selection
DEFAULT_MODEL = SELECTED_MODEL.value

# chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# file paths
SAMPLE_TEXT_PATH = "sample_text.txt"
OUTPUT_PATH = "output.txt"

# API configuration
API_BASE = "http://localhost:11434/api"

# model runtime settings
MODEL_SETTINGS = {
    "options": {
        # core processing settings
        "num_gpu": 16,           # number of layers to offload to GPU (reduce memmory usage)
        "num_ctx": 4096,         # context window size
        "num_batch": 1024,        # batch size (increase speed)
        "num_thread": 16,         # CPU threads
        
        # generation settings
        "temperature": 0.3,      # lower temp for more focused output
        "top_p": 0.7,           # original top_p from Modelfile
        "repeat_penalty": 1.1,   # prevent repetition
        
        # memory settings
        "seed": 42              # optional: for reproducibility
    }
}

MODEL_SETTINGS_V2 = {
    "options": {
        # core processing settings
        "num_gpu": 16,           # number of layers to offload to GPU (reduce memmory usage)
        "num_ctx": 4096,         # context window size
        "num_batch": 1024,        # batch size (increase speed)
        "num_thread": 16,         # CPU threads
        
        # generation settings
        "temperature": 0.3,      # lower temp for more focused output
        "top_p": 0.7,           # original top_p from Modelfile
        "repeat_penalty": 1.1,   # prevent repetition
        
        # memory settings
        "seed": None,              # optional: for reproducibility
        "max_tokens": None

    }
}

MODEL_SETTINGS_V3 = {
    "options": {
        "num_gpu": 8,            # start conservatively; check your GPU specs.
        "num_ctx": 2048,         # safe default context size.
        "num_batch": 128,        # common stable batch size.
        "num_thread": 8,         # match CPU cores.

        "temperature": 0.3,
        "top_p": 0.7,
        "repeat_penalty": 1.1,
        
        "seed": 42,          # optional: for reproducibility
        "max_tokens": None

    }
}

MODEL_SETTINGS_V4 = {
    "options": {
        "num_gpu": 8,
        "num_ctx": 4096,
        "num_batch": 256,
        "num_thread": 8,

        "temperature": 0.4,
        "top_p": 0.6,
        "repeat_penalty": 1.1,

        "max_tokens": 800,
        "seed": 42
    }
}