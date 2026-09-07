import os

# Stage routing: "api" uses the cloud model, "ft" uses local fine-tuned adapters.

# "align" → gold + variant knowledge basetype 6 field
# Formal Expression base fieldverbatimdistribution only"variant
# base diff" parameter/constraintdescriptioncover constraint deterministic
# gold base 100% Stage-2 fine-tunedtraining distribution variantonlycontrolled
# "Stage-1 Stage-2 fine-tuned"
# "kb" → type Formal Expression verbatimtraining set
# buteachvariant only base
# "free" → generate Formal Expression onlyrevert/
# "llm" → etc. "free"
STAGE1_MODE = os.environ.get("STAGE1_MODE", "free")   # "align" | "kb" | "free" | "llm"
STAGE1_VERIFY = os.environ.get("STAGE1_VERIFY", "1") != "0"

STAGE_ROUTING = {
    "nl_structuring":   "api",
    "math_modeling":    "ft",
    "code_generation":  "ft",
    "feedback_fix":     "api",
    "dynamic_constraint": "ft",
    "validation":       "api",
}

# Cloud API. Do not bake credentials into the image or source tree; pass
# CPAM_API_KEY at runtime.
API_BASE_URL = os.environ.get(
    "CPAM_API_BASE_URL",
    os.environ.get("API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
API_KEY      = os.environ.get("CPAM_API_KEY", "")
# Set CPAM_API_MODEL to a model available at the configured endpoint, e.g. qwen-plus.
API_MODEL    = os.environ.get("CPAM_API_MODEL", os.environ.get("API_MODEL", "qwen-XXX"))

# Fine-tuned local adapters.
FT_API_KEY  = os.environ.get("FT_API_KEY", "0")
FT_MODEL    = os.environ.get("FT_MODEL", "stage2")  # fallback model
FT_URL_STAGE2 = os.environ.get("FT_URL_STAGE2", "http://localhost:6006/v1")
FT_URL_STAGE3 = os.environ.get("FT_URL_STAGE3", "http://localhost:6008/v1")

# Stage-3 compatibility toggles.
FT_STAGE3_USE_GOLD_LOADER = os.environ.get("FT_STAGE3_USE_GOLD_LOADER", "1") != "0"
FT_STAGE3_USE_SKELETON_CONTRACT = os.environ.get("FT_STAGE3_USE_SKELETON_CONTRACT", "0") != "0"
FT_STAGE3_FIX_CONSTANTS = os.environ.get("FT_STAGE3_FIX_CONSTANTS", "1") != "0"
FT_STAGE3_POSTPROCESS_FIX = os.environ.get("FT_STAGE3_POSTPROCESS_FIX", "1") != "0"

FT_STAGE_ENDPOINTS = {
    "math_modeling":   (FT_URL_STAGE2, "stage2"),
    "code_generation": (FT_URL_STAGE3, "stage3"),
}

FT_BASE_URL    = FT_URL_STAGE2
FT_STAGE_MODELS = {k: v for k, (_, v) in FT_STAGE_ENDPOINTS.items()}

# Shared generation params.
MAX_TOKENS  = 6000
TEMPERATURE = 0.0

# Request timeouts in seconds.
FT_REQUEST_TIMEOUT  = int(os.environ.get("FT_REQUEST_TIMEOUT",  "300"))
API_REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "120"))

# Knowledge-base update policy: "strict", "relaxed", or "manual".
KB_UPDATE_POLICY         = "strict"
KB_UPDATE_SCORE_STRICT   = 0.85
KB_UPDATE_SCORE_RELAXED  = 0.60
KB_DEDUP_THRESHOLD       = 0.85
KB_REBUILD_BATCH         = 5

# Paths.
DATA_PATH     = os.path.join(os.path.dirname(__file__), "data", "KB.xlsx")
DATASET_SHEET = "Sheet1"
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output")

KB_STORE_DIR  = os.path.join(os.path.dirname(__file__), "kb_store")
KB_JSON_PATH  = os.path.join(KB_STORE_DIR, "knowledge_base.json")

# Retrieval and feedback.
MAX_FEEDBACK_ROUNDS = int(os.environ.get("MAX_FEEDBACK_ROUNDS", "3"))
TOP_K_RETRIEVAL     = 3
MIN_SIMILARITY      = 0.10

# Solver.
SOLVER_TIME_LIMIT   = 60
SOLVER_PROC_TIMEOUT = 120

# Validation switches.
ENABLE_LLM_SEMANTIC_CHECK = os.environ.get("ENABLE_LLM_SEMANTIC_CHECK", "0") != "0"
ENABLE_DYNAMIC_VALIDATION = os.environ.get("ENABLE_DYNAMIC_VALIDATION", "1") != "0"
DYNAMIC_SOLVE_TIME_LIMIT  = int(os.environ.get("DYNAMIC_SOLVE_TIME_LIMIT", "30"))

GATE_ON_MISSING_GLOBAL_CONSTRAINTS = os.environ.get(
    "GATE_ON_MISSING_GLOBAL_CONSTRAINTS", "1") != "0"

# Chaos-map augmentation.
CHAOS_R   = 3.9
CHAOS_X0  = 0.4
ALPHA_MAX = 0.3
BETA_MAX  = 0.3
GAMMA_MAX = 0.3

PROBLEM_TYPE_ATTRIBUTES = {
    "Aircraft Skin Processing": [
        "scheduling", "makespan", "machine", "job", "operation",
        "precedence", "no_overlap", "interval_var", "alternative",
        "time_window", "tight_constraint", "loose_constraint", "flexible_job_shop",
    ],
    "Battery Pack Design": [
        "battery", "energy", "voltage", "current", "module",
        "reconfigurable", "parallel", "series", "energy_loss",
        "switching", "load", "photovoltaic", "period",
    ],
    "Charging Station Location": [
        "facility_location", "charging_station", "demand", "coverage",
        "capacity", "cost", "distance", "allocation", "electric_vehicle",
        "open", "pack",
    ],
    "DNA Sequence Design": [
        "dna", "sequence", "base", "gc_content", "hamming_distance",
        "watson_crick", "complement", "reverse_complement",
        "nucleotide", "feasibility", "count",
    ],
    "VRP": [
        "vehicle", "routing", "customer", "depot", "capacity",
        "distance", "flow_conservation", "sub_tour_elimination",
        "arc", "visiting_order", "time_window", "vrp",
    ],
}

# Fine-tuned model system prompts. Must match training YAML.
FT_STAGE2_SYSTEM = (
    "You are a mathematical modeling expert for constraint programming. "
    "Given a structured problem formulation in JSON, produce a formal "
    "mathematical model in JSON using CPLEX CP Optimizer notation. "
    "Output ONLY valid JSON, no markdown, no prose."
)

FT_STAGE3_SYSTEM = (
    "You are a constraint programming expert specialising in IBM CP Optimizer "
    "(docplex.cp). Given a mathematical model in JSON, generate complete "
    "executable Python code. No comments, no markdown, no prose. "
    "Output Python code only."
)
