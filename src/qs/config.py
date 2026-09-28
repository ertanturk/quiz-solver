from pathlib import Path

# Config
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Credentials
BB_SERVICE_NAME = "quiz-solver-bb"
API_SERVICE_NAME = "quiz-solver-api"
API_KEY_ACCOUNT = "gemini_api_key"
GOOGLE_API_KEY_MIN_LENGTH = 39
MIN_API_KEY_LENGTH = 39

# LLM
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
FALLBACK_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_RETRY_DELAY = 1.0

# Playwright & Browser
BB_LINK = "https://mef.blackboard.com/"
DEFAULT_BROWSER_PROFILE_DIR = Path.home() / ".config" / "quiz-solver" / "browser_profile"
DEFAULT_BROWSER_CHANNEL = "chrome"
DEFAULT_QUESTION_WAIT_TIMEOUT_MS = 10000

# Rate Limiting (Google Gemini Free Tier)
FREE_TIER_RPM = 5
FREE_TIER_TPM = 250_000
FREE_TIER_RPD = 20

# Batching
DEFAULT_BATCH_SIZE = 2
