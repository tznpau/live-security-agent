
# Security Assistant App
# ----------------------

# This Streamlit app is powered by an LLM agent that can call trained ML models as tools, in order to analyse network traffic.



# Virtual environment config:
# ---------------------------
#
#
# python -m venv .venv
# .venv\Scripts\activate
# pip install streamlit anthropic joblib pandas scikit-learn numpy python-dotenv
# .env file



# Running the Streamlit app
# -------------------------
# streamlit run app.py



# General Config 
# --------------
import streamlit as st 
import anthropic
import json 
import joblib 
import pandas as pd  
import numpy as np 
import os 

import paramiko
import queue
import threading
import time


from dotenv import load_dotenv
load_dotenv()

# anthropic config 
ANTHROPIC_API_KEY = None
MODEL_NAME = "claude-sonnet-4-6" 

# paths to trained ML models
MODELS_DIR = r"C:\Users\pauline\dissertation\repos\live-agent\models"

RF_PATH     = os.path.join(MODELS_DIR, "random_forest_iot23.joblib")
XGB_PATH    = os.path.join(MODELS_DIR, "xgboost_iot23.joblib")
IF_PATH     = os.path.join(MODELS_DIR, "isolation_forest_iot23.joblib")
AE_PATH     = os.path.join(MODELS_DIR, "autoencoder_iot23.joblib")
AE_SCALER_PATH = os.path.join(MODELS_DIR, "autoencoder_scaler.joblib")
# autoencoder_scaler.joblib is also needed: the autoencoder was trained on scaled data,
# so any new input must be scaled the same way before inference.


# Model Loading
# -------------
# @st.cache_resource loads these once when the app starts, then caches them.
# Without this, Streamlit would reload all four models from disk on every single user interaction.

@st.cache_resource
def load_models():
    """
    Load all four trained models and the autoencoder scaler from disk.
    Returns a dict so the rest of the app can access each model by name.
    Returns None for any model that fails to load, so the app doesn't crash
    if one file is missing.
    """
    models = {}

    for name, path in [
        ("random_forest",     RF_PATH),
        ("xgboost",           XGB_PATH),
        ("isolation_forest",  IF_PATH),
        ("autoencoder",       AE_PATH),
        ("ae_scaler",         AE_SCALER_PATH),
    ]:
        try:
            models[name] = joblib.load(path)
            print(f"[OK] Loaded {name} from {path}")
        except FileNotFoundError:
            print(f"[WARN] Model not found: {path}")
            models[name] = None
        except Exception as e:
            print(f"[ERROR] Failed to load {name}: {e}")
            models[name] = None

    return models

# models are ready before any user interaction
MODELS = load_models()


# Feature Bridge
# --------------

# Copied and adapted from feature_bridge.ipynb.
# The logic is identical to the notebook — same maps, same 13 columns, same order.
# The only difference: load_conn_log() here accepts a string (from Streamlit's
# file uploader) rather than a file path.

# --- Fixed encoding maps ---
# These are the exact mappings used during training.
# Any value not in the map becomes -1 (unknown to the model — honest behaviour).
PROTO_MAP = {'tcp': 0, 'udp': 1, 'icmp': 2, 'icmp6': 3}

SERVICE_MAP = {
    '-': 0, 'http': 1, 'dns': 2, 'ssl': 3, 'ssh': 4,
    'ftp': 5, 'smtp': 6, 'dhcp': 7, 'mqtt': 8
}

CONN_STATE_MAP = {
    'S0': 0,    # SYN sent, no response — classic scan signature
    'S1': 1,    # established, not closed
    'SF': 2,    # normal full close
    'REJ': 3,   # rejected
    'S2': 4,    # established, close attempted by originator
    'S3': 5,    # established, close attempted by responder
    'RSTO': 6,  # reset by originator
    'RSTR': 7,  # reset by responder
    'SH': 8,    # SYN + FIN by originator (half-open)
    'SHR': 9,   # SYN + FIN by responder
    'OTH': 10   # no SYN, midstream
}

# The 13 feature columns the models were trained on, in order.
FEATURE_COLUMNS = [
    'id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes',
    'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
    'proto', 'service', 'conn_state'
]

# Live Watcher
# ------------
PI_HOST = "pifive"
PI_USER = "piadmin"
PI_LOG_PATH = "/home/piadmin/zeek-logs/live/conn.log"

# A line that can never be a real Zeek flow (those start with a unix timestamp).
# tail_worker pushes this onto the queue if the SSH connection fails, so the
# main thread can show the real cause instead of waiting out the whole window.
CAPTURE_ERROR_SENTINEL = "__CAPTURE_ERROR__:"

LIVE_CAPTURE_DURATION = 15
LIVE_CAPTURE_FLOW_CAP = 20

RAW_CONN_FIELDS = [
    'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
    'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
    'conn_state', 'local_orig', 'local_resp', 'missed_bytes',
    'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts',
    'resp_ip_bytes', 'tunnel_parents', 'ip_proto'
]

# tail -F never sees the real #fields header — it only watches lines
# appended after it starts. This synthetic header lets the existing
# load_conn_log() pipeline handle live lines exactly like an uploaded file.
LIVE_LOG_HEADER = (
    "#separator \\x09\n"
    "#set_separator ,\n"
    "#empty_field\t(empty)\n"
    "#unset_field\t-\n"
    "#path\tconn\n"
    "#fields\t" + "\t".join(RAW_CONN_FIELDS) + "\n"
)


def build_log_content(lines: list) -> str:
    """Reassembles a full conn.log text block from raw tailed lines,
    dropping any stray Zeek metadata lines that slipped through."""
    body = [l for l in lines if l.strip() and not l.startswith('#')]
    return LIVE_LOG_HEADER + "\n".join(body)

# Caches the feature matrix and autoencoder intermediates from the most
# recent classify_traffic call, so explain_connection can look up a
# specific row by index without the agent needing to resend the entire
# log as input 
LAST_ANALYSIS = {"X": None, "X_scaled": None, "X_reconstructed": None}


def tail_worker(out_queue: queue.Queue, stop_event: threading.Event, password: str):
    """Background thread — SSH only. No Streamlit calls allowed in here;
    it only feeds raw lines into out_queue."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=PI_HOST, username=PI_USER, password=password,
            look_for_keys=False, allow_agent=False,
            timeout=5,
        )
        _, stdout, _ = client.exec_command(f"tail -F {PI_LOG_PATH}", get_pty=False)
    except Exception as e:
        out_queue.put(f"{CAPTURE_ERROR_SENTINEL}{e}")
        client.close()
        return
    channel = stdout.channel
    channel.settimeout(1.0)

    buffer = ""
    while not stop_event.is_set():
        if channel.recv_ready():
            buffer += channel.recv(4096).decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.startswith("#"):
                    continue  # Zeek metadata, not a flow
                out_queue.put(line)
        else:
            time.sleep(0.1)

    channel.close()
    client.close()


def run_live_capture(password: str, line_placeholder) -> tuple:
    """Main thread: starts the SSH tail in the background, then drains
    the queue HERE for LIVE_CAPTURE_DURATION seconds or until
    LIVE_CAPTURE_FLOW_CAP lines arrive — whichever comes first. Updates
    line_placeholder (an st.empty()) live as each line lands.
    Returns (collected_lines, error_string_or_None)."""
    q = queue.Queue()
    stop_event = threading.Event()
    thread = threading.Thread(target=tail_worker, args=(q, stop_event, password), daemon=True)
    thread.start()

    collected = []
    error = None
    start = time.time()
    while time.time() - start < LIVE_CAPTURE_DURATION and len(collected) < LIVE_CAPTURE_FLOW_CAP:
        try:
            line = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if line.startswith(CAPTURE_ERROR_SENTINEL):
            error = line[len(CAPTURE_ERROR_SENTINEL):]
            break
        collected.append(line)
        line_placeholder.code("\n".join(collected))

    stop_event.set()
    thread.join(timeout=2)
    return collected, error


def load_conn_log(content: str) -> pd.DataFrame:
    """
    Parses a Zeek conn.log from a string (already read by Streamlit's uploader).
    Handles the IoT-23 quirk where the last column (tunnel_parents label detailed-label)
    is space-separated rather than tab-separated.
    Returns a raw DataFrame with original Zeek column names.
    """
    fields = None
    data_lines = []

    for line in content.splitlines():
        line = line.strip()

        if line.startswith('#fields'):
            # Drop the '#fields' prefix, split on tabs.
            # Last token may be 'tunnel_parents   label   detailed-label' space-joined —
            # split each token further and flatten.
            raw_fields = line.split('\t')[1:]
            fields = []
            for token in raw_fields:
                fields.extend(token.split())

        elif line.startswith('#'):
            continue  # skip all other comment lines (#types, #separator, etc.)

        elif line:
            # Split row on tabs; last part may be space-separated — split it too.
            parts = line.split('\t')
            last = parts[-1].split()
            row = parts[:-1] + last
            data_lines.append(row)

    if fields is None:
        raise ValueError("No #fields header found. Is this a valid Zeek conn.log?")
    if len(data_lines) == 0:
        raise ValueError("No data rows found in log file.")

    # Pad or trim rows to match field count — handles minor column count mismatches
    # between IoT-23 (23 cols) and homelab (21 cols) logs gracefully.
    n = len(fields)
    data_lines = [row[:n] if len(row) >= n else row + ['']*(n - len(row))
                  for row in data_lines]

    return pd.DataFrame(data_lines, columns=fields)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw Zeek DataFrame from load_conn_log() and returns
    the 13-column numeric feature matrix the models expect.
    Identical logic to feature_bridge.ipynb — same maps, same columns.
    """
    features = pd.DataFrame()

    # Numeric columns: '-' and '(empty)' are Zeek's missing value markers → 0
    numeric_cols = [
        'id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes',
        'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes'
    ]
    for col in numeric_cols:
        raw = df[col] if col in df.columns else pd.Series(['0'] * len(df))
        features[col] = pd.to_numeric(
            raw.replace({'-': '0', '(empty)': '0'}), errors='coerce'
        ).fillna(0)

    # Categorical columns: apply fixed maps, unknowns become -1
    features['proto']      = df['proto'].map(PROTO_MAP).fillna(-1).astype(int)
    features['service']    = df['service'].map(SERVICE_MAP).fillna(-1).astype(int)
    features['conn_state'] = df['conn_state'].map(CONN_STATE_MAP).fillna(-1).astype(int)

    return features[FEATURE_COLUMNS]



# Tool Definitions
# ----------------

# These JSON schemas are sent to the Claude API with every request.
# They tell the LLM what tools exist, what each one does, and what
# arguments to pass. The LLM never calls Python directly — it emits
# a structured tool_use block, and our agent loop (run_agent) executes
# the matching Python function and returns the result.

TOOL_DEFINITIONS = [
    {
        "name": "classify_traffic",
        "description": (
            "Run all four trained ML models (Random Forest, XGBoost, Isolation Forest, "
            "Autoencoder) on a Zeek conn.log that has already been uploaded. "
            "Returns prediction counts, anomaly scores, and reconstruction error "
            "so you can interpret what each model found and explain any disagreements. "
            "Call this when the user wants traffic analysed or asks what the models think."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "log_content": {
                    "type": "string",
                    "description": "The full text content of the Zeek conn.log file to analyse."
                }
            },
            "required": ["log_content"]
        }
    },
    {
        "name": "explain_connection",
        "description": (
            "Explain why one specific connection received its prediction. "
            "Returns that connection's actual feature values, each "
            "supervised model's global feature-importance ranking, and the "
            "autoencoder's per-feature reconstruction error for that exact "
            "connection. Only call this AFTER classify_traffic has already "
            "been run on the current log — it looks up a connection by its "
            "index in that result, it does not take new log content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "connection_index": {
                    "type": "integer",
                    "description": "Zero-based index of the connection to explain (e.g. the outlier_index or an index from error_sample in classify_traffic's results)."
                }
            },
            "required": ["connection_index"]
        }
    }
]


# --- Tool Execution ---

def run_classify_traffic(log_content: str, status_callback=None) -> dict:
    """
    The Python function that runs when the LLM calls classify_traffic.
    Parses the log, extracts features, runs all four models, and returns
    a structured result dict the LLM can reason about.

    status_callback: optional function(str) -> None. If provided, it is
    called with a short progress message right before each model runs,
    so a caller (e.g. Streamlit) can display live progress. If None,
    behaviour is identical to before this parameter was added.
    """

    # Step 1: parse the log and extract the 13-feature matrix
    try:
        df_raw = load_conn_log(log_content)
        X = extract_features(df_raw)
    except Exception as e:
        return {"error": f"Failed to parse log: {str(e)}"}

    n_connections = len(X)
    if n_connections == 0:
        return {"error": "No connections found in log file."}
    LAST_ANALYSIS["X"] = X 

    results = {"n_connections": n_connections, "models": {}}

    # Step 2: Random Forest — supervised classifier
    # predict() returns 0 (Benign) or 1 (Malicious) per connection.
    # predict_proba() returns the probability of each class — tells us confidence.
    if MODELS["random_forest"] is not None:
        if status_callback:
            status_callback("Running Random Forest...")
        try:
            rf_pred = MODELS["random_forest"].predict(X)
            rf_prob = MODELS["random_forest"].predict_proba(X)
            results["models"]["random_forest"] = {
                "benign_count":    int((rf_pred == 0).sum()),
                "malicious_count": int((rf_pred == 1).sum()),
                # mean confidence across all connections for each class
                "avg_prob_benign":    round(float(rf_prob[:, 0].mean()), 3),
                "avg_prob_malicious": round(float(rf_prob[:, 1].mean()), 3),
            }
        except Exception as e:
            results["models"]["random_forest"] = {"error": str(e)}

    # Step 3: XGBoost — second supervised classifier
    # Same interface as Random Forest — lets us compare how two supervised
    # approaches degrade differently under domain shift (RQ2).
    if MODELS["xgboost"] is not None:
        if status_callback:
            status_callback("Running XGBoost...")
        try:
            xgb_pred = MODELS["xgboost"].predict(X)
            xgb_prob = MODELS["xgboost"].predict_proba(X)
            results["models"]["xgboost"] = {
                "benign_count":    int((xgb_pred == 0).sum()),
                "malicious_count": int((xgb_pred == 1).sum()),
                "avg_prob_benign":    round(float(xgb_prob[:, 0].mean()), 3),
                "avg_prob_malicious": round(float(xgb_prob[:, 1].mean()), 3),
            }
        except Exception as e:
            results["models"]["xgboost"] = {"error": str(e)}

    # Step 4: Isolation Forest — anomaly-based model
    # predict() returns 1 (normal) or -1 (anomaly) — note the inverted convention.
    # decision_function() returns a score: more negative = more anomalous.
    # We flip the sign so higher score = more suspicious, which is more intuitive, so the LLM can explain correctly in plain language.
    if MODELS["isolation_forest"] is not None:
        if status_callback:
            status_callback("Running Isolation Forest...")
        try:
            if_pred  = MODELS["isolation_forest"].predict(X)
            if_score = MODELS["isolation_forest"].decision_function(X)
            results["models"]["isolation_forest"] = {
                # IF uses 1 for normal, -1 for anomaly — convert to counts
                "normal_count":  int((if_pred == 1).sum()),
                "anomaly_count": int((if_pred == -1).sum()),
                # flip sign: higher = more anomalous (easier to explain)
                "avg_anomaly_score": round(float(-if_score.mean()), 4),
            }
        except Exception as e:
            results["models"]["isolation_forest"] = {"error": str(e)}

    # Step 5: Autoencoder — anomaly-based via reconstruction error
    # The autoencoder (MLPRegressor) was trained to reconstruct benign traffic.
    # High reconstruction error = the connection looks unlike anything in training.
    # We must scale the input first using the same scaler fitted during training.
    # the 0.077 threshold is computed from the IoT-23 benign data
    if MODELS["autoencoder"] is not None and MODELS["ae_scaler"] is not None:
        if status_callback:
            status_callback("Running Autoencoder...")
        try:
            X_scaled = MODELS["ae_scaler"].transform(X)
            X_reconstructed = MODELS["autoencoder"].predict(X_scaled)
            LAST_ANALYSIS["X_scaled"] = X_scaled
            LAST_ANALYSIS["X_reconstructed"] = X_reconstructed
            # Mean squared error per connection across all 13 features
            reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
            # Sample errors: first 10, last 5, and the max outlier index
            # This gives the agent enough variation to reason about connection types
            sample_indices = list(range(min(10, len(reconstruction_errors)))) + \
                             list(range(max(0, len(reconstruction_errors)-5), len(reconstruction_errors)))
            sample_indices = sorted(set(sample_indices))

            results["models"]["autoencoder"] = {
                "avg_reconstruction_error": round(float(reconstruction_errors.mean()), 4),
                "max_reconstruction_error": round(float(reconstruction_errors.max()), 4),
                "min_reconstruction_error": round(float(reconstruction_errors.min()), 4),
                "threshold": 0.077,
                "anomaly_count": int((reconstruction_errors > 0.077).sum()),
                "error_sample": {
                    str(i): round(float(reconstruction_errors[i]), 4)
                    for i in sample_indices
                },
                "outlier_index": int(np.argmax(reconstruction_errors)),
            }
        except Exception as e:
            results["models"]["autoencoder"] = {"error": str(e)}

    return results


def explain_connection(connection_index: int) -> dict:
    """
    Explains one connection's prediction. Returns three things, and they are
    NOT all the same kind of evidence — this matters:

    - feature_values: this connection's actual numbers.
    - random_forest_top_global_features / xgboost_top_global_features:
      what each model relies on most OVERALL, across all training data.
      This is a GLOBAL property of the model, not a per-instance
      explanation — sklearn/XGBoost's feature_importances_ doesn't tell you
      "why did the model flag THIS row," only "what does this model
      generally pay attention to." We hand the agent both this and the
      actual values so it can connect them itself, but it's important the
      agent (and you, in the write-up) doesn't conflate the two.
    - autoencoder_top_contributing_features: the per-feature squared
      reconstruction error for THIS specific row. This one genuinely is
      instance-level — it's not "what the model generally cares about,"
      it's "which of these 13 numbers, for this exact connection, looked
      least like anything the model saw during training."
    """
    X = LAST_ANALYSIS["X"]
    if X is None:
        return {"error": "No traffic has been classified yet. Call classify_traffic first."}

    if connection_index is None or connection_index < 0 or connection_index >= len(X):
        return {"error": f"connection_index out of range. Valid range: 0-{len(X)-1}."}

    row = X.iloc[connection_index]
    result = {
        "connection_index": connection_index,
        # .item() converts numpy float64/int64 to plain Python types —
        # json.dumps chokes on numpy scalars otherwise.
        "feature_values": {k: (v.item() if hasattr(v, "item") else v) for k, v in row.to_dict().items()},
    }

    if MODELS["random_forest"] is not None:
        importances = dict(zip(FEATURE_COLUMNS, MODELS["random_forest"].feature_importances_))
        top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:3]
        result["random_forest_prediction"] = int(MODELS["random_forest"].predict(X.iloc[[connection_index]])[0])
        result["random_forest_top_global_features"] = [
            {"feature": f, "importance": round(float(v), 4)} for f, v in top
        ]

    if MODELS["xgboost"] is not None:
        importances = dict(zip(FEATURE_COLUMNS, MODELS["xgboost"].feature_importances_))
        top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:3]
        result["xgboost_prediction"] = int(MODELS["xgboost"].predict(X.iloc[[connection_index]])[0])
        result["xgboost_top_global_features"] = [
            {"feature": f, "importance": round(float(v), 4)} for f, v in top
        ]

    X_scaled = LAST_ANALYSIS["X_scaled"]
    X_reconstructed = LAST_ANALYSIS["X_reconstructed"]
    if X_scaled is not None and X_reconstructed is not None:
        per_feature_error = (X_scaled[connection_index] - X_reconstructed[connection_index]) ** 2
        paired = sorted(zip(FEATURE_COLUMNS, per_feature_error), key=lambda kv: kv[1], reverse=True)
        result["autoencoder_top_contributing_features"] = [
            {"feature": f, "squared_error": round(float(v), 4)} for f, v in paired[:3]
        ]

    return result


def build_verdict_chart_df(raw_results: dict) -> pd.DataFrame:
    """
    One-column DataFrame of 'connections flagged as suspicious' per model,
    for st.bar_chart. This is the visual of the central finding: the two
    supervised models disagree (one flags many, the other few) while both
    anomaly detectors flag everything. Computed only from numbers already in
    raw_results — no model re-runs, no LLM call, so it adds no agent time.
    """
    models = raw_results.get("models", {})
    flagged = {}
    rf  = models.get("random_forest", {})
    xgb = models.get("xgboost", {})
    iso = models.get("isolation_forest", {})
    ae  = models.get("autoencoder", {})
    if "malicious_count" in rf:  flagged["Random Forest"]    = rf["malicious_count"]
    if "malicious_count" in xgb: flagged["XGBoost"]          = xgb["malicious_count"]
    if "anomaly_count" in iso:   flagged["Isolation Forest"] = iso["anomaly_count"]
    if "anomaly_count" in ae:    flagged["Autoencoder"]      = ae["anomaly_count"]
    return pd.DataFrame(
        {"Flagged as suspicious": list(flagged.values())},
        index=list(flagged.keys()),
    )


# will grow as more tools are developed
# TO DO: explain_connection tool that takes 1 row and explain which features drove the prediction
def execute_tool(tool_name: str, tool_input: dict, status_callback=None) -> str:
    """
    Dispatcher: maps a tool name to its Python function and runs it.
    Returns the result as a JSON string — the API requires tool results
    to be strings, not dicts.

    status_callback: optional function(str) -> None, forwarded to the
    underlying tool function so it can report per-step progress.
    """
    if tool_name == "classify_traffic":
        result = run_classify_traffic(tool_input.get("log_content", ""), status_callback=status_callback)
    elif tool_name == "explain_connection":
        if status_callback:
            status_callback("Explaining connection...")
        result = explain_connection(tool_input.get("connection_index"))
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result)




# AI Agent setup with Anthropic
# -----------------------------

# The client reads ANTHROPIC_API_KEY from the environment automatically.
# load_dotenv() at the top of the file already loaded the .env file,
# so by the time this line runs, the key is in os.environ.
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a tier-2 SOC analyst reviewing IoT network flow data. Run the detection tools, interpret their output, and produce a short, evidence-grounded assessment.

DETECTORS (all run together by classify_traffic):
- Random Forest & XGBoost — supervised classifiers; each outputs Benign/Malicious + a confidence. Treat them as two independent opinions.
- Isolation Forest & Autoencoder — unsupervised, trained on normal traffic only. They judge FAMILIARITY, not attack type. Autoencoder's normal threshold is 0.077; higher reconstruction error = the further the flow deviates from normal IoT traffic.

Extra tool:
- explain_connection — given a connection index, returns that connection's feature values, each supervised model's GLOBAL feature importances, and the autoencoder's PER-CONNECTION reconstruction error.

INTERPRETING DISAGREEMENT:
- Supervised vs supervised: don't average or dismiss — the disagreement is itself the signal. Explain what each model is keying on.
- Supervised vs anomaly: not a contradiction. Supervised claims attack type; anomaly claims unfamiliarity. Both can be true at once.

THINK BEFORE ANSWERING: in your thinking phase, work through what each model reported, reason about WHY any disagreement follows from model design, and rule out at least one alternative reading before settling. Reasoning, not narration — don't announce tool calls.

OUTPUT — write for a SOC analyst under time pressure: verdict first, skimmable in under 1 minute. Exactly these four sections, every time:
1. **Verdict** — severity emoji + one of [Likely Benign / Likely Malicious / Ambiguous / Requires Investigation] + one or two sentences on what the traffic most likely IS.
2. **Why** — 4-6 short bullets, one piece of evidence each. Say whether the models agree; if the supervised models split, give the structural reason in one sentence. Don't walk through every connection. Describe anomalies as deviations from normal/baseline traffic behaviour — never as distance from the models' training data or what a model has "seen".
3. **Key connection** — the autoencoder's worst outlier, and only that one. If its per-connection detail wasn't already provided to you, call explain_connection on it first. In 2-3 sentences, say what about ITS feature values stands out. Note: RF/XGBoost feature importances are GLOBAL (what the model relies on across all training, not why THIS row was flagged) — only the autoencoder's per-feature error is instance-level. Don't conflate the two.
4. **Actions** — one markdown table, max 5 rows, columns Action | Why; prefix each Action with a severity emoji.

Severity: 🔴 isolate/escalate now · 🟡 investigate · 🟢 no action.
One table only (the Actions table). State ambiguity in the Verdict, don't hedge throughout. Be precise; don't speculate beyond the data."""


def run_agent(user_message: str, history: list, uploaded_log: str = None,
              precomputed_results: dict = None, precomputed_explanation: dict = None,
              status_callback=None) -> tuple:
    """
    The agent loop. Sends the user message to Claude, handles any tool calls
    Claude makes, then returns the final text response, the accumulated
    extended-thinking text, and the updated history.

    Arguments:
        user_message    : the text the user typed in the chat input
        history          : the conversation so far as a list of Anthropic message dicts
        uploaded_log     : the full text of an uploaded conn.log, or None
        precomputed_results : classify_traffic results already computed in
                              Python (Live Watch path). When provided, the
                              agent is handed the numbers directly and given
                              no tools, guaranteeing a single turn instead of
                              calling classify_traffic again on data it
                              already has the answer to.
        status_callback  : optional function(str) -> None. If provided, it is
                            called with a short status string right before each
                            tool call, so the caller (e.g. Streamlit) can show
                            live progress. If None, behaviour is identical to
                            before this parameter was added.

    Returns:
        (response_text, thinking_text, updated_history)
        response_text  : Claude's final answer, same as before.
        thinking_text  : the model's extended-thinking content, concatenated
                          across every turn of the loop (there can be more
                          than one — e.g. one round before calling the tool,
                          another round after seeing the tool's result).
                          Empty string if no thinking content was produced.
        updated_history: unchanged in meaning from before — the full
                          conversation, ready to be stored in session_state.
    """

    if precomputed_results is not None:
        explanation_block = ""
        if precomputed_explanation is not None:
            explanation_block = (
                f"\n\nexplain_connection has already been run on the worst outlier "
                f"(connection {precomputed_explanation.get('connection_index')}). Result:\n\n"
                f"<explanation>\n{json.dumps(precomputed_explanation)}\n</explanation>\n\n"
                f"Use these numbers directly. Do NOT call explain_connection again — "
                f"build your report around this single connection as the worked example. "
                f"Analysing further connections individually is unnecessary and slows the response."
            )
        full_message = (
            f"{user_message}\n\n"
            f"classify_traffic has already been run on this capture. Here are its results:\n\n"
            f"<results>\n{json.dumps(precomputed_results)}\n</results>"
            f"{explanation_block}\n\n"
            f"Interpret these results directly in your final answer."
        )
        # The worst outlier is already explained in-message, so the agent has everything
        # it needs — hand it no tools, forcing a single-turn answer (no extra round trip,
        # consistent timing). Only fall back to leaving explain_connection available if,
        # in some degraded run, we have no precomputed explanation to give it.
        active_tools = [] if precomputed_explanation is not None \
            else [t for t in TOOL_DEFINITIONS if t["name"] != "classify_traffic"]
    elif uploaded_log:
        full_message = (
            f"{user_message}\n\n"
            f"[A Zeek conn.log has been uploaded. "
            f"Call the classify_traffic tool with the following content to analyse it.]\n\n"
            f"<log_content>\n{uploaded_log}\n</log_content>"
        )
        active_tools = TOOL_DEFINITIONS
    else:
        full_message = user_message
        active_tools = TOOL_DEFINITIONS

    # Add the user turn to history
    history = history + [{"role": "user", "content": full_message}]

    # Collects thinking text from every turn of the loop, in order.
    # We use a list and join at the end rather than concatenating strings
    # in place, since string concatenation in a loop is wasteful for
    # anything beyond a couple of iterations.
    thinking_parts = []

    # Agent loop: keep going until Claude stops calling tools.
    # Claude may call a tool, receive the result, then respond — or call
    # another tool. The loop exits when stop_reason is "end_turn".
    while True:

        # Send the full conversation history to Claude.
        # thinking={"type": "enabled", ...} turns on extended thinking:
        # Claude emits a separate "thinking" content block working through
        # the problem before its tool calls or final answer. budget_tokens
        # caps how many tokens it can spend thinking — it counts against
        # the same max_tokens ceiling as the rest of the response, so
        # max_tokens must comfortably exceed budget_tokens.
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=16384,
            thinking={
                "type": "enabled",
                "budget_tokens": 2000
            },
            system=SYSTEM_PROMPT,
            tools=active_tools,
            messages=history
        )

        # Add Claude's response to history so the next turn has full context.
        # IMPORTANT: this must include the thinking block exactly as returned —
        # the API requires the full content list (thinking + tool_use, or
        # thinking + text) to be preserved verbatim in history. We are not
        # changing this line's behaviour, just noting why it matters now.
        history = history + [{"role": "assistant", "content": response.content}]

        # Pull out any thinking block from this turn and keep it, regardless
        # of whether this turn ends in a tool call or a final answer —
        # thinking can precede either.
        for block in response.content:
            if block.type == "thinking":
                thinking_parts.append(block.thinking)

        # If Claude is done — no tool calls — extract the text and return.
        if response.stop_reason == "end_turn":
            text = next(
                (block.text for block in response.content if block.type == "text"),
                "No response generated."
            )
            full_thinking = "\n\n---\n\n".join(thinking_parts)
            return text, full_thinking, history

        # If Claude wants to call a tool, execute it and feed the result back
        if response.stop_reason == "tool_use":
            # Build the tool_result messages for every tool Claude called.
            # Claude can request multiple tools in one turn — handle all of them.
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if status_callback:
                        status_callback(f"Calling tool: {block.name}...")
                    result_str = execute_tool(block.name, block.input, status_callback=status_callback)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,  # must match the request id
                        "content": result_str
                    })

            # Add tool results as a user turn — this is what the API requires.
            # The next iteration sends this back to Claude so it can reason
            # about the results and either call another tool or respond.
            history = history + [{"role": "user", "content": tool_results}]

        elif response.stop_reason == "max_tokens":
            # Model hit the token ceiling — return whatever it produced so far
            partial = next(
                (block.text for block in response.content if block.type == "text"),
                "Response was cut off before completion."
            )
            full_thinking = "\n\n---\n\n".join(thinking_parts)
            return partial + "\n\n*Response truncated — token limit reached. Consider asking a more specific question.*", full_thinking, history

        else:
            return f"Unexpected stop reason: {response.stop_reason}", "\n\n---\n\n".join(thinking_parts), history




# STREAMLIT Interface
# -------------------

# Note: Streamlit reruns this entire script every time the user interacts with the UI -> st.session_state preserves conversations between reruns



def main():

    st.set_page_config(page_title="Network Security Assistant", layout="wide")
    st.title("Network Security Assistant")
    st.caption("AI-powered IoT traffic analysis and threat detection")

    with st.sidebar:
        st.header("Log Analysis")
        uploaded_file = st.file_uploader(
            "Upload file (conn.log /conn.log.labeled):",
            type=["labeled", "log", "txt"],
            help="Upload a Zeek conn.log file to analyze"
        )

        # if file upload is successful
        uploaded_log_content = None
        if uploaded_file is not None:
            uploaded_log_content = uploaded_file.read().decode('utf-8', errors='replace')

            # preview
            lines = uploaded_log_content.strip().split('\n')
            data_lines = [l for l in lines if not l.startswith('#')]
            st.success(f"Loaded {len(data_lines)} data rows")

            # obs !!
            # for now, only send the first N rows to the agent to avoid token limits
            MAX_ROWS = 50
            if len(data_lines) > MAX_ROWS:
                st.info(f"Sending first {MAX_ROWS} rows to agent (total rows: {len(data_lines)})")
                header_lines = [l for l in lines if l.startswith("#")]
                truncated = header_lines + data_lines[:MAX_ROWS]
                uploaded_log_content ='\n'.join(truncated)

        st.divider()

        st.header("About")
        st.markdown("""
        **Capabilities**
        - analyse batch Zeek logs
        - reason possible threats and botnet behaviour
        - suggest defensive actions
        """)



        # reset conversation 
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.agent_history = []
            st.rerun()



    # INIT - session state
    # persistence across streamlit reruns
    if "messages" not in st.session_state:
        st.session_state.messages = [] # chat display
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = [] #anthropic api conversation

    tab1, tab2 = st.tabs(["Chat & Upload", "Live Watch"])

    with tab1:
        # existing chats
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # chat INPUT
        if prompt := st.chat_input("request analysis ..."):

            # user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # run agent
            with st.chat_message("assistant"):
                # st.empty() creates a placeholder we can overwrite repeatedly —
                # this is what lets the status text update live instead of being
                # stuck on one static "Analyzing..." message for the whole wait.
                status_placeholder = st.empty()

                def update_status(msg):
                    status_placeholder.info(msg)

                update_status("Starting analysis...")

                response_text, thinking_text, updated_history = run_agent(
                    prompt,
                    st.session_state.agent_history,
                    uploaded_log=uploaded_log_content,
                    status_callback=update_status
                )

                # clear the status line now that we have the final answer
                status_placeholder.empty()

                # Show the model's extended thinking in a collapsible section,
                # above the final answer. expanded=True so it's visible by
                # default during the live demo — change to False for normal use
                # if it gets in the way of quick back-and-forth chatting.
                st.markdown(response_text)

            # save to session state
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.agent_history = updated_history

            pass 
    
    with tab2:
        st.subheader("Live Network Watch")
        st.caption(
            f"Tails the lab network on for "
            f"{LIVE_CAPTURE_DURATION}s or {LIVE_CAPTURE_FLOW_CAP} flows, whichever comes first."
        )

        pi_password = st.text_input("Enter admin password", type="password", key="pi_password_input")
        start_clicked = st.button("Start Listening", disabled=not pi_password)

        if start_clicked:
            line_placeholder = st.empty()
            with st.spinner("Listening..."):
                captured_lines, capture_error = run_live_capture(pi_password, line_placeholder)

            if capture_error:
                st.error(f"Could not connect to the Pi: {capture_error}")
            elif len(captured_lines) == 0:
                st.warning("No flows captured — check Zeek is running on the Pi.")
            else:
                st.success(f"Captured {len(captured_lines)} flows.")
                log_content = build_log_content(captured_lines)

                # Run the models first (raw_results must exist before any visual uses it)...
                with st.spinner("Running models..."):
                    raw_results = run_classify_traffic(log_content)

                st.subheader("Raw model outputs")
                st.json(raw_results)

                chart_df = build_verdict_chart_df(raw_results)
                n = raw_results["n_connections"]

                # Build metrics from chart_df so their order matches the chart exactly.
                cols = st.columns(len(chart_df))
                for col, (model_name, row) in zip(cols, chart_df.iterrows()):
                    col.metric(model_name, f"{int(row['Flagged as suspicious'])}/{n} flagged")

                if not chart_df.empty:
                    import altair as alt
                    st.subheader("Model agreement at a glance")
                    n_total = raw_results.get("n_connections", int(chart_df["Flagged as suspicious"].max()))
                    plot_df = chart_df.reset_index().rename(columns={"index": "Model"})
                    chart = (
                        alt.Chart(plot_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Model:N", sort=None, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("Flagged as suspicious:Q",
                                    scale=alt.Scale(domain=[0, n_total]),
                                    title=f"Flagged (of {n_total})"),
                            tooltip=["Model", "Flagged as suspicious"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart, use_container_width=True)
                    st.caption("Connections each model flagged as suspicious — uneven bars mean the models disagree.")

                st.subheader("Agent interpretation")
                status_placeholder = st.empty()
                def update_status(msg):
                    status_placeholder.info(msg)

                update_status("Examining anomalies...")
                outlier_idx = raw_results.get("models", {}).get("autoencoder", {}).get("outlier_index")
                precomputed_explanation = explain_connection(outlier_idx) if outlier_idx is not None else None

                update_status("Reasoning over the model outputs and writing the report...")

                response_text, thinking_text, _ = run_agent(
                    "A live traffic capture has just completed. Analyse it.",
                    [],
                    precomputed_results=raw_results,
                    precomputed_explanation=precomputed_explanation,
                    status_callback=update_status,
                )
                status_placeholder.empty()

                st.markdown(response_text)



if __name__ == "__main__":
    main()