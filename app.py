
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

LIVE_CAPTURE_DURATION = 25
LIVE_CAPTURE_FLOW_CAP = 30

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


def tail_worker(out_queue: queue.Queue, stop_event: threading.Event, password: str):
    """Background thread — SSH only. No Streamlit calls allowed in here;
    it only feeds raw lines into out_queue."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=PI_HOST, username=PI_USER, password=password,
        look_for_keys=False, allow_agent=False,
    )
    _, stdout, _ = client.exec_command(f"tail -F {PI_LOG_PATH}", get_pty=False)
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


def run_live_capture(password: str, line_placeholder) -> list:
    """Main thread: starts the SSH tail in the background, then drains
    the queue HERE for LIVE_CAPTURE_DURATION seconds or until
    LIVE_CAPTURE_FLOW_CAP lines arrive — whichever comes first. Updates
    line_placeholder (an st.empty()) live as each line lands."""
    q = queue.Queue()
    stop_event = threading.Event()
    thread = threading.Thread(target=tail_worker, args=(q, stop_event, password), daemon=True)
    thread.start()

    collected = []
    start = time.time()
    while time.time() - start < LIVE_CAPTURE_DURATION and len(collected) < LIVE_CAPTURE_FLOW_CAP:
        try:
            line = q.get(timeout=0.5)
            collected.append(line)
            line_placeholder.code("\n".join(collected))
        except queue.Empty:
            continue

    stop_event.set()
    thread.join(timeout=2)
    return collected


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
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result)




# AI Agent setup with Anthropic
# -----------------------------

# The client reads ANTHROPIC_API_KEY from the environment automatically.
# load_dotenv() at the top of the file already loaded the .env file,
# so by the time this line runs, the key is in os.environ.
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a tier-2 network security analyst reviewing IoT network flow data. You have access to four ML-based detection tools trained on IoT network traffic. Your job is to run these tools, interpret their outputs, and produce a clear, evidence-grounded assessment.

TOOLS AVAILABLE:
- Random Forest: a supervised classifier that outputs a binary label (Benign/Malicious) and a confidence probability for each connection.
- XGBoost: a second supervised classifier with the same interface. Use it to corroborate or challenge Random Forest findings.
- Isolation Forest: an unsupervised anomaly detector trained on normal traffic only. It returns an anomaly score — more negative means more anomalous. It has no knowledge of what attacks look like; it only knows what normal looks like.
- Autoencoder: a reconstruction-based anomaly detector, also trained on normal traffic only. It returns a reconstruction error per connection. The threshold for normal is 0.077. Connections above this threshold could not be reconstructed from patterns the model has seen.

HOW TO INTERPRET DISAGREEMENT:
If supervised models disagree with each other, do not average or dismiss — investigate. Disagreement between classifiers is itself a signal worth explaining.
If supervised models disagree with anomaly detectors, explain the structural difference: supervised models make claims about attack type, anomaly detectors make claims about familiarity. Both can be simultaneously correct.

USE OF YOUR THINKING PHASE:
After the classify_traffic tool returns its results, use your thinking phase to genuinely work through the numbers before composing your final answer. Specifically:
- Walk through what each model actually reported, in your own words, as if you were puzzling it out for the first time.
- When two models disagree, reason about *why* — what about their design (supervised vs. unsupervised, classification vs. reconstruction) would make them diverge on this particular traffic.
- Consider at least one alternative interpretation of the evidence before settling on your assessment, and say why you ruled it in or out.
- Only move to your final answer once you've reasoned through the disagreement, not before.
Do not simply restate that you are about to call a tool or that results have arrived — that is not reasoning, it's narration. The thinking phase is where the real interpretive work happens; the final answer is where you report the conclusion of that work.

OUTPUT REQUIREMENTS:
1. Summarise what each model found, using the exact numbers from the tool output.
2. Identify whether the models agree or disagree, and what the disagreement pattern suggests.
3. Interpret anomaly scores and reconstruction error magnitudes — not just whether they exceed a threshold, but by how much and what that implies.
4. State your overall assessment of the traffic: likely benign, likely malicious, ambiguous, or requires further investigation.
5. If you recommend action, specify what kind (monitor, isolate, escalate) and what evidence supports it.

Be technically precise. Do not speculate beyond the data. If the evidence is ambiguous, say so explicitly."""


def run_agent(user_message: str, history: list, uploaded_log: str = None,
              status_callback=None) -> tuple:
    """
    The agent loop. Sends the user message to Claude, handles any tool calls
    Claude makes, then returns the final text response, the accumulated
    extended-thinking text, and the updated history.

    Arguments:
        user_message    : the text the user typed in the chat input
        history          : the conversation so far as a list of Anthropic message dicts
        uploaded_log     : the full text of an uploaded conn.log, or None
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

    # If a log file was uploaded, append it to the user message so the agent
    # knows it exists and can pass it to the classify_traffic tool.
    # We don't pass it automatically — the LLM decides whether to call the tool.
    if uploaded_log:
        full_message = (
            f"{user_message}\n\n"
            f"[A Zeek conn.log has been uploaded. "
            f"Call the classify_traffic tool with the following content to analyse it.]\n\n"
            f"<log_content>\n{uploaded_log}\n</log_content>"
        )
    else:
        full_message = user_message

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
                "budget_tokens": 6000
            },
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
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
                if thinking_text:
                    with st.expander("Agent reasoning", expanded=True):
                        st.markdown(thinking_text)

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
                captured_lines = run_live_capture(pi_password, line_placeholder)

            if len(captured_lines) == 0:
                st.warning("No flows captured — check Zeek is running on the Pi.")
            else:
                st.success(f"Captured {len(captured_lines)} flows.")
                log_content = build_log_content(captured_lines)

                st.subheader("Raw model outputs")
                with st.spinner("Running models..."):
                    raw_results = run_classify_traffic(log_content)
                st.json(raw_results)

                st.subheader("Agent interpretation")
                status_placeholder = st.empty()
                def update_status(msg):
                    status_placeholder.info(msg)

                response_text, thinking_text, _ = run_agent(
                    "A live traffic capture has just completed. Analyse it.",
                    [],
                    uploaded_log=log_content,
                    status_callback=update_status,
                )
                status_placeholder.empty()

                if thinking_text:
                    with st.expander("Agent reasoning", expanded=True):
                        st.markdown(thinking_text)
                st.markdown(response_text)



if __name__ == "__main__":
    main()