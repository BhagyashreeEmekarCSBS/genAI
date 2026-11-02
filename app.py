"""
AI Multi-Agent Software Development System
Streamlit Interactive Dashboard & Orchestration UI
Built with Google Gemini (`google-genai`), LangGraph, and Python AST validation.
"""

import os
import sys
import json
import time
from datetime import datetime
import streamlit as st

# Ensure multi_agent_development and root are in python search path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from workflow.graph import stream_development_workflow, create_development_workflow
from workflow.state import AgentWorkflowState
from utils.llm import (
    get_api_key,
    get_model_name,
    validate_api_connection,
    list_content_generation_models,
    DEFAULT_GEMINI_MODEL
)
from utils.validators import validate_code, validate_srs, validate_source_code

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Multi-Agent Software Development System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sleek Dark Theme Styling with Purple and Pink Neon Gradients
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.25;
        margin-top: 0;
        padding-top: 0;
        background: linear-gradient(135deg, #A855F7 0%, #EC4899 50%, #F43F5E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 1.6rem;
        font-weight: 400;
    }

    /* Agent Cards */
    .agent-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.2rem;
        margin-bottom: 2rem;
    }

    .agent-card {
        background: rgba(30, 27, 46, 0.7);
        border: 1px solid rgba(168, 85, 247, 0.25);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 20px -2px rgba(15, 10, 30, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }

    .agent-card:hover {
        border-color: rgba(236, 72, 153, 0.6);
        transform: translateY(-2px);
    }

    .agent-icon {
        font-size: 1.7rem;
        margin-bottom: 0.4rem;
        display: inline-block;
    }

    .agent-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.3rem;
    }

    .agent-desc {
        font-size: 0.83rem;
        color: #CBD5E1;
        line-height: 1.4;
    }

    .agent-status-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        margin-top: 0.6rem;
    }

    .status-pending {
        background: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }

    .status-running {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }

    .status-completed {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .status-failed {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    /* Verdict Boxes */
    .verdict-box {
        padding: 1.1rem;
        border-radius: 12px;
        margin-bottom: 1.3rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-size: 1.1rem;
    }

    .verdict-approved {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
    }

    .verdict-suggestions {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #FBBF24;
    }

    .verdict-needs-improvement {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #F87171;
    }

    /* AST Banner */
    .ast-card {
        background: rgba(20, 24, 40, 0.85);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(168, 85, 247, 0.3);
        margin-bottom: 1.5rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 27, 46, 0.4);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(168, 85, 247, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.25), rgba(236, 72, 153, 0.25)) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(236, 72, 153, 0.4) !important;
    }

    /* Primary Gradient Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #9333EA 0%, #DB2777 100%);
        color: white;
        font-weight: 700;
        border: none;
        padding: 0.65rem 1.8rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(219, 39, 119, 0.4);
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #A855F7 0%, #EC4899 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.6);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sample Test Cases
SAMPLE_TEST_CASES = {
    "🎓 Student Management System": {
        "name": "Student Management System",
        "language": "Python",
        "description": "Create a Student Management System using Python. It should allow students to register, view student details, update information and delete student records.",
        "instructions": "Implement full CRUD operations using an in-memory dictionary or SQLite. Include error handling for non-existent IDs, email validation, and a command-line demo menu."
    },
    "📚 Library Management System": {
        "name": "Library Management System",
        "language": "Python",
        "description": "Create a Library Management System using Python. It should manage books, issue books, return books and maintain student records.",
        "instructions": "Model Book and Student entities. Track issued book status with due dates, calculate overdue fines, and prevent issuing already borrowed books."
    },
    "🛒 Online Shopping System": {
        "name": "Online Shopping System",
        "language": "Python",
        "description": "Create an Online Shopping System using Python. It should allow users to register, browse products, add products to cart and place orders.",
        "instructions": "Implement User, Product, Cart, and Order models. Support inventory stock deduction, cart total calculation with taxes, and order invoice generation."
    }
}

# Initialize Session State
if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "api_connection_status" not in st.session_state:
    st.session_state.api_connection_status = None
if "input_project_name" not in st.session_state:
    st.session_state.input_project_name = ""
if "input_language" not in st.session_state:
    st.session_state.input_language = "Python"
if "input_description" not in st.session_state:
    st.session_state.input_description = ""
if "input_instructions" not in st.session_state:
    st.session_state.input_instructions = ""

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚡ Multi-Agent Control Center")
    st.caption("Configured with `google-genai` SDK & Python AST.")

    # API Key Configuration
    env_key = get_api_key()
    api_key_input = st.text_input(
        "Gemini API Key (GEMINI_API_KEY)",
        value=env_key if env_key and env_key != "your_gemini_api_key_here" else "",
        type="password",
        help="Reads GEMINI_API_KEY from .env by default. You can also paste your key here."
    )

    # Dynamic Model Retrieval
    if "dynamic_models" not in st.session_state or not st.session_state.dynamic_models:
        if api_key_input and api_key_input != "your_gemini_api_key_here":
            st.session_state.dynamic_models = list_content_generation_models(api_key_input)
        else:
            st.session_state.dynamic_models = [DEFAULT_GEMINI_MODEL, "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    available_models = st.session_state.dynamic_models or [DEFAULT_GEMINI_MODEL]

    # Model Selection
    current_default_model = get_model_name()
    if current_default_model in available_models:
        default_index = available_models.index(current_default_model)
    elif DEFAULT_GEMINI_MODEL in available_models:
        default_index = available_models.index(DEFAULT_GEMINI_MODEL)
    else:
        default_index = 0

    selected_model = st.selectbox(
        "Select Gemini Model",
        options=available_models,
        index=default_index,
        help="Dynamically loaded from client.models.list() for your configured API key."
    )

    # API Connection and Model Sync Buttons
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        test_conn_btn = st.button("🔌 Test Connection", use_container_width=True)
    with col_t2:
        sync_models_btn = st.button("🔄 Sync Models", use_container_width=True, help="Refresh available models from Gemini API")

    if sync_models_btn:
        with st.spinner("Querying Gemini Models API (client.models.list())..."):
            refreshed = list_content_generation_models(api_key_input)
            st.session_state.dynamic_models = refreshed
            st.success(f"Discovered {len(refreshed)} generation models.", icon="✅")
            st.rerun()

    if test_conn_btn:
        with st.spinner(f"Testing generation on '{selected_model}'..."):
            is_valid, msg = validate_api_connection(api_key=api_key_input, model=selected_model)
            st.session_state.api_connection_status = (is_valid, msg)
            if is_valid:
                # Also refresh model list with verified key
                refreshed = list_content_generation_models(api_key=api_key_input)
                st.session_state.dynamic_models = refreshed

    if st.session_state.api_connection_status:
        is_conn_ok, conn_msg = st.session_state.api_connection_status
        if is_conn_ok:
            st.success(conn_msg, icon="✅")
        else:
            st.error(conn_msg, icon="❌")
    elif api_key_input and api_key_input != "your_gemini_api_key_here":
        st.info("API Key entered. Click 'Test Connection' to verify.", icon="ℹ️")
    else:
        st.warning("⚠️ GEMINI_API_KEY is unconfigured.", icon="⚠️")

    st.markdown("---")

    # Sample Test Cases Loader
    st.markdown("#### 🧪 Load Sample Test Cases")
    selected_sample = st.selectbox(
        "Choose a pre-configured test case:",
        options=["-- Select a Test Case --"] + list(SAMPLE_TEST_CASES.keys())
    )

    if selected_sample != "-- Select a Test Case --":
        sample_data = SAMPLE_TEST_CASES[selected_sample]
        st.session_state.input_project_name = sample_data["name"]
        st.session_state.input_language = sample_data["language"]
        st.session_state.input_description = sample_data["description"]
        st.session_state.input_instructions = sample_data["instructions"]
        st.info(f"Loaded: {selected_sample}", icon="ℹ️")

    st.markdown("---")
    st.markdown("#### 🔄 LangGraph Execution Flow")
    st.markdown(
        """
        1. **Agent 1: Requirement Analysis**  
        2. **Gate: SRS Validation**  
        3. **Agent 2: Code Generation**  
        4. **Gate: Code Validation**  
        5. **Agent 3: Code Review & AST Validation**  
        """
    )

# Main Header
st.markdown('<h1 class="hero-title">⚡ AI Multi-Agent Software Development System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">An autonomous 3-agent software engineering team powered by Google Gemini, '
    'LangGraph StateGraph, and Python AST deterministic code verification.</p>',
    unsafe_allow_html=True
)

# Derive status of each agent from state
state = st.session_state.workflow_state or {}
srs_st = state.get("srs_status", "Pending")
code_st = state.get("code_status", "Pending")
rev_st = state.get("review_status", "Pending")

def get_status_html(status: str) -> str:
    css_class = f"status-{status.lower()}"
    icon_map = {"Pending": "⏳", "Running": "🔄", "Completed": "✅", "Failed": "❌"}
    icon = icon_map.get(status, "⏳")
    return f'<span class="agent-status-badge {css_class}">{icon} {status}</span>'

# 3 Agent Cards with Dynamic Status
st.markdown(
    f"""
    <div class="agent-grid">
        <div class="agent-card">
            <div class="agent-icon">📋</div>
            <div class="agent-title">Agent 1: Requirements</div>
            <div class="agent-desc">Analyzes objectives, extracts functional & non-functional requirements, inputs/outputs, and formulates the comprehensive SRS.</div>
            {get_status_html(srs_st)}
        </div>
        <div class="agent-card">
            <div class="agent-icon">💻</div>
            <div class="agent-title">Agent 2: Code Generation</div>
            <div class="agent-desc">Translates validated SRS into modular source code with defensive error handling and executable demo harness.</div>
            {get_status_html(code_st)}
        </div>
        <div class="agent-card">
            <div class="agent-icon">🛡️</div>
            <div class="agent-title">Agent 3: Code Review</div>
            <div class="agent-desc">Performs dual-layer audit: Python AST deterministic compiler check followed by deep architectural and security scrutiny.</div>
            {get_status_html(rev_st)}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Project Input Form
st.markdown("### 📝 Project Requirements Specification")
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        project_name = st.text_input(
            "Project Name",
            value=st.session_state.input_project_name,
            placeholder="e.g. Student Management System, Cloud File Vault, E-Commerce API",
            key="proj_name_field"
        )
    with col2:
        language_options = ["Python", "JavaScript", "TypeScript", "C++", "Java", "Go", "Rust"]
        current_lang_idx = language_options.index(st.session_state.input_language) if st.session_state.input_language in language_options else 0
        preferred_language = st.selectbox(
            "Target Language",
            options=language_options,
            index=current_lang_idx,
            key="proj_lang_field"
        )

    requirements_description = st.text_area(
        "Software Requirement Description",
        value=st.session_state.input_description,
        placeholder="Describe what the software should do, its target audience, key operations, and constraints...",
        height=130,
        key="proj_desc_field"
    )

    with st.expander("⚙️ Optional Additional Instructions & Architectural Constraints"):
        additional_instructions = st.text_area(
            "Specific guidelines (e.g., persistence mechanism, specific frameworks, security expectations)",
            value=st.session_state.input_instructions,
            height=70,
            key="proj_instr_field"
        )

# Action Buttons
btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 3])
with btn_col1:
    start_btn = st.button("🚀 Start Development Workflow", type="primary", use_container_width=True)
with btn_col2:
    reset_btn = st.button("🔄 Reset System", use_container_width=True)

if reset_btn:
    st.session_state.workflow_state = None
    st.session_state.input_project_name = ""
    st.session_state.input_description = ""
    st.session_state.input_instructions = ""
    st.session_state.api_connection_status = None
    st.rerun()

# Execution Pipeline Handling
if start_btn:
    # Pre-flight Validation
    if not api_key_input or api_key_input == "your_gemini_api_key_here":
        st.error("🛑 GEMINI_API_KEY is missing or invalid! Please configure it in your .env file or enter a valid key in the sidebar.", icon="🚫")
    elif not project_name.strip():
        st.warning("⚠️ Please provide a Project Name before starting.", icon="⚠️")
    elif not requirements_description.strip():
        st.warning("⚠️ Please provide a Software Requirement Description.", icon="⚠️")
    else:
        # Pre-validate connection before invoking pipeline
        with st.spinner("Verifying Gemini API authentication before starting..."):
            is_valid_conn, conn_err = validate_api_connection(api_key=api_key_input, model=selected_model)

        if not is_valid_conn:
            st.error(f"🛑 Pre-Execution Connection Check Failed: {conn_err}", icon="🚫")
            # Mark Agent 1 as failed and downstream as pending/blocked
            st.session_state.workflow_state = {
                "project_name": project_name.strip(),
                "requirements_description": requirements_description.strip(),
                "preferred_language": preferred_language,
                "srs_status": "Failed",
                "srs_error": conn_err,
                "code_status": "Pending",
                "review_status": "Pending",
                "workflow_status": "Failed",
                "error_message": conn_err
            }
            st.rerun()
        else:
            st.session_state.is_running = True
            
            initial_state: AgentWorkflowState = {
                "project_name": project_name.strip(),
                "requirements_description": requirements_description.strip(),
                "preferred_language": preferred_language,
                "additional_instructions": additional_instructions.strip() if additional_instructions else None,
                "api_key": api_key_input.strip(),
                "model_name": selected_model,
                "srs_status": "Running",
                "code_status": "Pending",
                "review_status": "Pending",
                "ast_status": "Pending",
                "workflow_status": "Running",
                "errors": [],
                "current_step": "Workflow Initialized"
            }

            progress_box = st.status("🚀 Executing Multi-Agent Software Development Pipeline...", expanded=True)
            workflow_state = initial_state

            try:
                progress_box.update(label="📋 Agent 1: Generating Software Requirement Specification (SRS)...", state="running")
                
                # Execute StateGraph in streaming mode
                for node_name, state_update in stream_development_workflow(initial_state):
                    workflow_state = {**workflow_state, **state_update}

                    if node_name == "requirement_analysis":
                        if workflow_state.get("srs_status") == "Completed":
                            progress_box.write("✅ **Agent 1 (Requirement Analysis):** SRS successfully created and verified.")
                            progress_box.update(label="💻 Agent 2: Synthesizing complete source code from validated SRS...", state="running")
                        else:
                            progress_box.write(f"❌ **Agent 1 Failed:** {workflow_state.get('srs_error', 'Unknown error')}")
                            progress_box.update(label="🛑 Workflow halted: Downstream agents blocked by SRS failure.", state="error")
                            break

                    elif node_name == "code_generation":
                        if workflow_state.get("code_status") == "Completed":
                            progress_box.write(f"✅ **Agent 2 (Code Generation):** Source code generated ({workflow_state.get('code_language', preferred_language)}).")
                            progress_box.update(label="🛡️ Agent 3: Conducting AST compiler checks and security review...", state="running")
                        else:
                            progress_box.write(f"❌ **Agent 2 Failed:** {workflow_state.get('code_error', 'Unknown error')}")
                            progress_box.update(label="🛑 Workflow halted: Downstream code review blocked by generation failure.", state="error")
                            break

                    elif node_name == "code_review":
                        if workflow_state.get("review_status") == "Completed":
                            verdict = workflow_state.get("review_verdict", "Completed")
                            progress_box.write(f"✅ **Agent 3 (Code Review & AST):** Completed with verdict: **{verdict}**.")

                if workflow_state.get("workflow_status") == "Completed":
                    progress_box.update(label="🎉 Multi-Agent Development Pipeline Completed Successfully!", state="complete", expanded=False)
                else:
                    progress_box.update(label="⚠️ Pipeline execution halted on validation error.", state="error", expanded=True)

                st.session_state.workflow_state = workflow_state

                # Save artifacts locally if code exists
                if workflow_state.get("generated_code"):
                    try:
                        outputs_dir = os.path.join(current_dir, "outputs")
                        os.makedirs(outputs_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '_', '-')).rstrip().replace(" ", "_")
                        
                        ext_map = {"python": ".py", "javascript": ".js", "typescript": ".ts", "c++": ".cpp", "java": ".java", "go": ".go", "rust": ".rs"}
                        ext = ext_map.get(workflow_state.get("code_language", "python").lower(), ".py")
                        with open(os.path.join(outputs_dir, f"{safe_name}_{timestamp}{ext}"), "w", encoding="utf-8") as f:
                            f.write(workflow_state.get("generated_code", ""))
                    except Exception:
                        pass

            except Exception as e:
                progress_box.update(label=f"❌ Pipeline Exception: {str(e)}", state="error", expanded=True)
                st.error(f"Workflow Exception: {str(e)}")
            finally:
                st.session_state.is_running = False
                st.rerun()

# Display Deliverables
state = st.session_state.workflow_state

if state:
    st.markdown("---")
    st.markdown("## 📊 Multi-Agent Workflow Deliverables")

    tab_srs, tab_code, tab_review = st.tabs([
        "📋 Tab 1: Requirement Analysis (SRS)",
        "💻 Tab 2: Generated Source Code",
        "🛡️ Tab 3: Code Review & AST Validation"
    ])

    # Tab 1: Requirement Analysis
    with tab_srs:
        srs_status = state.get("srs_status", "Pending")
        srs_report = state.get("srs_report")
        
        st.markdown(f"### 📋 Software Requirement Specification: {state.get('project_name', '')}")
        st.markdown(f"**Generation Status:** {get_status_html(srs_status)}", unsafe_allow_html=True)
        
        if srs_status == "Completed" and srs_report:
            st.markdown(srs_report)
            st.download_button(
                label="📥 Download SRS Document (.md)",
                data=srs_report,
                file_name=f"{state.get('project_name', 'project').lower().replace(' ', '_')}_srs.md",
                mime="text/markdown",
                key="dl_srs_btn"
            )
        elif srs_status == "Failed":
            st.error(f"❌ **SRS Generation Failed:** {state.get('srs_error', 'Unknown error')}", icon="🛑")
            st.info("💡 Please verify your GEMINI_API_KEY in the sidebar and ensure the model has active quota.", icon="💡")
        else:
            st.info("⏳ Requirement analysis has not executed yet. Click 'Start Development Workflow' to begin.", icon="ℹ️")

    # Tab 2: Generated Source Code
    with tab_code:
        code_status = state.get("code_status", "Pending")
        generated_code = state.get("generated_code")
        code_lang = state.get("code_language", state.get("preferred_language", "python"))

        st.markdown(f"### 💻 Source Code Implementation ({code_lang.upper()})")
        st.markdown(f"**Generation Status:** {get_status_html(code_status)}", unsafe_allow_html=True)

        if code_status == "Completed" and generated_code:
            lines_count = len(generated_code.splitlines())
            st.caption(f"Language: **{code_lang.capitalize()}** | Total Lines: **{lines_count}** | Encoding: **UTF-8**")
            st.code(generated_code, language=code_lang.lower())

            ext_map = {"python": "py", "javascript": "js", "typescript": "ts", "c++": "cpp", "java": "java", "go": "go", "rust": "rs"}
            file_ext = ext_map.get(code_lang.lower(), "py")
            
            st.download_button(
                label=f"📥 Download Source Code (.{file_ext})",
                data=generated_code,
                file_name=f"{state.get('project_name', 'solution').lower().replace(' ', '_')}.{file_ext}",
                mime="text/plain",
                key="dl_code_btn"
            )
        elif code_status == "Failed":
            st.error(f"❌ **Code Generation Failed:** {state.get('code_error', 'Code could not be produced.')}", icon="🛑")
        else:
            st.info("⏳ Code generation has not completed yet. Waiting for validated SRS from Agent 1.", icon="ℹ️")

    # Tab 3: Code Review & AST Validation
    with tab_review:
        review_status = state.get("review_status", "Pending")
        review_report = state.get("review_report")
        ast_result = state.get("syntax_validation") or {}
        verdict = state.get("review_verdict")

        st.markdown("### 🛡️ Code Review & Parser Verification")
        st.markdown(f"**Audit Status:** {get_status_html(review_status)}", unsafe_allow_html=True)

        # Show AST Validation Status
        st.markdown("#### 🔬 Deterministic Python AST Compiler Validation")
        st.caption("Concrete syntax validation using Python's built-in Abstract Syntax Tree parser, completely independent of LLM reasoning.")

        has_code = ast_result.get("has_code", False)
        is_ast_valid = ast_result.get("is_valid", False)

        if not has_code:
            st.info(f"ℹ️ {ast_result.get('summary', 'Code generation has not completed yet. No code available for AST validation.')}", icon="ℹ️")
        elif is_ast_valid:
            st.success("✅ **Python AST Parser Result: Syntax is 100% Valid and Parsable.**", icon="✔️")
            with st.expander("🔍 View AST Compiler Structure", expanded=True):
                st.text(ast_result.get("summary", ""))
                if ast_result.get("classes"):
                    st.write("**Classes Detected:**", ", ".join(ast_result["classes"]))
                if ast_result.get("functions"):
                    st.write("**Functions / Methods Detected:**", ", ".join(ast_result["functions"]))
                if ast_result.get("imports"):
                    st.write("**Imports Detected:**", ", ".join(ast_result["imports"]))
        else:
            st.error(f"❌ **Python AST Parser Result: Syntax Error Detected at Line {ast_result.get('error_line')}!**", icon="⚠️")
            st.warning(f"**Error Details:** {ast_result.get('error')}")
            if ast_result.get("error_snippet"):
                st.code(ast_result.get("error_snippet"), language="python")
            if ast_result.get("suggested_fix"):
                st.info(f"💡 **Suggested Fix:** {ast_result.get('suggested_fix')}")

        st.markdown("---")

        # Show AI Review Feedback and Verdict
        if review_status == "Completed" and review_report:
            if verdict == "Approved":
                verdict_class, verdict_icon = "verdict-approved", "🟢"
            elif verdict == "Approved with Suggestions":
                verdict_class, verdict_icon = "verdict-suggestions", "🟡"
            else:
                verdict_class, verdict_icon = "verdict-needs-improvement", "🔴"

            st.markdown(
                f"""
                <div class="verdict-box {verdict_class}">
                    <span>{verdict_icon}</span>
                    <span>Final Review Verdict: {verdict.upper()}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("#### 📑 Comprehensive AI Code Review Report")
            st.markdown(review_report)
        elif review_status == "Failed":
            st.error(f"❌ **Code Review Failed:** {state.get('review_error', 'Review could not be completed.')}")
        else:
            st.info("⏳ Code review has not completed yet. Waiting for generated source code from Agent 2.", icon="ℹ️")

    # Global Dossier Export
    if state.get("srs_report") or state.get("generated_code"):
        st.markdown("---")
        st.markdown("### 📦 Export Complete Project Package")
        full_dossier = (
            f"# Multi-Agent AI Software Development System - Deliverables Dossier\n"
            f"**Project:** {state.get('project_name')}\n"
            f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Target Language:** {state.get('code_language', state.get('preferred_language'))}\n"
            f"**Workflow Status:** {state.get('workflow_status')}\n\n"
            f"---\n\n"
            f"## SECTION 1: Software Requirement Specification (SRS)\n\n"
            f"{state.get('srs_report') or 'N/A'}\n\n"
            f"---\n\n"
            f"## SECTION 2: Generated Source Code\n\n"
            f"```{state.get('code_language', 'python')}\n"
            f"{state.get('generated_code') or '# No code generated'}\n"
            f"```\n\n"
            f"---\n\n"
            f"## SECTION 3: Deterministic AST Syntax Verification\n\n"
            f"```\n{state.get('syntax_validation', {}).get('summary', 'N/A')}\n```\n\n"
            f"---\n\n"
            f"## SECTION 4: Code Review Audit\n\n"
            f"{state.get('review_report') or 'N/A'}\n"
        )
        st.download_button(
            label="📥 Download Complete Project Dossier (Markdown)",
            data=full_dossier,
            file_name=f"{state.get('project_name', 'project').lower().replace(' ', '_')}_full_dossier.md",
            mime="text/markdown",
            key="dl_dossier_btn"
        )
