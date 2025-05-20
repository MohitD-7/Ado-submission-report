import streamlit as st
import pandas as pd
import re
import time
import threading
import logging
from io import BytesIO
import os # For path.splitext

# Import configuration
try:
    import config
except ImportError:
    st.error("Configuration file 'config.py' not found. Please create it in the same directory as the app.")
    st.stop()

# --- Global Logger (for console/file, not Streamlit UI directly from thread) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions (No st.* calls here) ---
def extract_file_info_from_name(filename_str: str, log_fn_for_worker):
    de_code, submission_date, sku_count, sku_type = None, None, 0, 'Regular'
    de_match = re.search(r'(DE-\d+)', filename_str, re.IGNORECASE)
    if de_match: de_code = de_match.group(1).upper()
    date_match = re.search(r'(\w+\s+\d{1,2},\s+\d{4})', filename_str)
    if date_match: submission_date = date_match.group(1)
    sku_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*[-]?\s*(\b(?:Multi Urgent|Multi-Urgent|Multi-Regular|Multi Regular|Urgent|Regular|Priority)\b)?\s*SKUs?', filename_str, re.IGNORECASE)
    if sku_match:
        sku_count_str = sku_match.group(1)
        try: sku_count = int(sku_count_str.replace(',', ''))
        except ValueError: log_fn_for_worker(f"Could not convert SKU count '{sku_count_str}' from: {filename_str}", "warning"); sku_count = 0
        extracted_type_str = sku_match.group(2)
        if extracted_type_str:
            temp_type = extracted_type_str.strip().lower()
            type_map = {'multi urgent': 'Multi-Urgent', 'multi regular': 'Multi-Regular', 'urgent': 'Urgent', 'priority': 'Priority'}
            sku_type = type_map.get(temp_type, extracted_type_str.strip().capitalize())
    if not de_code or not submission_date or sku_count <= 0: return None, None, 0, sku_type
    return de_code, submission_date, sku_count, sku_type

def should_process_file(filename_str: str):
    is_valid = filename_str.lower().endswith(config.VALID_EXTENSIONS)
    not_excluded = not any(term.lower() in filename_str.lower() for term in config.EXCLUDE_FILENAME_TERMS)
    not_temp = not filename_str.lower().startswith('~$')
    return is_valid and not_excluded and not_temp

def count_terms_in_comments(comments_text: str, terms_to_search: list[str]):
    counts = {term: 0 for term in terms_to_search}
    current_comments = comments_text.lower()
    lower_terms = [term.lower() for term in terms_to_search]
    neg_pattern = r'\b(?:not|no|non)\s*(?:' + "|".join([re.escape(t) for t in lower_terms]) + r')\b'
    cleaned = re.sub(neg_pattern, ' ', current_comments, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    priority = (config.LARGE_BUNDLE_TERMS_CONFIG + config.SIMILAR_TERMS_CONFIG + config.PARENT_TERMS_CONFIG +
                [t for t in terms_to_search if t not in (config.LARGE_BUNDLE_TERMS_CONFIG +
                                                        config.SIMILAR_TERMS_CONFIG +
                                                        config.PARENT_TERMS_CONFIG)])
    for comment in re.split(r'[\n\.]+', cleaned):
        comment = comment.strip()
        if not comment: continue
        for term in priority:
            if re.search(r'\b' + re.escape(term.lower()) + r'\b', comment, re.IGNORECASE):
                counts[term] += 1; break
    return counts

# --- Worker Thread Function ---
def process_uploaded_files_task_worker(uploaded_files: list, results_container: dict):
    worker_logs, worker_status_data, detailed_list, summary_list, skipped_list = [], [], [], [], []
    final_msg = "Processing initiated..."

    def _log(message, level="info"): # Worker's internal logger
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        worker_logs.append(f"{timestamp} - {level.upper()} - {message}")
        getattr(logger, level, logger.info)(message) # Also log to console/file

    try:
        _log(f"Processing {len(uploaded_files)} uploaded files.")
        _log(f"Terms to count: {', '.join(config.TERMS_TO_COUNT)}")
        _log("-" * 50)
        processed_c = 0
        candidates = []

        for up_file in uploaded_files:
            fname = up_file.name
            if should_process_file(fname):
                candidates.append((up_file, fname))
                worker_status_data.append({"Filename": fname, "Status": "Pending", "Reason": "Queued"})
            else:
                skipped_list.append((fname, "Excluded by initial file criteria (name/type/temp)."))
                worker_status_data.append({"Filename": fname, "Status": "Skipped", "Reason": "Initial filter"})
        
        _log(f"{len(candidates)} files passed initial filter for processing.")
        sorted_candidates = sorted(candidates, key=lambda x: x[1])

        for up_file, fname in sorted_candidates:
            for item in worker_status_data:
                if item["Filename"] == fname: item.update({"Status": "Processing", "Reason": "Extracting info..."}); break
            _log(f"Processing: {fname}")
            had_error = False
            de, date, skus, type_ = extract_file_info_from_name(fname, _log)

            if not de or not date or skus <= 0:
                reason = "Invalid filename format or missing DE/Date/SKU count."
                _log(f"SKIPPING {fname}: {reason}", "warning")
                skipped_list.append((fname, reason))
                for item in worker_status_data:
                    if item["Filename"] == fname: item.update({"Status": "Skipped", "Reason": reason}); break
                had_error = True
            
            if not had_error:
                try:
                    up_file.seek(0) # Reset buffer
                    xl_file = pd.ExcelFile(up_file, engine='openpyxl')
                    de_sheets = [s_name for s_name in xl_file.sheet_names if s_name.upper().startswith('DE-')]
                    if not de_sheets:
                        reason = "No 'DE-' prefixed sheet found."
                        _log(f"SKIPPING {fname}: {reason}", "warning"); skipped_list.append((fname, reason)); had_error = True
                        for item in worker_status_data:
                            if item["Filename"] == fname: item.update({"Status": "Skipped", "Reason": reason}); break
                    else:
                        df = pd.read_excel(xl_file, sheet_name=de_sheets[0], keep_default_na=False)
                        _log(f"Read sheet '{de_sheets[0]}' from {fname}")
                        if df.empty:
                            reason = "Sheet is empty."
                            _log(f"SKIPPING {fname}: {reason}", "warning"); skipped_list.append((fname, reason)); had_error = True
                            for item in worker_status_data:
                                if item["Filename"] == fname: item.update({"Status": "Skipped", "Reason": reason}); break
                    
                    if not had_error: # Comments processing
                        comments_series, col_name_used = None, None
                        if config.COMMENTS_COLUMN_PRIMARY in df.columns:
                            comments_series, col_name_used = df[config.COMMENTS_COLUMN_PRIMARY], config.COMMENTS_COLUMN_PRIMARY
                        elif df.shape[1] > config.COMMENTS_COLUMN_FALLBACK_INDEX:
                            try:
                                comments_series = df.iloc[:, config.COMMENTS_COLUMN_FALLBACK_INDEX]
                                col_name_used = str(df.columns[config.COMMENTS_COLUMN_FALLBACK_INDEX])
                                _log(f"Used fallback column '{col_name_used}' for {fname}.")
                            except IndexError: _log(f"Fallback index out of bounds for {fname}.", "error")
                        
                        if comments_series is None:
                            reason = "Comments column not found."
                            _log(f"SKIPPING {fname}: {reason}", "warning"); skipped_list.append((fname, reason)); had_error = True
                            for item in worker_status_data:
                                if item["Filename"] == fname: item.update({"Status": "Skipped", "Reason": reason}); break
                        
                        if not had_error: # Actual processing
                            comments_text = ' '.join(comments_series.fillna('').astype(str).tolist()).strip()
                            if not comments_text: _log(f"No comments in '{col_name_used}' for {fname}.")
                            term_counts = count_terms_in_comments(comments_text, config.TERMS_TO_COUNT)
                            detailed_list.append([de, date, fname, comments_text] + [term_counts.get(t, 0) for t in config.TERMS_TO_COUNT])
                            
                            lb_count = sum(term_counts.get(t, 0) for t in config.LARGE_BUNDLE_TERMS_CONFIG)
                            sim_count = sum(term_counts.get(t, 0) for t in config.SIMILAR_TERMS_CONFIG)
                            reg_pri_val, urg_val = 0, 0
                            if 'urgent' in type_.lower():
                                urg_val = skus - lb_count - sim_count
                                if urg_val < 0: _log(f"Neg urg count for {fname}", "warning"); urg_val = 0
                            else:
                                reg_pri_val = skus - lb_count - sim_count
                                if reg_pri_val < 0: _log(f"Neg reg/pri count for {fname}", "warning"); reg_pri_val = 0
                            
                            s_fname_base, s_fname_ext = os.path.splitext(fname)
                            s_fname = re.sub(r'^.*? - ', '', s_fname_base) + s_fname_ext
                            if type_.lower() in ['multi-regular', 'multi-urgent'] or not date or not de:
                                s_fname = f"{skus} {type_} SKUs - {date}{s_fname_ext}" if date else f"{skus} {type_} SKUs{s_fname_ext}"
                            
                            summary_list.append([de, date, skus, reg_pri_val or '', lb_count or '', sim_count or '', urg_val or '', s_fname])
                            _log(f"SUCCESS: {fname} processed.")
                            processed_c += 1
                            for item in worker_status_data:
                                if item["Filename"] == fname: item.update({"Status": "Processed", "Reason": "Completed"}); break
                except Exception as e_file:
                    reason = f"Error during processing of {fname}: {str(e_file)}"
                    _log(reason, "error"); logger.exception(f"Error details for {fname}:")
                    if not had_error: skipped_list.append((fname, reason)) # Avoid double add
                    for item in worker_status_data:
                        if item["Filename"] == fname: item.update({"Status": "Error", "Reason": str(e_file)[:100]}); break
        
        _log("-" * 50)
        _log("Processing Summary:")
        _log(f"Total files considered: {len(uploaded_files)}")
        _log(f"Candidates for processing: {len(sorted_candidates)}")
        _log(f"Successfully processed: {processed_c}")
        _log(f"Skipped/Errored: {len(skipped_list)}")
        _log("-" * 50)

        detailed_df = pd.DataFrame(detailed_list, columns=['DE Code', 'Submission Date', 'Source Filename', 'Combined Comments'] + config.TERMS_TO_COUNT)
        summary_df = pd.DataFrame(summary_list, columns=['JIRA/Input', 'Submission Date', 'Total SKUs', 'Regular / Priority', 'Large / Bundle', 'Similar', 'Urgent', 'File Name'])
        skipped_df = pd.DataFrame(skipped_list, columns=['Filename', 'Reason'])

        if not detailed_df.empty: # Add Totals
            totals_d = {'DE Code': 'Total'}
            for col in config.TERMS_TO_COUNT: totals_d[col] = pd.to_numeric(detailed_df[col], errors='coerce').sum()
            detailed_df = pd.concat([detailed_df, pd.DataFrame([totals_d])], ignore_index=True)
        if not summary_df.empty:
            cols_s = ['Total SKUs', 'Regular / Priority', 'Large / Bundle', 'Similar', 'Urgent']
            totals_s = {'JIRA/Input': 'Total'}
            for col in cols_s: totals_s[col] = pd.to_numeric(summary_df[col], errors='coerce').sum()
            summary_df = pd.concat([summary_df, pd.DataFrame([totals_s])], ignore_index=True)

        final_msg = "Processing complete. See results below." if (processed_c > 0 or skipped_list) else "No files processed or data generated."
        results_container.update({
            "log_messages": worker_logs, "file_status_data": worker_status_data,
            "detailed_df": detailed_df, "summary_df": summary_df, "skipped_df": skipped_df,
            "final_status_message": final_msg, "error_occurred": False
        })
    except Exception as e_critical:
        _log(f"CRITICAL TASK ERROR: {str(e_critical)}", "critical")
        logger.exception("Critical error in worker thread:")
        results_container.update({
            "log_messages": worker_logs, "file_status_data": worker_status_data, # Partial results
            "detailed_df": pd.DataFrame(), "summary_df": pd.DataFrame(), "skipped_df": pd.DataFrame(skipped_list, columns=['Filename', 'Reason']),
            "final_status_message": f"Critical Task Error: {str(e_critical)}", "error_occurred": True
        })

# --- Streamlit UI ---
st.set_page_config(page_title="Excel Analyzer", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state
default_ss = {"is_processing": False, "initiated_once": False, "logs": [], "status_data": [],
              "df_detailed": pd.DataFrame(), "df_summary": pd.DataFrame(), "df_skipped": pd.DataFrame(),
              "final_message": "", "worker_thread": None, "thread_results": {}}
for k, v in default_ss.items():
    if k not in st.session_state: st.session_state[k] = v

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #007bff;'>📁 Advanced Excel File Analyzer 📁</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Upload Section ---
with st.container():
    st.subheader("1. Upload Your Excel Files")
    st.markdown("""
    Select all Excel files (`.xlsx`, `.xlsm`) from a single folder that you wish to analyze.
    You can typically use `Ctrl+A` (Windows) or `Cmd+A` (Mac) within the file dialog to select all files in the current directory.
    """)
    uploaded_files = st.file_uploader(
        "Drag and drop files here or click to browse",
        type=["xlsx", "xlsm"],
        accept_multiple_files=True,
        key="file_uploader",
        disabled=st.session_state.is_processing,
        label_visibility="collapsed"
    )

    if uploaded_files and not st.session_state.is_processing:
        st.caption(f"{len(uploaded_files)} file(s) selected. Ready to process.")

    process_button_disabled = st.session_state.is_processing or not uploaded_files
    if st.button("🚀 Analyze Selected Files", type="primary", disabled=process_button_disabled, use_container_width=True):
        if uploaded_files:
            st.session_state.is_processing = True
            st.session_state.initiated_once = True
            st.session_state.logs = [f"[{time.strftime('%H:%M:%S')}] Processing {len(uploaded_files)} files..."] # Initial log
            st.session_state.status_data, st.session_state.df_detailed, st.session_state.df_summary, st.session_state.df_skipped = [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            st.session_state.final_message, st.session_state.thread_results = "", {}

            st.session_state.worker_thread = threading.Thread(
                target=process_uploaded_files_task_worker,
                args=(uploaded_files, st.session_state.thread_results),
                name="FileProcessingThread"
            )
            st.session_state.worker_thread.start()
            st.rerun()
        else:
            st.warning("No files selected. Please upload files to analyze.")

# --- Processing & Results Display Logic ---
if st.session_state.is_processing:
    thread_is_alive = st.session_state.worker_thread and st.session_state.worker_thread.is_alive()
    if thread_is_alive:
        with st.spinner("⚙️ Analyzing files... Please hold on. This may take a few moments."):
            st.session_state.worker_thread.join(timeout=1.5) # Wait with timeout
        if st.session_state.worker_thread.is_alive(): # Still running
            st.rerun()
        else: # Finished within the timeout
            st.session_state.is_processing = False # Mark as done
            st.rerun() # Rerun to process results
    else: # Thread is confirmed not alive (or never started properly)
        st.session_state.is_processing = False
        results = st.session_state.thread_results
        if results: # Process results if they exist and haven't been processed
            st.session_state.logs.extend(results.get("log_messages", []))
            st.session_state.status_data = results.get("file_status_data", [])
            st.session_state.df_detailed = results.get("detailed_df", pd.DataFrame())
            st.session_state.df_summary = results.get("summary_df", pd.DataFrame())
            st.session_state.df_skipped = results.get("skipped_df", pd.DataFrame())
            st.session_state.final_message = results.get("final_status_message", "Processing finished.")
            
            if results.get("error_occurred", False): st.error(st.session_state.final_message)
            else: st.success(st.session_state.final_message)
            st.session_state.thread_results = {} # Clear processed results
            # No automatic rerun here, let the UI draw based on new states

# --- Display Sections (Status, Logs, Results Tables, Download) ---
if st.session_state.initiated_once and not st.session_state.is_processing:
    st.markdown("---")
    st.subheader("2. Processing Overview")

    col_status, col_logs = st.columns(2)
    with col_status:
        with st.expander("📊 File by File Status", expanded=True):
            if st.session_state.status_data:
                df_status_display = pd.DataFrame(st.session_state.status_data)
                def style_status(row):
                    colors = {'Processed': 'green', 'Skipped': 'orange', 'Error': 'red', 'Pending': 'grey'}
                    return [f'color: {colors.get(row.Status, "black")}'] * len(row)
                st.dataframe(df_status_display.style.apply(style_status, axis=1), height=300, use_container_width=True)
            else: st.caption("Status will appear here after processing.")
    with col_logs:
        with st.expander("📜 Detailed Log", expanded=True):
            log_text = "\n".join(st.session_state.logs)
            st.text_area("Log Output:", value=log_text, height=300, key="log_output_area", disabled=True)

    st.markdown("---")
    st.subheader("3. Analysis Results")

    if not st.session_state.df_summary.empty:
        with st.expander("📋 Summary Report", expanded=True):
            st.dataframe(st.session_state.df_summary, use_container_width=True)
    else: st.caption("Summary Report will appear here.")

    if not st.session_state.df_detailed.empty:
        with st.expander("📑 Detailed Term Counts", expanded=False): # Default collapsed
            st.dataframe(st.session_state.df_detailed, use_container_width=True)
    else: st.caption("Detailed Counts will appear here.")
    
    if not st.session_state.df_skipped.empty:
        with st.expander("🚫 Skipped/Errored Files", expanded=False): # Default collapsed
            st.dataframe(st.session_state.df_skipped.sort_values(by='Filename').reset_index(drop=True), use_container_width=True)
    
    # Download Button
    has_data = not st.session_state.df_summary.empty or not st.session_state.df_detailed.empty or not st.session_state.df_skipped.empty
    if has_data:
        st.markdown("---")
        st.subheader("4. Download Report")
        ts = int(time.time())
        out_fname = f"{config.OUTPUT_FILENAME_BASE}_{ts}.xlsx"
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            if not st.session_state.df_summary.empty: st.session_state.df_summary.to_excel(writer, sheet_name='Summary', index=False)
            if not st.session_state.df_detailed.empty: st.session_state.df_detailed.to_excel(writer, sheet_name='Detailed_Counts', index=False)
            if not st.session_state.df_skipped.empty: st.session_state.df_skipped.sort_values(by='Filename').reset_index(drop=True).to_excel(writer, sheet_name='Skipped_Files', index=False)
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Click to Download Excel Report",
            data=excel_buffer,
            file_name=out_fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    elif st.session_state.initiated_once: # If processing ran but no data
        st.info("No data was generated from the processed files to include in a downloadable report.")

elif not st.session_state.initiated_once:
    st.info("☝️ Upload files and click 'Analyze Selected Files' to begin.")

st.markdown("---")
st.caption("Excel Analyzer v1.1")
