import streamlit as st
import os # Still useful for config.OUTPUT_FILENAME_BASE if you want to suggest a save location
import pandas as pd
import re
import time
import threading
import logging
from io import BytesIO # For in-memory Excel file and for UploadedFile objects

# Import configuration
try:
    import config
except ImportError:
    st.error("Configuration file 'config.py' not found. Please create it in the same directory.")
    st.stop()

# Standard Python logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def extract_file_info_from_name(filename_str: str, log_fn):
    de_code = None
    submission_date = None
    sku_count = 0
    sku_type = 'Regular'

    de_match = re.search(r'(DE-\d+)', filename_str, re.IGNORECASE)
    if de_match:
        de_code = de_match.group(1).upper()

    date_match = re.search(r'(\w+\s+\d{1,2},\s+\d{4})', filename_str)
    if date_match:
        submission_date = date_match.group(1)

    sku_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*[-]?\s*(\b(?:Multi Urgent|Multi-Urgent|Multi-Regular|Multi Regular|Urgent|Regular|Priority)\b)?\s*SKUs?', filename_str, re.IGNORECASE)
    if sku_match:
        sku_count_str = sku_match.group(1)
        try:
            sku_count = int(sku_count_str.replace(',', ''))
        except ValueError:
            log_fn(f"Could not convert SKU count '{sku_count_str}' from filename: {filename_str}", "warning")
            sku_count = 0

        extracted_type_str = sku_match.group(2)
        if extracted_type_str:
            temp_type = extracted_type_str.strip().lower()
            if 'multi urgent' in temp_type: sku_type = 'Multi-Urgent'
            elif 'multi regular' in temp_type: sku_type = 'Multi-Regular'
            elif 'urgent' in temp_type: sku_type = 'Urgent'
            elif 'priority' in temp_type: sku_type = 'Priority'
            else: sku_type = extracted_type_str.strip().capitalize()

    if not de_code or not submission_date or sku_count <= 0:
        # log_fn(f"Filename '{filename_str}' did not yield all required info (DE, Date, SKU Count). DE: {de_code}, Date: {submission_date}, Count: {sku_count}", "debug")
        return None, None, 0, sku_type
    return de_code, submission_date, sku_count, sku_type

def should_process_file(filename_str: str):
    is_valid_extension = filename_str.lower().endswith(config.VALID_EXTENSIONS)
    contains_exclude_term = any(term.lower() in filename_str.lower() for term in config.EXCLUDE_FILENAME_TERMS)
    is_temp_file = filename_str.lower().startswith('~$') # Check for temp file prefix
    return is_valid_extension and not contains_exclude_term and not is_temp_file

def count_terms_in_comments(comments_text: str, terms_to_search: list[str]):
    counts = {term: 0 for term in terms_to_search}
    current_comments = comments_text.lower()
    lower_terms = [term.lower() for term in terms_to_search]
    simple_negation_pattern = r'\b(?:not|no|non)\s*(?:' + "|".join([re.escape(t) for t in lower_terms]) + r')\b'
    cleaned_comments = re.sub(simple_negation_pattern, ' ', current_comments, flags=re.IGNORECASE)
    cleaned_comments = re.sub(r'\s+', ' ', cleaned_comments).strip()
    priority_terms = (
        config.LARGE_BUNDLE_TERMS_CONFIG +
        config.SIMILAR_TERMS_CONFIG +
        config.PARENT_TERMS_CONFIG +
        [t for t in terms_to_search if t not in config.LARGE_BUNDLE_TERMS_CONFIG
                                      and t not in config.SIMILAR_TERMS_CONFIG
                                      and t not in config.PARENT_TERMS_CONFIG]
    )
    comment_list = re.split(r'[\n\.]+', cleaned_comments)
    for comment in comment_list:
        comment = comment.strip()
        if not comment: continue
        for term in priority_terms:
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            if re.search(pattern, comment, re.IGNORECASE):
                counts[term] += 1
                break
    return counts

# --- Main Processing Task (for uploaded files from memory) ---
def process_uploaded_files_task_worker(uploaded_files: list, results_container: dict):
    local_log_messages = []
    local_file_status_data = []
    detailed_data_list = []
    summary_data_list = []
    skipped_files_list = [] # Store (filename_str, reason) tuples
    final_status_msg = "Processing started..."

    def _log(message, level="info"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {level.upper()} - {message}"
        local_log_messages.append(log_entry)
        if level == "info": logger.info(message)
        elif level == "warning": logger.warning(message)
        elif level == "error": logger.error(message)
        elif level == "critical": logger.critical(message)

    try:
        _log(f"Starting processing for {len(uploaded_files)} uploaded files.")
        _log(f"Using terms to count: {', '.join(config.TERMS_TO_COUNT)}")
        _log("-------------------------------------------------")

        processed_count = 0
        candidate_files_info = [] # List of (UploadedFile, filename_str)

        for uploaded_file_obj in uploaded_files:
            filename_str = uploaded_file_obj.name # Get filename from UploadedFile object
            if should_process_file(filename_str):
                candidate_files_info.append((uploaded_file_obj, filename_str))
                local_file_status_data.append({
                    "Filename": filename_str, "Status": "Pending", "Reason": "Ready for processing"
                })
            else:
                reason = f"Excluded by initial criteria (extension: {any(filename_str.lower().endswith(ext) for ext in config.VALID_EXTENSIONS)}, exclude term: {any(term.lower() in filename_str.lower() for term in config.EXCLUDE_FILENAME_TERMS)}, temp file: {filename_str.lower().startswith('~$')})."
                skipped_files_list.append((filename_str, reason))
                local_file_status_data.append({
                    "Filename": filename_str, "Status": "Skipped", "Reason": reason
                })
        
        _log(f"Identified {len(candidate_files_info)} candidate files for processing.")
        initial_skips = len(uploaded_files) - len(candidate_files_info)
        if initial_skips > 0:
            _log(f"Skipped {initial_skips} files based on initial criteria (name/extension/temp).")

        # Sort by the filename string for consistent processing order
        sorted_candidate_files_info = sorted(candidate_files_info, key=lambda x: x[1])

        for uploaded_file_obj, filename_str in sorted_candidate_files_info:
            for f_dict in local_file_status_data:
                if f_dict["Filename"] == filename_str:
                    f_dict["Status"] = "Processing"; f_dict["Reason"] = "Extracting info..."; break
            _log(f"Attempting to process file: {filename_str}")
            file_had_error = False

            de_code, submission_date, total_skus, sku_type = extract_file_info_from_name(filename_str, _log)

            if not de_code or not submission_date or total_skus <= 0:
                reason = "Invalid filename format or missing essential info (DE code, Date, or SKU count <= 0)."
                _log(f"SKIPPING: {filename_str}. Reason: {reason}", "warning")
                skipped_files_list.append((filename_str, reason))
                for f_dict in local_file_status_data:
                    if f_dict["Filename"] == filename_str: f_dict["Status"] = "Skipped"; f_dict["Reason"] = reason; break
                file_had_error = True
            else:
                for f_dict in local_file_status_data:
                    if f_dict["Filename"] == filename_str: f_dict["Reason"] = "Loading Excel data..."; break
            
            try:
                if not file_had_error:
                    # uploaded_file_obj is already an in-memory BytesIO-like object
                    # Reset buffer position just in case it was read before (though unlikely for new UploadedFile)
                    uploaded_file_obj.seek(0)
                    xl = pd.ExcelFile(uploaded_file_obj, engine='openpyxl')
                    sheet_names = xl.sheet_names
                    de_sheets = [sheet for sheet in sheet_names if sheet.upper().startswith('DE-')]

                    if not de_sheets:
                        reason = "No sheets found starting with 'DE-'."
                        _log(f"SKIPPING: {filename_str}. Reason: {reason}", "warning")
                        skipped_files_list.append((filename_str, reason))
                        for f_dict in local_file_status_data:
                            if f_dict["Filename"] == filename_str: f_dict["Status"] = "Skipped"; f_dict["Reason"] = reason; break
                        file_had_error = True
                    else:
                        df = pd.read_excel(xl, sheet_name=de_sheets[0], engine='openpyxl', keep_default_na=False)
                        _log(f"Processing sheet '{de_sheets[0]}' from {filename_str}")

                    if not file_had_error and df.empty:
                        reason = "Excel sheet is empty."
                        _log(f"SKIPPING: {filename_str}. Reason: {reason}", "warning")
                        skipped_files_list.append((filename_str, reason))
                        for f_dict in local_file_status_data:
                            if f_dict["Filename"] == filename_str: f_dict["Status"] = "Skipped"; f_dict["Reason"] = reason; break
                        file_had_error = True

                    if not file_had_error:
                        for f_dict in local_file_status_data:
                            if f_dict["Filename"] == filename_str: f_dict["Reason"] = "Finding comments column..."; break
                        comments_series = None; comments_col_name_used = None
                        if config.COMMENTS_COLUMN_PRIMARY in df.columns:
                            comments_series = df[config.COMMENTS_COLUMN_PRIMARY]; comments_col_name_used = config.COMMENTS_COLUMN_PRIMARY
                        elif df.shape[1] > config.COMMENTS_COLUMN_FALLBACK_INDEX:
                            try:
                                comments_series = df.iloc[:, config.COMMENTS_COLUMN_FALLBACK_INDEX]
                                comments_col_name_used = str(df.columns[config.COMMENTS_COLUMN_FALLBACK_INDEX])
                                _log(f"INFO: Used fallback column '{comments_col_name_used}' (index {config.COMMENTS_COLUMN_FALLBACK_INDEX}) for comments in {filename_str}.")
                            except IndexError: _log(f"Fallback column index {config.COMMENTS_COLUMN_FALLBACK_INDEX} is out of bounds for {filename_str}.", "error")
                        
                        if comments_series is None:
                            reason = f"Comments column ('{config.COMMENTS_COLUMN_PRIMARY}' or index {config.COMMENTS_COLUMN_FALLBACK_INDEX}) not found."
                            _log(f"SKIPPING: {filename_str}. Reason: {reason}", "warning")
                            skipped_files_list.append((filename_str, reason))
                            for f_dict in local_file_status_data:
                                if f_dict["Filename"] == filename_str: f_dict["Status"] = "Skipped"; f_dict["Reason"] = reason; break
                            file_had_error = True
                        
                        if not file_had_error:
                            for f_dict in local_file_status_data:
                                if f_dict["Filename"] == filename_str: f_dict["Reason"] = "Analyzing comments..."; break
                            comments_series_str = comments_series.fillna('').astype(str)
                            combined_comments = ' '.join(comments_series_str.tolist()).strip()
                            if not combined_comments: _log(f"No non-empty comments found in column '{comments_col_name_used}' for {filename_str}. Proceeding with zero counts.")
                            
                            term_counts = count_terms_in_comments(combined_comments, config.TERMS_TO_COUNT)
                            detailed_row = [de_code, submission_date, filename_str, combined_comments] + \
                                           [term_counts.get(term, 0) for term in config.TERMS_TO_COUNT]
                            detailed_data_list.append(detailed_row)

                            for f_dict in local_file_status_data:
                                if f_dict["Filename"] == filename_str: f_dict["Reason"] = "Categorizing SKUs..."; break
                            large_bundle_count = sum(term_counts.get(term, 0) for term in config.LARGE_BUNDLE_TERMS_CONFIG)
                            similar_count = sum(term_counts.get(term, 0) for term in config.SIMILAR_TERMS_CONFIG)
                            regular_priority_count_val, urgent_count_val = 0, 0
                            file_sku_type_lower = sku_type.lower()

                            if 'urgent' in file_sku_type_lower: 
                                urgent_count_val = total_skus - large_bundle_count - similar_count
                                if urgent_count_val < 0: _log(f"Negative Urgent count for {filename_str} ({urgent_count_val}). Setting to 0.", "warning"); urgent_count_val = 0
                            else: 
                                regular_priority_count_val = total_skus - large_bundle_count - similar_count
                                if regular_priority_count_val < 0: _log(f"Negative Regular/Priority count for {filename_str} ({regular_priority_count_val}). Setting to 0.", "warning"); regular_priority_count_val = 0
                            
                            base_filename_str, ext = os.path.splitext(filename_str) # Use os.path for consistency
                            filename_for_summary = re.sub(r'^.*? - ', '', base_filename_str) + ext # Regex on filename_str
                            if sku_type.lower() in ['multi-regular', 'multi-urgent'] or not submission_date:
                                 filename_for_summary = f"{total_skus} {sku_type} SKUs - {submission_date}{ext}" if submission_date else f"{total_skus} {sku_type} SKUs{ext}"
                            elif not de_code:
                                 filename_for_summary = f"{total_skus} {sku_type} SKUs - {submission_date}{ext}" if submission_date else filename_str
                            
                            summary_row = [de_code, submission_date, total_skus,
                                           regular_priority_count_val if regular_priority_count_val > 0 else '', 
                                           large_bundle_count if large_bundle_count > 0 else '',
                                           similar_count if similar_count > 0 else '',
                                           urgent_count_val if urgent_count_val > 0 else '',
                                           filename_for_summary]
                            summary_data_list.append(summary_row)
                            
                            _log(f"Successfully processed: {filename_str}")
                            for f_dict in local_file_status_data:
                                if f_dict["Filename"] == filename_str: f_dict["Status"] = "Processed"; f_dict["Reason"] = "Data extracted and categorized"; break
                            processed_count += 1
            except Exception as e_file:
                reason = f"Processing error in {filename_str}: {type(e_file).__name__} - {e_file}"
                _log(reason, "error")
                logger.exception(f"Full traceback for error in {filename_str}:") # Standard logger for full traceback
                if not file_had_error:
                    skipped_files_list.append((filename_str, reason))
                    for f_dict in local_file_status_data:
                        if f_dict["Filename"] == filename_str: f_dict["Status"] = "Error"; f_dict["Reason"] = reason; break
        # End of for loop over sorted_candidate_files_info

        _log("-------------------------------------------------")
        _log("Processing Summary:")
        _log(f"Total files uploaded and considered: {len(uploaded_files)}")
        _log(f"Total candidate files meeting criteria: {len(sorted_candidate_files_info)}")
        _log(f"Total files processed successfully: {processed_count}")
        _log(f"Total files skipped or errored: {len(skipped_files_list)}")
        _log("-------------------------------------------------")

        detailed_columns = ['DE Code', 'Submission Date', 'Source Filename', 'Combined Comments'] + config.TERMS_TO_COUNT
        summary_columns = ['JIRA/Input', 'Submission Date', 'Total SKUs', 'Regular / Priority',
                           'Large / Bundle', 'Similar', 'Urgent', 'File Name']
        detailed_df = pd.DataFrame(detailed_data_list, columns=detailed_columns)
        summary_df = pd.DataFrame(summary_data_list, columns=summary_columns)
        skipped_df = pd.DataFrame(skipped_files_list, columns=['Filename', 'Reason'])

        if not detailed_df.empty:
            totals_detailed_values = {'DE Code': 'Total', 'Submission Date': '', 'Source Filename': '', 'Combined Comments': ''}
            for term in config.TERMS_TO_COUNT:
                if term in detailed_df.columns: totals_detailed_values[term] = pd.to_numeric(detailed_df[term], errors='coerce').sum()
            detailed_df = pd.concat([detailed_df, pd.DataFrame([totals_detailed_values])], ignore_index=True)
        if not summary_df.empty:
            numeric_cols_summary = ['Total SKUs', 'Regular / Priority', 'Large / Bundle', 'Similar', 'Urgent']
            totals_summary_values = {'JIRA/Input': 'Total', 'Submission Date': '', 'File Name': ''}
            for col in numeric_cols_summary:
                if col in summary_df.columns: totals_summary_values[col] = pd.to_numeric(summary_df[col], errors='coerce').sum()
            summary_df = pd.concat([summary_df, pd.DataFrame([totals_summary_values])], ignore_index=True)
        
        if not detailed_data_list and not summary_data_list and not skipped_files_list:
            final_status_msg = "No data processed or skipped from uploaded files."
        else:
            final_status_msg = "Processing of uploaded files complete. Results below."
        _log(final_status_msg)

        results_container.update({
            "log_messages": local_log_messages, "file_status_data": local_file_status_data,
            "detailed_df": detailed_df, "summary_df": summary_df, "skipped_df": skipped_df,
            "final_status_message": final_status_msg, "error_occurred": False
        })

    except Exception as e_task_critical:
        _log(f"UNHANDLED CRITICAL ERROR in processing task: {e_task_critical}", "critical")
        logger.exception("Unhandled critical error in process_uploaded_files_task_worker:")
        results_container.update({
            "log_messages": local_log_messages, "file_status_data": local_file_status_data,
            "detailed_df": pd.DataFrame(), "summary_df": pd.DataFrame(), "skipped_df": pd.DataFrame(skipped_files_list, columns=['Filename', 'Reason']),
            "final_status_message": f"Critical Error: {e_task_critical}. Check console logs.",
            "error_occurred": True
        })

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📁 Excel File Analyzer (Upload Files from Folder)")

# Initialize session state
default_session_state = {
    "is_processing": False,
    "processing_initiated_once": False,
    "log_messages": ["Welcome! Please upload Excel files from your target folder."],
    "file_status_data": [],
    "detailed_df": pd.DataFrame(),
    "summary_df": pd.DataFrame(),
    "skipped_df": pd.DataFrame(),
    "final_processing_message": "",
    "worker_thread": None,
    "thread_results": {}
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("""
**Instructions:**
1. Click "Browse files" below.
2. Navigate to the folder containing your Excel task files.
3. Select **all** the Excel files you want to process within that folder (e.g., using Ctrl+A or Cmd+A).
4. Click "Open".
5. Then, click the "🚀 Process Uploaded Files" button.
""")

uploaded_files = st.file_uploader(
    "Upload Excel files (.xlsx, .xlsm) from your folder",
    type=["xlsx", "xlsm"],
    accept_multiple_files=True,
    key="file_uploader_widget",
    disabled=st.session_state.is_processing
)

if st.button("🚀 Process Uploaded Files", disabled=st.session_state.is_processing or not uploaded_files):
    if uploaded_files:
        st.session_state.is_processing = True
        st.session_state.processing_initiated_once = True
        st.session_state.log_messages = [f"--- Starting processing for {len(uploaded_files)} uploaded files ---"]
        st.session_state.file_status_data = [] # Clear previous status for UI
        st.session_state.detailed_df = pd.DataFrame()
        st.session_state.summary_df = pd.DataFrame()
        st.session_state.skipped_df = pd.DataFrame()
        st.session_state.final_processing_message = ""
        st.session_state.thread_results = {} # Reset results container

        st.session_state.worker_thread = threading.Thread(
            target=process_uploaded_files_task_worker,
            args=(uploaded_files, st.session_state.thread_results)
        )
        st.session_state.worker_thread.start()
        st.rerun()
    else:
        st.warning("Please upload at least one Excel file.")

# Monitoring loop and results processing
if st.session_state.is_processing:
    thread_alive = st.session_state.worker_thread and st.session_state.worker_thread.is_alive()
    if thread_alive:
        with st.spinner("Processing files... Logs and status will update upon completion."):
            st.session_state.worker_thread.join(timeout=1.0) # Wait for 1 sec, then check again
        if st.session_state.worker_thread.is_alive(): # If still alive after timeout
             st.rerun()
        else: # Thread finished just after timeout check
            st.session_state.is_processing = False
            # (Results processing will happen in the next block due to is_processing now being False)
            st.rerun() # Rerun to process results
    else: # Thread is confirmed not alive
        st.session_state.is_processing = False # Ensure it's false
        results = st.session_state.thread_results
        if results: # Process results only if they exist
            st.session_state.log_messages.extend(results.get("log_messages", []))
            st.session_state.file_status_data = results.get("file_status_data", [])
            st.session_state.detailed_df = results.get("detailed_df", pd.DataFrame())
            st.session_state.summary_df = results.get("summary_df", pd.DataFrame())
            st.session_state.skipped_df = results.get("skipped_df", pd.DataFrame())
            st.session_state.final_processing_message = results.get("final_status_message", "Processing finished.")
            
            if results.get("error_occurred", False):
                st.error(st.session_state.final_processing_message)
            else:
                st.success(st.session_state.final_processing_message)
            st.session_state.thread_results = {} # Clear results after processing
            # No st.rerun() here, UI will update with current states. If needed, it happens after success/error.
        else:
            if st.session_state.processing_initiated_once and not st.session_state.final_processing_message:
                # This case can happen if thread finishes but results somehow weren't set, or app reran before results were fully processed
                st.warning("Processing seemed to finish, but no results were found. Please try again or check console logs.")


# --- Display Areas ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 File Processing Status")
    if st.session_state.file_status_data:
        status_df_display = pd.DataFrame(st.session_state.file_status_data)
        def style_status(row):
            color = 'black'
            if row['Status'] == 'Processed': color = 'green'
            elif row['Status'] == 'Skipped': color = 'orange'
            elif row['Status'] == 'Error': color = 'red'
            elif row['Status'] == 'Pending': color = 'grey'
            return [f'color: {color}'] * len(row)
        st.dataframe(status_df_display.style.apply(style_status, axis=1), height=300, use_container_width=True)
    elif st.session_state.processing_initiated_once and not st.session_state.is_processing:
         st.caption("Status table populated after processing.")
    elif not st.session_state.processing_initiated_once:
        st.caption("Upload files and click process to see status.")

with col2:
    st.subheader("📜 Processing Log")
    st.text_area("Logs:", value="\n".join(st.session_state.log_messages), height=300, key="log_display_uploader", disabled=True)

st.markdown("---")
st.subheader("📄 Output Results")

# Show results only if processing was initiated and is no longer active
if st.session_state.processing_initiated_once and not st.session_state.is_processing:
    if not st.session_state.summary_df.empty:
        st.markdown("#### Summary Report"); st.dataframe(st.session_state.summary_df, use_container_width=True)
    if not st.session_state.detailed_df.empty:
        st.markdown("#### Detailed Term Counts"); st.dataframe(st.session_state.detailed_df, use_container_width=True)
    if not st.session_state.skipped_df.empty:
        st.markdown("#### Skipped/Errored Files"); st.dataframe(st.session_state.skipped_df.sort_values(by='Filename').reset_index(drop=True), use_container_width=True)

    has_data_to_download = not st.session_state.summary_df.empty or \
                           not st.session_state.detailed_df.empty or \
                           not st.session_state.skipped_df.empty
    if has_data_to_download:
        output_filename = f"{config.OUTPUT_FILENAME_BASE}_uploaded_{int(time.time())}.xlsx"
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            if not st.session_state.detailed_df.empty: st.session_state.detailed_df.to_excel(writer, index=False, sheet_name='Detailed Counts')
            if not st.session_state.summary_df.empty: st.session_state.summary_df.to_excel(writer, index=False, sheet_name='Summary')
            if not st.session_state.skipped_df.empty: st.session_state.skipped_df.sort_values(by='Filename').reset_index(drop=True).to_excel(writer, index=False, sheet_name='Skipped Files')
        excel_buffer.seek(0)
        st.download_button(label="📥 Download Report as Excel", data=excel_buffer, file_name=output_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    elif not st.session_state.is_processing: # If processing done but no data
        st.caption("No data was generated to download for this run.")
elif not st.session_state.processing_initiated_once:
    st.caption("Process files to generate results and a downloadable report.")
