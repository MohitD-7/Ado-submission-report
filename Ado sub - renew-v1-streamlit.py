import streamlit as st
import os
import pandas as pd
import re
import time
import threading
import logging
import sys
from io import BytesIO

# Import configuration
try:
    import config
except ImportError:
    st.error("Configuration file 'config.py' not found. Please create it in the same directory.")
    st.stop()

# Standard Python logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper functions (extract_file_info, should_process_file, count_terms_in_comments) ---
# These remain largely the same but will use the local _log function inside the task.
def extract_file_info(filename: str, log_fn): # Pass logger function
    de_code = None
    submission_date = None
    sku_count = 0
    sku_type = 'Regular'

    de_match = re.search(r'(DE-\d+)', filename, re.IGNORECASE)
    if de_match:
        de_code = de_match.group(1).upper()

    date_match = re.search(r'(\w+\s+\d{1,2},\s+\d{4})', filename)
    if date_match:
        submission_date = date_match.group(1)

    sku_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*[-]?\s*(\b(?:Multi Urgent|Multi-Urgent|Multi-Regular|Multi Regular|Urgent|Regular|Priority)\b)?\s*SKUs?', filename, re.IGNORECASE)
    if sku_match:
        sku_count_str = sku_match.group(1)
        try:
            sku_count = int(sku_count_str.replace(',', ''))
        except ValueError:
            log_fn(f"Could not convert SKU count '{sku_count_str}' from filename: {filename}", "warning")
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
        return None, None, 0, sku_type
    return de_code, submission_date, sku_count, sku_type

def should_process_file(filename: str): # No logging needed here
    is_valid_extension = filename.lower().endswith(config.VALID_EXTENSIONS)
    contains_exclude_term = any(term.lower() in filename.lower() for term in config.EXCLUDE_FILENAME_TERMS)
    is_temp_file = filename.lower().startswith('~$')
    return is_valid_extension and not contains_exclude_term and not is_temp_file

def count_terms_in_comments(comments_text: str, terms_to_search: list[str]): # No logging needed here
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


# --- Main Processing Task ---
# This function will now store results in the 'results_container' dict passed to it.
def process_excel_files_task_worker(folder_path: str, results_container: dict):
    local_log_messages = []
    local_file_status_data = [] # List of dicts to build the status table
    detailed_data_list = []
    summary_data_list = []
    skipped_files_list = []
    final_status_msg = "Processing started..."

    # Internal logging function for this worker
    def _log(message, level="info"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {level.upper()} - {message}"
        local_log_messages.append(log_entry)
        if level == "info": logger.info(message)
        elif level == "warning": logger.warning(message)
        elif level == "error": logger.error(message)
        elif level == "critical": logger.critical(message)

    try:
        _log(f"Starting processing for folder: {folder_path}")
        _log(f"Using terms: {', '.join(config.TERMS_TO_COUNT)}")
        _log("-------------------------------------------------")

        processed_count = 0
        all_items = []

        try:
            all_items = os.listdir(folder_path)
            _log(f"Found {len(all_items)} items in folder.")

            candidate_files = []
            for item_name in sorted(all_items):
                item_path = os.path.join(folder_path, item_name)
                if os.path.isfile(item_path):
                    if should_process_file(item_name):
                        candidate_files.append(item_name)
                        local_file_status_data.append({
                            "Filename": item_name, "Status": "Pending", "Reason": "Ready for processing"
                        })
                    else:
                        reason = "Excluded by initial criteria (extension, name term, or temp file)."
                        skipped_files_list.append((item_name, reason))
                        local_file_status_data.append({
                            "Filename": item_name, "Status": "Skipped", "Reason": reason
                        })
            _log(f"Identified {len(candidate_files)} candidate Excel files for processing.")
            initial_skips = sum(1 for f_dict in local_file_status_data if f_dict["Status"] == "Skipped" and "initial criteria" in f_dict["Reason"])
            _log(f"Skipped {initial_skips} items based on initial criteria.")

            for filename in sorted(candidate_files):
                # Update local status (won't be visible live in UI)
                for f_dict in local_file_status_data:
                    if f_dict["Filename"] == filename:
                        f_dict["Status"] = "Processing"
                        f_dict["Reason"] = "Extracting info..."
                        break
                _log(f"Attempting to process file: {filename}")
                file_had_error = False

                de_code, submission_date, total_skus, sku_type = extract_file_info(filename, _log) # Pass _log

                if not de_code or not submission_date or total_skus <= 0:
                    reason = "Invalid filename format or missing essential info (DE code, Date, or SKU count <= 0)."
                    _log(f"SKIPPING: {filename}. Reason: {reason}", "warning")
                    skipped_files_list.append((filename, reason))
                    for f_dict in local_file_status_data:
                        if f_dict["Filename"] == filename: f_dict["Status"] = "Skipped"; f_dict["Reason"] = reason; break
                    file_had_error = True
                # ... (rest of your core file processing logic, using _log and local_file_status_data) ...
                else: # Filename parsing OK
                    for f_dict in local_file_status_data:
                        if f_dict["Filename"] == filename: f_dict["Reason"] = "Loading Excel..."; break
                
                try: # Inner try for individual file processing
                    if not file_had_error:
                        file_path = os.path.join(folder_path, filename)
                        xl = pd.ExcelFile(file_path, engine='openpyxl')
                        sheet_names = xl.sheet_names
                        de_sheets = [sheet for sheet in sheet_names if sheet.upper().startswith('DE-')]
                        
                        if not de_sheets:
                            reason = "No sheets found starting with 'DE-'."
                            _log(f"SKIPPING: {filename}. Reason: {reason}", "warning")
                            skipped_files_list.append((filename, reason))
                            for f_dict in local_file_status_data:
                                if f_dict["Filename"] == filename: f_dict["Status"] = "Skipped"; f_dict["Reason"] = reason; break
                            file_had_error = True
                        else:
                            df = pd.read_excel(file_path, sheet_name=de_sheets[0], engine='openpyxl', keep_default_na=False)
                            _log(f"Processing sheet '{de_sheets[0]}' from {filename}")

                        if not file_had_error and df.empty:
                            reason = "File is empty."
                            _log(f"SKIPPING: {filename}. Reason: {reason}", "warning")
                            skipped_files_list.append((filename, reason))
                            for f_dict in local_file_status_data:
                                if f_dict["Filename"] == filename: f_dict["Status"] = "Skipped"; f_dict["Reason"] = reason; break
                            file_had_error = True

                        if not file_had_error:
                            for f_dict in local_file_status_data:
                                if f_dict["Filename"] == filename: f_dict["Reason"] = "Finding comments column..."; break
                            comments_series = None; comments_col_name_used = None
                            if config.COMMENTS_COLUMN_PRIMARY in df.columns:
                                comments_series = df[config.COMMENTS_COLUMN_PRIMARY]; comments_col_name_used = config.COMMENTS_COLUMN_PRIMARY
                            elif df.shape[1] > config.COMMENTS_COLUMN_FALLBACK_INDEX:
                                try:
                                    comments_series = df.iloc[:, config.COMMENTS_COLUMN_FALLBACK_INDEX]
                                    comments_col_name_used = str(df.columns[config.COMMENTS_COLUMN_FALLBACK_INDEX])
                                    _log(f"INFO: Used fallback column '{comments_col_name_used}' for comments in {filename}.")
                                except IndexError: _log(f"Fallback column index out of bounds for {filename}.", "error")
                            if comments_series is None:
                                reason = f"Comments column ('{config.COMMENTS_COLUMN_PRIMARY}' or index {config.COMMENTS_COLUMN_FALLBACK_INDEX}) not found."
                                _log(f"SKIPPING: {filename}. Reason: {reason}", "warning")
                                skipped_files_list.append((filename, reason))
                                for f_dict in local_file_status_data:
                                    if f_dict["Filename"] == filename: f_dict["Status"] = "Skipped"; f_dict["Reason"] = reason; break
                                file_had_error = True
                            
                            if not file_had_error:
                                for f_dict in local_file_status_data:
                                    if f_dict["Filename"] == filename: f_dict["Reason"] = "Analyzing comments..."; break
                                comments_series_str = comments_series.fillna('').astype(str)
                                combined_comments = ' '.join(comments_series_str.tolist()).strip()
                                if not combined_comments: _log(f"No non-empty comments in '{comments_col_name_used}' for {filename}.")
                                term_counts = count_terms_in_comments(combined_comments, config.TERMS_TO_COUNT)
                                detailed_row = [de_code, submission_date, filename, combined_comments] + [term_counts.get(term, 0) for term in config.TERMS_TO_COUNT]
                                detailed_data_list.append(detailed_row)

                                for f_dict in local_file_status_data:
                                    if f_dict["Filename"] == filename: f_dict["Reason"] = "Categorizing SKUs..."; break
                                large_bundle_count = sum(term_counts.get(term, 0) for term in config.LARGE_BUNDLE_TERMS_CONFIG)
                                similar_count = sum(term_counts.get(term, 0) for term in config.SIMILAR_TERMS_CONFIG)
                                regular_priority_count_val, urgent_count_val = 0, 0
                                if 'urgent' in sku_type.lower():
                                    urgent_count_val = total_skus - large_bundle_count - similar_count
                                    if urgent_count_val < 0: _log(f"Negative Urgent count for {filename}.", "warning"); urgent_count_val = 0
                                else:
                                    regular_priority_count_val = total_skus - large_bundle_count - similar_count
                                    if regular_priority_count_val < 0: _log(f"Negative Reg/Pri count for {filename}.", "warning"); regular_priority_count_val = 0
                                base_filename, ext = os.path.splitext(filename)
                                filename_for_summary = re.sub(r'^.*? - ', '', base_filename) + ext
                                if sku_type.lower() in ['multi-regular', 'multi-urgent'] or not submission_date or not de_code:
                                     filename_for_summary = f"{total_skus} {sku_type} SKUs - {submission_date}{ext}" if submission_date else f"{total_skus} {sku_type} SKUs{ext}"
                                summary_row = [de_code, submission_date, total_skus,
                                               regular_priority_count_val or '', large_bundle_count or '',
                                               similar_count or '', urgent_count_val or '', filename_for_summary]
                                summary_data_list.append(summary_row)
                                _log(f"Successfully processed: {filename}")
                                for f_dict in local_file_status_data:
                                    if f_dict["Filename"] == filename: f_dict["Status"] = "Processed"; f_dict["Reason"] = "Data extracted"; break
                                processed_count += 1
                except Exception as e_file:
                    reason = f"Processing error: {type(e_file).__name__} - {e_file}"
                    _log(f"ERROR processing {filename}: {e_file}", "error")
                    logger.exception(f"Full traceback for error in {filename}:")
                    if not file_had_error:
                         skipped_files_list.append((filename, reason))
                         for f_dict in local_file_status_data:
                             if f_dict["Filename"] == filename: f_dict["Status"] = "Error"; f_dict["Reason"] = reason; break
            # End of for loop for candidate_files
        except FileNotFoundError:
            _log(f"ERROR: Folder not found: {folder_path}", "critical")
            final_status_msg = f"Error: Folder not found: {folder_path}"
            # Populate results_container directly here as we are exiting
            results_container.update({
                "log_messages": local_log_messages, "file_status_data": local_file_status_data,
                "detailed_df": pd.DataFrame(), "summary_df": pd.DataFrame(), "skipped_df": pd.DataFrame(skipped_files_list, columns=['Filename', 'Reason']),
                "final_status_message": final_status_msg, "error_occurred": True
            })
            return
        except Exception as e_outer:
            _log(f"A critical error occurred during file listing/loop: {e_outer}", "critical")
            logger.exception("Critical error in processing task outer loop:")
            final_status_msg = f"Critical Error: {e_outer}"
            results_container.update({
                "log_messages": local_log_messages, "file_status_data": local_file_status_data,
                "detailed_df": pd.DataFrame(), "summary_df": pd.DataFrame(), "skipped_df": pd.DataFrame(skipped_files_list, columns=['Filename', 'Reason']),
                "final_status_message": final_status_msg, "error_occurred": True
            })
            return

        _log("-------------------------------------------------")
        _log("Processing Summary:")
        _log(f"Total items found in folder: {len(all_items)}")
        _log(f"Total candidate Excel files evaluated: {len(candidate_files) if 'candidate_files' in locals() else 0}")
        _log(f"Total files processed successfully: {processed_count}")
        _log(f"Total files skipped or errored: {len(skipped_files_list)}")
        _log("-------------------------------------------------")

        detailed_columns = ['DE Code', 'Submission Date', 'Source Filename', 'Combined Comments'] + config.TERMS_TO_COUNT
        summary_columns = ['JIRA/Input', 'Submission Date', 'Total SKUs', 'Regular / Priority',
                           'Large / Bundle', 'Similar', 'Urgent', 'File Name']
        detailed_df = pd.DataFrame(detailed_data_list, columns=detailed_columns)
        summary_df = pd.DataFrame(summary_data_list, columns=summary_columns)
        skipped_df = pd.DataFrame(skipped_files_list, columns=['Filename', 'Reason'])

        if not detailed_df.empty: # Add totals rows
            totals_detailed_values = {'DE Code': 'Total'}
            for term in config.TERMS_TO_COUNT:
                if term in detailed_df.columns: totals_detailed_values[term] = pd.to_numeric(detailed_df[term], errors='coerce').sum()
            detailed_df = pd.concat([detailed_df, pd.DataFrame([totals_detailed_values])], ignore_index=True)
        if not summary_df.empty:
            numeric_cols_summary = ['Total SKUs', 'Regular / Priority', 'Large / Bundle', 'Similar', 'Urgent']
            totals_summary_values = {'JIRA/Input': 'Total'}
            for col in numeric_cols_summary:
                if col in summary_df.columns: totals_summary_values[col] = pd.to_numeric(summary_df[col], errors='coerce').sum()
            summary_df = pd.concat([summary_df, pd.DataFrame([totals_summary_values])], ignore_index=True)
        
        if not detailed_data_list and not summary_data_list and not skipped_files_list:
            final_status_msg = "No data processed or skipped."
            _log("No data to generate an output file.")
        else:
            final_status_msg = "Processing complete. Results below."
            _log("Output ready for download.")
        
        results_container.update({
            "log_messages": local_log_messages, "file_status_data": local_file_status_data,
            "detailed_df": detailed_df, "summary_df": summary_df, "skipped_df": skipped_df,
            "final_status_message": final_status_msg, "error_occurred": False
        })

    except Exception as e_task_critical: # Catch-all for the entire task function
        _log(f"UNHANDLED CRITICAL ERROR in processing task: {e_task_critical}", "critical")
        logger.exception("Unhandled critical error in process_excel_files_task_worker:")
        final_status_msg = f"Critical Error: {e_task_critical}. Check console logs."
        results_container.update({
            "log_messages": local_log_messages, # Include any logs captured so far
            "file_status_data": local_file_status_data, # Include any statuses captured
            "detailed_df": pd.DataFrame(), "summary_df": pd.DataFrame(), "skipped_df": pd.DataFrame(),
            "final_status_message": final_status_msg, "error_occurred": True
        })


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📁 Excel File Analyzer")

# Initialize session state
default_session_state = {
    "folder_path": "",
    "is_processing": False,
    "processing_initiated_once": False,
    "log_messages": ["Welcome! Enter a folder path and click 'Process'."],
    "file_status_data": [],
    "detailed_df": pd.DataFrame(),
    "summary_df": pd.DataFrame(),
    "skipped_df": pd.DataFrame(),
    "final_processing_message": "",
    "worker_thread": None,
    "thread_results": {} # To store results from the thread
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

folder_path_input = st.text_input(
    "Enter the full path to the folder containing Excel files:",
    value=st.session_state.folder_path,
    key="folder_path_widget",
    disabled=st.session_state.is_processing
)
st.session_state.folder_path = folder_path_input

if st.button("🚀 Process Files", disabled=st.session_state.is_processing):
    if not st.session_state.folder_path or not os.path.isdir(st.session_state.folder_path):
        st.error("Invalid or no folder path provided. Please enter a valid folder path.")
    else:
        st.session_state.is_processing = True
        st.session_state.processing_initiated_once = True
        # Clear previous results and logs for the new run
        st.session_state.log_messages = [f"--- Starting new processing task for: {st.session_state.folder_path} ---"]
        st.session_state.file_status_data = []
        st.session_state.detailed_df = pd.DataFrame()
        st.session_state.summary_df = pd.DataFrame()
        st.session_state.skipped_df = pd.DataFrame()
        st.session_state.final_processing_message = ""
        st.session_state.thread_results = {} # Reset results container

        st.session_state.worker_thread = threading.Thread(
            target=process_excel_files_task_worker,
            args=(st.session_state.folder_path, st.session_state.thread_results) # Pass results dict
        )
        st.session_state.worker_thread.start()
        st.rerun()

if st.session_state.is_processing:
    if st.session_state.worker_thread and st.session_state.worker_thread.is_alive():
        with st.spinner("Processing files... Please wait. Logs and status will update upon completion."):
            # We don't call join() here in the spinner to keep UI responsive if needed,
            # but for this simple case, join() is fine. Let's use a timeout on join.
            st.session_state.worker_thread.join(timeout=1.0) # Wait for 1 sec
        if st.session_state.worker_thread.is_alive(): # If still alive after timeout
             st.rerun() # Rerun to continue showing spinner and checking
        else: # Thread finished
            st.session_state.is_processing = False # Mark as not processing
             # Process results now that thread is done
            results = st.session_state.thread_results
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
            st.rerun() # Rerun one last time to display all results and enable button

    elif not (st.session_state.worker_thread and st.session_state.worker_thread.is_alive()):
        # This case handles if thread finishes very quickly or if state is inconsistent
        st.session_state.is_processing = False
        # Attempt to process results if they exist, similar to above
        if st.session_state.thread_results:
            results = st.session_state.thread_results
            st.session_state.log_messages.extend(results.get("log_messages", []))
            st.session_state.file_status_data = results.get("file_status_data", [])
            # ... (update other dfs and final message as above) ...
            st.session_state.final_processing_message = results.get("final_status_message", "Processing finished.")
            if results.get("error_occurred", False): st.error(st.session_state.final_processing_message)
            else: st.success(st.session_state.final_processing_message)
        # No st.rerun() here, or it might loop if results are not fully processed.
        # The main UI drawing below should handle it.


# --- Display Areas ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 File Processing Status")
    if st.session_state.file_status_data:
        status_df_display = pd.DataFrame(st.session_state.file_status_data)
        def style_status(row): # Styling function
            color = 'black'
            if row['Status'] == 'Processed': color = 'green'
            elif row['Status'] == 'Skipped': color = 'orange'
            elif row['Status'] == 'Error': color = 'red'
            elif row['Status'] == 'Pending': color = 'grey'
            # 'Processing' status won't be visible live with this model
            return [f'color: {color}'] * len(row)
        st.dataframe(status_df_display.style.apply(style_status, axis=1), height=300)
    elif st.session_state.processing_initiated_once and not st.session_state.is_processing:
         st.caption("Status table will populate after processing completes.")
    elif not st.session_state.processing_initiated_once:
        st.caption("No files processed yet.")


with col2:
    st.subheader("📜 Processing Log")
    log_text_area = st.text_area(
        "Logs:",
        value="\n".join(st.session_state.log_messages),
        height=300,
        key="log_display_deferred", # New key
        disabled=True
    )

st.markdown("---")
st.subheader("📄 Output Results")

if st.session_state.processing_initiated_once and not st.session_state.is_processing: # Show results only if processing finished
    if not st.session_state.summary_df.empty:
        st.markdown("#### Summary Report")
        st.dataframe(st.session_state.summary_df)
    if not st.session_state.detailed_df.empty:
        st.markdown("#### Detailed Term Counts")
        st.dataframe(st.session_state.detailed_df)
    if not st.session_state.skipped_df.empty:
        st.markdown("#### Skipped/Errored Files")
        st.dataframe(st.session_state.skipped_df.sort_values(by='Filename').reset_index(drop=True))

    has_data_to_download = not st.session_state.summary_df.empty or \
                           not st.session_state.detailed_df.empty or \
                           not st.session_state.skipped_df.empty
    if has_data_to_download:
        output_filename = f"{config.OUTPUT_FILENAME_BASE}_{int(time.time())}.xlsx"
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