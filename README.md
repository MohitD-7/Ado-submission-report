# ado-sub---renew-v1-streamlit

## Description
This Python script is a Streamlit application that processes Excel files in a specified folder, extracts relevant information, and generates a summary and detailed report. It analyzes the comments in the Excel files and counts the occurrences of specific terms.

## Features
- Processes Excel files in a specified folder
- Extracts information such as DE code, submission date, SKU count, and SKU type
- Analyzes the comments in the Excel files and counts the occurrences of specific terms
- Generates a summary report and a detailed report
- Allows the user to download the reports as an Excel file
- Provides a processing status table and a log of the processing activities

## Prerequisites/Dependencies
The script requires the following Python libraries:
- `streamlit`
- `os`
- `pandas`
- `re`
- `time`
- `threading`
- `logging`
- `sys`
- `io`

Additionally, the script requires a configuration file named `config.py` in the same directory as the script.

## How to Use/Run
1. Ensure you have Python and the required libraries installed.
2. Create a `config.py` file in the same directory as the script and define the necessary configuration variables (e.g., `VALID_EXTENSIONS`, `EXCLUDE_FILENAME_TERMS`, `COMMENTS_COLUMN_PRIMARY`, `COMMENTS_COLUMN_FALLBACK_INDEX`, `TERMS_TO_COUNT`, etc.).
3. Run the script using the following command:
   ```
   streamlit run ado-sub---renew-v1-streamlit.py
   ```
4. In the Streamlit application, enter the full path to the folder containing the Excel files and click the "Process Files" button.
5. The application will start processing the files and display the processing status, logs, and the generated reports.
6. Once the processing is complete, you can download the reports as an Excel file by clicking the "Download Report as Excel" button.

## Input Format
The script expects the Excel files in the specified folder to have a specific naming convention that includes the DE code, submission date, and SKU count/type information. The script uses regular expressions to extract this information from the filenames.

## License
To be determined.