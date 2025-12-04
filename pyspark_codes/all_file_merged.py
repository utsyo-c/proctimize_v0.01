
import streamlit as st
import pandas as pd
import polars as pl
import requests
import io
import json
import time
from datetime import datetime
from azure.storage.blob import BlobServiceClient, ContentSettings

# --- CONFIGURATION ---
DATABRICKS_INSTANCE = "https://adb-2333119529108287.7.azuredatabricks.net/"
JOB_ID = "237790322087536"    #to join/stack files
JOB_ID_1 = "800883880120325"  #to detect granularity
JOB_ID_2 = "314211451745919"  #to modify granularity
TOKEN = "dapi1e301874313e3e0dac76e98d72534367-3"
BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=mmixstorage;AccountKey=UZTHs33FPYTUvC9G51zk+DQQp/FWf31YOteoW+dEnKuprRgxvk53yS+IpEiLn1062IBpOyoKaXp4+AStRcA1Cw==;EndpointSuffix=core.windows.net"
BLOB_CONTAINER = "pre-processing"
BLOB_BASE_URL = f"https://mmixstorage.blob.core.windows.net/{BLOB_CONTAINER}"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def kpi_table(df: pl.DataFrame):
    numerical_config_dict = {}
    categorical_config_dict = {}
    try:
        st.success("File loaded successfully!")
        st.write("### Preview of Data")
        st.dataframe(df.head().to_pandas())  # Convert polars df head to pandas for Streamlit
        #st.write("Type of df", type(df))

        # Polars dtype to python type mapping for filtering
        # Numeric types in Polars: Int32, Int64, Float32, Float64, etc.
        numeric_dtypes = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]
        category_dtypes = [pl.Utf8, pl.Categorical]

        # Step 2: Numerical Column Selection
        num_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in numeric_dtypes]
        st.subheader("🔢 Numerical Columns")
        selected_num_cols = st.multiselect("Select numerical columns:", num_cols)

        if selected_num_cols:
            num_ops_df = pd.DataFrame({
                "Column": selected_num_cols,
                "Operation": [""] * len(selected_num_cols)
            })

            operation_options = ["average", "sum", "product", "min", "max"]
            edited_num_df = st.data_editor(
                num_ops_df,
                column_config={
                    "Operation": st.column_config.SelectboxColumn("Operation", options=operation_options)
                },
                num_rows="fixed",
                key="numerical_editor"
            )

            st.write("📋 Numerical Operations")
            st.dataframe(edited_num_df)

        # Step 3: Categorical Column Selection
        cat_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in category_dtypes]
        st.subheader("🔠 Categorical Columns")
        selected_cat_cols = st.multiselect("Select categorical columns:", cat_cols)

        if selected_cat_cols:
            cat_ops_df = pd.DataFrame({
                "Column": selected_cat_cols,
                "Operation": [""] * len(selected_cat_cols)
            })

            cat_operation_options = ["pivot", "count", "distinct count"]
            edited_cat_df = st.data_editor(
                cat_ops_df,
                column_config={
                    "Operation": st.column_config.SelectboxColumn("Operation", options=cat_operation_options)
                },
                num_rows="fixed",
                key="categorical_editor"
            )

            st.write("📋 Categorical Operations")
            st.dataframe(edited_cat_df)

        if selected_num_cols or selected_cat_cols:
            if st.button("💾 Save All  Configurations"):
                if selected_num_cols:
                    numerical_config_dict = edited_num_df.set_index("Column")["Operation"].to_dict()
                if selected_cat_cols:
                    categorical_config_dict = edited_cat_df.set_index("Column")["Operation"].to_dict()
                st.success("✅ Configuration saved!")
                return numerical_config_dict, categorical_config_dict

    except Exception as e:
        st.error(f"Error processing KPI Table: {e}")


def upload_to_blob(file_path, blob_name):
    connection_string =  BLOB_CONNECTION_STRING
    container_name = BLOB_CONTAINER
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    return blob_name

def download_from_blob(folder_path):
    connection_string =  BLOB_CONNECTION_STRING
    container_name = BLOB_CONTAINER
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container=container_name)

    # List all .csv part files inside the folder
    blob_list = container_client.list_blobs(name_starts_with=folder_path)
    csv_files = [b.name for b in blob_list if b.name.endswith(".csv")]

    if not csv_files:
        return None

    dfs = []
    for blob_name in csv_files:
        blob_data = container_client.get_blob_client(blob_name).download_blob().readall()
        dfs.append(pd.read_csv(io.BytesIO(blob_data)))

    return pd.concat(dfs, ignore_index=True)

def download_from_blob_part_file(blob_folder_path: str):
    connection_string =  BLOB_CONNECTION_STRING
    container_name = BLOB_CONTAINER
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # List blobs in the folder
    blob_list = container_client.list_blobs(name_starts_with=blob_folder_path)

    for blob in blob_list:
        if "part-" in blob.name and blob.name.endswith(".csv"):
            blob_client = container_client.get_blob_client(blob)
            stream = blob_client.download_blob()
            data = stream.readall()
            df = pd.read_csv(io.BytesIO(data))
            return df

    return None  # If no part file is found

def trigger_databricks_job_detect_granularity(blob_input_path, blob_output_path, date_col):
    url = f"{DATABRICKS_INSTANCE}/api/2.1/jobs/run-now"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    payload = {
        "job_id": JOB_ID_1,
        "notebook_params": {
            "input_path": blob_input_path,
            "output_path": blob_output_path,
            "date_col": date_col
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

def trigger_databricks_job_modify_granularity(blob_input_path, blob_output_path, geo_col, date_col,
                                              granularity_level_df, granularity_level_user_input,
                                              work_days, numerical_config_dict, categorical_config_dict):
    url = f"{DATABRICKS_INSTANCE}/api/2.1/jobs/run-now"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    payload = {
        "job_id": JOB_ID_2,
        "job_parameters": {
            "input_path": blob_input_path,
            "output_path": blob_output_path,
            "date_col": date_col,
            "geo_col": geo_col,
            "granularity_level_df": granularity_level_df,
            "granularity_level_user_input": granularity_level_user_input,
            "work_days": work_days,
            "numerical_config_dict": json.dumps(numerical_config_dict),
            "categorical_config_dict": json.dumps(categorical_config_dict)
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    # Add this logging for debug
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    if response.status_code != 200:
        raise Exception(f"Failed to trigger Databricks job: {response.text}")
    
    return response.json()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Unified Preprocessing & Filtering Tool", layout="wide")
st.title("📊 Upload, Preprocess, and Filter Your Data")

# --- FILE UPLOAD ---
uploaded_files = st.file_uploader("📄 Upload one or more CSV files", type="csv", accept_multiple_files=True)

df_final = None

if uploaded_files:
    if len(uploaded_files) == 1:
        st.subheader("🧱 Standardize Columns for Single File")
        file = uploaded_files[0]
        df = pd.read_csv(file)
        st.markdown(f"**File: {file.name}**")

        # ✅ Step 2a: Column Selection and Renaming
        selected_cols = st.multiselect(
            "Select columns to keep:", 
            df.columns.tolist(), 
            default=df.columns.tolist()
        )
        rename_dict = {}
        for col in selected_cols:
            new_col = st.text_input(f"Rename '{col}'", value=col, key=f"{col}_single")
            rename_dict[col] = new_col

        df = df[selected_cols].rename(columns=rename_dict)

        # ✅ Step 2b: Date Standardization
        date_cols = st.multiselect(
            "Select Date Columns (if any):", 
            df.columns.tolist(), 
            key="date_cols_single"
        )

        for date_col in date_cols:
            st.markdown(f"📅 **Standardizing `{date_col}`**")
            date_format = st.selectbox(
                f"Select current date format for `{date_col}` being used:",
                ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "Custom"],
                key=f"date_format_single_{date_col}"
            )

            if date_format == "Custom":
                date_format = st.text_input(
                    "Enter custom date format (e.g., %d-%b-%Y):", 
                    key=f"custom_date_format_single_{date_col}"
                )

            if date_format:
                try:
                    df[date_col] = pd.to_datetime(
                        df[date_col], 
                        format=date_format, 
                        errors='coerce'
                    ).dt.strftime("%Y/%m/%d")
                    st.success(f"✅ Date column `{date_col}` standardized from `{date_format}` to `YYYY/MM/DD` format!")
                except Exception as e:
                    st.error(f"❌ Failed to parse date column `{date_col}`: {e}")

        # Final clean-up before conversion
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        df[str_cols] = df[str_cols].fillna("")

        # Convert to Polars for filtering
        df_final = pl.from_pandas(df)


    else:
        st.subheader("🧱Standardize Columns for Multiple Files")
        column_mappings = []
        dfs = []
        renamed_columns_list = []

        for i, file in enumerate(uploaded_files):
            st.markdown(f"**File {i+1}: {file.name}**")
            df = pd.read_csv(file)
            dfs.append(df)
            selected_cols = st.multiselect(
                f"Select columns from `{file.name}`:", 
                df.columns.tolist(), 
                default=df.columns.tolist(), 
                key=f"select_cols_{i}"
            )
            df = df[selected_cols]

            rename_dict = {}
            renamed_cols = []

            for col in selected_cols:
                new_col = st.text_input(f"Rename '{col}'", value=col, key=f"{col}_{i}")
                rename_dict[col] = new_col
                renamed_cols.append(new_col)

            column_mappings.append(rename_dict)
            renamed_columns_list.append(set(renamed_cols))

            # 📅 Step: Ask for Date Format and Standardize Date Columns
            date_cols = st.multiselect(
                f"Select Date Columns in `{file.name}` (if any):", 
                selected_cols, 
                key=f"date_cols_{i}"
            )

            for date_col in date_cols:
                st.markdown(f"📅 **Standardizing `{date_col}` in file `{file.name}`**")
                date_format = st.selectbox(
                    f"Select the current date format used for `{date_col}`:",
                    ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "Custom"],
                    key=f"date_format_{i}_{date_col}"
                )

                if date_format == "Custom":
                    date_format = st.text_input(
                        "Enter custom date format (e.g., %d-%b-%Y):", 
                        key=f"custom_date_format_{i}_{date_col}"
                    )

                # Convert and standardize to YYYY/MM/DD format
                if date_format:
                    try:
                        df[date_col] = pd.to_datetime(
                            df[date_col], 
                            format=date_format, 
                            errors='coerce'
                        ).dt.strftime("%Y/%m/%d")
                        df[date_col] = df[date_col].fillna("")

                        st.success(f"✅ Date column `{date_col}` standardized from `{date_format}` to `YYYY/MM/DD` format!")
                    except Exception as e:
                        st.error(f"❌ Failed to parse date column `{date_col}`: {e}")



        common_columns = set.intersection(*renamed_columns_list)
        st.subheader("🔗Select Merge Strategy")
        merge_strategy = st.radio("Merge type:", ["Vertical Stack", "Horizontal Join"])

        if merge_strategy == "Horizontal Join":
            join_keys = st.multiselect("Select join key(s):", list(common_columns))
            join_type = st.selectbox("Join type:", ["inner", "left", "right", "outer"])
        else:
            join_keys = join_type = None

        st.subheader("🚀Trigger Databricks Job to Merge Files")
        output_filename = st.text_input("Enter the Output Filename (with .csv extension):")
        if st.button("📤 Trigger Databricks Job for Merging"):
            blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
            raw_folder = "pre_processing_csv_new"
            blob_urls = []

            for i, file in enumerate(uploaded_files):
                blob_name = f"{raw_folder}/uploaded_{output_filename}_{i+1}.csv"
                blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER, blob=blob_name)
                file.seek(0)
                blob_client.upload_blob(file, overwrite=True, content_settings=ContentSettings(content_type='application/octet-stream'))
                blob_urls.append(f"{BLOB_BASE_URL}/{blob_name}")

            job_input = {
                "file_urls": blob_urls,
                "merge_strategy": merge_strategy.lower(),
                "column_mappings": column_mappings,
                "join_keys": join_keys,
                "join_type": join_type,
                "output_filename": output_filename,
            }

            run_url = f"{DATABRICKS_INSTANCE}/api/2.1/jobs/run-now"
            payload = {
                "job_id": JOB_ID,
                "notebook_params": {k: json.dumps(v) for k, v in job_input.items()}
            }

            response = requests.post(run_url, headers=HEADERS, json=payload)
            if response.status_code == 200:
                run_id = response.json()["run_id"]
                st.success(f"Job triggered! Run ID: {run_id}")

                status_url = f"{DATABRICKS_INSTANCE}/api/2.1/jobs/runs/get"
                with st.spinner("⏳ Waiting for Databricks job to complete..."):
                    for _ in range(60):
                        status_resp = requests.get(status_url, headers=HEADERS, params={"run_id": run_id})
                        if status_resp.status_code == 200:
                            state = status_resp.json()["state"]
                            if state["life_cycle_state"] == "TERMINATED":
                                if state.get("result_state") == "SUCCESS":
                                    st.success("✅ Job completed successfully!")
                                    break
                                else:
                                    st.error(f"❌ Job failed: {state.get('result_state')}")
                                    st.stop()
                            time.sleep(5)
                        else:
                            st.error("Failed to fetch job status.")
                            st.stop()
                    else:
                        st.warning("⏱ Job timed out.")
                        st.stop()

                sample_folder = f"correct_csv_outputs/{output_filename}"
                try:
                    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
                    container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
                    
                    # Dynamically list blobs and find the correct merged CSV
                    blob_list = container_client.list_blobs(name_starts_with=sample_folder)
                    part_csv_blob = None
                    for blob in blob_list:
                        if "part-" in blob.name and blob.name.endswith(".csv"):
                            part_csv_blob = blob.name
                            break

                    if part_csv_blob:
                        blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER, blob=part_csv_blob)
                        download_stream = blob_client.download_blob()
                        merged_df = pd.read_csv(io.BytesIO(download_stream.readall()))
                        if merged_df.empty:
                            st.error("❌ Merged CSV is empty. Cannot proceed to filtering.")
                            st.stop()
                        else:
                            str_cols = df.select_dtypes(include=["object", "string"]).columns
                            df[str_cols] = df[str_cols].fillna("")

                            df_final = pl.from_pandas(merged_df)
                            df_final = df_final.fill_null("")
                            st.session_state["df_final"] = df_final
                            if df_final is None or df_final.shape[0] == 0:
                                st.error("❌ Failed to convert merged CSV to Polars DataFrame or no data available.")
                                st.stop()
                            else:
                                st.success(f"✅ Loaded merged file: `{part_csv_blob}`")
                            
                            # ✅ Attempt to parse Date Columns immediately after loading
                            date_cols = [col for col in df_final.columns if "date" in col.lower()]
                            for date_col in date_cols:
                                try:
                                    df_final = df_final.with_columns(
                                        pl.col(date_col).str.strptime(pl.Date, strict=False).alias(date_col)
                                    )
                                    st.info(f"📅 Column `{date_col}` parsed to Date format.")
                                except Exception:
                                    st.warning(f"⚠️ Could not parse `{date_col}` to Date format after merging. Proceeding with raw data.")
                    else:
                        st.warning("No merged CSV file found in the output folder.")
                        st.stop()
                except Exception as e:
                    st.warning(f"Could not load merged file: {e}")
                    st.stop()

            else:
                st.error(f"Failed to trigger job: {response.text}")
    
    # ---------------------- FILTERING SECTION ----------------------
    st.subheader("🔍Filter Data")

    if "df_final" in st.session_state:
        df_final = st.session_state["df_final"]

    if df_final is None or df_final.shape[0] == 0:
        st.error("❌ No data available for filtering. Please check the input files.")
        st.stop()
        
    if df_final is not None:
        st.write("### Sample of the Dataset")
        st.dataframe(df_final.head().to_pandas())

        date_column = st.selectbox("Select the column representing Date", df_final.columns)

        if date_column:
            if df_final[date_column].dtype in [pl.Utf8, pl.Object]:
                try:
                    df_final = df_final.with_columns(
                        pl.col(date_column).str.strptime(pl.Date, strict=False).alias(date_column)
                    )
                except Exception:
                    st.warning("⚠️ Valid date column not selected or parsing failed.")
                    st.stop()
            elif df_final[date_column].dtype != pl.Date:
                st.warning("⚠️ Selected column is not a valid date column.")
                st.stop()

            min_date = df_final[date_column].min()
            max_date = df_final[date_column].max()

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

            if start_date > end_date:
                st.error("Start date must be before or equal to end date.")
                st.stop()
            else:
                df_final = df_final.filter((pl.col(date_column) >= start_date) & (pl.col(date_column) <= end_date))

        df_filtered = df_final.clone()
        categorical_cols = [col for col in df_filtered.columns if df_filtered[col].dtype == pl.Utf8]

        if categorical_cols:
            st.write("### Categorical Column Filters")
            selected_cat_cols = st.multiselect("Select categorical columns to filter:", categorical_cols)

            for col in selected_cat_cols:
                unique_vals = df_filtered[col].unique().to_list()
                display_vals = []
                for val in unique_vals:
                    if val is None or str(val).strip() == "":
                        display_vals.append("<BLANK>")
                    else:
                        display_vals.append(str(val))
                display_vals = sorted(display_vals)

                selected_vals = st.multiselect(
                    f"Select values to retain in '{col}'",
                    options=display_vals,
                    default=display_vals,
                    key=f"filter_{col}"
                )

                filter_conditions = []

                if "<BLANK>" in selected_vals:
                    filter_conditions.append(pl.col(col).is_null() | (pl.col(col).cast(str).str.strip_chars().eq("")))

                selected_non_blank_vals = [val for val in selected_vals if val != "<BLANK>"]
                if selected_non_blank_vals:
                    filter_conditions.append(pl.col(col).cast(str).is_in(selected_non_blank_vals))

                if filter_conditions:
                    combined_condition = filter_conditions[0]
                    for cond in filter_conditions[1:]:
                        combined_condition = combined_condition | cond
                    df_filtered = df_filtered.filter(combined_condition)
                    st.session_state["df_filtered"] = df_filtered


        st.write("### Final Filtered Dataset")
        st.dataframe(df_filtered.head(100).to_pandas())


        csv_bytes = df_filtered.write_csv()
        st.download_button("📥 Download CSV", data=csv_bytes, file_name="final_filtered_data.csv", mime="text/csv")

        output_filename = st.text_input("Enter output filename (with .csv extension):")
        if st.button("📤 Upload to Azure Blob Storage"):
            if not output_filename.endswith(".csv"):
                st.warning("⚠️ Filename must end with `.csv`")
            else:
                try:
                    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
                    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER, blob=output_filename)

                    csv_data = df_filtered.write_csv().encode("utf-8")
                    blob_client.upload_blob(csv_data, overwrite=True)

                    st.success(f"✅ File uploaded as `{output_filename}` to container `{BLOB_CONTAINER}`")
                    blob_url = f"https://mmixstorage.blob.core.windows.net/{BLOB_CONTAINER}/{output_filename}"
                    st.session_state["filtered_input_path"] = blob_url
                    #st.markdown(f"[🔗 View Uploaded File]({blob_url})")

                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")

        

    # ---------------------- GRANULARITY SECTION ----------------------
    st.title("Modify Time Granularity")

    if "df_filtered" and "filtered_input_path" in st.session_state:
        df_filtered = st.session_state["df_filtered"]
        filtered_input_path = st.session_state["filtered_input_path"]
     
    if df_filtered is not None:
        st.write("📊 Sample of Uploaded Data", df_filtered.head(10))
        st.write("Blob link", filtered_input_path)
        #st.write("Type of df_filtered:", type(df_filtered))

        column_names = df_filtered.columns
        geo_col = st.selectbox("Select the Grouping Column", options=column_names)
        date_col = st.selectbox("Select the Date Column", options=column_names)
        # dependent_variable = st.selectbox("Select the Dependent Variable", options=column_names)

        if st.button("Detect Granularity"):
            if date_col.strip() == "":
                st.warning("Please enter a date column name.")
            else:
                with st.spinner("Detecting granularity..."):
                    trigger_databricks_job_detect_granularity(
                        blob_input_path = filtered_input_path,
                        blob_output_path=f"https://mmixstorage.blob.core.windows.net/{BLOB_CONTAINER}/granular_data_transformed.csv",
                        date_col=date_col
                    )
                st.success("✅ Databricks job triggered. You can refresh below when it's done.")

        # Cache the granularity
        @st.cache_data(ttl=60)  # Cache expires every 60 seconds
        def load_granularity_string(blob_path: str):
            df = download_from_blob(blob_path)
            if df is not None and not df.empty:
                return df.iloc[0, 0]  # Return the granularity string
            return None

        # Initialize granularity in session state
        if "granularity" not in st.session_state:
            st.session_state["granularity"] = ""

        # Load Granularity Result
        if st.button("Load Granularity Result"):
            with st.spinner("Loading result from Blob..."):
                granularity = str(load_granularity_string("granular_data_transformed.csv/"))
                st.session_state["granularity"] = granularity
            if granularity:
                st.success(f"📈 Detected Granularity: **{granularity}**")
            else:
                st.warning("⚠️ No result found or blob is still processing.")



        # Get granularity from session state
        granularity = st.session_state.get("granularity", "")


        #NEW 
        # Calling the KPI function
        numerical_config_dict={}
        categorical_config_dict={}
        st.subheader("Creating KPI Table")
        config_result = kpi_table(df_filtered)
        if config_result is not None:
            numerical_config_dict, categorical_config_dict = config_result
            st.session_state["numerical_config_dict"] = numerical_config_dict
            st.session_state["categorical_config_dict"] = categorical_config_dict
            st.json(numerical_config_dict)
            st.json(categorical_config_dict)

        # Integrated Analytics Database
        st.subheader("Creating Integrated Analytics Database")

        if granularity:
            if granularity == 'Daily':
                granularity_options = ['Weekly','Monthly']
            elif granularity == 'Weekly':
                granularity_options = ['Weekly','Monthly']
            elif granularity == 'Monthly':
                granularity_options = ['Monthly']

            time_granularity_user_input = st.selectbox('Choose the time granularity level', granularity_options)
            st.write('You selected time granularity: ', time_granularity_user_input)
            work_days = st.selectbox('Choose the number of business days in a week', [5,6,7])
            st.write('You selected time granularity: ', time_granularity_user_input)


            # Modifying the granularity
            if st.button("Modify Granularity"):
                st.write("🚀 Going to trigger with:")
                numerical_config_dict = st.session_state.get('numerical_config_dict', {})
                categorical_config_dict = st.session_state.get('categorical_config_dict', {})

                st.json(numerical_config_dict)
                st.json(categorical_config_dict)

                if date_col.strip() == "" or geo_col.strip() == "" or time_granularity_user_input.strip() == "" :
                    st.warning("Please enter the required columns")
                else:
                    with st.spinner("Modifying granularity..."):
                        trigger_databricks_job_modify_granularity(
                            blob_input_path=filtered_input_path,
                            blob_output_path= f"https://mmixstorage.blob.core.windows.net/{BLOB_CONTAINER}/modified_granular_data_transformed.csv",
                            date_col=date_col,
                            geo_col=geo_col,
                            granularity_level_df = granularity,
                            granularity_level_user_input = time_granularity_user_input,
                            work_days=5,    
                            numerical_config_dict=numerical_config_dict,
                            categorical_config_dict=categorical_config_dict
                        )
                        st.cache_data.clear()  # 💡 Clear cached blob data

                    st.success("✅ Databricks job triggered. You can refresh below when it's done.")


        else:
            st.warning("⚠️ Please load granularity before selecting granularity level.")


        # Cache the granularity
        def load_transformed_blob(blob_path: str):
            blob_data = download_from_blob_part_file(blob_path)
            if blob_data is None or blob_data.empty:
                return None
            return blob_data

        # Load Transformed File
        if st.button("Load Transformed Data"):
            with st.spinner("Loading modified granular data from Blob..."):
                df_transformed = load_transformed_blob("modified_granular_data_transformed.csv/")
            if df_transformed is not None and not df_transformed.empty:
                st.success("Modified Granular data loaded from Blob Storage")
                st.dataframe(df_transformed)
            else:
                st.warning("⚠️ No data found or blob may still be processing.")



