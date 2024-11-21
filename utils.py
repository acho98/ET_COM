import requests
import json
import pandas as pd
from tqdm import tqdm
import time

def call_clova_api(api_key, api_gw_key, messages):
    url = 'https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003'

    headers = {
        'X-NCP-CLOVASTUDIO-API-KEY': api_key,
        'X-NCP-APIGW-API-KEY': api_gw_key,
        'Content-Type': 'application/json',
    }

    data = {
        "topK": 0,
        "includeAiFilters": True,
        "maxTokens": 1000,
        "temperature": 0.25,
        "messages": messages,
        "repeatPenalty": 4,
        "topP": 0.8
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            return None, f"API 호출 실패: {response.status_code}, {response.text}"

        if not response.text:
            return None, "Empty response received"

        try:
            response_json = response.json()
        except json.JSONDecodeError as e:
            return None, f"JSON decoding error: {e} - Response text: {response.text[:100]}"

        if isinstance(response_json, dict):
            return response_json, None
        else:
            return None, "Unexpected response format"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"

def process_single_row(row, prompt, api_key, api_gw_key):
    try:
        context = row['formatted_text']
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
        ]

        response_json, error = call_clova_api(api_key, api_gw_key, messages)

        if error:
            return {"error": error}

        if 'result' not in response_json or 'message' not in response_json['result']:
            return {"error": "Unexpected response format", "raw_response": response_json}

        result_content = response_json['result']['message'].get('content', '')
        return json.loads(result_content)
    except Exception as e:
        return {"error": str(e)}

def add_clova_results_to_dataframe(df, prompt, api_key, api_gw_key, max_retries=3):
    results = []
    errors = []
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row"):
        try:
            result = process_single_row(row, prompt, api_key, api_gw_key)
            if "error" in result:
                raise Exception(result["error"])
            results.append(result)
        except Exception as e:
            errors.append({
                "idx": idx,
                "row": row.to_dict(),
                "error": str(e),
            })
            results.append(None)  # Append None for failed rows

        elapsed_time = time.time() - start_time
        avg_time_per_row = elapsed_time / (idx + 1)
        remaining_time = avg_time_per_row * (len(df) - idx - 1)
        tqdm.write(f"Row {idx + 1}/{len(df)} processed. Elapsed: {elapsed_time:.2f}s, "
                   f"Estimated remaining: {remaining_time:.2f}s")

    error_df = pd.DataFrame(errors)
    df['result'] = results

    return df, error_df

def retry_failed_rows(result_df, error_df, prompt, api_key, api_gw_key, max_retries=3):
    retries = 0
    while retries < max_retries and not error_df.empty:
        new_errors = []
        retried_results = []
        start_time = time.time()

        for idx, row_data in tqdm(
            error_df.iterrows(),
            total=len(error_df),
            desc=f"Retrying (Attempt {retries + 1})",
            unit="row"
        ):
            row = pd.Series(row_data["row"])
            try:
                result = process_single_row(row, prompt, api_key, api_gw_key)
                if "error" in result:
                    raise Exception(result["error"])
                retried_results.append({"idx": idx, "result": result})
            except Exception as e:
                new_errors.append({
                    "idx": idx,
                    "row": row.to_dict(),
                    "error": str(e),
                })

            elapsed_time = time.time() - start_time
            avg_time_per_row = elapsed_time / max(1, len(error_df))
            remaining_time = avg_time_per_row * (len(error_df) - len(retried_results) - len(new_errors))
            tqdm.write(f"Row {idx + 1}/{len(error_df)} retried. Elapsed: {elapsed_time:.2f}s, "
                       f"Estimated remaining: {remaining_time:.2f}s")

        error_df = pd.DataFrame(new_errors)
        for retried in retried_results:
            result_df.loc[retried['idx'], 'result'] = retried['result']
        retries += 1

    return result_df, error_df

def prettify_result_column(df, result_column='result', pretty_column='pretty_result'):
    df = df.copy()
    df[pretty_column] = df[result_column].apply(
        lambda x: json.dumps(x, indent=2, ensure_ascii=False) if isinstance(x, dict) else x
    )
    return df
