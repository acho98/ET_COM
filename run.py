import sys
import pandas as pd
from utils import (
    add_clova_results_to_dataframe,
    retry_failed_rows,
    prettify_result_column
)

def main(input_file, prompt_file, output_file, api_key, api_gw_key):
    # CSV 파일 읽기
    df = pd.read_csv(input_file)

    # 프롬프트 파일 읽기
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()

    # 데이터 처리
    df, error_df = add_clova_results_to_dataframe(df, prompt, api_key, api_gw_key)

    # 실패한 행 재처리
    if not error_df.empty:
        df, error_df = retry_failed_rows(df, error_df, prompt, api_key, api_gw_key)

    # 결과를 예쁘게 포맷팅
    result_df = prettify_result_column(df, result_column='result', pretty_column='pretty_result')

    # 결과 저장
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # 입력 인수: input_file, prompt_file, output_file, api_key, api_gw_key
    if len(sys.argv) != 6:
        print("Usage: python run.py <input_file> <prompt_file> <output_file> <api_key> <api_gw_key>")
        sys.exit(1)

    input_file = sys.argv[1]
    prompt_file = sys.argv[2]
    output_file = sys.argv[3]
    api_key = sys.argv[4]
    api_gw_key = sys.argv[5]

    main(input_file, prompt_file, output_file, api_key, api_gw_key)
