import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

INPUT_CSV = "data/filtered_it_video_info.csv"
OUTPUT_CSV = "youtube_redirect_results_it.csv"
PROGRESS_INTERVAL = 500

# Pass 1: fast bulk classification
PASS1_WORKERS = 50
PASS1_TIMEOUT_SECONDS = 5
PASS1_MAX_ATTEMPTS = 2

# Pass 2: slower retry for retryable errors only
PASS2_WORKERS = 3
PASS2_TIMEOUT_SECONDS = 10
PASS2_MAX_ATTEMPTS = 3

# Pass 3: final single-worker retry for remaining retryable errors
PASS3_WORKERS = 1
PASS3_TIMEOUT_SECONDS = 10
PASS3_MAX_ATTEMPTS = 2

RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def classify_video_once(video_id: str, timeout_seconds: int) -> dict:
    url = f"https://www.youtube.com/shorts/{video_id}"
    try:
        response = requests.get(url, timeout=timeout_seconds, headers=HEADERS)
        response.raise_for_status()

        final_url = response.url
        raw_html = response.text.replace(" ", "")

        if (
            '"playabilityStatus":{"status":"ERROR"' in raw_html
            or '"playabilityStatus":{"status":"UNPLAYABLE"' in raw_html
        ):
            verdict = "error"
            reason = "video is missing, deleted, private, or unavailable"
        elif "/watch" in final_url:
            verdict = "longform"
            reason = "redirected to /watch"
        elif "/shorts" in final_url:
            verdict = "shorts"
            reason = "kept /shorts route"
        else:
            verdict = "unknown"
            reason = f"unexpected final_url: {final_url}"

        return {
            "video_id": video_id,
            "verdict": verdict,
            "reasoning": reason,
            "final_url": final_url,
            "retryable": False,
        }
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        retryable = status_code in RETRYABLE_HTTP_STATUS
        return {
            "video_id": video_id,
            "verdict": "error",
            "reasoning": f"http_error({status_code}): {exc}",
            "final_url": "N/A",
            "retryable": retryable,
        }
    except requests.exceptions.Timeout as exc:
        return {
            "video_id": video_id,
            "verdict": "error",
            "reasoning": f"timeout: {exc}",
            "final_url": "N/A",
            "retryable": True,
        }
    except requests.exceptions.ConnectionError as exc:
        return {
            "video_id": video_id,
            "verdict": "error",
            "reasoning": f"connection_error: {exc}",
            "final_url": "N/A",
            "retryable": True,
        }
    except requests.exceptions.RequestException as exc:
        return {
            "video_id": video_id,
            "verdict": "error",
            "reasoning": f"request_error: {exc}",
            "final_url": "N/A",
            "retryable": False,
        }


def classify_with_retry(index: int, video_id: str, timeout_seconds: int, max_attempts: int) -> tuple[int, dict]:
    last_result: dict | None = None
    for attempt in range(1, max_attempts + 1):
        result = classify_video_once(video_id, timeout_seconds)
        result["attempt"] = attempt
        last_result = result
        if result["verdict"] != "error" or not result["retryable"]:
            break
        if attempt < max_attempts:
            sleep_seconds = (2 ** (attempt - 1)) + random.uniform(0.0, 0.3)
            time.sleep(sleep_seconds)
    return index, last_result if last_result is not None else {
        "video_id": video_id,
        "verdict": "error",
        "reasoning": "unexpected_retry_failure",
        "final_url": "N/A",
        "retryable": True,
        "attempt": max_attempts,
    }


def run_pass(
    indexed_video_ids: list[tuple[int, str]],
    workers: int,
    timeout_seconds: int,
    max_attempts: int,
    pass_name: str,
) -> list[tuple[int, dict]]:
    total = len(indexed_video_ids)
    if total == 0:
        return []
    print(f"{pass_name} start: total={total}, workers={workers}, timeout={timeout_seconds}s, attempts={max_attempts}")
    indexed_results: list[tuple[int, dict]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(classify_with_retry, idx, video_id, timeout_seconds, max_attempts)
            for idx, video_id in indexed_video_ids
        ]
        for done_count, future in enumerate(as_completed(futures), start=1):
            indexed_results.append(future.result())
            if done_count % PROGRESS_INTERVAL == 0 or done_count == total:
                print(f"{pass_name} progress: {done_count}/{total}")
    return indexed_results


def main() -> None:
    source_df = pd.read_csv(INPUT_CSV)
    video_ids = (
        source_df["video_id"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .tolist()
    )

    # Remove duplicates while keeping original order.
    video_ids = list(dict.fromkeys(video_ids))

    total = len(video_ids)
    indexed_input = list(enumerate(video_ids))
    pass1_results = run_pass(
        indexed_input,
        workers=PASS1_WORKERS,
        timeout_seconds=PASS1_TIMEOUT_SECONDS,
        max_attempts=PASS1_MAX_ATTEMPTS,
        pass_name="pass1",
    )

    result_map = {idx: result for idx, result in pass1_results}
    retry_targets = [
        (idx, result_map[idx]["video_id"])
        for idx in sorted(result_map.keys())
        if result_map[idx]["verdict"] == "error" and result_map[idx].get("retryable", False)
    ]
    print(f"pass1 done: total={total}, error={sum(1 for r in result_map.values() if r['verdict'] == 'error')}, retryable_error={len(retry_targets)}")

    if retry_targets:
        pass2_results = run_pass(
            retry_targets,
            workers=PASS2_WORKERS,
            timeout_seconds=PASS2_TIMEOUT_SECONDS,
            max_attempts=PASS2_MAX_ATTEMPTS,
            pass_name="pass2",
        )
        for idx, result in pass2_results:
            result_map[idx] = result

    retry_targets = [
        (idx, result_map[idx]["video_id"])
        for idx in sorted(result_map.keys())
        if result_map[idx]["verdict"] == "error" and result_map[idx].get("retryable", False)
    ]
    print(f"pass2 done: remaining_retryable_error={len(retry_targets)}")

    if retry_targets:
        pass3_results = run_pass(
            retry_targets,
            workers=PASS3_WORKERS,
            timeout_seconds=PASS3_TIMEOUT_SECONDS,
            max_attempts=PASS3_MAX_ATTEMPTS,
            pass_name="pass3",
        )
        for idx, result in pass3_results:
            result_map[idx] = result

    results = [result_map[idx] for idx in sorted(result_map.keys())]
    for row in results:
        row.pop("retryable", None)
        row.pop("attempt", None)

    output_df = pd.DataFrame(results)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_CSV)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"done: {output_path}")


if __name__ == "__main__":
    main()
