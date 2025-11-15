"""Streamlit UI for generating synthetic PII datasets."""

from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from pii_pipeline.config import FINAL_DATASET_DIR, FINAL_JSONL_FILENAME
from pii_pipeline.dataset_generator import generate_full_dataset
from pii_pipeline.utils import write_jsonl

st.set_page_config(page_title="PII Dataset Generator", page_icon="🧪")

st.title("PII Dataset Generator")
st.write(
    "Use this interface to run the synthetic PII pipeline and create a JSONL dataset. "
    "Choose how many records you want and download the results directly."
)

num_records = st.number_input(
    "Number of records to generate",
    min_value=1,
    value=50,
    step=1,
    help="The pipeline may generate fewer records if the underlying clean samples are limited.",
)

if st.button("Create dataset"):
    with st.spinner("Running pipeline…"):
        try:
            records = generate_full_dataset(max_records=int(num_records))
        except Exception as exc:  # pragma: no cover - surfaced in UI
            st.error(f"Failed to generate dataset: {exc}")
            st.stop()

    total_records = len(records)

    if total_records == 0:
        st.warning("No records were generated. Check that clean samples are available.")
        st.stop()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if FINAL_JSONL_FILENAME.endswith(".jsonl"):
        filename = FINAL_JSONL_FILENAME.replace(
            ".jsonl", f"_{total_records}_{timestamp}.jsonl"
        )
    else:
        filename = f"pii_dataset_{total_records}_{timestamp}.jsonl"

    out_path = FINAL_DATASET_DIR / filename
    write_jsonl(out_path, records)

    st.success(f"Generated {total_records} records. Saved to {out_path}.")

    preview_count = min(5, total_records)
    st.caption(f"Previewing the first {preview_count} record(s):")
    for idx, record in enumerate(records[:preview_count], start=1):
        with st.expander(f"Record {idx}"):
            st.json(record)

    jsonl_payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    st.download_button(
        label="Download dataset (.jsonl)",
        data=jsonl_payload.encode("utf-8"),
        file_name=filename,
        mime="application/json",
    )
