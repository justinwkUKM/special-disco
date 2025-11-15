"""Streamlit UI for generating synthetic PII datasets."""

from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from pii_pipeline.config import FINAL_DATASET_DIR, FINAL_JSONL_FILENAME
from pii_pipeline.dataset_generator import (
    generate_full_dataset,
    get_prompt_scenario,
    list_prompt_scenarios,
    load_clean_samples,
)
from pii_pipeline.utils import write_jsonl

st.set_page_config(page_title="PII Dataset Generator", page_icon="🧪")

st.title("PII Dataset Generator")
st.write(
    "Use this interface to run the synthetic PII pipeline and create a JSONL dataset. "
    "Select a teacher prompt scenario, choose how many records you want, and download "
    "the results directly."
)


@st.cache_data(show_spinner=False)
def _sample_context_map() -> dict[str, str]:
    """Return a mapping of clean sample IDs to their raw contexts."""

    contexts: dict[str, str] = {}
    for sample in load_clean_samples():
        contexts[sample.id] = sample.context
    return contexts


scenario_metadata = list_prompt_scenarios()
scenario_options = {item["key"]: item for item in scenario_metadata}
scenario_keys = [item["key"] for item in scenario_metadata]

selected_scenario_key = st.selectbox(
    "Prompt scenario",
    options=scenario_keys,
    format_func=lambda key: scenario_options[key]["name"],
)

selected_scenario_meta = scenario_options[selected_scenario_key]
selected_scenario = get_prompt_scenario(selected_scenario_key)
sample_contexts = _sample_context_map()

st.markdown(
    f"**Scenario description:** {selected_scenario_meta['description']}"
)

example_contexts = [
    sample_contexts[sample_id]
    for sample_id in selected_scenario.sample_ids
    if sample_id in sample_contexts
]
if example_contexts:
    st.caption("Example clean context used for this scenario:")
    for sample_id, context in zip(selected_scenario.sample_ids, example_contexts):
        st.text_area(
            f"{sample_id}",
            context,
            height=120,
            disabled=True,
            key=f"context_preview_{sample_id}",
        )

    prompt_preview = selected_scenario.prompt_factory(example_contexts[0])
    st.caption("Teacher prompt preview:")
    st.code(prompt_preview, language="text")

num_records = st.number_input(
    "Number of records to generate",
    min_value=1,
    value=50,
    step=1,
    help="The pipeline may generate fewer records if the underlying clean samples are limited.",
)

if st.button("Create dataset"):
    with st.spinner(
        f"Running pipeline for '{scenario_options[selected_scenario_key]['name']}'…"
    ):
        try:
            records = generate_full_dataset(
                max_records=int(num_records),
                scenario_keys=[selected_scenario_key],
            )
        except Exception as exc:  # pragma: no cover - surfaced in UI
            st.error(f"Failed to generate dataset: {exc}")
            st.stop()

    total_records = len(records)

    if total_records == 0:
        st.warning(
            "No records were generated for the selected scenario. "
            "Check that clean samples are available."
        )
        st.stop()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    scenario_suffix = f"_{selected_scenario_key}" if selected_scenario_key else ""
    if FINAL_JSONL_FILENAME.endswith(".jsonl"):
        filename = FINAL_JSONL_FILENAME.replace(
            ".jsonl", f"{scenario_suffix}_{total_records}_{timestamp}.jsonl"
        )
    else:
        filename = f"pii_dataset{scenario_suffix}_{total_records}_{timestamp}.jsonl"

    out_path = FINAL_DATASET_DIR / filename
    write_jsonl(out_path, records)

    st.success(
        f"Generated {total_records} records for '{scenario_options[selected_scenario_key]['name']}'. "
        f"Saved to {out_path}."
    )

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
