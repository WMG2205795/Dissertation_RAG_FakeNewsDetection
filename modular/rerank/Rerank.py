import json
from pathlib import Path

from sentence_transformers.cross_encoder import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"
BATCH_SIZE = 32

def load_json(
    input_path: Path,
):
    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    output_path: Path,
    records,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )


def build_final_evidence(
    claim,
    retrieved_evidence,
    rerank,
    final_top_k,
    reranker=None,
):

    if not rerank:
        return retrieved_evidence[:final_top_k]

    pairs = [
        [claim, item["contents"]]
        for item in retrieved_evidence
    ]

    scores = reranker.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
    )

    reranked_evidence = []

    for item, score in zip(
        retrieved_evidence,
        scores,
    ):
        new_item = item.copy()
        new_item["rerank_score"] = float(score)

        reranked_evidence.append(
            new_item
        )

    reranked_evidence.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    for rank, item in enumerate(
        reranked_evidence,
        start=1,
    ):
        item["rerank_rank"] = rank

    return reranked_evidence[:final_top_k]


def main(reranker):

    retrieval_results = load_json(
        INPUT_PATH
    )

    final_results = []

    for item in retrieval_results:

        final_evidence = build_final_evidence(
            claim=item["claim"],
            retrieved_evidence=item[
                "retrieved_evidence"
            ],
            rerank=RERANK,
            final_top_k=FINAL_TOP_K,
            reranker=reranker,
        )

        output_record = item.copy()

        output_record["retrieved_evidence"] = final_evidence

        output_record["reranked"] = RERANK

        output_record["top_k"] = FINAL_TOP_K

        output_record["reranker_model"] = (
            RERANK_MODEL if RERANK else None
        )

        final_results.append(output_record)

    save_json(
        OUTPUT_PATH,
        final_results,
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]   
    CHUNK_MODE_LIST=["word_200_50"]#["sentence", "word_100_25", "word_200_50"] 
    RETRIEVE_MODE_LIST=["Hybrid"]#["BM25", "Dense", "Hybrid"]
    IS_DUPLICATED=[True]#[True, False]
    RERANK = True
    FINAL_TOP_K = 10
    reranker = None
    process="dev"
    if process == "dev":
        if RERANK:
            reranker = CrossEncoder(
                RERANK_MODEL,
                device="cuda",
            )
        for CHUNK_MODE in CHUNK_MODE_LIST:
            for RETRIEVE_MODE in RETRIEVE_MODE_LIST:
                for DEDUPLICATE in IS_DUPLICATED:

                    INPUT_PATH = (
                        BASE_DIR
                        / "output"
                        / "hybrid"
                        / RETRIEVE_MODE
                        / (
                            f"{CHUNK_MODE}_{RETRIEVE_MODE}_"
                            f"retrieval_cache_50_Dedup_{DEDUPLICATE}.json"
                        )
                    )

                    if not INPUT_PATH.exists():
                        print(f"Missing: {INPUT_PATH}")
                        continue

                    OUTPUT_PATH = (
                        BASE_DIR
                        / "output"
                        / "rerank"
                        / RETRIEVE_MODE
                        / (
                            f"{CHUNK_MODE}_retrieval_top{FINAL_TOP_K}_"
                            f"rerank_{RERANK}_Dedup_{DEDUPLICATE}.json"
                        )
                    )

    else:
        if RERANK:
            reranker = CrossEncoder(
                RERANK_MODEL,
                device="cuda",
            )
            INPUT_PATH = (
                    BASE_DIR
                    / "output"
                    / "hybrid"
                    / "Hybrid"
                    / "Test_result_word_200_50_Hybrid_retrieval_cache_50_Dedup_True.json"
                )
            OUTPUT_PATH = (
                BASE_DIR
                / "output"
                / "rerank"
                / "Hybrid"
                / "Chunk_200_overlap_50_retrieval_top10_rerank_True_Dedup_True.json"
            )
    main(reranker)
        
