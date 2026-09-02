import json

RERANK_ON_FILE = r"output\CrossEncoder\Hybrid\word_200_50_retrieval_top10_rerank_True_Dedup_False_qwen3_30b_result.json"
RERANK_OFF_FILE = r"output\NoRerank\Hybrid\word_200_50_retrieval_top10_rerank_False_Dedup_False_qwen3_30b_result.json"


def normalize(text):
    return " ".join(str(text).lower().split())


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        int(record["claim_id"]): [
            normalize(item["contents"])
            for item in record["retrieved_evidence"][:10]
        ]
        for record in data
    }


rerank_on = load_results(RERANK_ON_FILE)
rerank_off = load_results(RERANK_OFF_FILE)

claim_ids = sorted(set(rerank_on) & set(rerank_off))

claims_with_changed_top10 = 0
total_shared_chunks = 0
total_changed_positions = 0
claims_with_changed_rank1 = 0

for claim_id in claim_ids:
    on_chunks = rerank_on[claim_id]
    off_chunks = rerank_off[claim_id]

    shared_chunks = len(set(on_chunks) & set(off_chunks))
    changed_positions = sum(
        on != off
        for on, off in zip(on_chunks, off_chunks)
    )

    if set(on_chunks) != set(off_chunks):
        claims_with_changed_top10 += 1

    if on_chunks[0] != off_chunks[0]:
        claims_with_changed_rank1 += 1

    total_shared_chunks += shared_chunks
    total_changed_positions += changed_positions

print(f"Number of compared claims: {len(claim_ids)}")
print(
    "Claims with changed top-10 evidence set: "
    f"{claims_with_changed_top10}"
)
print(
    "Mean shared chunks per claim: "
    f"{total_shared_chunks / len(claim_ids):.3f}/10"
)
print(
    "Mean changed positions per claim: "
    f"{total_changed_positions / len(claim_ids):.3f}/10"
)
print(
    "Claims with changed rank-1 evidence: "
    f"{claims_with_changed_rank1}"
)