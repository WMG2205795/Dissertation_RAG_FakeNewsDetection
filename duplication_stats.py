import json

FILE_PATH = r"rerank\report\CrossEncoder\Dense\word_200_50_retrieval_top10_rerank_True_Dedup_True.json"


def normalize(text):
    return " ".join(str(text).lower().split())


with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


duplicate_counts = []

for record in data:
    claim_id = record["claim_id"]
    top10 = record["retrieved_evidence"][:10]

    contents = [
        normalize(item["contents"])
        for item in top10
        if item.get("contents")
    ]

    duplicate_count = len(contents) - len(set(contents))
    duplicate_counts.append(duplicate_count)

    print(
        f"Claim {claim_id}: "
        f"{duplicate_count} duplicate chunks"
    )


average_duplicates = sum(duplicate_counts) / len(duplicate_counts)
maximum_duplicates = max(duplicate_counts)

print("\nSummary")
print(f"Number of claims: {len(duplicate_counts)}")
print(f"Average duplicate chunks per claim: {average_duplicates:.3f}")
print(f"Maximum duplicate chunks in one claim: {maximum_duplicates}")