# Modular RAG for Evidence-Based Claim Verification

This repository contains the code developed for an MSc dissertation investigating how individual Retrieval-Augmented Generation (RAG) components affect evidence-based claim verification. The project uses AVeriTeC claims and evidence to compare chunking strategies, retrievers and LLM generators, together with ablations of normalised exact-text deduplication and Cross-Encoder re-ranking.

The repository provides the complete component code and a fixed executable development-set reference pipeline:

```text
AVeriTeC development claims
        -> 200-word chunks with 50-word overlap
        -> BM25 and dense BGE retrieval
        -> Hybrid retrieval using Reciprocal Rank Fusion
        -> normalised exact-text deduplication
        -> Cross-Encoder re-ranking from 50 candidates to top 10
        -> Qwen3:30b claim verification
        -> classification evaluation
```

The complete experimental matrix used in the dissertation is substantially more computationally and storage intensive. Some scripts retain commented configuration lists documenting the broader chunking, retrieval, deduplication, re-ranking and generator experiments, while their active configuration runs the reference pipeline above.

## Repository structure

```text
.
├── AVeriTeC/
│   ├── data/
│   │   └── internal_split/
│   │       ├── dev_ids_200.json
│   │       └── test_ids_600.json
│   └── evidence_store/                 # generated locally
├── data_aggregating&cleaning/
│   ├── jsonl_aggregation.py
│   ├── test_dev_split.py
│   └── evidence_store_split.py
├── modular/
│   ├── Chunking/
│   ├── retreiver/
│   │   ├── BM25/
│   │   ├── Embedding/
│   │   └── Hybrid/
│   └── rerank/
├── src/                                # prompt, schema and LLM loading
├── output/                             # generated locally
├── run_RAG.py
├── run_NoRAG.py
├── Batch_evaluation.py
└── requirements.txt
```

The folder name `retreiver` is retained because it is used by the existing project structure.

## Data availability

This repository does not redistribute the AVeriTeC claims, evidence documents, processed evidence stores, SQLite databases, indexes, embeddings, retrieval caches or prediction outputs. Download the AVeriTeC training data and full training evidence store from the official AVeriTeC source and place them under `AVeriTeC/data/`.

The expected data layout after download and preprocessing is:

```text
AVeriTeC/
├── data/
│   ├── train.json
│   ├── aggregated_raw.jsonl
│   ├── <downloaded evidence directory>/
│   └── internal_split/
│       ├── dev_claims_200.json
│       ├── dev_ids_200.json
│       ├── test_claims_600.json
│       └── test_ids_600.json
└── evidence_store/
    ├── dev_evidence_200.jsonl
    └── test_evidence_600.jsonl
```

The repository includes `dev_ids_200.json` and `test_ids_600.json` to record the claim indices used for the dissertation experiments. `test_dev_split.py` documents the stratified random sampling procedure. Running that script regenerates the claims and ID files and may produce a split that differs from the provided experimental ID records if the source data order or sampling environment differs.

## Environment

The pipeline requires:

- Python and the packages listed in `requirements.txt`;
- a compatible Java runtime for Pyserini/Lucene;
- an NVIDIA CUDA environment for the default dense embedding and re-ranking configuration;
- Ollama with the required Qwen model installed and running.

Create and activate a virtual environment, then install the Python dependencies:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The `torch` build in `requirements.txt` reflects the CUDA environment used by the project. A different PyTorch build may be required for another CUDA version or for CPU-only execution. `installed_packages.txt` is retained as a snapshot of the broader development environment rather than as the recommended installation file.

Verify Java and Ollama before running the retrieval and generation stages:

```bash
java -version
ollama --version
```

The reference generator is loaded through the Ollama model tag used in `run_RAG.py`:

```bash
ollama pull qwen3:30b
```

### External dependencies

The Python dependencies can be installed from `requirements.txt`. Java and Ollama must be installed separately.

- Java is required by Pyserini/Lucene for BM25 indexing and retrieval. (JDK 21) 
  Installation: https://adoptium.net/

Pyserini/Lucene requires a complete JDK. After installing the JDK, configure `JAVA_HOME` and add the JDK `bin` directory to `PATH`.

On Windows:

1. Open **Environment Variables**.
2. Create the system variable:

```text
JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-<installed-version>
```
- Ollama is required to run the local Qwen generators.  
  Installation: https://ollama.com/download

After installing Ollama, download the generator used by the reference pipeline:


## Reproducing the development-set reference pipeline

Run all commands from the repository root.

### 1. Aggregate the complete evidence store

`jsonl_aggregation.py` accepts a flexible evidence input directory. Replace `<downloaded-evidence-directory>` with the directory containing the downloaded AVeriTeC training evidence:

```bash
python "data_aggregating&cleaning/jsonl_aggregation.py" \
  --input_dir "AVeriTeC/data/<downloaded-evidence-directory>" \
  --output_path "AVeriTeC/data/aggregated_raw.jsonl" \
  --no_delete

```
The `--no_delete` option is important: without it, the aggregation script deletes each source file after it has been merged successfully.

### 2. Construct the development and held-out splits

```bash
python "data_aggregating&cleaning/test_dev_split.py"
```

This reads `AVeriTeC/data/train.json` and writes the sampled claim and ID files to `AVeriTeC/data/internal_split/`. The configured development distribution is 54 Supported, 114 Refuted, 12 Conflicting Evidence/Cherrypicking and 20 Not Enough Evidence claims. The configured held-out distribution is 162, 342, 36 and 60 claims respectively.

The development and held-out claim sets are non-overlapping. The supplied ID files document the samples used for the reported dissertation experiments; regenerated splits may differ as described above. You can ignore this and directly run the evidence_store_split file with the supplied ID  files. 

### 3. Split the aggregated evidence store

```bash
python "data_aggregating&cleaning/evidence_store_split.py"
```

This reads the aggregated evidence and the generated development/test IDs, then writes:

```text
AVeriTeC/evidence_store/dev_evidence_200.jsonl
AVeriTeC/evidence_store/test_evidence_600.jsonl
```

### 4. Build the 200-word, 50-word-overlap development database

```bash
python modular/Chunking/word_overlap2DB.py
```

Output:

```text
output/chunking/dev_chunks_200_overlap_50.db
```

The sentence-based script and the commented 100-word configuration are retained for the component comparison experiments but are not required by the active reference pipeline.

### 5. Build the BM25 index

```bash
python modular/retreiver/BM25/BM25_indexing.py \
  --database_path output/chunking/dev_chunks_200_overlap_50.db \
  --index_path output/bm25/dev_chunks_200_overlap_50_index \
  --chunk_type word
```

Output:

```text
output/bm25/dev_chunks_200_overlap_50_index/
```

`BM25_indexing.py` accepts flexible input and output paths. The paths above are required by the active Hybrid reference configuration.

### 6. Generate dense embeddings

```bash
python modular/retreiver/Embedding/embedding.py \
  --database_path output/chunking/dev_chunks_200_overlap_50.db \
  --embedding_path output/embedding/dev_chunks_200_overlap_50_embedding \
  --chunk_type word
```

The script generates sharded `vectors_*.npy` and corresponding `keys_*.jsonl` files under:

```text
output/embedding/dev_chunks_200_overlap_50_embedding/
```

The active embedding model is `BAAI/bge-small-en-v1.5`. Embedding the complete evidence store can require substantial time, GPU resources and storage.

### 7. Run Hybrid retrieval and exact-text deduplication

```bash
python modular/retreiver/Hybrid/hybrid_retrieve.py
```

The active configuration uses BM25 and dense retrieval, Reciprocal Rank Fusion with `RRF_K = 60`, normalised exact-text deduplication, and retains 50 candidates for re-ranking.

Output:

```text
output/hybrid/Hybrid/word_200_50_Hybrid_retrieval_cache_50_Dedup_True.json
```

In Hybrid retrieval, identical normalised text from BM25 and dense retrieval must be matched during rank fusion so that contributions from both retrievers can be combined. The optional deduplication setting controls removal of repeated normalised evidence content within the retrieved candidate lists.

### 8. Run Cross-Encoder re-ranking

```bash
python modular/rerank/Rerank.py
```

The active reference configuration uses `cross-encoder/ms-marco-MiniLM-L12-v2` to re-rank the 50 Hybrid candidates and retain the top 10 chunks.

Output:

```text
output/rerank/Hybrid/word_200_50_retrieval_top10_rerank_True_Dedup_True.json
```

### 9. Run RAG claim verification

Ensure Ollama is running and the model tag in `run_RAG.py` is available, then run:

```bash
python run_RAG.py
```

The script reads the re-ranked evidence, formats the ten retrieved chunks, and sends each claim and its evidence to the downstream LLM using the fixed prompt and structured four-label output schema.

Output:

```text
output/RAG_prediction/Cross Encoder/Hybrid/word_200_50_retrieval_top10_rerank_True_Dedup_True_qwen3_30b_result.json
```

The four labels are:

- Supported;
- Refuted;
- Not Enough Evidence;
- Conflicting Evidence/Cherrypicking.

### 10. Evaluate predictions

Open `evaluation.ipynb` in Jupyter Notebook or JupyterLab. In the result-path cell, set `RESULT_PATH` to the prediction file produced by the reference pipeline:

```python
RESULT_PATH = Path(
    "output/RAG_prediction/Cross Encoder/Hybrid/"
    "word_200_50_retrieval_top10_rerank_True_"
    "Dedup_True_qwen3_30b_result.json"
)
```

Then run the notebook cells in order. The notebook loads the prediction records, removes invalid or failed outputs, constructs the four-class confusion matrix, and reports Accuracy together with class-level precision, recall and F1. Macro-F1 is included in the classification report through the `macro avg` row.

The notebook displays its evaluation tables and confusion matrix interactively and does not save `evaluation_summary.csv` by default. Its later bootstrap-comparison cells require multiple result files and are only needed when comparing several pipeline configurations; they are not required for evaluating the fixed reference pipeline.

The evaluation notebook additionally requires `pandas`, `matplotlib` and Jupyter Notebook or JupyterLab if these packages are not already installed:

```bash
pip install pandas matplotlib jupyter
```

## No-RAG comparison

`run_NoRAG.py` uses the same prompt, output schema and downstream LLM interface while passing an empty evidence string. It was used to assess performance without claim-specific retrieved evidence.

The active public example currently points to the development claims at:

```text
AVeriTeC/data/internal_split/dev_claims_200.json
```

Run it with:

```bash
python run_NoRAG.py
```

The model tag and output filename can be selected in the script's active configuration. To repeat the held-out comparison, use `AVeriTeC/data/internal_split/test_claims_600.json` together with the same downstream LLM used by the corresponding held-out RAG run. No-RAG is not part of the default development-set Hybrid reference sequence above.

## Full component experiments

The dissertation additionally compared:

- sentence, 100-word/25-word-overlap and 200-word/50-word-overlap chunking;
- BM25, Dense and Hybrid retrieval;
- deduplication enabled and disabled;
- Cross-Encoder re-ranking enabled and disabled;
- different Qwen generator configurations;
- RAG and No-RAG prediction.

The complete matrix requires separate databases, BM25 indexes, embedding stores, retrieval caches and LLM outputs for each configuration. Commented configuration lists in the scripts preserve the organisation of these experiments, but the active blocks have been reduced to the fixed Hybrid reference pipeline because repeating the complete matrix requires substantial computation, storage and LLM inference time.

To repeat another configuration, generate all required upstream artifacts using matching file names and enable the corresponding configuration values in the relevant scripts.

## Reproducibility notes and limitations

- Run scripts from the repository root so that fixed relative paths resolve correctly.
- The repository records the claim indices used for the reported experiments but does not redistribute the corresponding claims or evidence.
- Re-running the random sampling script may generate different samples if the input order or environment differs.
- Exact numerical reproduction also depends on the AVeriTeC release, Java/Pyserini environment, model versions, Ollama model build and available hardware.
- Large generated files are intentionally excluded from version control.
- The default scripts target CUDA devices for dense retrieval and Cross-Encoder re-ranking. CPU execution requires changing the relevant device settings and will be substantially slower.
- The repository reproduces the implemented experimental workflow; it does not provide a one-command reproduction of the full dissertation experiment matrix.

## Generated files

Do not commit the following generated or downloaded content:

```text
AVeriTeC/data/train.json
AVeriTeC/data/aggregated_raw.jsonl
AVeriTeC/data/internal_split/dev_claims_200.json
AVeriTeC/data/internal_split/test_claims_600.json
AVeriTeC/evidence_store/
output/
```

The two supplied ID files under `AVeriTeC/data/internal_split/` should remain version-controlled because they document the development and held-out samples used in the dissertation.

## Licence and attribution

AVeriTeC is an external dataset and remains subject to its own licence and attribution requirements. No AVeriTeC claims or evidence documents are redistributed in this repository. Users are responsible for downloading and using the dataset under the terms provided by its maintainers.

The source code in this repository accompanies the dissertation implementation. Add an explicit repository licence file if the code is intended for reuse beyond assessment and reproducibility.
