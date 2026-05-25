# benchmark/query_generator.py

"""
Query generation utilities.

Generates three types of queries:
1. In-dataset: Direct questions from PubMedQA
2. Edge cases: Paraphrased/similar questions
3. Out-of-distribution: Novel medical questions
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from benchmark.data_loader import DataLoader
from benchmark.benchmark_config import config as benchmark_config


class QueryGenerator:
    """
    Generates test queries for benchmarking.
    """

    def __init__(self, data_loader: DataLoader):
        """
        Initialize query generator.

        Args:
            data_loader: DataLoader instance
        """
        self.data_loader = data_loader

        # Paraphrase templates for edge cases
        self.paraphrase_templates = [
            "What is known about {}?",
            "Can you explain {}?",
            "Tell me about {}",
            "What are the main aspects of {}?",
            "How does {} manifest?",
            "What research exists on {}?",
            "Describe the features of {}",
            "What causes {}?",
            "How is {} treated?",
            "What are symptoms of {}?",
            "What is the prognosis for {}?",
            "How can {} be prevented?",
        ]

        # OOD query templates
        self.ood_templates = [
            "What is the latest research on {}?",
            "How effective is {} treatment?",
            "What are risk factors for {}?",
            "Can {} be prevented?",
            "What diagnostic tests exist for {}?",
            "How does {} affect quality of life?",
            "What medications treat {}?",
            "What are complications of {}?",
            "How is {} diagnosed in children?",
            "What lifestyle changes help with {}?",
        ]

        # Medical topics for OOD
        self.medical_topics = [
            "COVID-19",
            "long COVID",
            "diabetes",
            "hypertension",
            "cancer",
            "Alzheimer's disease",
            "Parkinson's disease",
            "heart disease",
            "stroke",
            "obesity",
            "mental health",
            "depression",
            "anxiety",
            "arthritis",
            "asthma",
            "COPD",
            "kidney disease",
            "liver disease",
            "autoimmune disorders",
            "multiple sclerosis",
            "lupus",
            "rheumatoid arthritis",
            "osteoporosis",
            "thyroid disorders",
            "sleep apnea",
            "chronic pain",
            "fibromyalgia",
            "migraine",
            "epilepsy",
            "dementia",
            "tuberculosis",
            "HIV",
            "hepatitis",
        ]

    def load_manifest_queries(
        self, file_path: str, limit: Optional[int] = None
    ) -> Tuple[
        List[str], List[str], np.ndarray, List[str], List[List[str]], List[dict]
    ]:
        """Load labeled queries from a JSONL manifest."""
        print(f"[QUERIES] Loading manifest from {file_path}...")

        query_ids = []
        query_texts = []
        query_types = []
        expected_ids = []
        query_metadata = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                payload = record.get("payload") or record.get("text")
                if payload is None:
                    raise ValueError(
                        f"Manifest line {line_number} is missing payload/text."
                    )
                query_ids.append(str(record.get("query_id", f"manifest_{line_number}")))
                query_texts.append(str(payload))
                query_types.append(str(record.get("query_type", "manifest")))
                expected_ids.append(
                    [str(item_id) for item_id in record.get("expected_ids", [])]
                )
                metadata = dict(record.get("metadata", {}) or {})
                for field_name in ("session_id", "turn_id", "trace_id", "order_index"):
                    if (
                        record.get(field_name) is not None
                        and field_name not in metadata
                    ):
                        metadata[field_name] = record[field_name]
                query_metadata.append(metadata)
                if limit and len(query_ids) >= limit:
                    break
        # Check for cached embeddings
        cache_path = Path(file_path).parent / (Path(file_path).stem.replace('_manifest', '') + '_query_embeddings.npy')
        if cache_path.exists():
            print(f"[QUERIES] Loading cached embeddings from {cache_path}")
            query_embeddings = np.load(str(cache_path))
            if query_embeddings.shape[0] == len(query_ids):
                print(f"[QUERIES] Cache hit: {query_embeddings.shape}")
            else:
                print(f"[QUERIES] Cache shape mismatch ({query_embeddings.shape[0]} vs {len(query_ids)}), re-encoding...")
                query_embeddings = self.data_loader.generate_embeddings(query_texts, prefix="query: ")
        else:
            query_embeddings = self.data_loader.generate_embeddings(query_texts, prefix="query: ")

        print(f"[QUERIES] Loaded {len(query_ids)} labeled manifest queries")
        return (
            query_ids,
            query_texts,
            query_embeddings,
            query_types,
            expected_ids,
            query_metadata,
        )

    def load_custom_queries(
        self, file_path: str, limit: Optional[int] = None
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Load queries from text file.

        Args:
            file_path: Path to query file (one query per line)
            limit: Maximum number of queries to load

        Returns:
            (query_ids, query_texts, query_embeddings)
        """
        print(f"[QUERIES] Loading from {file_path}...")

        with open(file_path, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]

        if limit:
            queries = queries[:limit]

        # Generate IDs
        query_ids = [f"custom_{i}" for i in range(len(queries))]

        # Generate embeddings
        query_embeddings = self.data_loader.generate_embeddings(
            queries, prefix="query: "
        )

        print(f"[QUERIES] [OK] Loaded {len(queries)} custom queries")

        return query_ids, queries, query_embeddings

    def generate_quick_queries(
        self, count: int
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Generate quick test queries (sampled from dataset).

        Args:
            count: Number of queries

        Returns:
            (query_ids, query_texts, query_embeddings)
        """
        print(f"[QUERIES] Generating {count} quick test queries...")

        # Load dataset
        dataset = self.data_loader.load_dataset()

        # Sample indices (skip TIER 1)
        indices = self.data_loader.get_sample_indices(
            count,
            skip_first=min(benchmark_config.TIER1_SIZE, max(len(dataset) - count, 0)),
        )

        # Extract questions
        query_texts = [dataset[i]["question"] for i in indices]
        query_ids = [f"quick_{i}" for i in range(len(query_texts))]

        # Generate embeddings
        query_embeddings = self.data_loader.generate_embeddings(
            query_texts, prefix="query: "
        )

        print(f"[QUERIES] [OK] Generated {len(query_texts)} quick queries")

        return query_ids, query_texts, query_embeddings

    def generate_full_test_queries(
        self,
        total: int,
        in_dataset_ratio: float,
        edge_case_ratio: float,
        ood_ratio: float,
    ) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
        """
        Generate full test with realistic distribution.

        Args:
            total: Total number of queries
            in_dataset_ratio: Ratio of in-dataset queries (e.g., 0.65)
            edge_case_ratio: Ratio of edge case queries (e.g., 0.25)
            ood_ratio: Ratio of OOD queries (e.g., 0.10)

        Returns:
            (query_ids, query_texts, query_embeddings, query_types)
        """
        # Calculate counts
        in_count = int(total * in_dataset_ratio)
        edge_count = int(total * edge_case_ratio)
        ood_count = total - in_count - edge_count

        print(f"[QUERIES] Generating {total} queries:")
        print(f"  In-dataset:   {in_count}")
        print(f"  Edge cases:   {edge_count}")
        print(f"  Out-of-dist:  {ood_count}")

        all_ids = []
        all_texts = []
        all_types = []

        # 1. In-dataset queries
        print("\n[QUERIES] Generating in-dataset queries...")
        in_ids, in_texts, _ = self._generate_in_dataset(in_count)
        all_ids.extend(in_ids)
        all_texts.extend(in_texts)
        all_types.extend(["in_dataset"] * len(in_ids))

        # 2. Edge case queries
        print("[QUERIES] Generating edge case queries...")
        edge_ids, edge_texts = self._generate_edge_cases(edge_count)
        all_ids.extend(edge_ids)
        all_texts.extend(edge_texts)
        all_types.extend(["edge_case"] * len(edge_ids))

        # 3. OOD queries
        print("[QUERIES] Generating OOD queries...")
        ood_ids, ood_texts = self._generate_ood(ood_count)
        all_ids.extend(ood_ids)
        all_texts.extend(ood_texts)
        all_types.extend(["out_of_distribution"] * len(ood_ids))

        # Shuffle
        print("\n[QUERIES] Shuffling queries (realistic interleaving)...")
        indices = np.arange(len(all_ids))
        np.random.shuffle(indices)

        all_ids = [all_ids[i] for i in indices]
        all_texts = [all_texts[i] for i in indices]
        all_types = [all_types[i] for i in indices]

        # Generate embeddings
        print("[QUERIES] Generating embeddings...")
        embeddings = self.data_loader.generate_embeddings(all_texts, prefix="query: ")

        print(f"[QUERIES] [OK] Generated {len(all_ids)} mixed queries")

        return all_ids, all_texts, embeddings, all_types

    def _generate_in_dataset(
        self, count: int
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """Generate in-dataset queries."""
        dataset = self.data_loader.load_dataset()

        indices = self.data_loader.get_sample_indices(
            count,
            skip_first=min(benchmark_config.TIER1_SIZE, max(len(dataset) - count, 0)),
        )

        query_texts = [dataset[i]["question"] for i in indices]
        query_ids = [f"in_dataset_{i}" for i in range(len(query_texts))]

        embeddings = self.data_loader.generate_embeddings(query_texts, prefix="query: ")

        return query_ids, query_texts, embeddings

    def _generate_edge_cases(self, count: int) -> Tuple[List[str], List[str]]:
        """Generate edge case (paraphrased) queries."""
        dataset = self.data_loader.load_dataset()

        indices = self.data_loader.get_sample_indices(
            count,
            skip_first=min(benchmark_config.TIER1_SIZE, max(len(dataset) - count, 0)),
            seed=43,  # Different seed
        )

        query_ids = []
        query_texts = []

        for i, idx in enumerate(indices):
            item = dataset[idx]
            original = item["question"]

            # Extract key medical term (simple heuristic)
            words = original.split()
            medical_term = " ".join(words[-3:]) if len(words) >= 3 else words[-1]

            # Paraphrase
            template = self.paraphrase_templates[i % len(self.paraphrase_templates)]
            paraphrased = template.format(medical_term)

            query_ids.append(f"edge_case_{i}")
            query_texts.append(paraphrased)

        return query_ids, query_texts

    def _generate_ood(self, count: int) -> Tuple[List[str], List[str]]:
        """Generate out-of-distribution queries."""
        query_ids = []
        query_texts = []

        for i in range(count):
            topic = np.random.choice(self.medical_topics)
            template = np.random.choice(self.ood_templates)
            query = template.format(topic)

            query_ids.append(f"ood_{i}")
            query_texts.append(query)

        return query_ids, query_texts
