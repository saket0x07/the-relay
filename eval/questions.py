"""
15+ Curated Evaluation Questions for The Relay.
Structured across categories:
- Single-Paper Factual
- Numeric / Table-Specific
- Ambiguous & Multi-Paper Terms (e.g. "chunking")
- Unanswerable / Traps (mandating "The papers do not say")
"""

import yaml

QUESTIONS = [
    # 1. Single-Paper Factual
    {
        "id": "q01",
        "category": "factual",
        "question": "What is the dimensionality d_model of the base Transformer model in Attention Is All You Need?",
        "target_paper": "1706.03762v7",
        "expected_answer": "512",
        "is_answerable": True
    },
    {
        "id": "q02",
        "category": "factual",
        "question": "What is the total latency of Moshi including the neural audio codec?",
        "target_paper": "2410.00037v1",
        "expected_answer": "around 200ms or 160ms frame latency",
        "is_answerable": True
    },
    {
        "id": "q03",
        "category": "factual",
        "question": "What primary mechanism does Dynamic Chunking introduce to optimize sequence modeling?",
        "target_paper": "2507.07955v2",
        "expected_answer": "hierarchical sequence modeling with dynamic boundaries",
        "is_answerable": True
    },
    {
        "id": "q04",
        "category": "factual",
        "question": "How many attention heads does the base Transformer use in Attention Is All You Need?",
        "target_paper": "1706.03762v7",
        "expected_answer": "8 heads",
        "is_answerable": True
    },
    {
        "id": "q05",
        "category": "factual",
        "question": "What audio tokenizer or codec does Moshi train on for speech representation?",
        "target_paper": "2410.00037v1",
        "expected_answer": "Mimi (neural audio codec)",
        "is_answerable": True
    },

    # 2. Numeric / Table Lookups
    {
        "id": "q06",
        "category": "numeric_table",
        "question": "What BLEU score did the big Transformer model achieve on the WMT 2014 English-to-German translation task in Table 2?",
        "target_paper": "1706.03762v7",
        "expected_answer": "28.4",
        "is_answerable": True
    },
    {
        "id": "q07",
        "category": "numeric_table",
        "question": "What BLEU score did the base Transformer model achieve on WMT 2014 English-to-French in Table 2?",
        "target_paper": "1706.03762v7",
        "expected_answer": "38.1",
        "is_answerable": True
    },
    {
        "id": "q08",
        "category": "numeric_table",
        "question": "What is the parameter size of the text foundation model used in Moshi?",
        "target_paper": "2410.00037v1",
        "expected_answer": "7B parameters (Helium-7B)",
        "is_answerable": True
    },

    # 3. Disambiguation (Cross-Paper)
    {
        "id": "q09",
        "category": "disambiguation",
        "question": "How does chunking work in the hierarchical sequence modeling paper compared to traditional RAG text splitting?",
        "target_paper": "2507.07955v2",
        "expected_answer": "Dynamic boundary learning within model layers, not fixed character-length text splitting",
        "is_answerable": True
    },
    {
        "id": "q10",
        "category": "disambiguation",
        "question": "Does Moshi use full-context bidirectional attention or causal streaming attention during spoken dialogue generation?",
        "target_paper": "2410.00037v1",
        "expected_answer": "Causal / streaming attention for real-time operation",
        "is_answerable": True
    },

    # 4. Unanswerable & Traps (Mandates "The papers do not say")
    {
        "id": "q11",
        "category": "unanswerable",
        "question": "What is the training cost of Moshi in US dollars on AWS cloud instances?",
        "target_paper": "2410.00037v1",
        "expected_answer": "The papers do not say",
        "is_answerable": False
    },
    {
        "id": "q12",
        "category": "unanswerable",
        "question": "What GPU memory footprint does Dynamic Chunking require when deployed on an Apple iPhone 15?",
        "target_paper": "2507.07955v2",
        "expected_answer": "The papers do not say",
        "is_answerable": False
    },
    {
        "id": "q13",
        "category": "unanswerable",
        "question": "Which specific university cafeteria did the authors of Attention Is All You Need visit while writing the paper?",
        "target_paper": "1706.03762v7",
        "expected_answer": "The papers do not say",
        "is_answerable": False
    },
    {
        "id": "q14",
        "category": "unanswerable",
        "question": "What is the human evaluation preference score of Moshi compared directly against OpenAI GPT-4o voice mode?",
        "target_paper": "2410.00037v1",
        "expected_answer": "The papers do not say",
        "is_answerable": False
    },
    {
        "id": "q15",
        "category": "unanswerable",
        "question": "Does Dynamic Chunking utilize a quantum annealing optimizer to select sequence split points?",
        "target_paper": "2507.07955v2",
        "expected_answer": "The papers do not say",
        "is_answerable": False
    },
    {
        "id": "q16",
        "category": "factual",
        "question": "What optimizer and learning rate schedule was used to train the models in Attention Is All You Need?",
        "target_paper": "1706.03762v7",
        "expected_answer": "Adam optimizer with warmup steps and inverse square root decay",
        "is_answerable": True
    }
]

if __name__ == "__main__":
    with open("eval/questions.yaml", "w", encoding="utf-8") as f:
        yaml.dump(QUESTIONS, f, indent=2, sort_keys=False)
    print(f"Saved {len(QUESTIONS)} curated questions to eval/questions.yaml")

