# Code Learning Roadmap (This Project + LeetCode 100)

## 1. Purpose

This roadmap is for someone who still feels close to ?I barely understand code,
but I want to genuinely understand this project.?

The goal is not to make you instantly capable of rewriting the full repository.
The goal is to move through three layers:

1. understand the core scripts on the mainline
2. explain why the main code is written that way
3. use Python fundamentals, data structures, and common algorithm problems to
   build real coding ability underneath the project story

This roadmap addresses two problems at once:

- understanding the project itself
- building the coding foundation needed to read and modify it safely

## 2. Learning Goals By Level

### 2.1 Level 1: you can read it

At this stage you should be able to:

- understand the overall structure of a Python script
- understand config, paths, functions, inputs, and outputs
- understand what a DataFrame transformation is doing
- understand where a metric comes from

### 2.2 Level 2: you can change small logic

At this stage you should be able to:

- change constants and parameters yourself
- adjust filtering or sorting logic
- add a small helper function
- read an error message and locate the failing layer

### 2.3 Level 3: you can write small modules yourself

At this stage you should be able to:

- write a small evaluation script
- write a small data-processing script
- write a simple TopK, retrieval, or scoring routine
- explain the implementation logic behind more complex code

## 3. Learning Principles

### 3.1 Learn what the project actually uses first

Do not start by trying to finish a whole Python textbook. Start from the syntax,
data structures, and patterns that already appear in this repository.

### 3.2 Build code-reading ability before code-writing ability

If you are still near the beginner stage, reading code matters more than writing
from scratch. Your first win is learning how to recognize what a piece of code
is doing and where a change will have downstream impact.

### 3.3 Tie every topic back to a real project file

For each topic, answer:

1. where does it appear in this repository?
2. what problem does it solve here?
3. where would you get stuck in an interview if you did not understand it?

## 4. Full Map Of Skills To Build

The real coding skill set behind this repository has seven parts:

1. Python basics and data structures
2. file and data handling
3. pandas / numpy basics
4. Spark DataFrame thinking
5. ranking and evaluation code
6. SFT / DPO data construction code
7. engineering scripts and command-line execution

The first four are foundational. The last three are where the project mainline
becomes concrete.

## 5. Stage 1: Python Basics And Data Structures

### 5.1 Why start here

Without this layer, the code will still look opaque because even the basic
control flow and container usage will be hard to read.

### 5.2 Topics you must know

- variables, strings, numbers, booleans
- `if / elif / else`
- `for / while`
- function definition and return values
- `list / tuple / dict / set`
- list comprehensions and dictionary basics
- `try / except`
- modules and `import`
- `Path`

### 5.3 Direct mapping inside this project

Start with:

- [internal_pilot_runner.py](../scripts/pipeline/internal_pilot_runner.py)

This is friendlier than Spark-heavy scripts and helps you understand:

- script entry structure
- function decomposition
- JSON read/write flow
- path organization
- how commands are launched

### 5.4 Focus on understanding

- `read_json` / `write_json`
- `sanitize_name`
- `base_env`
- `run_python_script`
- `current_release_context`

### 5.5 Suggested exercises

1. write a function that reads a JSON file and prints one field
2. write a function that saves a dictionary as JSON
3. build an output path with `Path`
4. mimic a minimal environment-variable merge function

### 5.6 Relevant LeetCode topics

- arrays
- hash maps
- strings
- simulation

Suggested problems:

- Two Sum
- Valid Anagram
- Group Anagrams
- Top K Frequent Elements
- Valid Parentheses
- Longest Substring Without Repeating Characters

### 5.7 Why those problems help

- `dict` and `set` appear constantly in real scripts
- TopK thinking maps directly to candidate truncation and reranking
- string handling appears in prompts, paths, configs, and text cleaning

## 6. Stage 2: File Handling And Data Handling

### 6.1 Why this stage matters

This is not only an algorithm project. It is also a data project. You need to
understand where files come from, what outputs mean, and why manifests exist.

### 6.2 Topics you must know

- `json`
- `csv`
- basic text-file read/write
- `Path.glob`
- directory structure organization
- `run_meta.json` as run metadata

### 6.3 Direct mapping inside this project

Read:

- [internal_pilot_runner.py](../scripts/pipeline/internal_pilot_runner.py)
- [check_release_readiness.py](../tools/check_release_readiness.py)

### 6.4 Suggested exercises

1. scan all JSON files in a directory
2. count one field across many `run_meta.json` files
3. merge two JSON files into a summary JSON

### 6.5 Relevant LeetCode topics

- arrays and strings
- maps and sets
- sorting basics

### 6.6 Why this helps

A lot of engineering confidence comes from knowing how data is stored, located,
and summarized rather than from fancy modeling alone.

## 7. Stage 3: pandas / numpy Basics

### 7.1 Why this stage matters

Before Spark becomes readable, you need a clean mental model for tabular data.

### 7.2 Topics you must know

- selecting columns and rows
- filtering
- sorting
- group-by thinking
- merge / join basics
- vectorized operations
- missing values

### 7.3 Direct mapping inside this project

Look for smaller local data-processing sections in stage09 through stage11 and
trace how CSV or metrics outputs are assembled before they are saved.

### 7.4 Suggested exercises

1. load a CSV and sort by one score column
2. group by user and keep top-k items
3. join two tables on a shared key
4. compute a small summary table yourself

### 7.5 Relevant LeetCode topics

- sorting
- prefix sums
- arrays
- hash maps
- heaps / TopK

### 7.6 Why this helps

Tabular thinking is a bridge between ?basic Python? and ?production-style data
pipelines.?

## 8. Stage 4: Spark DataFrame Thinking

### 8.1 Why this is the most important stage

A lot of the repository value sits in Spark-based data work and local-machine
resource constraints. If you cannot think in DataFrames and expensive actions,
you will misread many important design decisions.

### 8.2 Topics you must know

- lazy execution
- transformations vs actions
- joins, aggregations, and windows
- partitions and shuffles
- why repeated `toPandas()` is dangerous
- why local hardware guardrails matter

### 8.3 Learn it together with repository constraints

Read Spark code together with the repository rules in `AGENTS.md` so you can see
why repeated actions, repeated scans, and large local materializations are
explicitly discouraged.

### 8.4 Direct mapping inside this project

Focus on:

- `09_candidate_fusion.py`
- `10_1_rank_train.py`
- `10_2_rank_infer_eval.py`

### 8.5 Suggested exercises

1. trace one DataFrame from load to save
2. mark which lines are transformations and which lines trigger actions
3. identify where a shuffle happens and why
4. explain why a given `persist` is or is not justified

### 8.6 Relevant LeetCode topics

- heaps and TopK
- intervals
- prefix / cumulative logic
- sorting and grouping

### 8.7 Why this helps

These topics sharpen the same instincts needed to reason about ranking windows,
top-k truncation, and grouped evaluation.

## 9. Stage 5: Ranking And Evaluation Code

### 9.1 Why this stage matters

This is where the project stops being ?data prep? and becomes a recommender
pipeline with measurable outcomes.

### 9.2 Topics you must know

- top-k ranking logic
- Recall@K
- NDCG@K
- candidate truncation
- cohort consistency
- parity between evaluation runs

### 9.3 Direct mapping inside this project

Read together:

- `10_1_rank_train.py`
- `10_2_rank_infer_eval.py`
- `11_3_qlora_sidecar_eval.py`
- release-readiness and alignment docs

### 9.4 Suggested exercises

1. manually compute Recall@K for a tiny example
2. manually compute a simple DCG / NDCG example
3. explain why two runs are not comparable when the candidate pool changes

### 9.5 Relevant LeetCode topics

- heaps / TopK
- custom sorting
- arrays and prefix logic

### 9.6 Why this helps

Once ranking logic becomes intuitive, the project story becomes much easier to
explain in both engineering and interview contexts.

## 10. Stage 6: SFT / DPO Data Construction Code

### 10.1 Why this stage matters

This is where the repository connects recommender evidence to LLM fine-tuning.

### 10.2 Topics you must know

- how prompts are built
- how training pairs are constructed
- why sequence length matters
- why pairwise training differs from pointwise training
- what makes a preference pair valid or noisy

### 10.3 Direct mapping inside this project

Read:

- `11_1_qlora_build_dataset.py`
- `11_2_qlora_train.py`
- `11_2_dpo_train.py`

### 10.4 What you really need to understand

You do not need to understand every `if` branch first. You need to understand:

- what the input sample looks like
- how it becomes prompt text
- how the output directory is organized
- how evaluation reconnects training outputs to the ranking pipeline

### 10.5 Suggested exercises

1. trace one sample from dataset build to training input
2. explain chosen vs rejected text in DPO
3. inspect how prompt length is controlled

### 10.6 Relevant LeetCode topics

- strings
- arrays
- maps
- simple simulation

### 10.7 Why this helps

The hard part here is not algorithm trivia. The hard part is connecting pipeline
evidence, prompt construction, and training constraints.

## 11. Stage 7: Engineering Scripts And CLI Execution

### 11.1 Why this stage matters

A lot of real project work happens in wrappers, validators, manifests, and
release helpers rather than in the model file alone.

### 11.2 Topics you must know

- shell and batch basics
- command invocation
- environment variables
- run directory structure
- validation scripts
- release pointers and manifests

### 11.3 Direct mapping inside this project

Read:

- `scripts/run_internal_pilot.bat`
- `scripts/pipeline/internal_pilot_runner.py`
- `tools/check_release_readiness.py`
- `tools/check_release_monitoring.py`

### 11.4 Suggested exercises

1. run a validation command and explain what it checks
2. trace how one wrapper passes environment variables to a Python script
3. summarize what a release pointer file means

### 11.5 Relevant LeetCode topics

There is no direct one-to-one algorithm mapping here. The real skill is system
reading and operational reasoning.

### 11.6 Why this helps

This is the layer that turns experiment work into something reviewable and
re-runnable.

## 12. Mapping LeetCode 100 To This Project

### 12.1 Highest-priority problem families

#### First priority

- arrays
- hash maps
- strings
- heaps / TopK

#### Second priority

- sorting
- intervals
- sliding window
- binary search on answers or thresholds

#### Third priority

- trees and graphs only at a basic level
- dynamic programming only in common interview forms

### 12.2 Problem families you should not overinvest in too early

Do not spend your earliest energy on advanced graph theory or very abstract DP
if the main goal is to understand this repository quickly.

## 13. Suggested LeetCode 100 Order

### 13.1 Group 1: lock the basics first

- Two Sum
- Valid Anagram
- Group Anagrams
- Contains Duplicate
- Valid Parentheses

### 13.2 Group 2: build TopK and sorting intuition

- Top K Frequent Elements
- Kth Largest Element in an Array
- Merge Intervals
- Sort Colors

### 13.3 Group 3: build window and sequence intuition

- Longest Substring Without Repeating Characters
- Minimum Window Substring
- Product of Array Except Self

### 13.4 Group 4: fill standard interview gaps

- binary search templates
- prefix sum problems
- simple BFS / DFS templates

## 14. Eight-Week Learning Route

## Weeks 1-2: Python basics plus arrays / hash maps

Goal: read helper scripts and small utilities comfortably.

## Weeks 3-4: pandas, file handling, sorting, and heaps

Goal: become comfortable with metrics files, manifests, and top-k logic.

## Weeks 5-6: Spark plus evaluation code

Goal: understand stage09 and stage10 mainline behavior.

## Week 7: SFT / DPO data construction

Goal: explain how recommendation evidence becomes LLM training data.

## Week 8: engineering and review

Goal: connect validators, release flow, rollback, and project delivery.

## 15. Recommended Project Code Reading Order

1. `scripts/pipeline/internal_pilot_runner.py`
2. `tools/check_release_readiness.py`
3. `scripts/09_candidate_fusion.py`
4. `scripts/10_1_rank_train.py`
5. `scripts/10_2_rank_infer_eval.py`
6. `scripts/11_1_qlora_build_dataset.py`
7. `scripts/11_2_dpo_train.py`
8. `scripts/11_3_qlora_sidecar_eval.py`

## 16. Eight Questions To Answer For Every File You Read

1. What does this file take as input?
2. What does it produce as output?
3. What problem does it solve in the mainline?
4. Which functions are the real entry points?
5. Which parameters are most important?
6. Which parts are data logic vs evaluation logic vs orchestration logic?
7. What would break if this file changed?
8. How would you explain it in plain language?

## 17. Things Not To Dive Into Too Early

- every historical experiment branch
- every environment variable before understanding the mainline
- micro-optimizations before learning the data flow
- line-by-line memorization of large scripts

## 18. Minimum Completion Standard

You can consider the roadmap minimally complete when you can:

- explain the full pipeline in plain language
- map the main stages to the key scripts
- explain the core retrieval and ranking metrics
- explain why the LLM is a sidecar reranker
- read a script and identify input, output, and core logic
- make a small safe code change without getting lost immediately

## 19. One-Sentence Summary

The fastest path is not ?learn everything about coding first.? The fastest path
is ?use the project to learn the exact coding concepts that this repository
actually depends on, and let that grow into real engineering ability.?
