mergekit-evolve ./examples/genomic_1.yml --storage-path ./mistralv02_mammoth7b_ties_1_synthetic_task_and_safety_2k_100_1_0p3 --task-search-path workspace/eval_tasks/ --merge-cuda --max-fevals 100
mergekit-evolve ./examples/genomic_8.yml --storage-path ./mistralv02_mammoth7b_ties_8_synthetic_task_and_safety_2k_100_1_0p3 --task-search-path workspace/eval_tasks/ --merge-cuda --max-fevals 100
mergekit-evolve ./examples/genomic_16.yml --storage-path ./mistralv02_mammoth7b_ties_16_synthetic_task_and_safety_2k_100_1_0p3 --task-search-path workspace/eval_tasks/ --merge-cuda --max-fevals 100
