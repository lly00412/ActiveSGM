###### ablation for semantic loss
bash scripts/activesgm/run_replica.sh office0 1 ablation_w_kl 0 0,1
bash scripts/activesgm/run_replica.sh room0 1 ablation_w_kl 0 0,1

bash scripts/evaluation/eval_replica_semantic.sh office0 1 ablation_w_kl 0 0 0 exploration_stage_1
bash scripts/evaluation/eval_replica_semantic.sh room0 1 ablation_w_kl 0 0 0 exploration_stage_1