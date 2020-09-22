Aryan Luthra's CS285 HW1 ReadMe

In order to replicate my results please run the following commands for each Question and Environment.

--- Q1 ANT

python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --num_agent_train_steps_per_iter 2000 --size 128 --eval_batch_size 10000


--- Q1 Human

python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl --num_agent_train_steps_per_iter 2000 --size 128 --eval_batch_size 10000

--- Q1.3 Ant Hyperparams

python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant_1000 --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --num_agent_train_steps_per_iter 1000 --size 128 --eval_batch_size 10000

python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant_3000 --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --num_agent_train_steps_per_iter 3000 --size 128 --eval_batch_size 10000

python cs285/scripts/run_****hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant_5000 --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --num_agent_train_steps_per_iter 5000 --size 128 --eval_batch_size 10000


--- Q2 ANT

python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name dagger_ant --n_iter 15 --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --size 128 --eval_batch_size 10000

-- Q2 Human

python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name dagger_human --n_iter 15 --do_dagger --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl --size 128 --eval_batch_size 10000

