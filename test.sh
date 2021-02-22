# experiments on k-regular dataset
# --model choose from [GNN, GIN, 1-2-3-GNN,powerful-GNN, fingerprint, or fingerprint-no-clustering]
# fingerprint-no-clustering avoid using clustering coefficients as initial attributes of node, which is a better choice to run in this setting compared with fingerprint
python run_k_regular.py --model=1-2-3-GNN

#experiments for time and memory evaluation
# --model choose from [GIN, GNN, 1-2-3-GNN, fingerprint or powerful-GNN]
# --num_node takes arbitrary integer value
python run_evaluation_complexity.py --model=fingerprint --num_node=8

#experiments on TU dataset for baseline fingerprinting methods
#--dataset choose from [MUTAG NCI1 PROTEINS IMDB-BINARY IMDB-MULTI]
#--fingerprint_type choose from [NetLSD GeoScattering FEATHER]
python run_TU_baseline.py --dataset=MUTAG --fingerprint_type=NetLSD

#experiments on TU dataset for MP-GNN methods
#--dataset choose from [MUTAG NCI1 PROTEINS IMDB-BINARY IMDB-MULTI]
# --model choose from [GIN, GNN, 1-2-3-GNN]
python run_TU_gnn.py --dataset=MUTAG --model=1-2-3-GNN

#experiments on TU dataset for fingerprint methods
#--dataset choose from [MUTAG NCI1 PROTEINS IMDB-BINARY IMDB-MULTI]
#--fingerprint_type choose from [NetLSD GeoScattering FEATHER]
#--kernel_type choose from [square gaussian silverman]
#--initialization_type choose from [Uniform GMM]
#--if_learn choose from [true false]
python run_TU_fingerprints.py --dataset=MUTAG --kernel_type=gaussian --initialization_type=Uniform --fingerprint_type=NetLSD --if_learn=false

#experiments on karateclub dataset for fingerprint baseline methods
#--dataset choose from [deezer_ego_nets github_stargazers twitch_egos reddit_threads]
#--fingerprint_type choose from [NetLSD GeoScattering FEATHER]
#--seed can be any integer
python run_karateclub_baseline.py --dataset=deezer_ego_nets --fingerprint_type=NetLSD --seed=1


#experiments on karateclub dataset for MP-GNN methods
#--dataset choose from [deezer_ego_nets github_stargazers twitch_egos reddit_threads]
#--model choose from [GIN, GNN, 1-2-3-GNN]
#--seed can be any integer
python run_karateclub_gnn.py --dataset=deezer_ego_nets  --model=1-2-3-GNN --seed=1

#experiments on karateclub dataset for fingerprint methods
#--dataset choose from [deezer_ego_nets github_stargazers twitch_egos reddit_threads]
#--fingerprint_type choose from [NetLSD GeoScattering FEATHER]
#--seed can be any integer
#--kernel_type choose from [square gaussian silverman]
#--initialization_type choose from [Uniform GMM]
#--if_learn choose from [true false]
python run_karateclub_fingerprints.py --dataset=deezer_ego_nets --kernel_type=gaussian --initialization_type=Uniform --fingerprint_type=NetLSD --if_learn=false  --seed=1

#Introduce different model configuration, take NetLSD as example
#NetLSD-R
python run_TU_fingerprints.py --dataset=MUTAG --kernel_type=square --initialization_type=Uniform --fingerprint_type=NetLSD --if_learn=false
#NetLSD-G
python run_TU_fingerprints.py --dataset=MUTAG --kernel_type=gaussian --initialization_type=Uniform --fingerprint_type=NetLSD --if_learn=false
#NetLSD-S
python run_TU_fingerprints.py --dataset=MUTAG --kernel_type=silverman --initialization_type=Uniform --fingerprint_type=NetLSD --if_learn=false
#NetLSD-G-L
python run_TU_fingerprints.py --dataset=MUTAG --kernel_type=gaussian --initialization_type=Uniform --fingerprint_type=NetLSD --if_learn=true
#NetLSD-S-L
python run_TU_fingerprints.py --dataset=MUTAG --kernel_type=silverman --initialization_type=Uniform --fingerprint_type=NetLSD --if_learn=true
#NetLSD-G-IL
python run_TU_fingerprints.py --dataset=MUTAG --kernel_type=gaussian --initialization_type=GMM --fingerprint_type=NetLSD --if_learn=true
#NetLSD-S-IL
python run_TU_fingerprints.py --dataset=MUTAG --kernel_type=silverman --initialization_type=GMM --fingerprint_type=NetLSD --if_learn=true
