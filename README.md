# Graphite Tools

This repository contains files for interacting with Bittensor and Subnet 43.

### Syncing data from huggingface

Our subnet now pushes all the past problems to a huggingface dataset repository as .tsv files which contain all the problems for each given day. Miners can download it by running the `sync_past_data.py` file.

As a default behavior, the data is saved in a folder under the local 'subnet_tools' repository called 'past_data'. Users can change this behavior by adding `LOCAL_DIR` under the .env file. Refer to `.example.env` for reference.

Note that the `selected_uids` field for wandb was rolled out on the 18th of September 2024, so only the .tsv files for after 18th September contain that field.

Users can load in tsv files using pandas via `pd.read_csv(file_path, sep='\t')`.
