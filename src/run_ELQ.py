import elq.main_dense as main_dense
import argparse

models_path = "models/" # the path where you stored the ELQ models

config = {
    "interactive": False,
    "biencoder_model": models_path+"elq_wiki_large.bin",
    "biencoder_config": models_path+"elq_large_params.txt",
    "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "output_path": "logs/", # logging directory
    "faiss_index": "hnsw",
    "index_path": models_path+"faiss_hnsw_index.pkl",
    "num_cand_mentions": 10,
    "num_cand_entities": 10,
    "threshold_type": "joint",
    "threshold": -4.5,
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)

data_to_link = [{
                    "id": 0,
                    "text": "paris is capital of which country?".lower(),
                },
                {
                    "id": 1,
                    "text": "paris is great granddaughter of whom?".lower(),
                },
                {
                    "id": 2,
                    "text": "who discovered o in the periodic table?".lower(),
                },
                ]

predictions = main_dense.run(args, None, *models, test_data=data_to_link)