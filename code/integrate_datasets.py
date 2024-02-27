from datasets import *
import json
import numpy as np
import pandas as pd


def convert_answer(d) :
    return {"answer_start" : np.array([d["answer_start"]]).astype(int), 'text' : np.array([d["text"]]).astype(object)}


### KorQuad
def get_korquad_arrow(in_path, out_path) -> None :
    
    # load
    with open(in_path, 'r') as f:
        data_dict = json.load(f)
    korquad_datalist = data_dict["data"]

    # dictify
    korquad_data = []
    for document in korquad_datalist:
        for paragraph in document["paragraphs"]:
            for qa in paragraph["qas"]:
                korquad_data.append({
                    "title": document["title"],
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "id": qa["id"],
                    "answers": convert_answer(qa["answers"][0])
                })

    # dataframe
    korquad_df = pd.DataFrame(korquad_data)
    korquad_df["document_id"] = 0
    korquad_df["__index_level_0__"] = 0

    # arrow
    korquad_arrow = DatasetDict({'train' : Dataset.from_dict(korquad_df)})
    korquad_arrow.save_to_disk(out_path)
    print("Arrow saved in {}".format(out_path))


### ETRI_MRC
def get_etri_arrow(in_path, out_path) -> None :

    # load
    with open(in_path, 'r', encoding='UTF-8') as f:
        data_dict = json.load(f)
    etri_datalist = data_dict["data"]

    # dictify
    etri_data = []
    for document in etri_datalist:
        for paragraph in document["paragraphs"]:
            for qa in paragraph["qas"]:
                etri_data.append({
                    "title": document["title"],
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "id": qa["id"],
                    "answers": convert_answer(qa["answers"][0])
                })

    # dataframe
    etri_df = pd.DataFrame(etri_data)
    etri_df["document_id"] = 0
    etri_df["__index_level_0__"] = 0

    # arrow
    etri_arrow = DatasetDict({'train' : Dataset.from_dict(etri_df)})
    etri_arrow.save_to_disk(out_path)
    print("Arrow saved in {}".format(out_path))


def merge_dataset(arrow1_path, arrow2_path, out_path) -> None :
    
    arrow1 = Dataset.from_file(arrow1_path)
    arrow1_df = arrow1.to_pandas()

    arrow2 = Dataset.from_file(arrow2_path)
    arrow2_df = arrow2.to_pandas()

    new_df = pd.concat([arrow1_df, arrow2_df])
    new_arrow = DatasetDict({'train' : Dataset.from_dict(new_df)})
    new_arrow.save_to_disk(out_path)
    print("Arrow saved in {}".format(out_path))


if __name__ == "__main__" :

    IN_PATH = ""
    OUT_PATH = ""
    # get_etri_arrow(IN_PATH, OUT_PATH)
    # get_korquad_arrow(IN_PATH, OUT_PATH)
    print("Done!")