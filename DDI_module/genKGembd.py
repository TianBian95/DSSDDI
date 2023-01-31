

import numpy as np

import os
import csv

def getEmbd(id_list):
    entity2id = {}
    id2entity = {}
    with open("../data/entities.tsv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['entity', 'id'])
        for row_val in reader:
            id = row_val['id']
            entity = row_val['entity']

            entity2id[entity] = int(id)
            id2entity[int(id)] = entity

    print("Number of entities: {}".format(len(entity2id)))

    entity_emb = np.load('../data/DRKG_TransE_l2_entity.npy')
    dataset_id = {}
    for entity_name, i in entity2id.items():
        entity_key = entity_name.split('::')[0]
        compdID = entity_name.split('::')[1]
        if entity_key=='Compound' and dataset_id.get(compdID, None) is None:
            dataset_id[compdID] = [i]
    emb_list=[]
    for cmpdID in id_list:
        try:
            index=dataset_id[cmpdID]
        except KeyError:
            cmpdID='MESH:'+cmpdID
            index=dataset_id[cmpdID]
        val = np.asarray(index, dtype=np.long)
        emb = entity_emb[val]
        emb_list.extend(emb)

    return emb_list





if __name__ == '__main__':
    id_list=['C011462','CHEMBL1200879','CHEMBL1200772']
    emb_list=getEmbd(id_list)
    print(1)