from annoy import AnnoyIndex
from scipy import spatial
from nltk import ngrams
import random, json, glob, os, codecs, random, sys
import numpy as np

# class cluster_vector:
def cluster(prefix, database_files, input_file):
    ifile = input_file
    infiles = database_files
    infiles = [prefix+s for s in infiles]
    # data structures
    file_index_to_file_name = {}
    file_index_to_file_vector = {}

    # config
    dims = 16
    n_nearest_neighbors = 3
    trees = 10000

    # build ann index
    t = AnnoyIndex(dims)
    for file_index, i in enumerate(infiles):
      file_vector = np.loadtxt(i)
      file_name = os.path.basename(i).split('.')[0]
      file_index_to_file_name[file_index] = file_name
      file_index_to_file_vector[file_index] = file_vector
      t.add_item(file_index, file_vector)
    t.build(trees)

    # create a nearest neighbors json file for each input
    if not os.path.exists('nearest_neighbors'):
      os.makedirs('nearest_neighbors')

    for i in file_index_to_file_name.keys():
      if file_index_to_file_name[i] == ifile:
        master_file_name = file_index_to_file_name[i]
        master_vector = file_index_to_file_vector[i]

        named_nearest_neighbors = []
        out = []
        nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)
        for j in nearest_neighbors:
          neighbor_file_name = file_index_to_file_name[j]
          neighbor_file_vector = file_index_to_file_vector[j]

          similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
          rounded_similarity = int((similarity * 10000)) / 10000.0
          itId = (''.join(filter(str.islower, neighbor_file_name)))
          out.append(itId)
          named_nearest_neighbors.append({
            'filename': neighbor_file_name,
            'similarity': rounded_similarity
          })
        return out
