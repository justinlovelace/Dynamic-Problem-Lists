"""
    Reads NOTEEVENTS file, finds the discharge summaries, preprocesses them and writes out the filtered dataset.
"""
from os.path import join
import pandas as pd
import gensim, logging
import yaml

class MySentences(object):
    def __init__(self, df_notes):
        self.df_notes = df_notes

    def __iter__(self):
        for index, row in self.df_notes.iterrows():
            text = row['TEXT']
            yield text.split()

def train_embeddings(config, df_notes, folder):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = MySentences(df_notes)  #iterator
    print('Training model...')
    model = gensim.models.Word2Vec(sentences, min_count=5, workers=16, size=100)

    word_vectors = model.wv
    local_dir = config['local_data']
    datasetPath = join(local_dir, folder, 'vectors.kv')
    word_vectors.save(datasetPath)

    local_dir = config['local_data']
    datasetPath = join(local_dir, folder, 'w2v_model')
    model.save(datasetPath)


if __name__ == "__main__":
    config = yaml.safe_load(open("../resources/config.yml"))
    local_dir = config['local_data']
    print('Loading NOTES...')
    datasetPath = join(local_dir, 'df_MASTER_NOTES_ALL.csv')
    df_notes = pd.read_csv(datasetPath)
    for i in range(5):
        datasetPath = join(local_dir, 'fold'+str(i), 'df_test_subjects.csv')
        df_test = pd.read_csv(datasetPath)
        df_temp_notes = df_notes[~(df_notes.SUBJECT_ID.isin(df_test.SUBJECT_ID))]
        # train word2vec ebeddings
        train_embeddings(config, df_temp_notes, 'fold'+str(i))
