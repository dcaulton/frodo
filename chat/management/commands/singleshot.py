from django.core.management.base import BaseCommand

#from guided_redaction.jobs.tasks import sweep

################# MOVE TO A NEW UTILS FILE SOON
# Function to extract BERT embeddings for text as a list
emb_model = ''
def calc_embeddings(some_text):
    text_embeddings = emb_model.encode(some_text,normalize_embeddings=True)
    return text_embeddings.tolist()
# calc_embeddings('Sitel Group is changing from using the Duo App on your smart phone')

# Function to create embeddings for each item in a list (row of a df column)
def embedding_list(df_column):
    column_embeddings_list = list(map(calc_embeddings, df_column))
    return column_embeddings_list

####
# Calculate CosSim between question embeddings and article embeddings
def cos_sim_list(embedding_question,embedding_list):
    from sentence_transformers import util
    list_cos_sim = []
    for i in embedding_list:
        sim_pair = util.cos_sim(embedding_question,i).numpy()
        list_cos_sim.append(sim_pair[0][0])
        
    return list_cos_sim

#Calculate outliers within cos_sim_max data set, identified as possible answers
def find_outliers_IQR(cos_sim_max):
   q1=cos_sim_max.quantile(0.25)
   q3=cos_sim_max.quantile(0.75)
   IQR=q3-q1
   outliers = cos_sim_max[((cos_sim_max>(q3+1.5*IQR)))]

   return outliers

#######
def K_BOT(input_question, embeddings_title, embeddings_Content, df_knowledge):
    import pandas as pd
    import numpy as np
    pd.set_option('display.max_colwidth', 5000)

    #question embeddings
    embeddings_q = calc_embeddings(input_question)

    #calculate cosSim for included fields
    cos_sim_max = list(map(max, cos_sim_list(embeddings_q,embeddings_title),
                                cos_sim_list(embeddings_q,embeddings_Content)))
    df_knowledge['cos_sim_max'] = cos_sim_max

    #calculate log cosSim
    cos_sim_log = np.log2(df_knowledge['cos_sim_max']+1)
    df_knowledge['cos_sim_log'] = cos_sim_log

    #Identify outliers
    df_outliers = find_outliers_IQR(df_knowledge['cos_sim_log']).to_frame().reset_index(level=0, inplace=False)
    
    #Create df of potential answers
    df_answers = df_knowledge[['index','title','Content','cos_sim_max','cos_sim_log',]].sort_values(by=['cos_sim_max'], 
                                                                        ascending = False).head(len(df_outliers['index']))
    
    #search_results = []
    return df_answers

#############################################################

class Command(BaseCommand):
    help = """Sweeps away old jobs to make sure there are not too many.
        Removes the oldest jobs, except those jobs that have an 
        auto_delete_age of "never", until only "keep" number of jobs remain.
        Also attempts to clear all detritus left behind by the job.
    """

    def add_arguments(self, parser):
        parser.add_argument("-k", "--keep", type=int, default=None,
            help="Number of jobs to keep."
        )

    def handle(self, *args, **options):
#        keep = options["keep"]
        import numpy as np
        import pandas as pd
        import os
        import re
        import html
        import json
        import seaborn as sns
        import openai

        import inspect

        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics.pairwise import euclidean_distances
        from sentence_transformers import SentenceTransformer

        #Load knowledgebase data
        df_knowledge = pd.read_csv("data/triboo/Triboo_knowledgebase.csv")
        df_knowledge = df_knowledge.fillna('none')
        df_knowledge.dropna(inplace=True)
        df_knowledge.reset_index(level=0, inplace=True)
#        print(max(df_knowledge['index']))

        # Load embedding model
        global emb_model
        emb_model=SentenceTransformer(
            "all-mpnet-base-v2"
        )

        #Create embeddings for each column we want to compare our text with
        embeddings_title   = embedding_list(df_knowledge['title'])
        embeddings_Content = embedding_list(df_knowledge['Content'])

        the_resp = K_BOT('how about PowerBi access for external clients?', embeddings_title, embeddings_Content, df_knowledge)
        print(the_resp)
#embedding_list(['Sitel Group is changing from using the Duo App on your smart phone','How to enable sound and video on WSP Workspaces'])



        print('THERE, I just did the thing you want')
