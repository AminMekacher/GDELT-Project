import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx

import pycountry
import seaborn as sns

import collections
import jellyfish

import bs4 as bs  
import urllib.request  
import re  
import nltk

import os
import simplejson as json

def social_graph_creation(G, dataframe):
    import numpy as np
    actor_tot_list = []
    actor_buffer_list = [] # List used to check if an article is a perfect replica of the previous one
    for actor_list, theme_list in zip(dataframe.V2ENHANCEDPERSONS.unique(), dataframe.V2ENHANCEDTHEMES.unique()):
        actor_temp_list, offset_temp_list = [], []
        
        if not isinstance(actor_list, float):
            max_offset_diff = maximum_offset_difference(actor_list, theme_list)
            for actor in actor_list.split(';'):
                [actor_temp, offset_temp] = actor.split(',')
            
                if offset_temp not in offset_temp_list:
                    offset_temp_list.append(offset_temp)
                    
                    # Compute similarity between actor_temp and all actors in the tot_list
                    if actor_tot_list:
                        similarity_max = np.max([jellyfish.jaro_winkler(actor_temp, actor2) for 
                                                 actor2 in actor_tot_list])
                        index_max = np.argmax([jellyfish.jaro_winkler(actor_temp, actor2) for 
                                                  actor2 in actor_tot_list])
                        actor_max = actor_tot_list[index_max]
                        
                        nb_identical_names = len(set(actor_temp.split(' ')) & set(actor_max.split(' ')))
                    
                    else:
                        similarity_max = 0
                        nb_identical_names = 0
                        
                    # Condition to correct the name if there is a misdetected 'A'
                    if actor_temp[0:2] == 'A ':
                        actor_temp = actor_temp[2:]   
                    
                    if 'Kanzler Joseph' in actor_temp:
                        actor_max = 'Youssef Chahed'
                        similarity_max, nb_identical_names = 1, 1 
                    
                    if similarity_max > 0.7 and nb_identical_names > 0: # This actor is already present in the list
                        actor_temp = actor_max
                    else:
                        actor_tot_list.append(actor_temp)
                        G.add_node(actor_temp)
                        
                    if actor_temp not in actor_temp_list:
                        actor_temp_list.append(actor_temp)
                    
        if actor_temp_list != actor_buffer_list:
            actor_buffer_list = actor_temp_list
            
            nb_actors = len(actor_temp_list)
            #print("Actor list: ", nb_actors, actor_temp_list)
            
            # Edge creation between the actors of the article
            for index1 in range(0, len(actor_temp_list)):
                actor1 = actor_temp_list[index1]
                offset1 = int(offset_temp_list[index1])
                for index2 in range(index1 + 1, len(actor_temp_list)):
                    actor2 = actor_temp_list[index2]
                    offset2 = int(offset_temp_list[index2])
                    weight_edge = np.abs(offset2 - offset1) / (max_offset_diff * nb_actors)
                    #print("Weight: ", weight_edge)

                    if G.has_edge(actor1, actor2):
                        G[actor1][actor2]['weight'] += weight_edge
                    else:
                        G.add_edge(actor1, actor2, weight = weight_edge)
                        
def network_edge_filtering(graph, threshold):
    '''
    Method used to remove the edges from a network whose weight is below a specific threshold
    Returns a graph with the remaining edges and all isolated nodes removed
    '''
    
    edges_to_remove, nodes_to_remove = [], []
    
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        if weight < threshold:
            edges_to_remove.append((u, v))
        
    print("Removed edge: ", len(edges_to_remove))
    graph.remove_edges_from(edges_to_remove)
        #print("Edge: ", u, v, a['weight'])
    for (node, degree) in graph.degree:
        #print("Node: ", node, degree)
        if degree == 0:
            nodes_to_remove.append(node)
            
    print("Removed nodes: ", len(nodes_to_remove))
    graph.remove_nodes_from(nodes_to_remove)
        
    return graph
        
def maximum_offset_difference(actor_list, theme_list):
    
    offset_max = 0
    offset_min = np.inf
    
    # 1: Actor loop
    for actor_temp in actor_list.split(';'):
        offset_temp = float(actor_temp.split(',')[1])
        if offset_temp < offset_min:
            offset_min = offset_temp
        if offset_temp > offset_max:
            offset_max = offset_temp
            
    # 2: Theme loop
    if not isinstance(theme_list, float):
        for theme_temp in theme_list.split(';'):
            if theme_temp:
                offset_temp = float(theme_temp.split(',')[1])
                if offset_temp < offset_min:
                    offset_min = offset_temp
                if offset_temp > offset_max:
                    offset_max = offset_temp
            
    # Computation of the maximum difference
    max_offset_diff = offset_max - offset_min
    return max_offset_diff
            

def theme_network_creation(G_themes, list_actor, dataframe, themes_of_interest, tf_idf):
    
    '''
    Creation of a graph between the actors and the themes. For each theme mentioned in the articles, 
    we draw an edge between this theme and the closest actor in terms of offset. This will give us a
    bipartite graph, with the actors on one side and the themes on the other side. The goal is to see if 
    some actors are strongly linked to very specific themes, as detected by GDELT
    '''
    uncommon_theme = ['GOV_DIVISIONOFPOWER', 'HATE_SPEECH', 'INFO_HOAX', 'POLITICAL_PRISONER', 'MEDIA_CENSORSHIP']
    for actor_list, theme_list, doc_id in zip(dataframe.V2ENHANCEDPERSONS.unique(), 
                                              dataframe.V2ENHANCEDTHEMES.unique(), dataframe.GKGRECORDID):
        
        actor_list_temp, offset_list_temp = [], []
        #print("begin: ", actor_list, theme_list, doc_id)
    
        if not isinstance(actor_list, float):
            for actor in actor_list.split(';'):
                actor_list_temp.append(actor.split(',')[0])
                offset_list_temp.append(int(actor.split(',')[1]))
    
        # First, we need to get the themes and their respective offset in two separate lists
        
        if not isinstance(theme_list, float) and not isinstance(actor_list, float):
            
            #print("Here: ", doc_id)
            number_theme = len(theme_list)
            max_offset_diff = maximum_offset_difference(actor_list, theme_list)
            
            for theme in theme_list.split(';'):
                if theme:
                    theme_temp = theme.split(',')[0]
                    offset_temp = int(theme.split(',')[1])
                    
                    if theme_temp in themes_of_interest:
                    
                        if not G_themes.has_node(theme_temp):
                            G_themes.add_node(theme_temp)

                        
                        index_actor = np.argmin(np.abs([offset - offset_temp for offset in offset_list_temp]))
                        actor_offset = actor_list_temp[index_actor]

                        # We need to find this actor in the nodes of the network

                        similarity_max = np.max([jellyfish.jaro_winkler(actor_offset, actor2) for 
                                                     actor2 in list_actor])
                        index_max = np.argmax([jellyfish.jaro_winkler(actor_offset, actor2) for 
                                                      actor2 in list_actor])
                        actor_max = list_actor[index_max]
                        
                        '''
                        for (actor, offset_actor) in zip(actor_list_temp, offset_list_temp):
                            offset_diff = np.abs(offset_actor - offset_temp)
                            
                            similarity_max = [jellyfish.jaro_winkler(actor, actor2) for 
                                                     actor2 in list_actor]
                            index_max = np.argmax(similarity_max)
                            actor_max = list_actor[index_max]
                        
                        # The weight associated with this theme and article is extracted from the tf-idf dictionary
                            weight_theme = tf_idf[doc_id][theme_temp] * (1 - offset_diff / max_offset_diff)

                        # Now that we have the theme and the actor, we can draw an edge between the two

                            if G_themes.has_edge(actor_max, theme_temp):
                                G_themes[actor_max][theme_temp]['weight'] += weight_theme
                            else:
                                #print("New edge! ", actor_max, theme_temp)
                                G_themes.add_edge(actor_max, theme_temp, weight = weight_theme)
                        '''
                        #print("Theme: ", doc_id, theme_temp)
                        weight_theme = tf_idf[doc_id][theme_temp]
                        
                        if G_themes.has_edge(actor_max, theme_temp):
                            G_themes[actor_max][theme_temp]['weight'] += weight_theme
                        else:
                            G_themes.add_edge(actor_max, theme_temp, weight=weight_theme)
                                      
        
    return G_themes


def gcam_extraction(dataframe, threshold):
    
    '''
    Method used to read the different dictionary dimensions present in each article and to increment their count if their
    density score (i.e number of reference words in the article divided by the word count) is greater than a pre-selected
    threshold
    '''
    
    gcam_dict = {}
    print(np.max([2, 3]))
    word_count = 0 # will be defined when reading the first entry of the GCAM feature
    
    for gcam_list in dataframe.V2GCAM.unique():
        for gcam in gcam_list.split(','):
            [dim_name, dim_count] = gcam.split(':')
            
            if dim_name == 'wc':
                word_count = int(dim_count)
            elif dim_name[0] == 'c': # we need to compute the density score for this dimension and compare it to the threshold
                
                density_score = int(dim_count) / word_count
                if density_score > threshold:
                    
                    # Check if the dimension has already been added to the dictionary
                    if dim_name in gcam_dict:
                        gcam_dict[dim_name] += 1
                    else:
                        gcam_dict[dim_name] = 1
                        
    return gcam_dict

def topk_dict_extraction(sorted_dict, k):

    topk_dict = {}

    for key, value in sorted_dict.items():
        if len(topk_dict) < k:
            topk_dict[key] = value
    
    return topk_dict

def topk_actor_extraction(actor_dict, k):
    
    topk_actor = {}
    
    for actor, pagerank in actor_dict.items():
        if len(topk_actor) < k:
            topk_actor[actor] = pagerank
            
    return topk_actor

def media_coverage_visualization(dataframe):
    
    '''
    Goal: display a donut pie for each analyzed day, displaying the number of articles published by each media outlet 
    covering the news in Tunisia
    '''
    
    dataframe.rename(columns={'V2.1DATE':'V2DATE'}, inplace=True)
    DAY_BEGINNING = 24
    
    dict_ocurrences = [{}, {}, {}, {}, {}] # Five dictionaries for now, because five days of news are present in the dataframe
    
    for date, media in zip(dataframe.V2DATE, dataframe.V2SOURCECOMMONNAME):
        day = int(str(date)[6:8])
        month = int(str(date)[4:6])
        
        index_dict = day - DAY_BEGINNING
        
        if media in dict_ocurrences[index_dict]:
            dict_ocurrences[index_dict][media] += 1
        else:
            dict_ocurrences[index_dict][media] = 1
            
    return dict_ocurrences

def download_gdelt_repo(filename, date_beginning, date_ending):
    
    start_reading = False
            
    with open(filename) as fileobject:
        for line in fileobject:
            
            if line != '\n':
                line = line.strip()
                time_line = line.split('v2/')[1][0:14]
                
                if time_line:
                    time_line = int(line.split('v2/')[1][0:14])
                
                    if time_line == date_beginning:
                        print("Start")
                        start_reading = True
                    elif time_line == date_ending:
                        print("End")
                        start_reading = False

                    # While we are in the time interval, look for the GKG files
                    if start_reading == True:
                        #print("Download: ", line)
                        gkg_string = line.split('translation.')[1][0:3]

                        if gkg_string == 'gkg':

                            url = line.split(' ')[2]
                            folder_dest = '/Users/aminmekacher/Documents/EPFL Master/MA2/GDELT Project/GDELT-Project/GKG Notebooks Tunisia'
                            file_dest = os.path.join(folder_dest, line.split('v2/')[1])
                            urllib.request.urlretrieve(url, file_dest) 


def theme_list_extraction(dataframe):
    '''
    Method used to create a list with all the themes identified by GDELT in the dataframe
    '''
    
    theme_list = []
    
    for theme_article in dataframe.V2ENHANCEDTHEMES.unique():
        if not isinstance(theme_article, float):
            for theme in theme_article.split(';'):
                theme_temp = theme.split(',')[0]
                if theme_temp not in theme_list and theme_temp != '':
                    theme_list.append(theme_temp)
    
    return theme_list
            

def tf_idf_computation(dataframe, themes_of_interest):
    '''
    Method used to compute, for each theme of interest, its tf-idf score with the articles present in the dataframe
    '''
    
    tf_score, idf_score, tf_idf = {}, {}, {}
    number_docs = len(dataframe.GKGRECORDID)
    #print("Num: ", number_docs)
    
    for theme_interest in themes_of_interest:
        #print("New theme: ", theme_interest)
        for theme_list, doc_id in zip(dataframe.V2ENHANCEDTHEMES.unique(), dataframe.GKGRECORDID):
            #print("Doc theme: ", doc_id, theme_list)
            theme_found = False
            if not isinstance(theme_list, float):
                for theme in theme_list.split(';'):
                    if theme:
                        theme_temp = theme.split(',')[0]
                        #print("New themeprev: ", theme_temp)
                        # Checking if the theme is the one we are currently looking for
                        if theme_temp == theme_interest:
                            #print("New theme: ", theme_temp)
                            
                            # We increment the tf_score of the theme in the current document
                            if doc_id not in tf_score:
                                #print("Adding doc: ", doc_id)
                                tf_score[doc_id] = {}
                            
                            if theme_interest not in tf_score[doc_id]:
                                tf_score[doc_id][theme_interest] = 0
                                
                            tf_score[doc_id][theme_interest] += 1
                            
                            #print("TF: ", tf_score)
                            
                            # We increment the idf score of the theme in the current document, if it has not been found yet
                            if theme_found == False:
                                    
                                if theme_interest not in idf_score:
                                    idf_score[theme_interest] = 0
                                
                                idf_score[theme_interest] += 1
                                #print("NEW: ", idf_score[theme_interest])
                                
                                theme_found = True
                                #print("IDF: ", idf_score)
                            
            #print("Finished doc: ", tf_score[doc_id], doc_id)
                                
    # Last step: compute the tf-idf score for each theme for each document
    for doc_id in dataframe.GKGRECORDID:
        #print("Doc out: ", doc_id)
        if doc_id in tf_score:
            #print("Doc in: ", doc_id)
            tf_temp = tf_score[doc_id]
            max_tf = np.max(list(tf_temp.values()))
            
            for theme_interest in tf_temp:
                tf_theme = tf_temp[theme_interest] / max_tf
                idf_theme = np.log10(number_docs / idf_score[theme_interest])
                if doc_id not in tf_idf:
                    tf_idf[doc_id] = {}
                if theme_interest not in tf_idf[doc_id]:
                    tf_idf[doc_id][theme_interest] = 0
                tf_idf[doc_id][theme_interest] = tf_theme * idf_theme
                #print("New value: ", tf_idf)
                                
    return (tf_score, idf_score, tf_idf)
                            
            
            
def word2vec_similarity(dataframe):
    '''
    Method used to define the word2vec algorithm for the dataset, in order to compute the similarity between words
    '''
    
    dataframe.rename(columns={'V2.1TRANSLATIONINFO':'TRANSLATIONINFO'}, inplace=True)
    tokenized_article_list = [] # List used to append each article after they have been translated and tokenized
    failures = 0
    
    for article_url, translation_info in zip(dataframe.V2DOCUMENTIDENTIFIER, dataframe.TRANSLATIONINFO):
        #print("Translation: ", translation_info, article_url)
        translator = Translator()
        try:
            #print("Works!!")
            scrapped_data = urllib.request.urlopen(article_url)

            article = scrapped_data.read()
            parsed_article = bs.BeautifulSoup(article,'lxml')

            paragraphs = parsed_article.find_all('p')
            article_text = ""

            for p in paragraphs:
                #print("Starting trans: ", p.text)
                text_translated = translator.translate(text=p.text, dest='en')
                #print("End trans")
                article_text += text_translated.text
            #print("Article: ", article_text)

            processed_article = article_text.lower()  
            processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)  
            processed_article = re.sub(r'\s+', ' ', processed_article)
            processed_article = re.sub(r'http\S+', '', processed_article)

            #print("Processed: ", processed_article)

            # Preparing the dataset
            all_sentences = nltk.sent_tokenize(processed_article)

            all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

            # Removing Stop Words
            from nltk.corpus import stopwords  
            for i in range(len(all_words)):  
                all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

            tokenized_article_list.append(all_words)
            
            print("New article tokenized: ", len(tokenized_article_list))

            #print("Works! ", all_words)
        except BaseException as e:
            print('Failed to do something: ' + str(e))
            failures += 1
            print("Doesn't work ", failures)
            pass
        
def check_word_presence(article_url, word):
    
    try:
        scrapped_data = urllib.request.urlopen(article_url)
    
        article = scrapped_data.read()
        parsed_article = bs.BeautifulSoup(article,'lxml')

        paragraphs = parsed_article.find_all('p')
        article_text = ""

        for p in paragraphs:
            article_text += p.text
        #print("Article: ", article_text)
    
        return word in article_text
    
    except:
        
        pass
        return False
    
    
def appendDFToCSV(df, csvFilePath, sep="\t"):
    import os
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)
        
def save_to_json(G, fname):
    json.dump(dict(nodes=[[n, G.node[n]] for n in G.nodes()],
                   edges=[[u, v, G.get_edge_data(u, v)] for u,v in G.edges()]),
              open(fname, 'w'), indent=2)
    

def load_json(fname):
    G = nx.DiGraph()
    d = json.load(open(fname))
    G.add_nodes_from(d['nodes'])
    G.add_edges_from(d['edges'])
    return G