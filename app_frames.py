#####################
#####################

# python -m streamlit run C:\Users\User\Downloads\NP_frames\app_frames.py


# data
cch_twi = r"NP-frames-up-ethos.xlsx"


colors = {
'negative':'#F90400', 'positive':'#107C18', 'neutral':'#076BE2', 'NA':'grey',
'attack':'#F90400', 'support':'#107C18', 'neutral agents':'blue',
'joy' : '#8DF903', 'anger' : '#FD7E00', 'sadness' : '#0798C3', 'fear' : '#000000', 'disgust' :'#840079', 'surprise' : '#E1CA01',
}

#####################
#####################
# imports
import streamlit as st
from PIL import Image
from collections import Counter
import pandas as pd
pd.set_option("max_colwidth", 400)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
#plt.style.use("seaborn-talk")

import time
import re

import spacy
nlp = spacy.load('en_core_web_sm')

pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


import plotly.express as px
import plotly
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

import wordcloud
from wordcloud import WordCloud, STOPWORDS

import nltk
from nltk.text import Text



#####################
#####################
# functions

def make_word_cloud(comment_words, width = 1100, height = 650, colour = "black", colormap = "brg", stops = True):
    stopwords = set(STOPWORDS)
    if stops:
            wordcloud = WordCloud(collocations=False, max_words=100, colormap=colormap, width = width, height = height,
                        background_color ='black',
                        min_font_size = 16, ).generate(comment_words) # , stopwords = stopwords
    else:
            wordcloud = WordCloud(collocations=False, max_words=100, colormap=colormap, width = width, height = height,
                        background_color ='black',
                        min_font_size = 16, stopwords = stopwords ).generate(comment_words) # , stopwords = stopwords

    fig, ax = plt.subplots(figsize = (width/ 100, height/100), facecolor = colour)
    ax.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()
    return fig, wordcloud.words_.keys()


import io
buffer = io.BytesIO()
@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False, sep = '\t').encode('utf-8')


def prepare_cloud_lexeme_data(data_neutral, data_support, data_attack):

  # neutral df
  neu_text = " ".join(data_neutral['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_neu_text = Counter(neu_text.split(" "))
  df_neu_text = pd.DataFrame( {"word": list(count_dict_df_neu_text.keys()),
                              'neutral #': list(count_dict_df_neu_text.values())} )
  df_neu_text.sort_values(by = 'neutral #', inplace=True, ascending=False)
  df_neu_text.reset_index(inplace=True, drop=True)
  #df_neu_text = df_neu_text[~(df_neu_text.word.isin(stops))]

  # support df
  supp_text = " ".join(data_support['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_supp_text = Counter(supp_text.split(" "))
  df_supp_text = pd.DataFrame( {"word": list(count_dict_df_supp_text.keys()),
                              'support #': list(count_dict_df_supp_text.values())} )

  df_supp_text.sort_values(by = 'support #', inplace=True, ascending=False)
  df_supp_text.reset_index(inplace=True, drop=True)
  #df_supp_text = df_supp_text[~(df_supp_text.word.isin(stops))]

  merg = pd.merge(df_supp_text, df_neu_text, on = 'word', how = 'outer')

  #attack df
  att_text = " ".join(data_attack['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_att_text = Counter(att_text.split(" "))
  df_att_text = pd.DataFrame( {"word": list(count_dict_df_att_text.keys()),
                              'attack #': list(count_dict_df_att_text.values())} )

  df_att_text.sort_values(by = 'attack #', inplace=True, ascending=False)
  df_att_text.reset_index(inplace=True, drop=True)
  #df_att_text = df_att_text[~(df_att_text.word.isin(stops))]

  df2 = pd.merge(merg, df_att_text, on = 'word', how = 'outer')
  df2.fillna(0, inplace=True)
  df2['general #'] = df2['support #'] + df2['attack #'] + df2['neutral #']
  df2['word'] = df2['word'].str.replace("'", "_").replace("”", "_").replace("’", "_")
  return df2


import random
def wordcloud_lexeme(dataframe, lexeme_threshold = 90, analysis_for = 'support', cmap_wordcloud = 'Greens', stops = False):
  '''
  analysis_for:
  'support',
  'attack',
  'both' (both support and attack)

  cmap_wordcloud: best to choose from:
  gist_heat, flare_r, crest, viridis

  '''
  if analysis_for == 'attack':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'Reds' #gist_heat
    dataframe['precis'] = (round(dataframe['attack #'] / dataframe['general #'], 3) * 100).apply(float) # att
  elif analysis_for == 'both':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'autumn' #viridis
    dataframe['precis'] = (round((dataframe['support #'] + dataframe['attack #']) / dataframe['general #'], 3) * 100).apply(float) # both supp & att
  else:
    #print(f'Analysis for: {analysis_for} ')
    dataframe['precis'] = (round(dataframe['support #'] / dataframe['general #'], 3) * 100).apply(float) # supp

  dfcloud = dataframe[(dataframe['precis'] >= int(lexeme_threshold)) & (dataframe['general #'] > 3) & (dataframe.word.map(len)>3)]
  #print(f'There are {len(dfcloud)} words for the analysis of language {analysis_for} with precis threshold equal to {lexeme_threshold}.')
  n_words = dfcloud['word'].nunique()
  text = []
  for i in dfcloud.index:
    w = dfcloud.loc[i, 'word']
    w = str(w).strip()
    if analysis_for == 'both':
      n = int(dfcloud.loc[i, 'support #'] + dfcloud.loc[i, 'attack #'])
    else:
      n = int(dfcloud.loc[i, str(analysis_for)+' #']) #  + dfcloud.loc[i, 'attack #']   dfcloud.loc[i, 'support #']+  general
    l = np.repeat(w, n)
    text.extend(l)

  import random
  random.shuffle(text)
  #st.write(f"There are {n_words} words.")
  if n_words < 1:
      st.error('No words with a specified threshold. \n Try lower value of threshold.')
      st.stop()
  figure_cloud, figure_cloud_words = make_word_cloud(" ".join(text), 1000, 620, '#1E1E1E', str(cmap_wordcloud), stops = stops) #gist_heat / flare_r crest viridis
  return figure_cloud, dfcloud, figure_cloud_words






def MainPage():
    add_spacelines(2)
    with st.expander("Read abstract"):
        add_spacelines(2)
        st.write("""XXX.""")

    with st.container():
        df_sum = pd.DataFrame(
                {
                        "Corpus": ['Covid', 'ElectionsSM'],
                        "# Words": [30014, 30099],
                        "# ADU": [2706, 3827],
                        "# Posts": [963, 1317],
                        "# Speakers": [465, 1317],
                }
        )

        df_iaa = pd.DataFrame(
                {
                        'Covid': [ 440, 59, 630, 1233, 653, 152, 0.752, 0.618, 0.417 ],
                        'ElectionsSM' : [ 847, 492, 581, 1144, 1294, 190, 0.793, 0.817, 0.573 ],

                }, index = ["# Ethos attack", "# Ethos support",  "# Logos attack",  "# Logos support",  "# Pathos negative", "# Pathos positive", 'IAA Logos', 'IAA Ethos', 'IAA Pathos' ]
        )
        #st.write("**Data summary**")
        #st.dataframe(df_sum.set_index("Corpus"))
        #st.dataframe(df_iaa)
        #add_spacelines(2)
        #st.write("**[The New Ethos Lab](https://newethos.org/)**")
        #add_spacelines(1)
        st.write(" ************************************************************* ")




def add_spacelines(number_sp=2):
    for xx in range(number_sp):
        st.write("\n")


@st.cache_data#(allow_output_mutation=True)
def load_data(file_path, indx = True, indx_col = 0, sheet = False, sheet_name = 'components'):
  '''Parameters:
  file_path: path to your excel or csv file with data,

  indx: boolean - whether there is index column in your file (usually it is the first column) --> default is True

  indx_col: int - if your file has index column, specify column number here --> default is 0 (first column)
  '''
  if indx == True and file_path.endswith(".xlsx"):
      if sheet:
          data = pd.read_excel(file_path, index_col = indx_col, sheet_name = sheet_name)
      else:
          data = pd.read_excel(file_path, index_col = indx_col)

  elif indx == False and file_path.endswith(".xlsx"):
      if sheet:
          data = pd.read_excel(file_path, sheet_name = sheet_name)
      else:
          data = pd.read_excel(file_path)

  elif indx == True and file_path.endswith(".csv"):
    data = pd.read_csv(file_path, index_col = indx_col)
  elif indx == False and file_path.endswith(".csv"):
    data = pd.read_csv(file_path)
  return data




@st.cache_data
def lemmatization(dataframe, text_column = 'sentence', name_column = False):
  '''Parameters:
  dataframe: dataframe with your data,

  text_column: name of a column in your dataframe where text is located
  '''
  df = dataframe.copy()
  lemmas = []
  for doc in nlp.pipe(df[text_column].astype('str')):
    lemmas.append(" ".join([token.lemma_ for token in doc if (not token.is_punct and not token.is_stop and not token.like_num and len(token) > 1) ]))

  if name_column:
      df[text_column] = lemmas
  else:
      df[text_column+"_lemmatized"] = lemmas
  return df




def distribution_plot_compare(data_list):
    c_contents1, c_contents2 = st.columns( 2 )
    with c_contents2:
        contents_radio_categories_val_units = st.radio("Choose the unit of statistic to display", ("percentage", "number" ) ) # horizontal=True, , label_visibility = 'collapsed'

    with c_contents1:
        contents_radio_categories = st.selectbox("Choose categories to display", ("frames components", "frames & ethos", "frames & sentiment", "frames & emotion" )) # horizontal=True,

    #st.write(df.columns)

    distributions_dict = {}
    df = data_list[-1]

    colsText = [c for c in df.columns if 'Text' in c]
    cols_frames_components = [ 'Component', 'CausationEffect', 'CausationPolarity', 'CausationType',  'InternalPolarity' ]
    cols2 = ['CausationEffect', 'CausationPolarity', 'CausationType', 'InternalPolarity', 'AgentNumerosity', 'CauseLength']


    if 'sentiment' in contents_radio_categories or 'ethos' in contents_radio_categories or 'emotion' in contents_radio_categories:
        contents_radio_categories = contents_radio_categories.split("&")[-1].strip()


        add_spacelines(1)
        contents_radio_categories_multiselect = st.multiselect(f"Select **frames components** for multi-level analysis of **{contents_radio_categories}** ", cols2, cols2[:3] )
        contents_radio_categories_multiselect_grouping = st.selectbox("Choose the grouping feature", contents_radio_categories_multiselect )

        #data_list[1]
        df[cols_frames_components+['AgentNumerosity', 'CauseLength']] = df[cols_frames_components+['AgentNumerosity', 'CauseLength']].fillna("NA")
        df[cols_frames_components+['AgentNumerosity', 'CauseLength']+colsText] = df[cols_frames_components+['AgentNumerosity', 'CauseLength']+colsText].astype('str')

        add_spacelines(1)
        df = df[df.Component == 'Causation']
        st.write()
        df_cause = df.groupby( contents_radio_categories_multiselect + [contents_radio_categories ], as_index=False ).size()
        notna = df[df.Component == 'Causation'].shape[0]
        df_cause['proportion'] = df_cause[ 'size' ] / notna
        df_cause['proportion'] = df_cause['proportion'].round(3) * 100
        #st.write(df_cause)
        df_cause2 = df.groupby( contents_radio_categories_multiselect )[ contents_radio_categories ].value_counts(normalize=True).round(3) * 100
        #st.write(df_cause2)

        for col in cols_frames_components[1:]+['AgentNumerosity', 'CauseLength']:

            if contents_radio_categories_val_units == "number":
                df_dist = df.groupby( [col, contents_radio_categories ], as_index=False ).size()
                df_dist['proportion'] = df_dist[ 'size' ].copy()
                #df_dist = df[col].value_counts(normalize=False)
                #df_dist = df_dist.reset_index()
                #df_dist = df_dist.rename(columns = {'count':'proportion'})
            else:
                df_dist = df.groupby( [col ], )[ contents_radio_categories ].value_counts(normalize=True).round(3) * 100
                df_dist = pd.DataFrame(df_dist)
                df_dist.columns = ['proportion']
                df_dist = df_dist.reset_index()
                #st.write(df_dist)
                #notna = df[ ~(df[col].isna()) ].shape[0]
                #df_dist['proportion'] = df_dist[ 'size' ] / notna
                #df_dist['proportion'] = df_dist['proportion'].round(3) * 100

            df_dist['Feature'] = col
            df_dist['Component'] = df_dist[col]
            distributions_dict[ col ] = df_dist[['Feature', 'Component', 'proportion', contents_radio_categories]]


        plot_tab, table_tab, case_tab = st.tabs( ['Plots', 'Tables', 'Cases'] )


        dist_all = pd.concat( distributions_dict.values(), axis=0, ignore_index=True )
        dist_all = dist_all.melt(['Feature', 'Component', contents_radio_categories ], value_vars =  'proportion')
        dist_all = dist_all.drop(columns = ['variable'])


        with plot_tab:

            sns.set(style = 'whitegrid', font_scale=2.5)
            if contents_radio_categories == 'emotion':
                h1 = 10
            else:
                h1 = 8.5
            if contents_radio_categories_val_units == "number":
                x1 = dist_all['value'].max() + 11
                x2 = 10
            else:
                x1 = 101
                x2 = 20


            plot1 = sns.catplot( data = dist_all, kind = 'bar', y = 'Component', x = 'value', col = 'Feature', col_wrap = 2,
                    aspect = 1.15, height=h1, sharey=False, sharex=False, hue = contents_radio_categories, palette = colors)
            plot1.set( xlim = (0, x1), xticks = np.arange(0, x1, x2),
                    ylabel='', xlabel=contents_radio_categories_val_units,
                    )
            plt.tight_layout(pad=1.5)
            sns.move_legend(plot1, loc='upper right', bbox_to_anchor = (0.8 - (len(contents_radio_categories_multiselect) / 30 ), 1.05), ncols = df[contents_radio_categories].nunique() )


            add_spacelines()
            st.write(" ##### Single-level analysis ")
            plt.show()
            st.pyplot( plot1 )
            #st.write(dist_all)
            add_spacelines(1)

            # fig 2
            sns.set(style = 'whitegrid', font_scale=2.1)
            df_cause2 = df_cause2.reset_index()
            #st.write(df_cause2)
            df_cause2 = df_cause2.rename(columns = {'proportion':'proportion feature'})
            colsc2 = list(df_cause2.columns)[:-1]
            #st.write(df_cause)
            df_cause = df_cause.merge(df_cause2, on = colsc2)
            #st.write(df_cause)
            if not contents_radio_categories_multiselect_grouping == 'AgentNumerosity' or contents_radio_categories_multiselect_grouping == 'CauseLength':
                df_cause = df_cause[df_cause[contents_radio_categories_multiselect_grouping] != 'NA']
            df_cause['grouping'] = contents_radio_categories_multiselect_grouping
            df_cause['Feature'] = df_cause[ list( set(contents_radio_categories_multiselect).difference([contents_radio_categories_multiselect_grouping]) ) ].apply( lambda x: ' '.join(x.values.astype('str')), axis=1)
            #st.write(df_cause)
            if contents_radio_categories_val_units == "number":
                x1 = df_cause['proportion'].max() + 3
                x2 = 10
            else:
                x1 = 101
                x2 = 20

            plot = sns.catplot( data = df_cause, kind = 'bar', y = 'Feature', x = 'proportion',
                    col = contents_radio_categories_multiselect_grouping, col_wrap = 2,
                    aspect = 1.15, sharex=False, height=h1, hue = contents_radio_categories, palette = colors)
            plot.set( xlim = (0, x1), xticks = np.arange(0, x1, x2),
                    ylabel='', xlabel=contents_radio_categories_val_units,
                    )
            plt.tight_layout(w_pad=7.5)
            for axis in plot.axes.flat:
                axis.tick_params(labelleft=True)
            sns.move_legend(plot, loc='upper right', bbox_to_anchor = (0.8 - (len(contents_radio_categories_multiselect) / 30 ), 1.1 - (len(contents_radio_categories_multiselect) / 200 ) ), ncols = df[contents_radio_categories].nunique() )
            add_spacelines(2)


            st.write( f" ##### Multi-level analysis of frames components & {contents_radio_categories} " )
            # fig 3
            if contents_radio_categories_val_units != "number":
                sns.set(style = 'whitegrid', font_scale=2.2)
                x1 = 101
                x2 = 20
                plot2 = sns.catplot( data = df_cause, kind = 'bar', y = 'Feature', x = 'proportion feature', col = contents_radio_categories_multiselect_grouping, col_wrap = 2,
                        aspect = 1.15, height=h1, sharex=False, hue = contents_radio_categories, palette = colors)
                plot2.set( xlim = (0, x1), xticks = np.arange(0, x1, x2),
                        ylabel='', xlabel=contents_radio_categories_val_units  )
                plt.tight_layout(w_pad=7.5)
                for axis in plot2.axes.flat:
                    axis.tick_params(labelleft=True)
                sns.move_legend(plot2, loc='upper right', bbox_to_anchor = (0.8- (len(contents_radio_categories_multiselect) / 30 ), 1.1 - (len(contents_radio_categories_multiselect) / 200 ) ), ncols = df[contents_radio_categories].nunique())

                st.write(" Figure: **proportion feature**")
                plt.show()
                st.pyplot( plot2)
                #st.write(df_cause)
                add_spacelines(2)


            st.write(" Figure: **proportion overall**")
            plt.show()
            st.pyplot( plot)
            df_cause = df_cause.rename(columns = {'proportion':'proportion overall'} )




        with table_tab:
            st.write(" **Single-level analysis** ")
            st.write(dist_all)
            add_spacelines(2)

            st.write( f" **Multi-level analysis of frames components & {contents_radio_categories}** " )
            df_cause = df_cause.rename( columns = { 'size':'number' } )
            if contents_radio_categories_val_units == "number":

                st.write(df_cause.drop(columns = ['proportion overall', 'proportion feature']).iloc[:, :-2] )
            else:
                st.write(df_cause.iloc[:, :-2])



        with case_tab:
            df = data_list[-1]
            df = df.fillna("NA")
            df[cols_frames_components+['AgentNumerosity', 'CauseLength']+colsText] = df[cols_frames_components+['AgentNumerosity', 'CauseLength']+colsText].astype('str')

            #st.write(df)
            dff_columns = [
                        'discussion', 'map', 'CausationText', 'CausationEffect',
                       'CausationPolarity', 'CausationType', 'Component', 'InternalPolarity',
                       'AgentNumerosity', 'CauseLength', 'speaker', 'turn',
                       'Causation begin', 'Causation end', 'CauseText', 'EffectText',
                       'AgentText', 'CircumstancesText',
                       'ethos',  'sentiment',
                       'emotion',
                         ]
            dff = df[dff_columns].copy()
            select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[-1])
            cols_columns = st.columns(len(select_columns))
            dict_cond = {}
            for n, c in enumerate(cols_columns):
                with c:
                    cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                           (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[1]))
                    dict_cond[select_columns[n]] = cond_col
            dff_selected = dff.copy()
            for i, k in enumerate(dict_cond.keys()):
                dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
            add_spacelines(2)
            st.dataframe(dff_selected.sort_values(by = select_columns).reset_index(drop=True).dropna(how='all', axis=1), width = None)
            st.write(f"No. of cases: {len(dff_selected)}.")


            csv_cc_desc2 = convert_to_csv(dff_selected.sort_values(by = select_columns).reset_index(drop=True).dropna(how='all', axis=1))
            # download button 1 to download dataframe as csv
            download1 = st.download_button(
                label="Download data as TSV",
                data=csv_cc_desc2,
                file_name='Cases.tsv',
                mime='text/csv'
            )

    else:
        #st.write( data_list[1] )
        #st.write( data_list[0] )
        df = data_list[0]

        plot_tab, table_tab, case_tab = st.tabs( ['Plots', 'Tables', 'Cases'] )

        for col in cols_frames_components:
            if contents_radio_categories_val_units == "number":
                df_dist = df[col].value_counts(normalize=False)
                df_dist = df_dist.reset_index()
                df_dist = df_dist.rename(columns = {'count':'proportion'})
                #st.dataframe(df_dist)
            else:
                df_dist = df[col].value_counts(normalize=True).round(3) * 100
                df_dist = df_dist.reset_index()

            df_dist['Feature'] = col
            df_dist['Component'] = df_dist[col]
            distributions_dict[ col ] = df_dist[['Feature', 'Component', 'proportion']]
            #st.write(col)
            #st.dataframe(df_dist)
            #sns.set(style = 'whitegrid', font_scale=1.4)
            ##plot = sns.catplot( data = df_dist, kind = 'bar', y = col, x = 'proportion',      aspect = 1.4, )
            #plot.set( xlim = (0, 101), xticks = np.arange(0, 101, 20) )
            #plt.show()
            #st.pyplot( plot)
        dist_all = pd.concat( distributions_dict.values(), axis=0, ignore_index=True )
        dist_all = dist_all.melt(['Feature', 'Component'], value_vars =  'proportion')
        dist_all = dist_all.drop(columns = ['variable'])
        #st.write(dist_all)
        with plot_tab:
            sns.set(style = 'whitegrid', font_scale=1.6)
            if contents_radio_categories_val_units == "number":
                x1 = dist_all['value'].max() + 11
                x2 = 10
            else:
                x1 = 101
                x2 = 20
            plot = sns.catplot( data = dist_all, kind = 'bar', y = 'Component', x = 'value', col = 'Feature', col_wrap = 2,
                    aspect = 1.5, sharey=False, sharex=False)
            plot.set( xlim = (0, x1), xticks = np.arange(0, x1, x2), ylabel='', xlabel=contents_radio_categories_val_units )
            plt.tight_layout(pad=1.1)
            plt.show()
            st.pyplot( plot)


        with table_tab:
            add_spacelines(1)
            st.dataframe(dist_all)

        with case_tab:
            colsText = [c for c in df.columns if 'Text' in c]
            cols_frames_components = [ 'Component', 'CausationEffect', 'CausationPolarity', 'CausationType',  'InternalPolarity' ]
            cols2 = ['CausationEffect', 'CausationPolarity', 'CausationType', 'InternalPolarity', 'AgentNumerosity', 'CauseLength']

            df = data_list[-1]
            dff_columns = [
                        'discussion', 'map', 'CausationText', 'CausationEffect',
                       'CausationPolarity', 'CausationType', 'Component', 'InternalPolarity',
                       'AgentNumerosity', 'CauseLength', 'speaker', 'turn',
                       'Causation begin', 'Causation end', 'CauseText', 'EffectText',
                       'AgentText', 'CircumstancesText',
                       'ethos',  'sentiment',
                       'emotion',
                         ]
            df = df.fillna("NA")
            df[cols_frames_components+['AgentNumerosity', 'CauseLength']+colsText] = df[cols_frames_components+['AgentNumerosity', 'CauseLength']+colsText].astype('str')

            dff = df[dff_columns].copy()
            dff = dff.fillna('NA')
            select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[1:2])
            cols_columns = st.columns(len(select_columns))
            dict_cond = {}
            for n, c in enumerate(cols_columns):
                with c:
                    cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                           (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[-1]))
                    dict_cond[select_columns[n]] = cond_col
            dff_selected = dff.copy()
            for i, k in enumerate(dict_cond.keys()):
                dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
            add_spacelines(2)
            st.dataframe(dff_selected.sort_values(by = select_columns).dropna(how='all', axis=1).reset_index(drop=True), width = None)
            st.write(f"No. of cases: {len(dff_selected)}.")

            csv_cc_desc2 = convert_to_csv(dff_selected.sort_values(by = select_columns).reset_index(drop=True).dropna(how='all', axis=1))
            # download button 1 to download dataframe as csv
            download1 = st.download_button(
                label="Download data as TSV",
                data=csv_cc_desc2,
                file_name='Cases.tsv',
                mime='text/csv'
            )



def Target_compare_freq(data_list):
    add_spacelines(1)
    contents_radio_categories_val_units = st.radio("Choose the unit of statistic to display", ("percentage", "number" ) ) # horizontal=True, , label_visibility = 'collapsed'
    contents_radio_unit = 'number'

    df = data_list[-1]
    ds = df['corpus'].iloc[0]
    #df = df[df['Component'] == 'Agent' ]
    df = df[ ~(df.AgentText.isna()) ]
    df['Target'] = df['AgentText']

    df1 = df['Target'].value_counts(normalize = True).round(3) * 100
    df1 = df1.reset_index()

    if contents_radio_categories_val_units == 'number':
        df1 = df['Target'].value_counts(normalize = False )
        df1 = df1.reset_index()
        df1 = df1.rename( columns = {'count': 'proportion'} )

    #st.write(df1)

    df1 = df1.rename(columns = {'Target':'Agent'})


    df2 = df.groupby( ['Target', 'ethos'], as_index = False ).size()
    #st.write(df2)

    dd_hero = df2[df2.ethos == 'support']
    dd_antihero = df2[df2.ethos == 'attack']
    #st.write(dd_antihero)
    dd_neu = df2[df2.ethos == 'neutral']

    dd2 = pd.DataFrame({ 'Target': df['Target'].unique() })

    dd2anti_scores = []
    dd2hero_scores = []
    up_data_dict_hist = {}

    dd2['score'] = np.nan
    dd2['number'] = np.nan
    dd2['appeals'] = np.nan

    #st.write( dd2.Target.unique())
    for i in dd2.index:

        t = dd2.loc[i, 'Target']

        try:
            ah = dd_antihero[dd_antihero.Target == t]['size'].iloc[0]
        except:
            ah = 0
        try:
            h = dd_hero[dd_hero.Target == t]['size'].iloc[0]

            if h > ah :
                dd2.loc[i, 'number'] = h
                dd2.loc[i, 'appeals'] = (ah + h)
                #dd2.loc[i, 'score'] = h

            elif h < ah:
                dd2.loc[i, 'number'] = ah
                #dd2.loc[i, 'score'] = ah
                dd2.loc[i, 'appeals'] = (ah + h)

            else:
                dd2.loc[i, 'number'] = 0
                dd2.loc[i, 'appeals'] = (ah + h)

        except:
            try:
                h = dd_neu[dd_neu.Target == t]['size'].iloc[0]

                if ah == 0:
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)

                elif h > ah :
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)
                    #dd2.loc[i, 'score'] = h

                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    #dd2.loc[i, 'score'] = ah
                    dd2.loc[i, 'appeals'] = (ah + h)

                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'appeals'] = (ah + h)

            except:
                h = 0
                if h > ah :
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)

                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    #dd2.loc[i, 'score'] = ah
                    dd2.loc[i, 'appeals'] = (ah + h)

                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'appeals'] = (ah + h)


        try:
            dd2.loc[i, 'score'] = ah / (ah + h)
        except:
            dd2.loc[i, 'score'] = 0
        #st.write(t, h, ah)



    dd2['category'] = np.where(dd2.score > 0.5, 'villains', 'heroes')
    dd2['category'] = np.where(dd2.score == 0, 'neutral agents', dd2['category'] )
    dd2 = dd2.sort_values(by = ['category', 'Target'])
    dd2['corpus'] = ds
    df_dist_hist_all = dd2.melt(['Target', 'category', 'number'])

    #st.write(dd2)
    df_dist_ethos_all = dd2.copy()

    if contents_radio_categories_val_units == 'number':
        df_dist_ethos_all = df_dist_ethos_all['category'].value_counts(normalize=False)
        df_dist_ethos_all = pd.DataFrame(df_dist_ethos_all).reset_index()
        df_dist_ethos_all = df_dist_ethos_all.rename( columns = {'count': 'proportion'} )
    else:
        df_dist_ethos_all = df_dist_ethos_all['category'].value_counts(normalize=True).round(3) * 100
        df_dist_ethos_all = pd.DataFrame(df_dist_ethos_all).reset_index()


    if contents_radio_categories_val_units == 'number':
        x1 = df_dist_ethos_all['proportion'].max() + 7
        x2 = 5
    else:
        x1 = 101
        x2 =10

    sns.set(font_scale=1, style='whitegrid')
    dist_agents = sns.catplot(kind='bar', data=df1, y = 'Agent', x = 'proportion', aspect = 1.3)
    if contents_radio_categories_val_units == 'number':
        dist_agents_xlabel = 'frequency'
    else:
        dist_agents_xlabel = 'frequency [%]'
    dist_agents.set( xlim = (0, x1), xticks = np.arange(0, x1, x2), xlabel = dist_agents_xlabel )

    #st.write(df_dist_ethos_all)
    sns.set(font_scale=1.2, style='whitegrid')
    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=5, aspect=1.2,
                    x = 'category', y = 'proportion', hue = 'category', dodge=False, legend = False,
                    palette = {'villains':'#EF0303', 'heroes':'#3DF94E', 'neutral agents':'blue' },
                    )

    f_dist_ethos.set(ylim=(0, x1), xlabel = '', ylabel = dist_agents_xlabel)
    #plt.legend(loc='upper right', fontsize=13, bbox_to_anchor = (1.03, 0.9) )
    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center', xytext = (0, 7), textcoords = 'offset points', fontsize=15)
    add_spacelines(1)

    #st.write(df_dist_hist_all)
    sns.set(font_scale=1, style='whitegrid')


    heroes_tab1, heroes_tab2, heroes_tab_explore = st.tabs(['Bar-chart', 'Tables',  'Cases'])

    with heroes_tab1:
        add_spacelines(1)
        st.write( "##### Frequency " )
        st.pyplot(dist_agents)
        add_spacelines(3)
        st.write( "##### Distribution of categories of Agents' ethotic profiles" )
        st.pyplot(f_dist_ethos)
        add_spacelines(3)
        #st.write( "##### Distribution of numbers of appeals to Agents" )
        #st.pyplot(f_dist_ethoshist)



    with heroes_tab2:
        add_spacelines(1)
        st.write( " Frequency " )
        st.write(df1.rename( columns = {'proportion': 'frequency'} ) )
        add_spacelines(2)
        st.write( " Distribution of categories of Agents' ethotic profiles" )
        st.write(df_dist_ethos_all)
        add_spacelines(2)
        st.write('Detailed summary')
        dd2 = dd2.rename(columns = {'appeals':'frequency', 'Target':'Agent'})
        dd2 = dd2.sort_values( by = ['score'] )
        dd2 = dd2.reset_index(drop=True)
        dd2.index += 1
        st.write(dd2)


    with heroes_tab_explore:
        add_spacelines(1)
        colsText = [c for c in df.columns if 'Text' in c]
        cols_frames_components = [ 'Component', 'CausationEffect', 'CausationPolarity', 'CausationType',  'InternalPolarity' ]
        cols2 = ['CausationEffect', 'CausationPolarity', 'CausationType', 'InternalPolarity', 'AgentNumerosity', 'CauseLength']

        #st.write(dd2)
        df = data_list[-1]
        dff_columns = [
                    'discussion', 'map', 'CausationText', 'CausationEffect',
                   'CausationPolarity', 'CausationType', 'Component', 'InternalPolarity',
                   'AgentNumerosity', 'CauseLength', 'speaker', 'turn',
                   'Causation begin', 'Causation end', 'CauseText', 'EffectText',
                   'AgentText', 'CircumstancesText',
                   'ethos',  'sentiment',
                   'emotion',
                     ]
        df = df.fillna("NA")
        df[cols_frames_components+['AgentNumerosity', 'CauseLength']+colsText] = df[cols_frames_components+['AgentNumerosity', 'CauseLength']+colsText].astype('str')

        dff = df[dff_columns].copy()
        dff = dff.fillna('NA')
        select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[1:2])
        cols_columns = st.columns(len(select_columns))
        dict_cond = {}
        for n, c in enumerate(cols_columns):
            with c:
                cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                       (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[-1]))
                dict_cond[select_columns[n]] = cond_col
        dff_selected = dff.copy()
        for i, k in enumerate(dict_cond.keys()):
            dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
        add_spacelines(2)
        st.dataframe(dff_selected.sort_values(by = select_columns).dropna(how='all', axis=1).reset_index(drop=True), width = None)
        st.write(f"No. of cases: {len(dff_selected)}.")

        csv_cc_desc2 = convert_to_csv(dff_selected.sort_values(by = select_columns).reset_index(drop=True).dropna(how='all', axis=1))
        # download button 1 to download dataframe as csv
        download1 = st.download_button(
            label="Download data as TSV",
            data=csv_cc_desc2,
            file_name='Cases.tsv',
            mime='text/csv'
        )



def Target_compare_scor(data_list):
    add_spacelines(1)
    #contents_radio_categories_val_units = st.radio("Choose the unit of statistic to display", ("percentage", "number" ) ) # horizontal=True, , label_visibility = 'collapsed'
    contents_radio_unit = 'number'

    df = data_list[-1]
    ds = df['corpus'].iloc[0]
    #df = df[df['Component'] == 'Agent' ]
    df = df[ ~(df.AgentText.isna()) ]
    df['Target'] = df['AgentText']
    df1 = df['Target'].value_counts(normalize = True).round(3) * 100
    #st.write(df1)
    df1 = df1.reset_index()


    df2 = df.groupby( ['Target', 'ethos'], as_index = False ).size()
    #st.write(df2)

    dd_hero = df2[df2.ethos == 'support']
    dd_antihero = df2[df2.ethos == 'attack']
    #st.write(dd_antihero)
    dd_neu = df2[df2.ethos == 'neutral']

    dd2 = pd.DataFrame({ 'Target': df['Target'].unique() })

    dd2anti_scores = []
    dd2hero_scores = []
    up_data_dict_hist = {}

    dd2['score'] = np.nan
    dd2['number'] = np.nan
    dd2['appeals'] = np.nan

    #st.write( dd2.Target.unique())
    for i in dd2.index:

        t = dd2.loc[i, 'Target']

        try:
            ah = dd_antihero[dd_antihero.Target == t]['size'].iloc[0]
        except:
            ah = 0
        try:
            h = dd_hero[dd_hero.Target == t]['size'].iloc[0]

            if h > ah :
                dd2.loc[i, 'number'] = h
                dd2.loc[i, 'appeals'] = (ah + h)
                #dd2.loc[i, 'score'] = h

            elif h < ah:
                dd2.loc[i, 'number'] = ah
                #dd2.loc[i, 'score'] = ah
                dd2.loc[i, 'appeals'] = (ah + h)

            else:
                dd2.loc[i, 'number'] = 0
                dd2.loc[i, 'appeals'] = (ah + h)

        except:
            try:
                h = dd_neu[dd_neu.Target == t]['size'].iloc[0]

                if ah == 0:
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)

                elif h > ah :
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)
                    #dd2.loc[i, 'score'] = h

                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    #dd2.loc[i, 'score'] = ah
                    dd2.loc[i, 'appeals'] = (ah + h)

                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'appeals'] = (ah + h)

            except:
                h = 0
                if h > ah :
                    dd2.loc[i, 'number'] = h
                    dd2.loc[i, 'appeals'] = (ah + h)

                elif h < ah:
                    dd2.loc[i, 'number'] = ah
                    #dd2.loc[i, 'score'] = ah
                    dd2.loc[i, 'appeals'] = (ah + h)

                else:
                    dd2.loc[i, 'number'] = 0
                    dd2.loc[i, 'appeals'] = (ah + h)


        try:
            dd2.loc[i, 'score'] = ah / (ah + h)
        except:
            dd2.loc[i, 'score'] = 0
        #st.write(t, h, ah)



    dd2['category'] = np.where(dd2.score > 0.5, 'villains', 'heroes')
    dd2['category'] = np.where(dd2.score == 0, 'neutral agents', dd2['category'] )

    #dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
    dd2 = dd2.sort_values(by = ['category', 'Target'])
    #dd2['score'] = dd2['score'] * 100
    #dd2['score'] = dd2['score''].round()
    #st.write(dd2)
    dd2['corpus'] = ds
    df_dist_hist_all = dd2.melt(['Target', 'category', 'score'])

    #st.write(dd2)
    df_dist_ethos_all = dd2.copy()
    df_dist_ethos_all = df_dist_ethos_all['category'].value_counts(normalize=True).round(3) * 100
    df_dist_ethos_all = pd.DataFrame(df_dist_ethos_all).reset_index()



    #st.write(df_dist_hist_all)
    sns.set(font_scale=1, style='whitegrid')
    f_dist_ethoshist = sns.catplot(kind='strip', data = df_dist_hist_all, height=4, aspect=1.15,
                    y = 'score', hue = 'category', dodge=False, s=35, alpha=0.8, legend=False,
                    palette = {'villains':'#EF0303', 'heroes':'#3DF94E', 'neutral agents':'blue'},
                    x = 'category')



    df_dist_hist_all = dd2[['Target', 'category', 'score']].melt( ['Target', 'category'], value_vars='score' )#.drop('variable')
    #st.write(df_dist_hist_all)
    df_dist_hist_all1 = df_dist_hist_all.copy()
    df_dist_hist_all1['value'] = np.where( df_dist_hist_all1['value'] == 0, 0.01, df_dist_hist_all1['value'] )

    sns.set(font_scale=1, style='whitegrid')
    height_n = 5 # int(df_dist_hist_all.Target.nunique() / 2.5 )
    f_dist_ethoshist_barh = sns.catplot(kind='bar',
                    data = df_dist_hist_all1.sort_values(by = 'value'), height=height_n, aspect=1.3,
                    x = 'value', y = 'Target', hue = 'category', dodge=False,# legend=False,
                    palette = {'villains':'#EF0303', 'heroes':'#3DF94E','neutral agents':'blue'  },
                    )
    f_dist_ethoshist_barh.set(ylabel = '', xticks = np.arange(0, 1.1, 0.2) )



    heroes_tab1, heroes_tab2, heroes_tab_explore = st.tabs(['Bar-chart', 'Tables',  'Cases'])

    with heroes_tab1:
        add_spacelines(1)
        st.pyplot(f_dist_ethoshist)
        add_spacelines(1)
        st.write( "##### Distribution of villain scores" )
        st.pyplot(f_dist_ethoshist_barh)



    with heroes_tab2:
        add_spacelines(1)
        st.write('Detailed summary')
        dd2 = dd2.rename(columns = {'appeals':'frequency'})
        dd2 = dd2.sort_values( by = ['score'] )
        dd2 = dd2.reset_index(drop=True)
        dd2.index += 1
        st.write(dd2)


    with heroes_tab_explore:
        add_spacelines(1)
        colsText = [c for c in df.columns if 'Text' in c]
        cols_frames_components = [ 'Component', 'CausationEffect', 'CausationPolarity', 'CausationType',  'InternalPolarity' ]
        cols2 = ['CausationEffect', 'CausationPolarity', 'CausationType', 'InternalPolarity', 'AgentNumerosity', 'CauseLength']

        #st.write(dd2)
        df = data_list[-1]
        dff_columns = [
                    'discussion', 'map', 'CausationText', 'CausationEffect',
                   'CausationPolarity', 'CausationType', 'Component', 'InternalPolarity',
                   'AgentNumerosity', 'CauseLength', 'speaker', 'turn',
                   'Causation begin', 'Causation end', 'CauseText', 'EffectText',
                   'AgentText', 'CircumstancesText',
                   'ethos',  'sentiment',
                   'emotion',
                     ]
        df = df.fillna("NA")
        df[cols_frames_components+['AgentNumerosity', 'CauseLength']+colsText] = df[cols_frames_components+['AgentNumerosity', 'CauseLength']+colsText].astype('str')

        dff = df[dff_columns].copy()
        dff = dff.fillna('NA')
        select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[1:2])
        cols_columns = st.columns(len(select_columns))
        dict_cond = {}
        for n, c in enumerate(cols_columns):
            with c:
                cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                       (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[-1]))
                dict_cond[select_columns[n]] = cond_col
        dff_selected = dff.copy()
        for i, k in enumerate(dict_cond.keys()):
            dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
        add_spacelines(2)
        st.dataframe(dff_selected.sort_values(by = select_columns).dropna(how='all', axis=1).reset_index(drop=True), width = None)
        st.write(f"No. of cases: {len(dff_selected)}.")

        csv_cc_desc2 = convert_to_csv(dff_selected.sort_values(by = select_columns).reset_index(drop=True).dropna(how='all', axis=1))
        # download button 1 to download dataframe as csv
        download1 = st.download_button(
            label="Download data as TSV",
            data=csv_cc_desc2,
            file_name='Cases.tsv',
            mime='text/csv'
        )




##################### page config  #####################
st.set_page_config(page_title="FrameAn", layout="centered") # centered wide

summary_corpora_list = []

#  *********************** sidebar  *********************
with st.sidebar:
    st.title("Contents")

    contents_radio_type = st.radio("", ('Home Page', 'Single Corpus Analysis',), ) # label_visibility='collapsed'   'Comparative Corpora Analysis'

    if contents_radio_type == 'Home Page':
        add_spacelines(2)


    else:
        add_spacelines(2)
        st.write('Choose corpora')
        #box_pol1_log = st.checkbox("Covid", value=True)
        #box_pol5_log = st.checkbox("ElectionsSM", value=True)
        box_pol4 = st.checkbox("Climate Change Twitter", value=True)

        #add_spacelines(1)
        st.write( " ********************************** " )

        corpora_list = []
        corpora_list_components = {}
        corpora_list_full_text = {}

        if box_pol4:
            cor11 = load_data(cch_twi, sheet = True, sheet_name = 'components', indx=False)
            cor1 = cor11.copy()
            cor1['corpus'] = "Climate Change Twitter"
            corpora_list.append(cor1)
            corpora_list_components[ cor1['corpus'].iloc[0] ] = cor1


            #cor11 = load_data(cch_twi, sheet = True, sheet_name = 'full', indx=False)
            cor11 = load_data(cch_twi, sheet = True, sheet_name = 'ethos', indx=False)
            cor1 = cor11.copy()
            cor1['ethos'] = cor1.ethos_label.map( {0:'neutral', 1:'support', 2:'attack'} )
            cor1['corpus'] = "Climate Change Twitter"
            corpora_list.append(cor1)



    if contents_radio_type != 'Home Page':
        st.write("### Analysis Units")

        contents_radio_an_cat = st.radio("Unit picker", ('Text-based', 'Entity-based'), label_visibility='collapsed')
        if contents_radio_an_cat == 'Entity-based':
            #contents_radio_an_cat_unit = st.radio("Next choose", ['Target-Based Analysis'] )
            st.write(" ******************************* ")
            st.write("#### Statistical module")
            contents_radio3 = st.radio("Statistic", [ 'Agent Frequency', "Agent Score"], label_visibility='collapsed') # '(Anti)-heroes',
            #contents_radio3 = st.radio("Statistic", [ 'Heroes & villainses', "Profiles"]) # '(Anti)-heroes',
        else:
            #contents_radio_an_cat_unit = st.radio("Next choose", [ 'Sentence' ])
            st.write(" ******************************* ")
            st.write("#### Statistical module")
            contents_radio3 = st.radio("Statistic", ('Distribution', ), label_visibility='collapsed') # , 'Odds ratio', 'Cases'





#####################  page content  #####################
st.title(f"Frames Analytics")
add_spacelines(1)



if contents_radio_type == 'Home Page':
    MainPage()


else:

    if contents_radio_type == 'Single Corpus Analysis' and contents_radio3 == 'Distribution':
        distribution_plot_compare( data_list = corpora_list )

    elif contents_radio_type == 'Single Corpus Analysis' and contents_radio3 == 'Agent Frequency':
        Target_compare_freq( data_list = corpora_list )

    elif contents_radio_type == 'Single Corpus Analysis' and contents_radio3 == 'Agent Score':
        Target_compare_scor( data_list = corpora_list )
