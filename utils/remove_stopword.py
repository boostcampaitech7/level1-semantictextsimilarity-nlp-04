#%%
def make_df():
    '''
    :return: Pandas DataFrame
    '''
    df = pd.read_csv('data/file_name.csv')
    return df
#%%
def make_csv(df):
    '''
    :param df: DataFrame to be saved
    '''
    df.to_csv('data/output_file_name.csv')
#%%
def custom_sentence(Sentence):
    '''
    :param Sentence: The input sentence to clean
    :return: Cleaned sentence
    '''

    Sentence = re.sub('[.!\^]', ' ', Sentence) # Replace .!^ with spaces
    Sentence = re.sub('[ㄱ-ㅎㅏ-ㅣ]|[^가-힣a-zA-Z0-9~\?\s:%]', ' ', Sentence) # Remove unnecessary characters

    Sentence = re.sub('([^가-힣a-zA-Z0-9])\\1{1,}', '\\1', Sentence)
    Sentence = re.sub('([가-힣a-zA-Z])\\1{4,}', '\\1' * 3, Sentence) # Repetition of letters, limit to 3
    Sentence = re.sub('^[~?\s:%]*|[~?\s:%]*$', '', Sentence) # Trim spaces and special characters at the start/end

    Sentence = Sentence

    if not Sentence: Sentence = ' '

    return Sentence
#%%
def drop_none(df):
    '''
    :param df: The input dataframe
    :return: Dataframe with missing rows dropped
    '''
    df = df.dropna(subset=['sentence_1'], axis=0)
    df = df.dropna(subset=['sentence_2'], axis=0)
    return df
#%%
def remove_stopword(df, *col_list):
    '''
    :param df: The input dataframe
    :param col_list: sentence1, sentence2
    :return: Cleaned dataframe
    '''
    df = drop_none(df)
    for col in col_list:
        df[col] = df[col].apply(custom_sentence)

    return df
#%%
train_df = make_df()
#%%
train_df = remove_stopword(train_df, 'sentence_1', 'sentence_2')
#%%
make_csv(train_df)