def swap(df: pd.DataFrame, label_min: float, label_max: float):
    '''
    '''
    swap_sentence1 = []
    swap_sentence2 = []
    id_list = []
    source_list = []
    label_list = []
    binary_label = []
    for i in range(len(df)):
      if df.loc[i].label >= label_min and df.loc[i].label <= label_max:
        swap_sentence1.append(df.loc[i].sentence_2)
        swap_sentence2.append(df.loc[i].sentence_1)
        id_list.append(df.loc[i].id+'_swap')
        source_list.append(df.loc[i].source+'_swap')
        label_list.append(df.loc[i].label)
        binary_label.append(df.loc[i]['binary-label'])
    df_swap = pd.DataFrame({
        'id': id_list,
        'source': source_list,
        'sentence_1': swap_sentence1,
        'sentence_2': swap_sentence2,
        'label': label_list,
        'binary-label': binary_label
    })
    return df_swap


def concat_data(data_path: str, *dataframes: pd.DataFrame):
    """
    """
    result = pd.concat(dataframes)
    result.to_csv(data_path, index=False)