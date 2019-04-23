# CS122: Linking restaurant records in Zagat and Fodor's data sets
#
# Alicia Chen


import numpy as np
import pandas as pd
import jellyfish
import util


def create_dataframe(file_name):
    '''
    Creates restaurant dataframes from CSV files

    Input: file_name: (string) name of the csv file
    Returns: df - pandas dataframe
    '''
    df = pd.read_csv(file_name, header=None, index_col=0,
                     names=["rest", "city", "address"])
    return df


def create_matches(df1, df2, links):
    '''
    Takes links and creates table of matches with name, city, and address
    for each restaurant taken from both zagat and fodors

    Inputs:
    - df1 (pandas dataframe): zagat df
    - df2 (pandas dataframe): fodor df
    - links (str): name of csv file with matched index pairs

    Returns: matches - a pandas dataframe
    '''
    known_links = pd.read_csv(links, header=None, names=["z", "f"])
    df1_o = df1.iloc[known_links["z"]].reset_index(drop=True)
    df2_o = df2.iloc[known_links["f"]].reset_index(drop=True)
    matches = pd.concat([df1_o, df2_o], axis=1)
    matches.columns = ["rest_z", "city_z", "address_z", 
                       "rest_f", "city_f", "address_f"]
    return matches


def create_unmatches(df1, df2):
    '''
    Creates list of 1000 pairs of unmatches by randomly sampling from data

    Inputs:
    - df1 (pandas dataframe): zagat df
    - df2 (pandas dataframe): fodor df

    Returns: unmatches - a pandas dataframe
    '''
    zs = df1.sample(1000, replace = True, random_state = 1234)
    fs = df2.sample(1000, replace = True, random_state = 5678)
    zs = zs.reset_index(drop=True)
    fs = fs.reset_index(drop=True)
    unmatches = pd.concat([zs, fs], axis=1)
    unmatches.columns = ["rest_z", "city_z", "address_z", 
                         "rest_f", "city_f", "address_f"]
    return unmatches


def tuple_probs(df):
    '''
    Takes matches or unmatches df and returns relative frequencies
    of every tuple possibility

    Input: df - pandas dataframe (either matches or unmatches)
    Returns: freqs - dictionary mapping tuple to relative frequency
    '''
    categories = ["low", "medium", "high"]
    counts = {}
    for cat_1 in categories:
        for cat_2 in categories:
            for cat_3 in categories:
                counts[(cat_1, cat_2, cat_3)] = 0

    for i, row in df.iterrows():
        score_r = jellyfish.jaro_winkler(row["rest_f"], row["rest_z"])
        score_c = jellyfish.jaro_winkler(row["city_f"], row["city_z"])
        score_a = jellyfish.jaro_winkler(row["address_f"], row["address_z"])
        cat_r = util.get_jw_category(score_r)
        cat_c = util.get_jw_category(score_c)
        cat_a = util.get_jw_category(score_a)
        tup = (cat_r, cat_c, cat_a)
        counts[tup] += 1

    freqs = counts.copy()
    for t, count in counts.items():
        freqs[t] = count / sum(counts.values())

    return freqs


def rank_tuples(front, tuples_to_rank, ratios):
    '''
    Ranks tuples in order of most match-like to least match-like

    Inputs:
    - front: (list) tuples with u(w) = 0
    - tuples_to_rank: (list) tuples that need to be sorted
    - ratios: (list) floats indicating ratio m(w)/u(w) for 
    the tuple with the matching index in tuples_to_rank

    Returns:
    - ranked: (list) ordered tuples
    '''
    order = list(np.argsort(ratios))
    order.reverse()
    ranked = [] 
    for rank in order:
        ranked.append(tuples_to_rank[rank])
    ranked = front + ranked
    return ranked


def sort_by_cum_pr(level, ranked, freqs, left_tail=True):
    '''
    Creates match_tuples or unmatch_tuples sets by finding
    appropriate cutoff in frequency distributions based on
    false positive and false negative rates

    Inputs:
    - level: (float) either mu or lambda, the false postiive or
             false negative rate
    - ranked: (list) ranked tuples based on m/u ratio
    - freqs: (dict) mapping of tuple to relative frequency within 
             match or unmatches
    - left_tail: (boolean) if true, cumulate sum from left. If 
                 false, accumulate sum from the right 
    '''
    tuples_set = set()
    if not left_tail:
        ranked.reverse()
    cumulative_p = 0
    for t in ranked:
        cumulative_p += freqs[t]
        if cumulative_p <= level:
            tuples_set.add(t)
        else:
            break
            
    return tuples_set


def create_sets(freqs_m, freqs_u, mu, lmbda):
    '''
    Create sets of tuples (possible_tuples, match_tuples and 
    unmatch_tuples) based on the frequency with which tuple appears in 
    the matches df and the unmatches df

    Inputs: 
    - freqs_m: (dict) maps tuples to relative frequency of occurence in 
               matches df
    - freqs_u: (dict) maps tuples to relative frequency of occurence in
               unmatches df
    - mu: (float) false positive rate
    - lmbda: (float) false negative rate
    '''
    possible_tuples = set()
    ratios = []
    tuples_to_rank = []
    front = []

    for t, u in freqs_u.items():
        m = freqs_m[t]
        if u == 0:
            if m == 0:
                possible_tuples.add(t)
            else:
                front.append(t)
        else:
            ratios.append(m / u)
            tuples_to_rank.append(t)

    ranked = rank_tuples(front, tuples_to_rank, ratios)
    match_tuples = sort_by_cum_pr(mu, ranked, freqs_u)
    unmatch_tuples_full = sort_by_cum_pr(lmbda, ranked, 
                                    freqs_m, left_tail=False)
    unmatch_tuples = unmatch_tuples_full.difference(match_tuples)

    ranked_set = set(ranked)
    leftover = ranked_set.difference(match_tuples.union(unmatch_tuples))
    possible_tuples = possible_tuples.union(leftover)

    return possible_tuples, match_tuples, unmatch_tuples


def make_final_dfs(df1, df2, match_z, match_f):
    '''
    Given indexes of matching or unmatching or possibly matching rows in
    zagats and fodors dataframes, concatenates corresponding rows to form 
    new dataframe. 

    Inputs: 
    - df1: (pd dataframe) zagats df
    - df2: (pd dataframe) fodors df
    - match_z (list) list of indexes in zagats df which are to be concatenated
    - match_f (list) list of indexes in fodors df which are to be concatenated

    Returns: 
    - concatenated (pd dataframe) - concatenated dataframe
    '''
    if (not match_z) and (not match_f):
        print("lists are empty")
        concatenated = pd.DataFrame(columns=["rest_z", "city_z", "address_z", 
                                             "rest_f", "city_f", "address_f"])
    else:
        df1_o = df1.iloc[match_z].reset_index(drop=True)
        df2_o = df2.iloc[match_f].reset_index(drop=True)
        concatenated = pd.concat([df1_o, df2_o], axis=1)
        concatenated.columns = ["rest_z", "city_z", "address_z", 
                                "rest_f", "city_f", "address_f"]

    print(concatenated.head(2), concatenated.shape)

    return concatenated


def find_matches(mu, lambda_, block_on_city=False):
    '''
    Given false positive and false negative rate thresholds, link records
    based on jaro-winkler text distance between restaurant entries in zagat
    and fodors based on restaurant name, city and address
    
    Inputs:
    - mu (float): false positive rate
    - lambda_ (float): false negative rate
    - block_on_city (boolean): indicates whether or not to block on city

    Returns: 
    - (dataframes) matches_out, possible_matches_out, unmatches_out listing
      pairs with restaurant name, city and address from zagat and fodors
    '''

    zagat = create_dataframe("zagat.csv")
    fodors = create_dataframe("fodors.csv")
    matches = create_matches(zagat, fodors, "known_links.csv")
    unmatches = create_unmatches(zagat, fodors)
    freqs_m, freqs_u = tuple_probs(matches), tuple_probs(unmatches)

    sets = create_sets(freqs_m, freqs_u, mu, lambda_)
    possible_tuples = sets[0]
    match_tuples = sets[1]
    unmatch_tuples = sets[2]

    match_z = []
    match_f = []
    unmatch_z = []
    unmatch_f = []
    poss_z = []
    poss_f = []
    fodors_to_iterate = fodors

    for row_z in zagat.iterrows():

        if block_on_city:
            city = row_z[1]["city"]
            city_filter = fodors["city"] == city
            fodors_to_iterate = fodors[city_filter]
            print(city)

        for row_f in fodors_to_iterate.iterrows():
            i_z, data_z = row_z
            i_f, data_f = row_f
            score_r = jellyfish.jaro_winkler(data_z["rest"], data_f["rest"])
            score_c = jellyfish.jaro_winkler(data_z["city"], data_f["city"])
            score_a = jellyfish.jaro_winkler(data_z["address"], 
                                             data_f["address"])
            cat_r = util.get_jw_category(score_r)
            cat_c = util.get_jw_category(score_c)
            cat_a = util.get_jw_category(score_a)
            t = (cat_r, cat_c, cat_a)
            print(t)

            if t in match_tuples:
                print("match")
                match_z.append(i_z)
                match_f.append(i_f)
            elif t in unmatch_tuples:
                print("unmatch")
                unmatch_z.append(i_z)
                unmatch_f.append(i_f)
            elif t in possible_tuples:
                print("possible")
                poss_z.append(i_z)
                poss_f.append(i_f)

    matches_out = make_final_dfs(zagat, fodors, match_z, match_f)
    unmatches_out = make_final_dfs(zagat, fodors, unmatch_z, unmatch_f)
    possible_matches_out = make_final_dfs(zagat, fodors, poss_z, poss_f)

    return matches_out, possible_matches_out, unmatches_out


if __name__ == '__main__':
    matches, possibles, unmatches = \
        find_matches(0.005, 0.005, block_on_city=True)

    print("Found {} matches, {} possible matches, and {} "
          "unmatches with no blocking.".format(matches.shape[0],
                                               possibles.shape[0],
                                               unmatches.shape[0]))
