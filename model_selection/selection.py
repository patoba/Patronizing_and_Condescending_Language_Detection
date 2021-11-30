import pandas as pd

def min_max(df, SCORE_TRAIN = "Score Train", SCORE_TEST = "Score Test",
            MAX_SCORE = "MAX_SCORE"):
    df[MAX_SCORE] = df[[SCORE_TRAIN, SCORE_TEST]].abs().max(axis = 1)
    idx = df[MAX_SCORE].argmin()
    return tuple(df.iloc[idx, [0, 1]])

def analyze_results(res, X_train, X_test, y_train, y_test):
    results_dict = {
                    "Tipo":[],
                    "Score Train":[], 
                    "Score Test":[]
                    }

    for key in res.keys():
        score_test = res[key].score(X_test, y_test)
        score_train = res[key].score(X_train, y_train)

        results_dict["Tipo"].append(" # ".join(map(str, key)))
        results_dict["Score Train"].append(score_train)
        results_dict["Score Test"].append(score_test)

    df_models = pd.DataFrame(results_dict)    
    return df_models