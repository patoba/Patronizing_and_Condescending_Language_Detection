from preprocessing import Limpieza, Tokenize, RemoveWords, \
                          GetTags, all_tag_list, verbs_adjectives, verbs_adjectives_nouns

base_pipe = [("limpieza", Limpieza()),
              ("tokenize", Tokenize()),
              ("remove_stop_words", RemoveWords())
            ]

all_words_pipe = base_pipe + [("tags", GetTags(tag_list = all_tag_list))]

verbs_adjectives_pipe = base_pipe + [("tags", GetTags(tag_list = verbs_adjectives))]

verbs_adjectives_nouns_pipe = base_pipe + [("tags", GetTags(tag_list = verbs_adjectives_nouns))]

preprocessesors = [("all_words", all_words_pipe),
                    ("verbs_adjectives", verbs_adjectives_pipe),
                    ("verbs_adjectives_nouns", verbs_adjectives_nouns_pipe)]
