# from transformers.pipelines import Pipeline
from typing import List # , Tuple, Union, Mapping, Any, Callable, Iterable
import numpy as np
from tqdm import tqdm


# languages = ['afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'assamese',
#              'azerbaijani', 'basque', 'belarusian', 'bengali', 'bengali romanize',
#              'bosnian', 'breton', 'bulgarian', 'burmese', 'burmese zawgyi font', 'catalan',
#              'chinese (simplified)', 'chinese (traditional)', 'croatian', 'czech', 'danish',
#              'dutch', 'english', 'esperanto', 'estonian', 'filipino', 'finnish', 'french',
#              'galician', 'georgian', 'german', 'greek', 'gujarati', 'hausa', 'hebrew', 'hindi',
#              'hindi romanize', 'hungarian', 'icelandic', 'indonesian', 'irish', 'italian', 'japanese',
#              'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz',
#              'lao', 'latin', 'latvian', 'lithuanian', 'macedonian', 'malagasy', 'malay', 'malayalam',
#              'marathi', 'mongolian', 'nepali', 'norwegian', 'oriya', 'oromo', 'pashto', 'persian',
#              'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'sanskrit', 'scottish gaelic',
#              'serbian', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese',
#              'swahili', 'swedish', 'tamil', 'tamil romanize', 'telugu', 'telugu romanize', 'thai',
#              'turkish', 'ukrainian', 'urdu', 'urdu romanize', 'uyghur', 'uzbek', 'vietnamese',
#              'welsh', 'western frisian', 'xhosa', 'yiddish']



class Embedder:

    def __init__(self, index:int = 0):
        self.index=index
        if index == 0:
            from ESG_BERT import model, tokenizer
            self.model = model
            self.tokenizer = tokenizer
        elif index == 1:
            from sbert import model
            self.model = model
        elif index == 2:
            from xlm_roberta import model, tokenizer
            self.model = model
            self.tokenizer = tokenizer
        

    def embed(self, documents: List[str], b_size = 1) -> np.ndarray:
        if self.index == 0:
            l = []
            for document in tqdm(documents, "Progress"):
                tokens = self.tokenizer(document, return_tensors='pt')
                output = self.model(**tokens)
                l += [output.logits.detach().numpy()]
            return np.vstack(l)
            
        elif self.index == 1:
            return np.array(self.model.encode(documents, batch_size=b_size, show_progress_bar=True))

        elif self.index == 2:
            l = []
            for document in tqdm(documents, "Progress"):
                tokens = self.tokenizer(document, return_tensors='pt')
                output = self.model(**tokens)
                l += [output["pooler_output"].detach().numpy()]
            return np.vstack(l)