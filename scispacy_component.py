import spacy

class ScispaCyComponent:
    def __init__(self, ner_models=("en_ner_jnlpba_md", "en_ner_bc5cdr_md"), text_model="en_core_sci_lg"):
        self.ner_models = [spacy.load(ner_model) for ner_model in ner_models]
        #self.text_model = spacy.load(text_model)

    def process_text_with_ner(self, text):
        ner_results = []

        for ner_model in self.ner_models:
            doc = ner_model(text)
            for ent in doc.ents:
                ner_results.append({
                    'text': ent.text,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                    'label': ent.label_
                })

        return ner_results

    #def process_text_with_text_model(self, text):
        #doc = self.text_model(text)
        # Do something with the text model, e.g., extracting keywords, etc.
        #return doc
        




