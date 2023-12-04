import spacy
from clu.clu_bridge import Bridge
from scispacy_component import ScispaCyComponent
from clu.bridge.conversion import ConversionUtils
import os
import json

def save_odinson_document(odinson_doc, output_file):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, 'w') as file:
        json.dump(odinson_doc, file, indent=4)

def main():
    # Load the SpaCy model
    nlp = spacy.load("en_core_sci_lg")

    # Load NER models
    ner_models=("en_ner_jnlpba_md", "en_ner_bc5cdr_md")
    ner_models = [spacy.load(ner_model) for ner_model in ner_models]

    bridge = Bridge()
    scispacy_component = ScispaCyComponent()

    # Load JSON file
    with open('/home/mdhv/Documents/Major_Project/B11_BERN_EntityType.json') as json_file:
        data = json.load(json_file)
        text = data['text']

    # Process text with each NER model
    ner_results = []
    for ner_model in ner_models:
        doc = ner_model(text)
        for ent in doc.ents:
            ner_results.append({
            'text': ent.text,
            'start_char': ent.start_char,
            'end_char': ent.end_char,
            'label': ent.label_
        })

    # Processed data from the bridge (assuming it's a function that processes data)
    processed_data = bridge.process_data(text)

    # Process text with the SciSpaCy component
    entities = scispacy_component.process_text_with_ner(processed_data)

    # Create a Clu Document
    clu_doc = ConversionUtils.spacy.to_clu_document(nlp(text))

    # Create an Odinson Document
    odinson_doc = ConversionUtils.processors.to_odinson_document(clu_doc)

    # Save the Odinson document to a file named 'odinson_document.json'
    save_odinson_document(odinson_doc, 'home/mdhv/Documents/Major_Project/output.json')

    print("Original Text:", text)
    print("Processed Data:", processed_data)
    print("Named Entities:", entities)

if __name__ == "__main__":
    main()

