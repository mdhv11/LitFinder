import spacy
from clu.clu_bridge import Bridge
from scispacy_component import ScispaCyComponent
from clu.bridge.conversion import ConversionUtils
import os

def save_odinson_document(odinson_doc, output_file):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open('home/mdhv/Documents/Major_Proj/odinson_document.txt', 'w') as file:
        file.write(str(odinson_doc))

def main():
    # Load the SpaCy model
    nlp = spacy.load("en_core_sci_lg")

    # Load NER models
    ner_models=("en_ner_jnlpba_md", "en_ner_bc5cdr_md")
    ner_models = [spacy.load(ner_model) for ner_model in ner_models]

    bridge = Bridge()
    scispacy_component = ScispaCyComponent()

    text = ("Patient was admitted to the hospital on 2023-03-15 with complaints of chest pain and shortness of breath. "
            "Initial assessment revealed elevated levels of troponin and other cardiac enzymes. "
            "The patient has a history of hypertension and diabetes."

            "During the hospital stay, the cardiology team performed an echocardiogram, "
            "which showed evidence of myocardial infarction. "
            "The patient was started on aspirin, beta-blockers, and statins for secondary prevention."

            "Laboratory results indicated abnormal liver function tests, "
            "and an abdominal ultrasound was ordered to investigate the cause. "
            "The ultrasound revealed gallstones, and the patient was referred to a gastroenterologist for further evaluation."

            "The patient's medication list includes metformin, lisinopril, and atorvastatin. "
            "The healthcare team is closely monitoring the patient's vital signs and adjusting medications as needed. "
            "Discharge planning is underway, and the patient will be scheduled for follow-up appointments with cardiology and gastroenterology.")

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

    # Save the Odinson document to a file named 'odinson_document.txt'
    save_odinson_document(odinson_doc, 'home/mdhv/Documents/Major_Proj/B3.json')

    print("Original Text:", text)
    print("Processed Data:", processed_data)
    print("Named Entities:", entities)

if __name__ == "__main__":
    main()


