import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json

cv_data = json.load(open('/home/mdhv/Documents/Major_Proj/NER/annotations_dataset/ner_rel_fulltext_full.json','r'))

for pmc_id, data in cv_data.items():
    annotations = data["annotations"]
    print(f"Annotations for {pmc_id}: {annotations}")


#function to create spaCy DocBin objects from the annotated data
def get_spacy_doc(file, data):
  #a blank spaCy pipeline
  nlp = spacy.blank('en')
  db = DocBin()

  # Iterate through the data
  for text, annot_list in tqdm(data):
    doc = nlp.make_doc(text)
    if isinstance(annot_list, dict) and 'entities' in annot:
             annot = item.get('entities', []) 

    ents = []
    entity_indices = []

     # Extract entities from the annotations
    for annot in annot_list:
        if len(annot) == 3:
            start, end, label = annot
            try:
                span = doc.char_span(start, end, label=label, alignment_mode='strict')
            except:
                continue

            if span is None:
                   # Log errors for annotations that couldn't be processed
                err_data = f"{[start, end]}    {text}\n"
                file.write(err_data)
            else:
                ents.append(span)

    try:
        doc.ents = ents
        db.add(doc)
    except:
        pass
    return db

# Split the annotated data into training and testing sets
from sklearn.model_selection import train_test_split
train, test = train_test_split(list(cv_data.values()), test_size=0.2, random_state=42)

# Display the number of items in the training and testing sets
len(train), len(test)

# Open a file to log errors during annotation processing
file = open('/home/mdhv/Documents/Major_Proj/NER/trained_models/train_file.txt','w')

# Create spaCy DocBin objects for training and testing data
db = get_spacy_doc(file, train)
db.to_disk('/home/mdhv/Documents/Major_Proj/NER/trained_models/train_data.spacy')

db = get_spacy_doc(file, test)
db.to_disk('/home/mdhv/Documents/Major_Proj/NER/trained_models/test_data.spacy')

# Close the error log file
file.close()
