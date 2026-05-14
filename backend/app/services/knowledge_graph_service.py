import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# In-memory graph storage
knowledge_graph = []


# ==========================================
# Clean Text
# ==========================================

def clean_text(text):

    return (
        text
        .replace("\n", " ")
        .replace("\t", " ")
        .strip()
    )


# ==========================================
# Extract Entities + Relationships
# ==========================================

def extract_entities_and_relationships(text):

    doc = nlp(text)

    entities = []

    relationships = []


    # ======================================
    # Extract Entities
    # ======================================

    for ent in doc.ents:

        entity_text = clean_text(
            ent.text
        )

        if len(entity_text) < 3:
            continue

        entities.append({

            "text": entity_text,

            "label": ent.label_
        })


    # ======================================
    # Extract Relationships
    # ======================================

    for sentence in doc.sents:

        words = sentence.text.split()

        if len(words) >= 3:

            subject = clean_text(
                words[0]
            )

            relation = clean_text(
                words[1]
            )

            obj = clean_text(
                words[2]
            )

            if (
                len(subject) < 2 or
                len(obj) < 2
            ):
                continue

            relationships.append({

                "subject": subject,

                "relation": relation,

                "object": obj
            })


    graph_data = {

        "entities": entities,

        "relationships": relationships
    }

    print(
        "GRAPH GENERATED:",
        graph_data
    )

    return graph_data


# ==========================================
# Store Graph
# ==========================================

def store_knowledge_graph(data):

    global knowledge_graph

    knowledge_graph.append(data)

    print(
        "GRAPH STORED SUCCESSFULLY"
    )


# ==========================================
# Get Graph
# ==========================================

def get_knowledge_graph():

    return knowledge_graph