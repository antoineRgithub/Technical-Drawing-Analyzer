from gliner2 import GLiNER2

# labels = ["BENDING RADIUS",
#               "UNDIMENSIONNED RADIUS",
#               "TOLERANCES",
#               "Relevant recommandations for manufacturing"]
# text = "BENDING RADIUS: 5mm\nUNDIMENSIONNED RADIUS: 3mm\nTOLERANCES: ±0.1mm\nRelevant recommandations for manufacturing: Use a fillet tool for bending operations to achieve the specified bending radius and ensure that the undimensionned radius is maintained within the tolerance limits."

def extract_entities(text: str, labels: list[str]) -> list[dict]:
    """
    Extract entities from text using GLiNER.

    Args:
        text: The input text to extract entities from.
        labels: A list of entity labels to extract.
    Returns:
        A list of dictionaries, each containing the extracted entity and its label.
    """
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    entities = model.extract_entities(text, labels)
    return entities

# print(extract_entities(text, labels))