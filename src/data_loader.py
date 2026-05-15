import os


def load_documents(data_folder="data"):
    """
    Load all text files from the disease folders.
    Returns a list of dictionaries with:
    - disease
    - source_file
    - text
    """
    all_documents = []

    diseases = ["glaucoma", "cataract", "amd", "dry_eye"]

    for disease in diseases:
        folder = os.path.join(data_folder, disease)

        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            all_documents.append({
                "disease": disease,
                "source_file": file_name,
                "text": text
            })

    return all_documents