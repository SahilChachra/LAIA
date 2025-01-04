def extract_document_info(documents):
    """
    Extract key information from a list of Document objects.
    
    Parameters:
    documents (list): A list of Document objects with metadata and page_content
    
    Returns:
    list: A list of dictionaries containing extracted information
    """
    extracted_docs = []
    
    for doc in documents:
        # Extract metadata safely using .get() to avoid KeyError
        doc_info = {
            'source': doc.metadata.get('source', 'N/A'),
            'page_number': doc.metadata.get('page', 'N/A'),
            'title': doc.metadata.get('Title', 'Untitled'),
            'page_content': doc.page_content
        }
        
        extracted_docs.append(doc_info)
    
    return extracted_docs