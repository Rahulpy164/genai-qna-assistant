import re
from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence endings within the last 200 characters
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size - 200:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text for better Q&A performance.
    
    Args:
        text: Raw text input
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()

def find_relevant_chunks(question: str, chunks: List[str], max_chunks: int = 3) -> List[str]:
    """
    Find the most relevant chunks for a given question using simple keyword matching.
    
    Args:
        question: User's question
        chunks: List of text chunks
        max_chunks: Maximum number of chunks to return
    
    Returns:
        List of relevant chunks
    """
    question_words = set(question.lower().split())
    
    # Score chunks based on keyword overlap
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        overlap = len(question_words.intersection(chunk_words))
        score = overlap / len(question_words) if question_words else 0
        chunk_scores.append((score, i, chunk))
    
    # Sort by score and return top chunks
    chunk_scores.sort(reverse=True)
    return [chunk for _, _, chunk in chunk_scores[:max_chunks]]

def enhance_context_with_prompt_engineering(question: str, context: str) -> str:
    """
    Enhance the context using prompt engineering techniques.
    
    Args:
        question: User's question
        context: Document context
    
    Returns:
        Enhanced context with prompt engineering
    """
    # Add instructional context to improve model performance
    enhanced_context = f"""
Document Content:
{context}

Instructions: Based on the document content above, please provide a comprehensive and accurate answer to the following question. If the answer is not explicitly stated in the document, indicate that the information is not available in the provided text.

Question: {question}
"""
    
    return enhanced_context.strip()

def extract_key_information(text: str) -> Dict[str, str]:
    """
    Extract key information from text for better context understanding.
    
    Args:
        text: Input text
    
    Returns:
        Dictionary with extracted information
    """
    info = {
        'word_count': str(len(text.split())),
        'char_count': str(len(text)),
        'has_numbers': str(bool(re.search(r'\d', text))),
        'has_dates': str(bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', text))),
        'has_emails': str(bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))),
    }
    
    return info

