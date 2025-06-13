from typing import List, Dict, Tuple
import requests
import json
from config import MODELS, DEFAULT_MODEL, API_BASE, CHUNK_SIZE, CHUNK_OVERLAP, MODEL_SETTINGS
from read_pdf import extract_text_from_pdf

class GreekSemanticChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.api_base = API_BASE
        # use configured model
        self.model = MODELS[DEFAULT_MODEL]["name"] + ":latest"
        
        # model settings
        self.settings = MODEL_SETTINGS
        
    def _check_ollama(self):
        """Check if Ollama is running and model is available"""
        try:
            # check server
            response = requests.get(f"{self.api_base}/tags")
            if response.status_code != 200:
                raise ConnectionError("Ollama server not responding")
            
            # check if model exists
            models_response = requests.get(f"{self.api_base}/tags")
            models = models_response.json().get('models', [])
            model_name = self.model
            if not any(m.get('name', '') == model_name for m in models):
                print(f"{model_name} model not found. Attempting to pull...")
                
                # pull model with all settings
                pull_response = requests.post(
                    f"{self.api_base}/pull",
                    json={
                        "name": model_name,
                        **self.settings
                    }
                )
                if pull_response.status_code != 200:
                    raise RuntimeError(f"Failed to pull {model_name} model")
                
            print(f"Ollama ready with model: {model_name}")
            return True
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return False

    def _summarize_chunk(self, text: str) -> str:
        """Generate a brief summary using Defined Models"""
        print(f"Making request to URL: {self.api_base}/generate")
        try:
            clean_text = text.strip().replace('\n', ' ').replace('"', '')
            
            prompt = f"""Κείμενο: {clean_text}

Δώσε μια σύντομη περίληψη του παραπάνω κειμένου σε 2-3 προτάσεις.
Γράψε την περίληψη χωρίς εισαγωγικές φράσεις ή επεξηγήσεις.
Η περίληψη πρέπει να είναι περιεκτική και να μην επαναλαμβάνει το αρχικό κείμενο."""
            
            response = requests.post(
                f"{self.api_base}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    **self.settings
                }
            )
            
            print(f"Ollama API response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()['response']
                    print("Successfully generated summary")
                    return result
                except Exception as e:
                    print(f"Error parsing response: {e}")
                    return response.text
            return text[:100] + "..."
        except Exception as e:
            print(f"Summarization error: {e}")
            return text[:100] + "..."

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts using Mistral"""
        try:
            prompt = f"""Κείμενο: {text}

Εξάγαγε τις 3-5 πιο βασικές έννοιες από το παραπάνω κείμενο.
Γράψε μόνο τις έννοιες, χωρίς αρίθμηση, προθέματα ή επεξηγήσεις.
Μην γράψεις μεγάλες προτάσεις ούτε έννοιες που αποτελούνται από έναν μόνο χαρακτήρα. 
Κάθε έννοια σε νέα γραμμή."""
            
            response = requests.post(
                f"{self.api_base}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    **self.settings
                }
            )
            if response.status_code == 200:
                try:
                    result = response.json()['response'].strip()
                    # clean up the concepts
                    concepts = [
                        line.strip().lstrip('- []1234567890. ')  # remove all prefixes
                        for line in result.split('\n')
                        if line.strip() and not any(
                            keyword in line.lower() 
                            for keyword in [
                                'έννοιες:', 'βασικές', 'κείμενο:', 'απάντηση:',
                                'απαντάς', 'χρησιμοποιείς', 'κάνεις', 'εξάγεις', 'παρακαλώ'
                            ]
                        )
                    ]
                    return [c for c in concepts[:5] if len(c) > 0]  # return only non-empty concepts
                except Exception as e:
                    print(f"Error parsing concepts: {e}")
                    return []
            return []
        except Exception as e:
            print(f"Concept extraction error: {e}")
            return []

    def create_semantic_chunks(self, paragraph_dict) -> Tuple[List[Dict], Dict]:
        """Split Greek text into chunks by paragraph."""
        
        if not self._check_ollama():
            raise RuntimeError("Ollama is not available")

        # as the "text" argument there is a paragraph_dict passed
        # paragraph_dict format is: paragraph_dict["paragraph text bla bla"] -> [1,2] (page numbers, one paragraph can be in more than one pages)
        # print(paragraph_dict)

        # paragraphs now contains all the paragraphs of the pdf 
        paragraphs = list(paragraph_dict.keys())

        # print(paragraphs)

        # i do this not to run the full sample text, just the first 6 paragraphs
        # because running the whole sample text takes so much time
        #paragraphs = paragraphs[:51]

        chunks = []

        # eurethrio dictionary
        # concepts_index["blabla concept"] = [1,5,8] -> page numbers list
        concepts_index = {}

        # key concept not found in text dictionary
        # concept_not_found["blabla concept"] = [2,4] -> chunk numbers that the key concept belongs
        concept_not_found = {}

        # process each paragraph as a separate chunk
        for i, para in enumerate(paragraphs, 1):
            print(f"\nProcessing paragraph {i}/{len(paragraphs)}")
            
           
            # page_num = int((i-1)/3) + 1

            try:
                chunk = {
                    "text": para,
                    "summary": self._summarize_chunk(para),
                    "key_concepts": self._extract_concepts(para)
                }
                chunks.append(chunk)
                print(f"Successfully processed chunk {i}")

                # iterate through key concepts to index them
                for concept in chunk["key_concepts"]:
                    # check if the concept exists in the text (case-sensitive now based on your code)
                    if concept.lower() in chunk["text"]:
                        # get the page number for this paragraph from our mapping dictionary
                        page_nums = paragraph_dict[para]
                        
                        # split the concept into individual words
                        words = concept.lower().split()
                        
                        # CASE 1: If it's a single word concept (e.g., "word1")
                        if len(words) == 1:
                            single_word = words[0]
                            # if this single word hasn't been indexed before, create a new entry with this page number
                            if single_word not in concepts_index:
                                concepts_index[single_word] = []
                            # add all page numbers that aren't already in the list
                            for page_num in page_nums:
                                if page_num not in concepts_index[single_word]:
                                    concepts_index[single_word].append(page_num)
                                        
                        # CASE 2: If it's a multi-word concept (e.g., "word3 word6")
                        else:
                            # extract the first word to use as the main category
                            first_word = words[0]
                            # join all remaining words to create the subcategory
                            remaining_words = " ".join(words[1:])
                            
                            # if this is the first time we're seeing this first word, initialize its entry
                            # as a dictionary (not a list) to hold subcategories
                            if first_word not in concepts_index:
                                concepts_index[first_word] = {}
                            
                            # if this is a new subcategory under this first word
                            if remaining_words not in concepts_index[first_word]:
                                concepts_index[first_word][remaining_words] = []
            
                            # add all page numbers that aren't already in the list
                            for page_num in page_nums:
                                if page_num not in concepts_index[first_word][remaining_words]:
                                        concepts_index[first_word][remaining_words].append(page_num) 


            except Exception as e:
                print(f"Error processing paragraph {i}: {e}")
        
        return chunks, concepts_index

def extract_key_terms(chunks: List[Dict]) -> List[Tuple[str, str]]:
    """Extract key terms and their context from chunks."""
    terms = []
    for chunk in chunks:
        for concept in chunk['key_concepts']:
            terms.append((concept, chunk['summary']))
    return terms

# example usage
# thelw na dw aplws se poio shmeio xwnei ta key concept
# kai ekei na krinw thn selida kai na ta kanw tuple
if __name__ == "__main__":
    chunker = GreekSemanticChunker()
    
    sample_text = """
   """
    
    pdf_path = "PROS-ETAIREIES.pdf"


    paragraph_dict = extract_text_from_pdf(pdf_path)
    print(f"Successfully extracted {len(paragraph_dict)} paragraphs")

    
    chunks, concepts_index, concept_not_found = chunker.create_semantic_chunks(paragraph_dict)
    key_terms = extract_key_terms(chunks)
