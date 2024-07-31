
# Invoice Matcher

A Python script to match similar invoices

## Usage/How to run?

To run the script locally on your system, follow the instructions.

 ### Pre-requisites: Python Packages/Libraries:-
 Install these using `pip` globally or in a virtual environment
   1. `pdfplumber`
   2. `scikit-learn`
   3. `tabulate`
 ### Instructions to run:-

1) Download the project files or use Git Clone using your Terminal at the desired directory
```bash
git clone https://github.com/arjunpalat/Invoice-Matcher.git
```  
2) Test files and Train files must be added into Test folder and Train folder respectively at the root of your directory (where the script is present)

3) Assuming you have all the packages mentioned in pre-requisites, run the script and follow the instructions

### Documentation

#### Overview:
The `main.py` file is designed to handle the extraction and processing of text data from PDF files, specifically for the purpose of matching invoices. The primary class in this file is `InvoiceMatcher`, which utilizes various libraries and techniques to achieve this goal.

#### Dependencies
The script imports several key libraries:

-   `pdfplumber`: For extracting text from PDF files.
-   `os`: For interacting with the operating system, such as file handling.
-   `re`: For regular expression operations.
-   `pickle`: For serializing and deserializing Python objects.
-   `sklearn.feature_extraction.text.TfidfVectorizer`: For converting text data into TF-IDF feature vectors.
-   `sklearn.metrics.pairwise.cosine_similarity`: For calculating the cosine similarity between feature vectors.
-   `collections.Counter`: For counting hashable objects.
-   `tabulate`: For creating formatted tables.

#### Approach
1.  **Text Extraction**: Using  `pdfplumber`, the script extracts text from PDF files.
2.  **Text Processing**: The extracted text is processed and converted into TF-IDF feature vectors using  `TfidfVectorizer`.
3.  **Cosine-similarity Calculation**: The cosine similarity between different invoice keywords is calculated using  `cosine_similarity`  to determine how similar they are.
4.  **Result Presentation**: The results, such as similarity scores, can be presented in a tabulated format using  `tabulate`.

#### Results
The expected results from this script include:

-   Extracted text data from PDF invoices.
-   TF-IDF feature vectors representing the text data.
-   Cosine similarity scores indicating the similarity between different invoices and returning the best match.
-   A tabulated summary of the results for easy interpretation.

