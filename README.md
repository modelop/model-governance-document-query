# model-governance-document-query
A model to interactively query model governance documents

The input to this model should be a data record that contains a column called 'question'.  The answer will be returned
in a column called answer, along with an appropriate chat_history element that can be used for subsequent queries.
Additionally, a list of source documents utilized to answer will be returned along with the excerpt of text that was
actually utilized by the LLM to respond.
