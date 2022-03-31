
# Autoflow Server

# Setup
Ensure you have `python 3.8` installed.

1.  `pip install -r requirements.txt`
2. Create `.env` file  containing your OpenAI secret key as follows
`OPENAI_API_KEY=`
3. [Download](https://drive.google.com/file/d/1a0e3m8TyyxRaPi73gdsSXSHjfWPbnRo_/view?usp=sharing) the intent embeddings to use `classify_intent.py`
4. Download the following models and place them in the mentioned folder:
	- [models/search](https://drive.google.com/file/d/1v3uXuPCpK4V5cnx5C0Q0zEteKgRV5B7B/view?usp=sharing)
	-  [models/defect](https://storage.googleapis.com/sfr-codet5-data-research/finetuned_models/defect_codet5_base.bin)
	- [models/refine](https://storage.googleapis.com/sfr-codet5-data-research/finetuned_models/refine_medium_codet5_base.bin)
## Start Server
	uvicorn server:app --reload --port 8080
