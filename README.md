# __Installation setup__
## Create a new conda environment 
```bash
conda create --name hcl_streamlit python=3.10
```

## Activate the conda environment
conda activate hcl_streamlit

## Check python version
python --version

## Install requirements.txt
pip install -r requirements.txt

## __Upgrade the pymupdf library__
pip install pymupdf --upgrade

## If streamlit not installed 
pip install streamlit

## Run the streamlit
streamlit run rule_gen_framework.py

# Streamlit runs on the browser
Select the required pdf from data folder in repository
Clicking on next will move to next step in pipeline
To extract images and tables click on its button on left side panel.

The streamlit and table extraction take some time to run.
