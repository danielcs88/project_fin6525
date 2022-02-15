echo "Python Conversion"
jupytext --to notebook FIN6525_Project_Daniel_Cárdenas_[6102358].py
# jupytext --to py:percent *.ipynb

echo "Running Notebook"
python run_notebooks.py
    
echo "Github"
git add --all && git commit -m "Update" && git push

echo "Done"