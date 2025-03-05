## Build the Model
    python glmapping.py train --csv glmapping.csv --input-cols number description --output-cols department type brand unit_type --model-path model.joblib

## Query the Model
    python glmapping.py predict --model-path model.joblib --text1 "41010" --text2 "Sales"