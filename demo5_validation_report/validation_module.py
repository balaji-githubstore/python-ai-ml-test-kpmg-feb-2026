import pandas as pd
from validation_report import ValidationReport

def load_data(file_path:str,report:ValidationReport):
    try:
        df=pd.read_csv(filepath_or_buffer=file_path,delimiter=",")
        report.add_result("Load Data","PASS")
        return df
    except Exception as e:
        report.add_result("Load Data","FAIL")
        return None
    
# logging the missing values
# logging the duplicate rows 
# logging the schema (column check)
# loggin the datatatype 

def validate_missing(df:pd.DataFrame,report:ValidationReport):
    missing_count= df.isnull().sum().sum()
    if missing_count == 0:
        report.add_result("Missing Value Check","PASS")
    else:
        report.add_result("Missing Value Check","FAIL",f"Missing count {missing_count}")
    

def validate_duplicate(df:pd.DataFrame,report:ValidationReport):
    dup_count=df.duplicated().sum()
    if dup_count == 0:
        report.add_result("Duplicate Check","PASS")
    else:
        report.add_result("Duplicate Check","FAIL",f"Duplicate rows: {dup_count}")

# Call load_data(), validate_missing(), validate_duplicate()

report_obj=ValidationReport()

# Call load_data()
df=load_data("files/sale_prices_practice.csv",report_obj)

# call validate_missing()
validate_missing(df,report_obj)

# call validate_duplicate()
validate_duplicate(df,report_obj)

report_obj.summary()

