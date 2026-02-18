import pandas as pd

class ValidationReport:
    def __init__(self):
        self.results=[]

    def add_result(self,test_name,status,message=""):
        dic={
            "Test":test_name,
            "Status":status,
            "Message":message
            }
        self.results.append(dic)
        
    def summary(self):
        # print(self.results)
        df_report=pd.DataFrame(self.results)
        print("\n-------------Validation Report----------------")
        print(df_report)
    
