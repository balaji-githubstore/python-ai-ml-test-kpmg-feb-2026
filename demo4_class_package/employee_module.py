class Employee:
    # static or class variable
    company_name=None
    company_location=None

    def __init__(self):
        # non-static variable or instance variable
        self.emp_id=None
        self.emp_name=None

# non-static method
    def print_employee_records(self):
        print(self.emp_id)
        print(self.emp_name)
        print(Employee.company_name)