from employee_module import Employee

Employee.company_name="KPMG"
Employee.company_location="London"
Employee.company_name="KPMG123"


print(Employee.company_name)
print(Employee.company_location)


# object 1 to keep 1st employee
emp1=Employee()
emp1.emp_id=101
emp1.emp_name="john"

# object 2 to keep 2nd  employee
emp2=Employee()
emp2.emp_id=102
emp2.emp_name="peter"


print(type(emp1))

emp1.print_employee_records()
