num=0

if num>0:
    print("positive: "+str(num))
elif num<0:
    print(f"negative {num}")
else:
    print("it's zero")


numbers=[45,68,89,54,89,88,25,24]
print(len(numbers))

# print only values <=50
for i in range(0,len(numbers)):
    if numbers[i]>=50:
        print(numbers[i])


for value in numbers:
    if value>=50:
        print(value)


ouput_list= [value for value in numbers if value>=50]
print(ouput_list)


colors=["red","green","yellow","red"] 
# get the final list in uppercase except red


