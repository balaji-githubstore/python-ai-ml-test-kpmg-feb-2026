colors=["red","green","yellow","red"]

print(colors)

print(type(colors))

print(len(colors))
print(colors[1])

# append value to the list
colors.append("blue")

print(colors)

# insert value at index 0 as black
colors.insert(0,"black")
print(colors)
# remove green from the list
colors.remove("green")

print(colors)

print(len(colors))

# print(colors.count("red"))

# Use tuple - when - fixed set of collections - immutable
signal=("red","green","yellow")
print(len(signal))

print(signal[0])

print(type(signal))


employee_data={
    "id":45,
    "name":"jack",
    "mobile":89899,
    "projects":["p1","p2","p3"]
}


print(employee_data)
print(type(employee_data))

print(employee_data["id"])

print(employee_data["projects"])
print(employee_data["projects"][2])

