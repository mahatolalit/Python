nums = [10,22,38,49,53]

for num in nums:
    if num <=25:
        continue
    if num % 2 != 0:
        print(f"The First odd num is greater than 25: {num}.")
        break