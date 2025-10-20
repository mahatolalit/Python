def sum_list(list):
    if not list:
        return 0
    else:
        return list[0] + sum_list(list[1:])
    
nums = [1,2,3,4,5]
result = sum_list(nums)
print(f"Sum of the list = {result}")