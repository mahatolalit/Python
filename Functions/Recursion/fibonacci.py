def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci( n - 1 ) + fibonacci ( n - 2 )
    
result = fibonacci(10)
print(f"10th fibonacci number = {result}")