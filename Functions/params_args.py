# Function definition: the names inside the parentheses are parameters.
# Parameters are placeholders used by the function to refer to the values
# it will receive when it's called.
def display_info(name, age, city):
        """
        Display a person's information.

        Parameters (placeholders used in the function definition):
            - name: parameter for the person's name
            - age: parameter for the person's age
            - city: parameter for the person's city
        """
        print(f"{name} is {age} years old and lives in {city}")

# The values we collect here (from input) are arguments — the actual values
# we pass into the function when we call it. In other words: parameters are
# defined by the function; arguments are supplied at the call site.
name = input("Enter name: ")
age = int(input("Enter age: "))
city = input("Enter city: ")

# Function call: `name`, `age`, `city` here are arguments (actual values).
display_info(name, age, city)