"""Run wanted tests on Markov. Modified to run easily from bash script on Markov."""
import sys
from systematic_tests4_test_functions import *

# Get filename and function name from command line.
filename = sys.argv[2]
function_name = sys.argv[1]
sgd_or_not = sys.argv[3]

function_dict = {
    "training_test_function1": training_test_function1,
    "testing_test_function1": testing_test_function1,
    "training_test_function2": training_test_function2,
    "testing_test_function2": testing_test_function2,
    "training_test_function3": training_test_function3,
    "testing_test_function3": testing_test_function3, 
    "training_test_function4": training_test_function4,
    "testing_test_function4": testing_test_function4,
}

if sgd_or_not == "t":
    sgd = True
else:
    sgd = False

function_dict[function_name](filename, sgd = sgd)
