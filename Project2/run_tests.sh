#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
nice python3 $DIR/run_systematic_tests.py training_test_function1 train_func_1_Reg f
nice python3 $DIR/run_systematic_tests.py training_test_function1 train_func_1_SGD t
nice python3 $DIR/run_systematic_tests.py testing_test_function1 test_func_1_Reg f
nice python3 $DIR/run_systematic_tests.py testing_test_function1 test_func_1_SGD f

nice python3 $DIR/run_systematic_tests.py training_test_function2 train_func_2_Reg f
nice python3 $DIR/run_systematic_tests.py training_test_function2 train_func_2_SGD t
nice python3 $DIR/run_systematic_tests.py testing_test_function2 test_func_2_Reg f
nice python3 $DIR/run_systematic_tests.py testing_test_function2 test_func_2_SGD f

nice python3 $DIR/run_systematic_tests.py training_test_function3 train_func_3_Reg f
nice python3 $DIR/run_systematic_tests.py training_test_function3 train_func_3_SGD t
nice python3 $DIR/run_systematic_tests.py testing_test_function3 test_func_3_Reg f
nice python3 $DIR/run_systematic_tests.py testing_test_function3 test_func_3_SGD f

nice python3 $DIR/run_systematic_tests.py training_test_function4 train_func_4_Reg f
nice python3 $DIR/run_systematic_tests.py training_test_function4 train_func_4_SGD t
nice python3 $DIR/run_systematic_tests.py testing_test_function4 test_func_4_Reg f
nice python3 $DIR/run_systematic_tests.py testing_test_function4 test_func_4_SGD f
