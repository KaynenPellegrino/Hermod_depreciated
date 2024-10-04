from concurrent.futures import ThreadPoolExecutor


def run_tests_in_parallel(test_functions):
    """""\"
Summary of run_tests_in_parallel.

Parameters
----------
test_functions : type
    Description of parameter `test_functions`.

Returns
-------
None
""\""""
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(lambda fn: fn(), test_functions)
