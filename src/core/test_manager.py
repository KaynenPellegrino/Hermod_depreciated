import unittest
import subprocess
from parallel_executor import run_tests_in_parallel


class TestManager:

    def run_tests(self):
        """
        Runs unit tests using unittest's discovery method and reports any failures.
        """
        try:
            result = subprocess.run(['python', '-m', 'unittest', 'discover'
                ], capture_output=True, text=True)
            if result.returncode == 0:
                print('All tests passed successfully.')
                return True
            else:
                print(f'Tests failed:\n{result.stdout}')
                return False
        except Exception as e:
            print(f'Error running tests: {e}')
            return False

    def run_individual_tests_in_parallel(self, test_functions):
        """
        Runs individual test functions in parallel.
        """
        run_tests_in_parallel(test_functions)


def test_func1():
    """Simulate a test case."""
    assert True == True, 'Test 1 passed.'


def test_func2():
    """Simulate a test case."""
    assert 1 == 1, 'Test 2 passed.'


def test_func3():
    """Simulate a test case."""
    assert 'abc' == 'abc', 'Test 3 passed.'


if __name__ == '__main__':
    test_manager = TestManager()
    all_tests_passed = test_manager.run_tests()
    if all_tests_passed:
        test_functions = [test_func1, test_func2, test_func3]
        test_manager.run_individual_tests_in_parallel(test_functions)
