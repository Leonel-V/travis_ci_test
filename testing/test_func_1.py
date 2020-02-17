import sys
sys.path.append("/Users/leovarvelo/Desktop/github_int_test")
from dynamics.func_1 import sort_list


def test_sort_list():
    list_random = [4,2,7,4,7,3,9,1]
    known_list_sorted = [1, 2, 3, 4, 4, 7, 7, 9]
    list_sorted = sort_list(list_random)
    assert list_sorted == known_list_sorted
