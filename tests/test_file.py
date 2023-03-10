import math


def test_method():
    assert math.log(100, 10) == 2, "Should be 2"
    
def test_broken_method():
    assert 2 == 3


if __name__ == "__main__":
    test_method()
    test_broken_method()
