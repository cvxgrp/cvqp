import numpy as np
from cvqp.libs import proj_sum_largest as proj_sum_largest_cpp


def test_cpp_extension_import():
    """Test that C++ extension can be imported."""
    assert proj_sum_largest_cpp is not None


def test_cpp_extension_basic():
    """Test basic functionality of C++ extension."""
    z = np.array([5.0, 3.0, 1.0, 0.0])
    k = 2
    alpha = 4.0
    untied = 2
    tied = 0
    cutoff = 4
    debug = False

    result = proj_sum_largest_cpp(z, k, alpha, untied, tied, cutoff, debug)

    # Should return (z, final_untied_count, final_tied_count, complete)
    assert len(result) == 4
    assert isinstance(result[1], int)  # final_untied_count
    assert isinstance(result[2], int)  # final_tied_count
    assert isinstance(result[3], bool)  # complete
