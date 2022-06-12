
def test_not_null(load_example_bank_data):
    """
    Test to ensure dataframe isn't empty
    """
    assert load_example_bank_data.empty is False
