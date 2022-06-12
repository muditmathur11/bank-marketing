import pytest
from pathlib import Path


from bank_marketing import load_data

@pytest.fixture
def load_example_bank_data():
    """
    Fixture to load example banking data from a csv file for testing
    """

    sample_data_path = Path(__file__).resolve().parent / '..' / 'data/sample'
    sample_data = sample_data_path / 'fake_data.csv'
    return load_data(sample_data)