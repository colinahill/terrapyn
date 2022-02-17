from terrapyn import __author__, __email__


def test_project_info():
    """Test __author__ value."""
    assert __author__ == "Colin Hill"
    assert __email__ == "colinalastairhill@gmail.com"
