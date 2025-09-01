"""Validation tests to ensure the testing infrastructure is working correctly."""

import pytest
import torch
from pathlib import Path


class TestInfrastructureValidation:
    """Test class to validate the testing infrastructure setup."""

    def test_pytest_is_working(self):
        """Basic test to ensure pytest is functioning."""
        assert True

    def test_fixtures_are_available(self, temp_dir, sample_config):
        """Test that shared fixtures are working."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert isinstance(sample_config, dict)
        assert "model" in sample_config

    def test_torch_integration(self, sample_tensor, mock_torch_device):
        """Test that PyTorch integration is working."""
        assert isinstance(sample_tensor, torch.Tensor)
        assert sample_tensor.shape == (10, 5)
        assert mock_torch_device == torch.device('cpu')

    def test_graph_data_fixture(self, sample_graph_data):
        """Test that graph data fixtures work correctly."""
        assert 'x' in sample_graph_data
        assert 'edge_index' in sample_graph_data
        assert sample_graph_data['num_nodes'] == 5
        assert isinstance(sample_graph_data['x'], torch.Tensor)
        assert isinstance(sample_graph_data['edge_index'], torch.Tensor)

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker is working."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker is working."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker is working."""
        assert True

    def test_mock_model_fixture(self, mock_model):
        """Test that mock model fixture is working."""
        assert hasattr(mock_model, 'forward')
        assert hasattr(mock_model, 'encode')
        assert hasattr(mock_model, 'decode')
        
        # Test mock model returns expected shapes
        mean, log_var = mock_model.forward()
        assert mean.shape == (5, 2)
        assert log_var.shape == (5, 2)

    def test_temp_file_operations(self, temp_file):
        """Test temporary file operations."""
        assert temp_file.exists()
        assert temp_file.read_text() == "test content"

    def test_reproducible_random_seeds(self, sample_tensor):
        """Test that random seeds are set for reproducible tests."""
        # Create another tensor with the same seed
        torch.manual_seed(42)
        other_tensor = torch.randn(10, 5)
        
        # They should be identical due to seed setting
        torch.testing.assert_close(sample_tensor, other_tensor)

    def test_isolated_environment(self, isolated_environment):
        """Test that isolated environment fixture works."""
        assert isinstance(isolated_environment, Path)
        assert isolated_environment.exists()

    def test_coverage_excludes_test_files(self):
        """Test that demonstrates coverage configuration."""
        # This test itself should be excluded from coverage
        uncovered_code = "This line should be excluded from coverage"
        assert uncovered_code is not None


def test_module_imports():
    """Test that we can import the main project modules."""
    try:
        import config
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import config module: {e}")
    
    # Skip other modules that have complex dependencies for now
    # This allows the testing infrastructure to be validated independently


def test_torch_geometric_import():
    """Test that torch_geometric can be imported (when available)."""
    try:
        import torch_geometric
        assert hasattr(torch_geometric, '__version__')
    except ImportError:
        pytest.skip("torch_geometric not available, skipping test")