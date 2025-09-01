"""Shared pytest fixtures for the test suite."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for testing."""
    temp_file = temp_dir / "test_file.txt"
    temp_file.write_text("test content")
    return temp_file


@pytest.fixture
def sample_config():
    """Provide a sample configuration dictionary for testing."""
    return {
        "model": {
            "hidden_dim": 128,
            "latent_dim": 64,
            "num_layers": 3,
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_epochs": 100,
        },
        "data": {
            "dataset_name": "test_dataset",
            "num_nodes": 1000,
        }
    }


@pytest.fixture
def mock_torch_device():
    """Mock torch device for consistent testing."""
    with patch('torch.cuda.is_available', return_value=False):
        yield torch.device('cpu')


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(10, 5)


@pytest.fixture
def sample_graph_data():
    """Create sample graph data for testing."""
    # Simple graph with 5 nodes and some edges
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], 
                               [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    x = torch.randn(5, 3)  # 5 nodes with 3 features each
    return {
        'x': x,
        'edge_index': edge_index,
        'num_nodes': 5,
        'num_edges': edge_index.size(1)
    }


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock = Mock()
    mock.forward.return_value = (torch.randn(5, 2), torch.randn(5, 2))  # mean, log_var
    mock.encode.return_value = torch.randn(5, 2)
    mock.decode.return_value = torch.randn(5, 3)
    mock.parameters.return_value = [torch.randn(10, 5, requires_grad=True)]
    return mock


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for testing."""
    mock = Mock()
    mock.step.return_value = None
    mock.zero_grad.return_value = None
    return mock


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    mock = Mock()
    mock.__len__.return_value = 100
    mock.__getitem__.return_value = {
        'x': torch.randn(10, 3),
        'edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    }
    return mock


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible testing."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture and return logs for testing."""
    return caplog


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing without actual I/O."""
    with patch('builtins.open', mock_open(read_data="test data")):
        with patch('os.path.exists', return_value=True):
            with patch('os.makedirs'):
                yield


def mock_open(read_data=""):
    """Helper function to create a mock open function."""
    from unittest.mock import mock_open as _mock_open
    return _mock_open(read_data=read_data)


@pytest.fixture
def isolated_environment(monkeypatch, temp_dir):
    """Create an isolated environment for testing."""
    # Change to temporary directory
    monkeypatch.chdir(temp_dir)
    
    # Set environment variables
    monkeypatch.setenv("TEST_MODE", "true")
    
    yield temp_dir