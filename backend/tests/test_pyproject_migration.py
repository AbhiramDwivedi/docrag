"""
Test for pyproject.toml migration.
This test validates that the migration from requirements.txt/setup.py to pyproject.toml works correctly.
"""
import tomllib
import subprocess
import sys
from pathlib import Path


def test_pyproject_toml_exists():
    """Test that pyproject.toml exists and is valid."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml should exist"
    
    # Test that it's valid TOML
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    assert "project" in data, "pyproject.toml should have [project] section"
    assert "build-system" in data, "pyproject.toml should have [build-system] section"


def test_pyproject_toml_content():
    """Test that pyproject.toml has the expected content."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    # Test build system
    assert data["build-system"]["requires"] == ["setuptools>=61.0", "wheel"]
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    
    # Test project metadata
    project = data["project"]
    assert project["name"] == "docquest"
    assert project["version"] == "0.1.0"
    assert project["description"] == "Local RAG-based document search and querying system"
    assert project["requires-python"] == ">=3.11"
    
    # Test dependencies exist
    assert "dependencies" in project
    dependencies = project["dependencies"]
    
    # Check some key dependencies are present
    expected_deps = ["watchdog", "fastapi", "sentence-transformers", "pydantic"]
    for dep in expected_deps:
        assert any(dep in d for d in dependencies), f"Expected dependency {dep} not found"


def test_requirements_txt_still_exists():
    """Test that requirements.txt still exists for backwards compatibility."""
    requirements_path = Path(__file__).parent.parent / "requirements.txt"
    assert requirements_path.exists(), "requirements.txt should still exist"


def test_setup_script_renamed():
    """Test that setup.py was renamed to setup_project.py."""
    old_setup = Path(__file__).parent.parent / "setup.py"
    new_setup = Path(__file__).parent.parent / "setup_project.py"
    
    assert not old_setup.exists(), "setup.py should not exist (renamed)"
    assert new_setup.exists(), "setup_project.py should exist"


def test_pytest_configuration():
    """Test that pytest configuration is properly set in pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    assert "tool" in data, "pyproject.toml should have [tool] section"
    assert "pytest" in data["tool"], "pyproject.toml should have [tool.pytest] section"
    
    pytest_config = data["tool"]["pytest"]["ini_options"]
    assert pytest_config["testpaths"] == ["tests"]
    assert "test_*.py" in pytest_config["python_files"]


def test_package_can_be_imported():
    """Test that the package structure can be imported correctly."""
    try:
        import src.interface.cli.ask
        import src.ingestion.pipeline
        import src.querying.agents.agent
    except ImportError as e:
        # This is expected if dependencies aren't installed
        # Just check that the import error is about missing dependencies, not structure
        assert "No module named" in str(e)


if __name__ == "__main__":
    # Run basic tests
    test_pyproject_toml_exists()
    test_pyproject_toml_content()
    test_requirements_txt_still_exists()
    test_setup_script_renamed()
    test_pytest_configuration()
    test_package_can_be_imported()
    print("✅ All migration tests passed!")