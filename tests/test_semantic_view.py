"""
Tests for semantic view utilities.

These tests verify the semantic view configuration management functions.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from deep_research_utils.semantic_view import (
    update_semantic_view_sample_values,
    validate_semantic_view_config,
    _get_dimension_sample_values
)


@pytest.fixture
def sample_yaml_config():
    """Sample semantic view configuration for testing."""
    return {
        'name': 'test_view',
        'description': 'Test semantic view',
        'tables': [
            {
                'name': 'test_table',
                'base_table': {
                    'database': 'TEST_DB',
                    'schema': 'TEST_SCHEMA',
                    'table': 'TEST_TABLE'
                },
                'dimensions': [
                    {
                        'name': 'status',
                        'description': 'Status field',
                        'expr': 'STATUS',
                        'data_type': 'string'
                    },
                    {
                        'name': 'category',
                        'description': 'Category field',
                        'expr': 'CATEGORY',
                        'data_type': 'string',
                        'sample_values': ['A', 'B']
                    }
                ]
            }
        ]
    }


@pytest.fixture
def temp_yaml_file(tmp_path, sample_yaml_config):
    """Create a temporary YAML file for testing."""
    yaml_file = tmp_path / "test_config.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(sample_yaml_config, f)
    return yaml_file


class TestValidateSemanticViewConfig:
    """Tests for validate_semantic_view_config function."""
    
    def test_valid_config(self, temp_yaml_file):
        """Test validation of a valid configuration."""
        result = validate_semantic_view_config(str(temp_yaml_file))
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert result['stats']['total_tables'] == 1
        assert result['stats']['total_dimensions'] == 2
        assert result['stats']['dimensions_with_sample_values'] == 1
    
    def test_missing_file(self):
        """Test validation with non-existent file."""
        result = validate_semantic_view_config("nonexistent.yaml")
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
    
    def test_invalid_yaml_structure(self, tmp_path):
        """Test validation with invalid YAML structure."""
        invalid_file = tmp_path / "invalid.yaml"
        with open(invalid_file, 'w') as f:
            f.write("not a dict: [1, 2, 3]")
        
        result = validate_semantic_view_config(str(invalid_file))
        
        assert result['valid'] is False
        assert any('dictionary' in error.lower() for error in result['errors'])
    
    def test_missing_tables_key(self, tmp_path):
        """Test validation with missing 'tables' key."""
        config_file = tmp_path / "no_tables.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'name': 'test'}, f)
        
        result = validate_semantic_view_config(str(config_file))
        
        assert result['valid'] is False
        assert any('tables' in error.lower() for error in result['errors'])


class TestGetDimensionSampleValues:
    """Tests for _get_dimension_sample_values helper function."""
    
    def test_dimension_with_few_values(self):
        """Test dimension with values under the limit."""
        mock_helper = Mock()
        
        count_df = pd.DataFrame({'UNIQUE_COUNT': [5]})
        values_df = pd.DataFrame({'VALUE': ['A', 'B', 'C', 'D', 'E']})
        
        mock_helper.execute_query_and_return_pandas_df.side_effect = [count_df, values_df]
        
        result = _get_dimension_sample_values(
            snowflake_helper=mock_helper,
            qualified_table='DB.SCHEMA.TABLE',
            column_expr='STATUS',
            max_unique_values=10,
            dimension_name='status'
        )
        
        assert result == ['A', 'B', 'C', 'D', 'E']
        assert mock_helper.execute_query_and_return_pandas_df.call_count == 2
    
    def test_dimension_with_too_many_values(self):
        """Test dimension with values over the limit."""
        mock_helper = Mock()
        
        count_df = pd.DataFrame({'UNIQUE_COUNT': [50]})
        mock_helper.execute_query_and_return_pandas_df.return_value = count_df
        
        result = _get_dimension_sample_values(
            snowflake_helper=mock_helper,
            qualified_table='DB.SCHEMA.TABLE',
            column_expr='ID',
            max_unique_values=10,
            dimension_name='id'
        )
        
        assert result is None
        assert mock_helper.execute_query_and_return_pandas_df.call_count == 1
    
    def test_dimension_with_zero_values(self):
        """Test dimension with no non-null values."""
        mock_helper = Mock()
        
        count_df = pd.DataFrame({'UNIQUE_COUNT': [0]})
        mock_helper.execute_query_and_return_pandas_df.return_value = count_df
        
        result = _get_dimension_sample_values(
            snowflake_helper=mock_helper,
            qualified_table='DB.SCHEMA.TABLE',
            column_expr='EMPTY_COL',
            max_unique_values=10,
            dimension_name='empty'
        )
        
        assert result is None
    
    def test_dimension_query_error(self):
        """Test handling of query errors."""
        mock_helper = Mock()
        mock_helper.execute_query_and_return_pandas_df.side_effect = Exception("Query failed")
        
        result = _get_dimension_sample_values(
            snowflake_helper=mock_helper,
            qualified_table='DB.SCHEMA.TABLE',
            column_expr='BAD_COL',
            max_unique_values=10,
            dimension_name='bad'
        )
        
        assert result is None


class TestUpdateSemanticViewSampleValues:
    """Tests for update_semantic_view_sample_values function."""
    
    @patch('deep_research_utils.semantic_view.SnowparkHelper')
    def test_successful_update(self, mock_snowpark_class, temp_yaml_file, tmp_path):
        """Test successful update of sample values."""
        output_file = tmp_path / "output.yaml"
        
        mock_helper = Mock()
        mock_snowpark_class.return_value = mock_helper
        
        count_df = pd.DataFrame({'UNIQUE_COUNT': [3]})
        values_df = pd.DataFrame({'VALUE': ['Active', 'Inactive', 'Pending']})
        
        mock_helper.execute_query_and_return_pandas_df.side_effect = [count_df, values_df]
        
        update_semantic_view_sample_values(
            yaml_path=str(temp_yaml_file),
            output_path=str(output_file),
            connection_type="programmatic",
            max_unique_values=10
        )
        
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            updated_config = yaml.safe_load(f)
        
        status_dim = updated_config['tables'][0]['dimensions'][0]
        assert 'sample_values' in status_dim
        assert status_dim['sample_values'] == ['Active', 'Inactive', 'Pending']
        
        mock_helper.close.assert_called_once()
    
    @patch('deep_research_utils.semantic_view.SnowparkHelper')
    def test_update_with_missing_base_table(self, mock_snowpark_class, tmp_path):
        """Test handling of tables without base_table definition."""
        yaml_file = tmp_path / "incomplete.yaml"
        output_file = tmp_path / "output.yaml"
        
        config = {
            'name': 'test',
            'tables': [
                {
                    'name': 'incomplete_table',
                    'dimensions': [
                        {'name': 'field1', 'expr': 'FIELD1'}
                    ]
                }
            ]
        }
        
        with open(yaml_file, 'w') as f:
            yaml.dump(config, f)
        
        mock_helper = Mock()
        mock_snowpark_class.return_value = mock_helper
        
        update_semantic_view_sample_values(
            yaml_path=str(yaml_file),
            output_path=str(output_file),
            max_unique_values=10
        )
        
        assert output_file.exists()
        
        mock_helper.execute_query_and_return_pandas_df.assert_not_called()
    
    def test_file_not_found(self):
        """Test error handling for missing input file."""
        with pytest.raises(FileNotFoundError):
            update_semantic_view_sample_values(
                yaml_path="nonexistent.yaml",
                output_path="output.yaml"
            )
    
    @patch('deep_research_utils.semantic_view.SnowparkHelper')
    def test_creates_output_directory(self, mock_snowpark_class, temp_yaml_file, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_file = tmp_path / "subdir" / "output.yaml"
        
        mock_helper = Mock()
        mock_snowpark_class.return_value = mock_helper
        mock_helper.execute_query_and_return_pandas_df.return_value = pd.DataFrame({'UNIQUE_COUNT': [100]})
        
        update_semantic_view_sample_values(
            yaml_path=str(temp_yaml_file),
            output_path=str(output_file),
            max_unique_values=10
        )
        
        assert output_file.parent.exists()
        assert output_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
