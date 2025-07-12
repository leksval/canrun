# Contributing to CanRun

Thank you for your interest in contributing to CanRun! This document provides guidelines for contributing to the project.

## 🚀 Quick Start for Contributors

### Setting Up Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/canrun.git
   cd canrun
   ```

2. **Install Dependencies**
   ```bash
   python setup.py
   ```

3. **Run Tests**
   ```bash
   python test/run_all_tests.py
   ```

## 🎯 Areas for Contribution

### High Priority
- **New Game Support**: Add requirements for popular games
- **Hardware Detection**: Improve GPU/CPU detection accuracy
- **Performance Prediction**: Enhance FPS prediction algorithms
- **Optimization Suggestions**: Add more actionable recommendations

### Medium Priority
- **UI/UX**: Improve G-Assist response formatting
- **Error Handling**: Better error messages and recovery
- **Documentation**: Code comments and user guides
- **Testing**: Additional test cases and edge cases

### Low Priority
- **Implement MCP tool version**: Support for architecture agnostic version
- **Internationalization**: Support for multiple languages

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings for all functions and classes
- Keep functions focused and under 50 lines when possible

### Example Code Style
```python
async def analyze_game_compatibility(
    self, 
    game_name: str, 
    settings: str = "High", 
    resolution: str = "1080p"
) -> Optional[Dict[str, Any]]:
    """
    Analyze game compatibility for the given parameters.
    
    Args:
        game_name: Name of the game to analyze
        settings: Graphics settings (Low, Medium, High, Ultra)
        resolution: Display resolution (1080p, 1440p, 4K)
        
    Returns:
        Dictionary containing compatibility analysis or None if failed
    """
    # Implementation here
```

### Testing Requirements
- All new features must include tests
- Tests should cover both success and failure cases
- Integration tests for G-Assist plugin functionality
- Performance tests for critical paths

### Documentation Requirements
- Update README.md for new features
- Add docstrings to all public functions
- Include usage examples
- Update API documentation

## 🔧 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Write clean, well-documented code
- Add appropriate tests
- Update documentation

### 3. Test Your Changes
```bash
# Run all tests
python test/run_all_tests.py

# Run specific test suite
python test/test_hardware_detection.py

# Test plugin functionality
python plugin/test.bat
```

### 4. Submit a Pull Request
- Describe your changes clearly
- Include before/after examples
- Reference any related issues

## 📊 Adding New Games

### Game Requirements Format
```json
{
  "game_name": {
    "minimum": {
      "cpu": {
        "name": "Intel Core i5-8400 / AMD Ryzen 5 1600",
        "cores": 6,
        "threads": 6,
        "base_clock": 2.8
      },
      "gpu": {
        "name": "NVIDIA GTX 1060 6GB / AMD RX 580 8GB",
        "memory": 6,
        "tier": "Medium"
      },
      "memory": {
        "total": 12
      },
      "storage": {
        "space": 70,
        "type": "SSD"
      }
    },
    "recommended": {
      // Similar structure for recommended requirements
    }
  }
}
```

### Adding a New Game
1. Research official system requirements
2. Add entry to `data/game_requirements.json`
3. Test compatibility analysis
4. Add test case in `test/test_compatibility_analysis.py`

## 🔍 Hardware Detection

### Adding New Hardware Support
```python
class NewHardwareDetector:
    async def detect_new_component(self) -> Dict[str, Any]:
        """Detect new hardware component."""
        try:
            # Detection logic here
            return {
                "name": "Component Name",
                "model": "Model Number",
                "specifications": {}
            }
        except Exception as e:
            self.logger.error(f"Failed to detect component: {e}")
            return {}
```

### GPU Performance Data
To add new GPU performance data:
1. Add GPU entry to `data/gpu_hierarchy.json`
2. Include performance tier, memory, and features
3. Add benchmark data if available
4. Update tests to include new GPU

## 🚨 Bug Reports

### Before Reporting
- Check existing issues
- Ensure you're using the latest version
- Test with minimal reproduction case

### Bug Report Template
```
**Bug Description**
Clear description of the issue

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: Windows 10/11
- Python version: 3.8+
- GPU: NVIDIA RTX 3080
- CanRun version: 1.0.0

**Logs**
Include relevant log output
```

## 📈 Performance Considerations

### Optimization Guidelines
- Use async/await for I/O operations
- Cache expensive computations
- Minimize API calls
- Profile critical paths

### Performance Testing
```python
import time
import asyncio

async def test_performance():
    start_time = time.time()
    result = await engine.analyze_game_compatibility("Test Game")
    end_time = time.time()
    
    assert (end_time - start_time) < 3.0, "Analysis took too long"
    assert result is not None, "Analysis failed"
```

## 🎖️ Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- Project documentation

## 📞 Getting Help

- Create an issue for questions
- Join discussions in pull requests
- Review existing code for patterns

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to CanRun! Together, we can make gaming more accessible for everyone.