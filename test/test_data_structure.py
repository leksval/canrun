#!/usr/bin/env python3

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from canrun.src.game_requirements_fetcher import GameRequirements

def test_game_requirements_dataclass():
    """Test that GameRequirements dataclass works with individual fields."""
    try:
        # Test creating GameRequirements with new field format
        requirements = GameRequirements(
            game_name="Test Game",
            minimum_cpu="Intel i5",
            minimum_gpu="GTX 1060",
            minimum_ram_gb=8,
            minimum_vram_gb=4,
            minimum_storage_gb=50,
            minimum_directx="DirectX 11",
            minimum_os="Windows 10",
            recommended_cpu="Intel i7",
            recommended_gpu="RTX 3070",
            recommended_ram_gb=16,
            recommended_vram_gb=8,
            recommended_storage_gb=100,
            recommended_directx="DirectX 12",
            recommended_os="Windows 11",
            source="Test",
            last_updated="123456789"
        )
        
        print("✅ GameRequirements dataclass creation: SUCCESS")
        print(f"Game: {requirements.game_name}")
        print(f"Min CPU: {requirements.minimum_cpu}")
        print(f"Min GPU: {requirements.minimum_gpu}")
        print(f"Min RAM: {requirements.minimum_ram_gb}GB")
        
        assert True
        
    except Exception as e:
        print(f"❌ GameRequirements dataclass creation: FAILED")
        print(f"Error: {e}")
        assert False

if __name__ == "__main__":
    print("Testing GameRequirements dataclass structure...")
    success = test_game_requirements_dataclass()
    sys.exit(0 if success else 1)