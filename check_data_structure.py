"""
Check the actual structure of data files to understand field names
"""

import json
import os

def check_game_requirements_structure():
    """Check what fields are actually present in game requirements"""
    print("="*60)
    print("CHECKING GAME REQUIREMENTS DATA STRUCTURE")
    print("="*60)
    
    if not os.path.exists("data/game_requirements.json"):
        print("❌ game_requirements.json not found!")
        return
    
    try:
        with open("data/game_requirements.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("📁 Top-level keys:", list(data.keys()))
        
        # Check if there's a nested games structure
        if "games" in data:
            games_data = data["games"]
            print("📁 Found nested 'games' structure")
        else:
            games_data = data
            print("📁 Direct games structure")
        
        # Get first game to check structure
        first_game = None
        for key, value in games_data.items():
            if key != "comment" and isinstance(value, dict):
                first_game = key
                break
        
        if first_game:
            print(f"🎮 Checking structure of '{first_game}':")
            game_data = games_data[first_game]
            
            print(f"   Game-level keys: {list(game_data.keys())}")
            
            if "minimum" in game_data:
                print(f"   Minimum requirements keys: {list(game_data['minimum'].keys())}")
            
            if "recommended" in game_data:
                print(f"   Recommended requirements keys: {list(game_data['recommended'].keys())}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error reading game requirements: {e}")

def check_gpu_hierarchy_structure():
    """Check what fields are actually present in GPU hierarchy"""
    print("="*60)
    print("CHECKING GPU HIERARCHY DATA STRUCTURE")
    print("="*60)
    
    if not os.path.exists("data/gpu_hierarchy.json"):
        print("❌ gpu_hierarchy.json not found!")
        return
    
    try:
        with open("data/gpu_hierarchy.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("📁 Top-level keys:", list(data.keys()))
        
        # Check if there's a nested nvidia structure
        if "nvidia" in data:
            gpu_data = data["nvidia"]
            print("📁 Found nested 'nvidia' structure")
        else:
            gpu_data = data
            print("📁 Direct GPU structure")
        
        # Get first GPU to check structure
        first_gpu = None
        for key, value in gpu_data.items():
            if key != "comment" and isinstance(value, dict):
                first_gpu = key
                break
        
        if first_gpu:
            print(f"🎮 Checking structure of '{first_gpu}':")
            gpu_info = gpu_data[first_gpu]
            
            print(f"   GPU fields: {list(gpu_info.keys())}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error reading GPU hierarchy: {e}")

if __name__ == "__main__":
    check_game_requirements_structure()
    check_gpu_hierarchy_structure()