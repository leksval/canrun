#!/usr/bin/env python3
"""
Test Enhanced LLM Integration for CanRun G-Assist Plugin
Tests the new structured LLM analysis context and integration.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.compatibility_analyzer import CompatibilityAnalyzer
from src.rtx_llm_analyzer import RTXLLMAnalyzer
from src.privacy_aware_hardware_detector import PrivacyAwareHardwareDetector
from src.game_requirements_fetcher import GameRequirementsFetcher
from src.data_sources.game_requirements_model import GameRequirements
from src.data_sources.compatibility_model import CompatibilityAnalysis, ComponentAnalysis, CompatibilityLevel, BottleneckType, ComponentType

async def test_enhanced_llm_integration():
    """Test the enhanced LLM integration with structured context."""
    print("🧪 Testing Enhanced LLM Integration...")
    
    try:
        # Initialize components
        print("📋 Initializing components...")
        llm_analyzer = RTXLLMAnalyzer()
        compatibility_analyzer = CompatibilityAnalyzer(llm_analyzer=llm_analyzer)
        hardware_detector = PrivacyAwareHardwareDetector()
        requirements_fetcher = GameRequirementsFetcher()
        
        # Get hardware specs
        print("🖥️ Detecting hardware...")
        hardware_specs = await hardware_detector.get_hardware_specs()
        print(f"   GPU: {hardware_specs.gpu_model}")
        print(f"   VRAM: {hardware_specs.gpu_vram_gb} GB")
        print(f"   CPU: {hardware_specs.cpu_model}")
        print(f"   RAM: {hardware_specs.ram_total_gb} GB")
        
        # Test game
        game_name = "Cyberpunk 2077"
        print(f"\n🎮 Testing with: {game_name}")
        
        # Get game requirements
        print("📊 Fetching game requirements...")
        requirements = await requirements_fetcher.fetch_requirements(game_name)
        
        if not requirements:
            print("⚠️ Creating mock requirements for testing...")
            requirements = GameRequirements(
                game_name=game_name,
                minimum_cpu="Intel Core i5-3570K",
                minimum_gpu="NVIDIA GTX 780",
                minimum_ram_gb=8,
                minimum_vram_gb=3,
                minimum_storage_gb=70,
                minimum_directx="DirectX 12",
                minimum_os="Windows 10",
                recommended_cpu="Intel Core i7-4790",
                recommended_gpu="NVIDIA RTX 2060",
                recommended_ram_gb=12,
                recommended_vram_gb=6,
                recommended_storage_gb=70,
                recommended_directx="DirectX 12",
                recommended_os="Windows 10",
                source="Mock Data"
            )
        
        print(f"   Source: {requirements.source}")
        print(f"   Min GPU: {requirements.minimum_gpu}")
        print(f"   Rec GPU: {requirements.recommended_gpu}")
        
        # Run compatibility analysis
        print("\n🔍 Running compatibility analysis...")
        analysis = await compatibility_analyzer.analyze_compatibility(hardware_specs, requirements)
        
        print(f"   Overall compatibility: {analysis.overall_compatibility.value}")
        print(f"   Can run minimum: {analysis.can_run_minimum}")
        print(f"   Can run recommended: {analysis.can_run_recommended}")
        print(f"   Overall score: {analysis.overall_score}")
        
        # Test enhanced LLM analysis context
        print("\n🤖 Testing enhanced LLM analysis context...")
        llm_context = await compatibility_analyzer.get_llm_analysis_context(
            game_name, hardware_specs, requirements, analysis
        )
        
        # Verify context structure
        print("✅ Verifying LLM context structure...")
        assert 'game_name' in llm_context
        assert 'hardware_specs' in llm_context
        assert 'game_requirements' in llm_context
        assert 'compatibility_analysis' in llm_context
        
        print(f"   Game name: {llm_context['game_name']}")
        print(f"   Hardware GPU: {llm_context['hardware_specs']['gpu_model']}")
        print(f"   Requirements source: {llm_context['game_requirements']['source']}")
        print(f"   Analysis score: {llm_context['compatibility_analysis']['overall_score']}")
        
        # Check for LLM analysis if available
        if 'llm_analysis' in llm_context:
            llm_analysis = llm_context['llm_analysis']
            print(f"   LLM confidence: {llm_analysis.get('confidence_score', 'N/A')}")
            print(f"   LLM processing time: {llm_analysis.get('processing_time_ms', 'N/A')} ms")
            print(f"   G-Assist used: {llm_analysis.get('g_assist_used', 'N/A')}")
            
            if 'analysis_text' in llm_analysis:
                analysis_text = llm_analysis['analysis_text']
                print(f"   LLM analysis preview: {analysis_text[:100]}...")
        else:
            print("   LLM analysis: Not available (expected in some environments)")
        
        # Test structured data extraction
        print("\n📊 Testing structured data extraction...")
        hardware_data = llm_context['hardware_specs']
        requirements_data = llm_context['game_requirements']
        analysis_data = llm_context['compatibility_analysis']
        
        # Verify all required fields are present
        required_hardware_fields = ['gpu_model', 'gpu_vram_gb', 'cpu_model', 'ram_total_gb']
        for field in required_hardware_fields:
            assert field in hardware_data, f"Missing hardware field: {field}"
            print(f"   ✓ Hardware {field}: {hardware_data[field]}")
        
        required_requirements_fields = ['minimum', 'recommended', 'source']
        for field in required_requirements_fields:
            assert field in requirements_data, f"Missing requirements field: {field}"
            print(f"   ✓ Requirements {field}: Available")
        
        required_analysis_fields = ['overall_compatibility', 'can_run_minimum', 'overall_score']
        for field in required_analysis_fields:
            assert field in analysis_data, f"Missing analysis field: {field}"
            print(f"   ✓ Analysis {field}: {analysis_data[field]}")
        
        # Test component analyses structure
        if 'component_analyses' in analysis_data:
            component_analyses = analysis_data['component_analyses']
            print(f"   ✓ Component analyses: {len(component_analyses)} components")
            
            for comp in component_analyses[:3]:  # Show first 3
                print(f"     - {comp['component']}: Score {comp['score']}, Meets min: {comp['meets_minimum']}")
        
        print("\n✅ Enhanced LLM Integration Test PASSED!")
        print("🎯 All structured data fields verified")
        print("🤖 LLM analysis context creation successful")
        print("📊 Comprehensive compatibility data available for LLM processing")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced LLM Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_plugin_llm_integration():
    """Test LLM integration in the plugin context."""
    print("\n🔌 Testing Plugin LLM Integration...")
    
    try:
        # Import plugin components
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from plugin import initialize_enhanced_plugin, handle_enhanced_compatibility_check, SteamCompareUI, EnhancedResponseFormatter
        
        # Initialize plugin
        print("🚀 Initializing plugin...")
        init_result = initialize_enhanced_plugin()
        
        if not init_result["success"]:
            print(f"❌ Plugin initialization failed: {init_result['message']}")
            return False
        
        print("✅ Plugin initialized successfully")
        
        # Test compatibility check with LLM integration
        print("🎮 Testing compatibility check with LLM...")
        
        # Import global components
        from plugin import canrun_engine, requirements_fetcher
        
        steam_compare = SteamCompareUI(requirements_fetcher)
        formatter = EnhancedResponseFormatter()
        
        # Test parameters
        params = {
            "game_name": "Diablo IV",
            "show_steam_data": True
        }
        
        print(f"   Testing with: {params['game_name']}")
        
        # Run enhanced compatibility check
        result = await handle_enhanced_compatibility_check(params, steam_compare, formatter)
        
        print(f"   Success: {result.get('success', False)}")
        
        if result.get('success'):
            message = result.get('message', '')
            print(f"   Response length: {len(message)} characters")
            
            # Check for LLM insights in response
            if "AI INSIGHTS" in message:
                print("   ✅ LLM insights included in response")
            else:
                print("   ℹ️ LLM insights not included (may be expected)")
            
            # Check for performance analysis
            if "PERFORMANCE ANALYSIS" in message:
                print("   ✅ Performance analysis included")
            
            # Check for Steam data
            if "STEAM" in message:
                print("   ✅ Steam data included")
            
            print(f"   Response preview: {message[:200]}...")
        else:
            print(f"   ❌ Compatibility check failed: {result.get('message', 'Unknown error')}")
            return False
        
        print("\n✅ Plugin LLM Integration Test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Plugin LLM Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all enhanced LLM integration tests."""
    print("🧪 Enhanced LLM Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Enhanced LLM Integration", test_enhanced_llm_integration),
        ("Plugin LLM Integration", test_plugin_llm_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        print("-" * 30)
        
        try:
            if await test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Enhanced LLM Integration tests PASSED!")
        return True
    else:
        print("⚠️ Some Enhanced LLM Integration tests FAILED!")
        return False

if __name__ == "__main__":
    asyncio.run(main())