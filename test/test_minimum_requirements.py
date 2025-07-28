"""Test the new minimum requirements logic in CanRun."""

import asyncio
import sys
sys.path.append('src')

from canrun_engine import CanRunEngine

async def test_minimum_requirements_logic():
    """Test the new minimum requirements messaging."""
    
    # Initialize engine with LLM disabled for testing
    engine = CanRunEngine(enable_llm=False)
    
    # Test with Diablo 4 (should exceed recommended for RTX 4090 system)
    print("Testing Diablo 4 compatibility...")
    result = await engine.check_game_compatibility("Diablo 4")
    
    # Test all the new methods
    print("\n=== MINIMUM REQUIREMENTS STATUS ===")
    min_req_status = result.get_minimum_requirements_status()
    print(f"Can run game: {min_req_status['can_run_game']}")
    print(f"Overall status: {min_req_status['overall_status']}")
    print(f"Summary message: {min_req_status['summary_message']}")
    
    print("\n=== COMPONENT DETAILS ===")
    for component in min_req_status['meeting_components']:
        print(f"✅ {component['component']}: {component['status']}")
    
    for component in min_req_status['failing_components']:
        print(f"❌ {component['component']}: {component['status']}")
        print(f"   Suggestion: {component['upgrade_suggestion']}")
    
    print("\n=== SIMPLE STATUS METHODS ===")
    print(f"Runnable status: {result.get_runnable_status_message()}")
    print(f"Can run minimum: {result.can_run_game()}")
    print(f"Exceeds recommended: {result.exceeds_recommended_requirements()}")
    
    print("\n=== FULL COMPATIBILITY ANALYSIS ===")
    print(f"Game: {result.compatibility_analysis.game_name}")
    print(f"Overall compatibility: {result.compatibility_analysis.overall_compatibility}")
    print(f"Overall score: {result.compatibility_analysis.overall_score:.2f}")
    print(f"Summary: {result.compatibility_analysis.summary}")
    
    # Test performance prediction
    print("\n=== PERFORMANCE PREDICTION ===")
    perf = result.performance_prediction
    print(f"Performance Tier: {perf.tier.name}")
    print(f"Tier Description: {perf.tier_description}")
    print(f"Performance Score: {perf.score}/100")
    print(f"Expected FPS: {perf.expected_fps}")
    print(f"Recommended Settings: {perf.recommended_settings}")
    print(f"Recommended Resolution: {perf.recommended_resolution}")
    print(f"Upgrade Suggestions: {perf.upgrade_suggestions}")

if __name__ == "__main__":
    asyncio.run(test_minimum_requirements_logic())