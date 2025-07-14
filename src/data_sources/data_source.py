"""Abstract base class for game requirements data sources."""
from abc import ABC, abstractmethod
from typing import Optional
from src.data_sources.game_requirements_model import GameRequirements

class DataSource(ABC):
    """Abstract base class for game requirements data sources."""
    
    @abstractmethod
    async def fetch(self, game_name: str) -> Optional[GameRequirements]:
        """Fetch game requirements from the source."""
        pass