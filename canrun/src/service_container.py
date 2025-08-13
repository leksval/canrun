"""
Service Container for CanRun Dependency Injection
Manages all service dependencies to eliminate circular imports.
"""

import logging
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod


class ServiceContainer:
    """
    Dependency injection container for CanRun services.
    Manages service instances and their dependencies.
    """
    
    def __init__(self):
        """Initialize the service container."""
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._singletons: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_singleton(self, name: str, factory: Callable[[], Any]) -> None:
        """
        Register a singleton service factory.
        
        Args:
            name: Service name
            factory: Factory function to create the service
        """
        self._factories[name] = factory
        self.logger.debug(f"Registered singleton service: {name}")
    
    def register_instance(self, name: str, instance: Any) -> None:
        """
        Register a service instance directly.
        
        Args:
            name: Service name
            instance: Service instance
        """
        self._services[name] = instance
        self.logger.debug(f"Registered service instance: {name}")
    
    def get(self, name: str) -> Any:
        """
        Get a service instance by name.
        
        Args:
            name: Service name
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service is not registered
        """
        # Check if instance already exists
        if name in self._services:
            return self._services[name]
        
        # Check if singleton already created
        if name in self._singletons:
            return self._singletons[name]
        
        # Create singleton from factory
        if name in self._factories:
            instance = self._factories[name]()
            self._singletons[name] = instance
            self.logger.debug(f"Created singleton service: {name}")
            return instance
        
        raise KeyError(f"Service '{name}' not registered")
    
    def has(self, name: str) -> bool:
        """
        Check if a service is registered.
        
        Args:
            name: Service name
            
        Returns:
            True if service is registered
        """
        return (name in self._services or 
                name in self._singletons or 
                name in self._factories)
    
    def clear(self) -> None:
        """Clear all services and factories."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self.logger.debug("Cleared all services from container")


class ServiceProvider(ABC):
    """
    Abstract base class for service providers.
    Services can inherit from this to access the container.
    """
    
    def __init__(self, container: ServiceContainer):
        """
        Initialize service provider.
        
        Args:
            container: Service container instance
        """
        self.container = container
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the service. Must be implemented by subclasses."""
        pass


# Global service container instance
_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """
    Get the global service container instance.
    
    Returns:
        Global service container
    """
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def reset_container() -> None:
    """Reset the global service container."""
    global _container
    _container = None


def inject(service_name: str) -> Callable[[Callable], Callable]:
    """
    Decorator to inject a service into a function or method.
    
    Args:
        service_name: Name of the service to inject
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            container = get_container()
            service = container.get(service_name)
            return func(service, *args, **kwargs)
        return wrapper
    return decorator