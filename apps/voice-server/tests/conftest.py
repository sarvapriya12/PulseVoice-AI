import sys
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from typing import AsyncGenerator

# 1. Setup Python Path for the Test Suite
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import application components AFTER path modification
from main import app
from core.models import Base

# 2. Test Client Fixture (for FastAPI routing tests)
@pytest.fixture(scope="module")
def client():
    '''Provides a TestClient for the FastAPI app.'''
    return TestClient(app)

# 3. Test Database Fixture (In-Memory SQLite)
@pytest.fixture
async def async_db_session() -> AsyncGenerator:
    '''Provides an isolated, in-memory SQLite database for testing db_tools.'''
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    
    # Create all tables in the memory db
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    TestSessionLocal = async_sessionmaker(bind=test_engine, expire_on_commit=False)
    
    async with TestSessionLocal() as session:
        yield session
        
    # Drop all tables after test
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
