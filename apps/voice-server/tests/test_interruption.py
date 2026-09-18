import pytest
from services.pipecat_pipeline.bot import LangGraphAdapter
from pipecat.frames.frames import UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

@pytest.mark.asyncio
async def test_buffer_clears_on_interruption():
    # Mock transport and app
    adapter = LangGraphAdapter(None, "test-sid", None)
    
    # Simulate some speech
    adapter._buffer.append("Hello there")
    assert len(adapter._buffer) == 1
    
    # Simulate interrupt
    interrupt_frame = UserStartedSpeakingFrame()
    # Note: in reality process_frame requires downstream direction, but we are testing internal state
    await adapter.process_frame(interrupt_frame, FrameDirection.DOWNSTREAM) # Downstream
    
    assert len(adapter._buffer) == 0
    assert adapter.was_interrupted == True
