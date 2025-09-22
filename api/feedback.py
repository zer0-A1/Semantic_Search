from fastapi import APIRouter, HTTPException
from database.database import Feedback, get_session
from database.database import SearchResult as ORMSearchResult
from sqlalchemy import select
from database.schemas import FeedbackRequest, FeedbackResponse
import uuid
from datetime import datetime

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback_request: FeedbackRequest):
    """
    Submit feedback for a search result.
    
    Action types:
    - "keep": User wants to keep this result
    - "reject": User wants to reject this result  
    - "compare": User wants to compare this result with others
    """
    try:
        # Validate action_type
        valid_actions = ["keep", "reject", "compare"]
        if feedback_request.action_type not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action_type. Must be one of: {valid_actions}")

        # Generate feedback ID and timestamp
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        async for session in get_session():
            try:
                # Verify result_id exists and matches query_id
                stmt = select(ORMSearchResult).where(
                    ORMSearchResult.id == feedback_request.result_id)
                result = await session.execute(stmt)
                db_result = result.scalar_one_or_none()

                if not db_result:
                    raise HTTPException(status_code=400,
                                        detail="Invalid result_id: not found")

                if db_result.query_id != feedback_request.query_id:
                    raise HTTPException(
                        status_code=400,
                        detail="result_id does not belong to query_id")

                # Create feedback record
                feedback = Feedback(id=feedback_id,
                                    query_id=feedback_request.query_id,
                                    result_id=feedback_request.result_id,
                                    action_type=feedback_request.action_type,
                                    timestamp=timestamp)

                session.add(feedback)
                await session.commit()

                return FeedbackResponse(
                    status="success",
                    message=
                    f"Feedback submitted successfully for action: {feedback_request.action_type}",
                    feedback_id=feedback_id)

            except Exception as e:
                await session.rollback()
                raise e

    except HTTPException:
        raise
    except Exception as e:
        return FeedbackResponse(status="failure",
                                message=f"Failed to submit feedback: {str(e)}")


# @router.get("/feedback/query/{query_id}")
# async def get_feedback_for_query(query_id: str):
#     """Get all feedback for a specific search query."""
#     try:
#         async for session in get_session():
#             # Get all feedback for this query
#             stmt = select(Feedback).where(Feedback.query_id == query_id)
#             result = await session.execute(stmt)
#             feedbacks = result.scalars().all()

#             return {
#                 "query_id":
#                 query_id,
#                 "feedback_count":
#                 len(feedbacks),
#                 "feedbacks": [{
#                     "id": feedback.id,
#                     "result_id": feedback.result_id,
#                     "action_type": feedback.action_type,
#                     "timestamp": feedback.timestamp
#                 } for feedback in feedbacks]
#             }

#     except Exception as e:
#         raise HTTPException(status_code=500,
#                             detail=f"Failed to get feedback: {str(e)}")

# @router.get("/feedback/stats")
# async def get_feedback_stats(query_id: Optional[str] = Query(
#     None, description="Filter by specific query ID"),
#                              action_type: Optional[str] = Query(
#                                  None, description="Filter by action type")):
#     """Get feedback statistics."""
#     try:
#         async for session in get_session():
#             # Build query with optional filters
#             stmt = select(Feedback)

#             if query_id:
#                 stmt = stmt.where(Feedback.query_id == query_id)
#             if action_type:
#                 stmt = stmt.where(Feedback.action_type == action_type)

#             result = await session.execute(stmt)
#             feedbacks = result.scalars().all()

#             # Calculate statistics
#             total_feedback = len(feedbacks)
#             action_counts = {}

#             for feedback in feedbacks:
#                 action = feedback.action_type
#                 action_counts[action] = action_counts.get(action, 0) + 1

#             return {
#                 "total_feedback": total_feedback,
#                 "action_breakdown": action_counts,
#                 "filters_applied": {
#                     "query_id": query_id,
#                     "action_type": action_type
#                 }
#             }

#     except Exception as e:
#         raise HTTPException(status_code=500,
#                             detail=f"Failed to get feedback stats: {str(e)}")

# @router.delete("/feedback/{feedback_id}")
# async def delete_feedback(feedback_id: str):
#     """Delete a specific feedback record."""
#     try:
#         async for session in get_session():
#             # Find the feedback record
#             stmt = select(Feedback).where(Feedback.id == feedback_id)
#             result = await session.execute(stmt)
#             feedback = result.scalar_one_or_none()

#             if not feedback:
#                 raise HTTPException(status_code=404,
#                                     detail="Feedback not found")

#             # Delete the feedback
#             await session.delete(feedback)
#             await session.commit()

#             return {
#                 "status": "success",
#                 "message": f"Feedback {feedback_id} deleted successfully"
#             }

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500,
#                             detail=f"Failed to delete feedback: {str(e)}")
