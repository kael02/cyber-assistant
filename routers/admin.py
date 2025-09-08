from datetime import datetime 
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dependencies import get_assistant
from services import CyberQueryAssistant

router = APIRouter()
security = HTTPBearer()

def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify admin authentication token."""
    # In production, validate against your auth system
    if credentials.credentials != "admin-secret-token":
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return credentials


@router.get("/stats/detailed")
async def get_detailed_stats(
    admin_user = Depends(verify_admin_token),
    assistant: CyberQueryAssistant = Depends(get_assistant)
):
    """Get detailed system statistics (admin only)."""
    try:
        # Get comprehensive analytics
        accuracy_stats = assistant.analyze_query_patterns("accuracy", "month")
        format_stats = assistant.analyze_query_patterns("formats", "month")
        
        return {
            # "system_info": {
            #     "uptime": "24h 15m", 
            #     "memory_usage": "45%",
            #     "cpu_usage": "12%",
            #     "active_sessions": 15
            # },
            # "query_analytics": {
            #     "total_queries_this_month": 1250,
            #     "success_rate": "94.5%",
            #     "average_response_time": "1.2s",
            #     "most_common_formats": ["splunk", "elastic", "graylog"]
            # },
            "performance_metrics": {
                "accuracy_analysis": accuracy_stats[:300],
                "format_distribution": format_stats[:300]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/config/update")
async def update_configuration(
    admin_user = Depends(verify_admin_token),
    config_data: dict = None
):
    """Update system configuration (admin only)."""
    if not config_data:
        raise HTTPException(status_code=400, detail="Configuration data required")
    
    return {
        "message": "Configuration updated successfully",
        "updated_fields": list(config_data.keys()),
        "timestamp": datetime.now().isoformat()
    }