from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.infrastructure.database.models import Query, Feedback
from app.utils.logger import get_logger
from typing import List, Dict

logger = get_logger(__name__)

class QueryCRUD:
    """CRUD operations for queries"""
    
    @staticmethod
    def create_query(db: Session, query_data: Dict) -> Query:
        """Save a new query to database"""
        query = Query(**query_data)
        db.add(query)
        db.commit()
        db.refresh(query)
        logger.info(f"Saved query to database: ID={query.id}")
        return query
    
    @staticmethod
    def get_recent_queries(db: Session, limit: int = 10) -> List[Query]:
        """Get recent queries"""
        return db.query(Query).order_by(desc(Query.created_at)).limit(limit).all()
    
    @staticmethod
    def get_query_stats(db: Session) -> Dict:
        """Get analytics statistics"""
        total = db.query(Query).count()
        successful = db.query(Query).filter(Query.success == True).count()
        avg_response_time = db.query(func.avg(Query.response_time)).scalar() or 0
        avg_quality = db.query(func.avg(Query.quality_score)).scalar() or 0
        cached_count = db.query(Query).filter(Query.cached == True).count()
        
        return {
            'total_queries': total,
            'successful_queries': successful,
            'success_rate': successful / total if total > 0 else 0,
            'avg_response_time': round(avg_response_time, 3),
            'avg_quality_score': round(avg_quality, 3),
            'cache_hit_rate': cached_count / total if total > 0 else 0
        }
    
    @staticmethod
    def get_most_frequent_questions(db: Session, limit: int = 10) -> List[Dict]:
        """Get most frequently asked questions"""
        results = db.query(
            Query.question,
            func.count(Query.id).label('count')
        ).group_by(Query.question).order_by(desc('count')).limit(limit).all()
        
        return [{'question': q, 'count': c} for q, c in results]

class FeedbackCRUD:
    """CRUD operations for feedback"""
    
    @staticmethod
    def create_feedback(db: Session, query_id: int, rating: int, comment: str = None):
        """Save user feedback"""
        feedback = Feedback(
            query_id=query_id,
            rating=rating,
            comment=comment
        )
        db.add(feedback)
        db.commit()
        logger.info(f"Saved feedback for query {query_id}")
        return feedback