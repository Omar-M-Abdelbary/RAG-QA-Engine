from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Query(Base):
    """Store query logs and analytics"""
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    success = Column(Boolean, default=True)
    response_time = Column(Float)  # Seconds
    quality_score = Column(Float)  # From validator
    num_contexts = Column(Integer)
    cached = Column(Boolean, default=False)
    similarity_score = Column(Float)  # Top result score
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Query(id={self.id}, question='{self.question[:30]}...')>"

class Feedback(Base):
    """Store user feedback on answers"""
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer)  # References queries.id
    rating = Column(Integer)  # 1-5 stars
    comment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)