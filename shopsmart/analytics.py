# Tracks user interactions in a database for reporting and dashboards
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from shopsmart.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class Interaction(Base):
    __tablename__ = "interaction"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String(100))
    user_query = Column(Text)
    bot_response = Column(Text)
    response_time_ms = Column(Float)
    products_retrieved = Column(Integer, default=0)
    category_detected = Column(String(100), nullable=True)


class AnalyticsTracker:
    """Manages the analytics database connection and logging"""

    def __init__(self):
        self.engine = create_engine(Config.ANALYTICS_DB_URL, echo=False)
        Base.metadata.create_all(self.engine)  # auto create table on first launch
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Analytics DB intialized: {Config.ANALYTICS_DB_URL}")

    def log_interaction(self, session_id, user_query, bot_response,
                        response_time_ms, products_retrieved=0,
                        category_detected=None):
        """Log a single user interaction to the database"""

        try:
            session = self.Session()
            interaction = Interaction(
                session_id=session_id,
                user_query=user_query,
                bot_response=bot_response,
                response_time_ms=response_time_ms,
                products_retrieved=products_retrieved,
                category_detected=category_detected
            )

            session.add(interaction)
            session.commit()
            session.close()
            logger.info(
                f"Logged interaction: query='{user_query[:50]}...' "
                f"time={response_time_ms:.0f}ms"
            )
        except Exception as e:
            logger.error(f"failed to log interaction: {e}")

    def get_total_queries(self):
        """Get total number of queries logged"""
        session = self.Session()
        count = session.query(Interaction).count()
        session.close()
        return count

    def get_recent_interactions(self, limit=20):
        """Return most recent interactions"""
        session = self.Session()
        results = (session.query(Interaction)
                   .order_by(Interaction.timestamp.desc())
                   .limit(limit)
                   .all())
        session.close()
        return results