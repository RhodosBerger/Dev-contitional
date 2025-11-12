"""
Database models for Dev-conditional Server Engine
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Workflow(Base):
    """Workflow model for storing node-based workflows"""
    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    workflow_data = Column(JSON, nullable=False)  # Node graph data
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    executions = relationship("WorkflowExecution", back_populates="workflow")
    generated_projects = relationship("GeneratedProject", back_populates="workflow")


class WorkflowExecution(Base):
    """Workflow execution model"""
    __tablename__ = "workflow_executions"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("workflows.id"), nullable=False)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

    # Relationships
    workflow = relationship("Workflow", back_populates="executions")


class GeneratedProject(Base):
    """Generated project model"""
    __tablename__ = "generated_projects"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("workflows.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    project_type = Column(String(100), nullable=False)  # fastapi, react, etc.
    template_used = Column(String(255))
    generated_code = Column(JSON)  # Structure of generated files
    file_path = Column(String(500))  # Path to generated project
    download_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    workflow = relationship("Workflow", back_populates="generated_projects")


class LLMConversation(Base):
    """LLM conversation history for context"""
    __tablename__ = "llm_conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    llm_response = Column(Text)
    context = Column(JSON)  # Additional context for the conversation
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class NodeTemplate(Base):
    """Node templates for workflow designer"""
    __tablename__ = "node_templates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)  # trigger, logic, action, llm
    node_type = Column(String(100), nullable=False)
    description = Column(Text)
    configuration_schema = Column(JSON)  # JSON schema for node configuration
    icon = Column(String(100))  # Icon name or path
    created_at = Column(DateTime(timezone=True), server_default=func.now())