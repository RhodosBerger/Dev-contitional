"""
Jinja2 template filters for code generation
"""

def sql_type_filter(field_type):
    """Convert field type to SQLAlchemy column type"""
    type_mapping = {
        "string": "String(255)",
        "text": "Text",
        "integer": "Integer",
        "float": "Float",
        "boolean": "Boolean",
        "datetime": "DateTime(timezone=True)",
        "email": "String(255)",
        "url": "String(500)"
    }
    return type_mapping.get(field_type.lower(), "String(255)")


def pydantic_type_filter(field_type):
    """Convert field type to Pydantic type"""
    type_mapping = {
        "string": "str",
        "text": "str",
        "integer": "int",
        "float": "float",
        "boolean": "bool",
        "datetime": "datetime",
        "email": "str",
        "url": "str"
    }
    return type_mapping.get(field_type.lower(), "str")