#!/usr/bin/env python3
"""
Database setup and initialization script
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragbot.models.base import db
from ragbot.config import Config
from ragbot.app import create_app


def setup_database():
    """Initialize database and create tables"""
    logging.info("Setting up database...")

    # Create Flask app context
    app = create_app()

    with app.app_context():
        try:
            # Create all tables
            db.create_all()
            logging.info("Database tables created successfully!")

            # Verify connection
            connection = db.engine.connect()
            connection.close()
            logging.info("Database connection verified!")

        except Exception as e:
            logging.error(f"Error setting up database: {e}")
            sys.exit(1)


def reset_database():
    """Reset database by dropping all tables and recreating them"""
    logging.info("Resetting database...")

    app = create_app()

    with app.app_context():
        try:
            # Drop all tables
            db.drop_all()
            logging.info("All tables dropped!")

            # Recreate all tables
            db.create_all()
            logging.info("Database tables recreated successfully!")

        except Exception as e:
            logging.error(f"Error resetting database: {e}")
            sys.exit(1)


def show_database_info():
    """Show database information and statistics"""
    print("Database Information:")

    app = create_app()

    with app.app_context():
        try:
            # Get table information
            from ragbot.models.document import Document, DocumentChunk
            from ragbot.models.chat import ChatSession, ChatMessage

            tables = [
                ('Documents', Document.__tablename__, Document.query.count()),
                ('Document Chunks', DocumentChunk.__tablename__, DocumentChunk.query.count()),
                ('Chat Sessions', ChatSession.__tablename__, ChatSession.query.count()),
                ('Chat Messages', ChatMessage.__tablename__, ChatMessage.query.count()),
            ]

            print("\nTable Statistics:")
            for name, table_name, count in tables:
                print(f"  {name} ({table_name}): {count} records")

            print(f"\n🔗 Database URL: {app.config['SQLALCHEMY_DATABASE_URI']}")
            print(f"📍 Database Path: {app.config['SQLALCHEMY_DATABASE_URI'].replace('postgresql://', '')}")

        except Exception as e:
            logging.error(f"Error getting database info: {e}")
            sys.exit(1)


def main():
    """Main function"""
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python setup_database.py [setup|reset|info]")
        print("  setup: Create database tables")
        print("  reset: Drop and recreate all tables")
        print("  info: Show database information")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'setup':
        setup_database()
    elif command == 'reset':
        confirm = input("This will delete all data. Are you sure? (yes/no): ")
        if confirm.lower() == 'yes':
            reset_database()
        else:
            print("Operation cancelled.")
    elif command == 'info':
        show_database_info()
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
