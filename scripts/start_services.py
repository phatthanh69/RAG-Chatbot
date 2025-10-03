#!/usr/bin/env python3
"""
Script to start database services using Docker Compose
"""

import subprocess
import sys
import time
from pathlib import Path
import logging


def start_database():
    """Start PostgreSQL and pgAdmin using Docker Compose"""
    logging.info("Starting database services...")

    try:
        # Start services
        result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logging.info("Database services started successfully!")
            print("\nServices:")
            print("  - PostgreSQL: localhost:5432")
            print("  - pgAdmin: http://localhost:5050")
            print("    Username: admin@rag.com")
            print("    Password: admin")
            logging.info("Waiting for services to be ready...")

            # Wait a bit for services to start
            time.sleep(10)

            # Check if services are running
            check_result = subprocess.run(
                ["docker", "compose", "ps"],
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True
            )

            if check_result.returncode == 0:
                print("Services are running:")
                print(check_result.stdout)
            else:
                logging.warning("Could not check service status")

        else:
            logging.error("Failed to start services:")
            print(result.stderr)
            sys.exit(1)

    except FileNotFoundError:
        logging.error("Docker Compose not found. Please install Docker and Docker Compose.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error starting services: {e}")
        sys.exit(1)


def stop_database():
    """Stop database services"""
    logging.info("Stopping database services...")

    try:
        result = subprocess.run(
            ["docker", "compose", "down"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logging.info("Database services stopped successfully!")
        else:
            logging.error("Failed to stop services:")
            print(result.stderr)
            sys.exit(1)

    except Exception as e:
        logging.error(f"Error stopping services: {e}")
        sys.exit(1)


def restart_database():
    """Restart database services"""
    logging.info("Restarting database services...")
    stop_database()
    time.sleep(2)
    start_database()


def main():
    """Main function"""
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python start_services.py [start|stop|restart]")
        print("  start: Start database services")
        print("  stop: Stop database services")
        print("  restart: Restart database services")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'start':
        start_database()
    elif command == 'stop':
        stop_database()
    elif command == 'restart':
        restart_database()
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
