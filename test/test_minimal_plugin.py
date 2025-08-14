#!/usr/bin/env python3
"""
Minimal G-Assist Plugin for Testing Communication
"""

import sys
import json
import logging

def setup_logging():
    """Setup logging for debugging."""
    logging.basicConfig(
        filename='minimal_plugin.log',
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding='utf-8'
    )

def read_command():
    """Read command from stdin."""
    try:
        line = sys.stdin.readline()
        logging.info(f'Raw input: {repr(line)}')
        if not line:
            logging.info('EOF received, shutting down')
            return "EOF"
            
        line = line.strip()
        if not line:
            logging.warning('Empty line received, continuing')
            return None
            
        command = json.loads(line)
        logging.info(f'Parsed command: {command}')
        return command
        
    except json.JSONDecodeError as e:
        logging.error(f'Invalid JSON received: {e}')
        return None
    except Exception as e:
        logging.error(f'Error in read_command: {e}')
        return None

def write_response(response):
    """Write response to stdout."""
    try:
        message = json.dumps(response) + '<<END>>'
        sys.stdout.write(message)
        sys.stdout.flush()
        logging.info(f'Response sent: {message}')
    except Exception as e:
        logging.error(f'Error writing response: {e}')

def is_g_assist_environment():
    """Check if running in G-Assist environment."""
    return not sys.stdin.isatty()

def main():
    """Main plugin execution loop."""
    setup_logging()
    logging.info("Minimal plugin started")
    
    # Check if running in G-Assist environment
    in_g_assist = is_g_assist_environment()
    logging.info(f"Running in G-Assist environment: {in_g_assist}")
    
    if not in_g_assist:
        print("This is a G-Assist plugin. Please run through G-Assist.")
        return
    
    # G-Assist protocol mode
    logging.info("Entering G-Assist protocol mode")
    while True:
        logging.info("Waiting for command...")
        command = read_command()
        
        if command == "EOF":
            logging.info("EOF received, exiting")
            break
        
        if command is None:
            logging.warning("Received None command, continuing...")
            continue
        
        logging.info(f"Processing command: {command}")
        
        # Handle G-Assist input
        if "tool_calls" in command:
            for tool_call in command.get("tool_calls", []):
                func = tool_call.get("func")
                params = tool_call.get("params", {})
                
                logging.info(f"Processing function: {func} with params: {params}")
                
                if func == "initialize":
                    response = {
                        "success": True,
                        "message": "Minimal plugin initialized successfully"
                    }
                    write_response(response)
                elif func == "canrun":
                    game_name = params.get("game_name", "Unknown")
                    response = {
                        "success": True,
                        "message": f"Minimal test: Would check compatibility for {game_name}"
                    }
                    write_response(response)
                elif func == "shutdown":
                    logging.info("Shutdown command received. Exiting.")
                    response = {
                        "success": True,
                        "message": "Plugin shutdown"
                    }
                    write_response(response)
                    return
                else:
                    logging.warning(f"Unknown function: {func}")
                    response = {
                        "success": False,
                        "message": f"Unknown function: {func}"
                    }
                    write_response(response)
        else:
            logging.warning(f"Unknown command format: {command}")
            response = {
                "success": False,
                "message": "Unknown command format"
            }
            write_response(response)

if __name__ == "__main__":
    main()