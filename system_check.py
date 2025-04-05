#!/usr/bin/env python
"""
SentinelDocs - System Check Script

Validates the environment setup and checks if all required dependencies are installed.
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path
import platform
import time

# ASCII Art
SENTINEL_ASCII = """
  ╭──────────────────────────────────────────╮
  │      ___  ___ _ __ | |_(_)_ __   ___| |  │
  │     / __|/ _ \\ '_ \\| __| | '_ \\ / _ \\ |  │
  │     \\__ \\  __/ | | | |_| | | | |  __/ |  │
  │     |___/\\___|_| |_|\\__|_|_| |_|\\___|_|  │
  │                  DOCS                     │
  ╰──────────────────────────────────────────╯
"""

# Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_colored(text, color=RESET, bold=False):
    """Print colored text."""
    if bold:
        print(f"{BOLD}{color}{text}{RESET}")
    else:
        print(f"{color}{text}{RESET}")

def print_header(text):
    """Print a section header."""
    print("\n" + "=" * 60)
    print_colored(f" {text}", BLUE, bold=True)
    print("=" * 60)

def print_status(name, status, notes=None):
    """Print a status line."""
    dots = "." * max(1, 30 - len(name))
    if status == "OK":
        status_text = f"{GREEN}[  OK  ]{RESET}"
    elif status == "WARNING":
        status_text = f"{YELLOW}[WARNING]{RESET}"
    elif status == "ERROR":
        status_text = f"{RED}[ ERROR ]{RESET}"
    else:
        status_text = f"[ {status} ]"
    
    print(f"  {name} {dots} {status_text}")
    if notes:
        print(f"     {YELLOW}→ {notes}{RESET}")

def check_python_version():
    """Check Python version."""
    version = platform.python_version()
    major, minor, _ = version.split(".")
    if int(major) < 3 or (int(major) == 3 and int(minor) < 8):
        print_status("Python version", "ERROR", f"Found {version}. Python 3.8+ required.")
        return False
    print_status("Python version", "OK", f"Running Python {version}")
    return True

def check_package(package_name, min_version=None):
    """Check if a package is installed and meets minimum version requirements."""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, "__version__"):
            version = module.__version__
        elif hasattr(module, "VERSION"):
            version = module.VERSION
        else:
            version = "Unknown"
        
        if min_version and version != "Unknown":
            version_tuple = tuple(map(int, version.split(".")))
            min_version_tuple = tuple(map(int, min_version.split(".")))
            if version_tuple < min_version_tuple:
                print_status(package_name, "WARNING", f"Version {version} found. {min_version}+ recommended.")
                return False
        
        print_status(package_name, "OK", f"Version {version}")
        return True
    except ImportError:
        print_status(package_name, "ERROR", "Not installed")
        return False
    except Exception as e:
        print_status(package_name, "ERROR", f"Error checking version: {str(e)}")
        return False

def check_spacy_model():
    """Check if spaCy model is installed."""
    try:
        import spacy
        model_name = "en_core_web_sm"
        if spacy.util.is_package(model_name):
            nlp = spacy.load(model_name)
            print_status("spaCy model", "OK", f"{model_name} loaded")
            return True
        else:
            print_status("spaCy model", "ERROR", f"{model_name} not found")
            return False
    except ImportError:
        print_status("spaCy", "ERROR", "Not installed")
        return False
    except Exception as e:
        print_status("spaCy model", "ERROR", f"Error loading model: {str(e)}")
        return False

def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        # Check if Ollama is in PATH
        result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        if result.returncode != 0:
            print_status("Ollama", "ERROR", "Not found in PATH")
            return False
        
        ollama_path = result.stdout.strip()
        
        # Check if Ollama is running by listing models
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print_status("Ollama", "ERROR", "Installed but not running")
            return False
        
        # Check if any models are available
        models = [line.split()[0] for line in result.stdout.split("\n")[1:] if line.strip()]
        if not models:
            print_status("Ollama", "WARNING", "Running but no models installed")
            return False
        
        model_list = ", ".join(models[:3]) + ("..." if len(models) > 3 else "")
        print_status("Ollama", "OK", f"Running with models: {model_list}")
        return True
    except FileNotFoundError:
        print_status("Ollama", "ERROR", "Not installed")
        return False
    except Exception as e:
        print_status("Ollama", "ERROR", f"Error checking: {str(e)}")
        return False

def check_file_permissions():
    """Check if we have the required file permissions."""
    # Check logs directory
    try:
        log_path = Path("logs")
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)
            print_status("Logs directory", "OK", "Created logs directory")
        else:
            # Try to write to the logs directory
            test_file = log_path / "test.log"
            test_file.write_text("Test")
            test_file.unlink()
            print_status("Logs directory", "OK", "Write permissions verified")
    except Exception as e:
        print_status("Logs directory", "ERROR", f"Cannot write to logs: {str(e)}")
        return False

    # Check config directory
    try:
        config_path = Path("config")
        if not config_path.exists():
            print_status("Config directory", "ERROR", "Config directory does not exist")
            return False
        
        yaml_files = list(config_path.glob("*.yaml"))
        if not yaml_files:
            print_status("Config files", "ERROR", "No YAML config files found")
            return False
        
        print_status("Config directory", "OK", f"Found {len(yaml_files)} config file(s)")
    except Exception as e:
        print_status("Config directory", "ERROR", f"Error: {str(e)}")
        return False
    
    return True

def main():
    """Run all checks."""
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Print header
    print_colored(SENTINEL_ASCII, PURPLE, bold=True)
    print_colored("\nSentinelDocs System Check", BLUE, bold=True)
    print_colored("=" * 60, BLUE)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Working Directory: {os.getcwd()}")
    print_colored("=" * 60, BLUE)
    
    all_checks_passed = True
    
    # Check Python version
    print_header("Checking Python Environment")
    all_checks_passed &= check_python_version()
    
    # Check required packages
    print_header("Checking Required Packages")
    packages = [
        ("streamlit", "1.22.0"),
        ("langchain_community", "0.0.13"),
        ("langchain_ollama", "0.0.1"),
        ("fpdf", "1.7.2"),
        ("faiss", None),  # Changed from faiss_cpu to faiss
        ("fitz", None),  # PyMuPDF module name
        ("docx", None),  # python-docx module name
        ("spacy", "3.6.0"),
        ("sentence_transformers", "2.2.2"),
        ("numpy", None),
    ]
    
    for package, min_version in packages:
        all_checks_passed &= check_package(package, min_version)
    
    # Check spaCy model
    print_header("Checking NLP Resources")
    all_checks_passed &= check_spacy_model()
    
    # Check Ollama
    print_header("Checking Ollama")
    all_checks_passed &= check_ollama()
    
    # Check file permissions
    print_header("Checking File Permissions")
    all_checks_passed &= check_file_permissions()
    
    # Print summary
    print_header("Summary")
    if all_checks_passed:
        print_colored("✓ All checks passed! SentinelDocs should work correctly.", GREEN, bold=True)
        print_colored("\nTo start the application, run:", YELLOW)
        print_colored("  streamlit run app.py", BLUE)
    else:
        print_colored("✗ Some checks failed. Please fix the issues above.", RED, bold=True)
        print_colored("\nFor more information, check the documentation or open an issue:", YELLOW)
        print_colored("  https://github.com/sentineldocs/sentineldocs/issues", BLUE)
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 