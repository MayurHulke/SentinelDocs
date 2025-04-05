.PHONY: install test run check clean

# Default target
all: check

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	@echo "Installation complete."

# Run tests
test:
	@echo "Running tests..."
	python run_tests.py

# Run the application
run:
	@echo "Starting SentinelDocs..."
	streamlit run app.py

# Run system checks
check:
	@echo "Running system checks..."
	python system_check.py

# Clean up temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
	@echo "Cleanup complete."

# Install for development
dev-install:
	@echo "Installing for development..."
	pip install -e .
	@echo "Development installation complete."

# Create a distribution package
dist:
	@echo "Creating distribution package..."
	python setup.py sdist bdist_wheel
	@echo "Distribution package created."

# Help
help:
	@echo "SentinelDocs Makefile"
	@echo "--------------------"
	@echo "make install     - Install dependencies"
	@echo "make test        - Run tests"
	@echo "make run         - Run the application"
	@echo "make check       - Run system checks"
	@echo "make clean       - Clean temporary files"
	@echo "make dev-install - Install for development"
	@echo "make dist        - Create a distribution package"
	@echo "make help        - Show this help message" 