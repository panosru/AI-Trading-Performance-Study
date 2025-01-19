# Makefile for managing the project

# Variables
PROJECT_DIR := $(shell pwd)
OUTPUT_DIR := output

# Create the output directory if it doesn't exist
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Install Python dependencies
install:
	@echo "Installing Python dependencies..."
	pip install pandas numpy matplotlib seaborn scipy statsmodels \
		scikit-learn tensorflow deap objproxies IPython dask jupyter \
		nbconvert tabulate

# Export a specific notebook to PDF
export_pdf: $(OUTPUT_DIR)
	@echo "Exporting $(file) to PDF..."
	@if [ -z "$(file)" ]; then \
		echo "Error: Please provide a notebook file to export using 'make export_pdf file=notebook.ipynb'"; \
		exit 1; \
	fi
	jupyter nbconvert --to pdf $(file) --output-dir=$(OUTPUT_DIR)

# Export a specific notebook to Markdown
export_md: $(OUTPUT_DIR)
	@echo "Exporting $(file) to Markdown..."
	@if [ -z "$(file)" ]; then \
		echo "Error: Please provide a notebook file to export using 'make export_md file=notebook.ipynb'"; \
		exit 1; \
	fi
	jupyter nbconvert --to markdown $(file) --output-dir=$(OUTPUT_DIR)

# Export a specific notebook to both PDF and Markdown
export_all: export_pdf export_md
	@echo "Exporting $(file) to both PDF and Markdown formats..."
	@if [ -z "$(file)" ]; then \
		echo "Error: Please provide a notebook file to export using 'make export_all file=notebook.ipynb'"; \
		exit 1; \
	fi
	make export_pdf file=$(file)
	make export_md file=$(file)

# Clean the contents of the output directory but not the directory itself
clean: $(OUTPUT_DIR)
	@echo "Cleaning the contents of the output directory..."
	rm -rf $(OUTPUT_DIR)/*

# Help message
help:
	@echo "Available commands:"
	@echo "  install       Install all dependencies"
	@echo "  export_pdf    Export a specific notebook to PDF format (usage: make export_pdf file=notebook.ipynb)"
	@echo "  export_md     Export a specific notebook to Markdown format (usage: make export_md file=notebook.ipynb)"
	@echo "  export_all    Export a specific notebook to both PDF and Markdown formats (usage: make export_all file=notebook.ipynb)"
	@echo "  clean         Clean the contents of the output directory"
